"""Scripts framework -- discovery and runner utilities.

Convention: each script is a Python module with:
    register_args(parser)  -- add script-specific CLI arguments
    run(db, args, apply)   -- execute the script, return stats dict

Discovery is **static**: it never imports (executes) candidate modules. It
parses each file with ``ast`` to confirm the convention interface and read the
module docstring. Only the single module the user actually invokes is imported
(executed), in ``load_script``. This prevents arbitrary code execution from any
``.py`` file dropped into ``~/.memex/scripts/`` merely from listing/discovery.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _builtin_dir() -> Path:
    return Path(__file__).parent


def _user_dir() -> Path:
    return Path.home() / ".memex" / "scripts"


def _load_module(name: str, path: Path):
    """Import (execute) a Python module from a file path.

    Registers the module in sys.modules so @dataclass and similar
    metaclass-based decorators can look up their own module.

    This is the ONLY place a script's top-level code runs. It is called for the
    single module the user invokes, never during discovery/listing.
    """
    mod_name = f"llm_memex_script_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _static_script_info(path: Path) -> Dict[str, Any] | None:
    """Inspect a candidate script file WITHOUT executing it.

    Parses the source with ``ast`` and returns
    ``{"path": Path, "description": str, "module": None}`` if the file defines
    both ``register_args`` and ``run`` at module top level, else ``None``.

    The ``"module"`` key is intentionally ``None``: the module is not imported
    during discovery. ``load_script`` imports it lazily when invoked.
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))

    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not ({"register_args", "run"} <= defined):
        return None

    doc = ast.get_docstring(tree) or ""
    return {
        "path": path,
        "description": doc.strip().split("\n")[0].strip(),
        "module": None,  # lazily imported by load_script; never exec'd on discovery
    }


def discover_scripts() -> Dict[str, Dict[str, Any]]:
    """Discover available scripts from built-in and user directories.

    Returns dict mapping script name to {"path": Path, "description": str, "module": mod}.
    User scripts shadow built-in scripts of the same name.

    Discovery is static (no import/execution). The ``"module"`` value is ``None``
    until the script is actually invoked via ``load_script``. Files that fail to
    parse are skipped and logged (not silently swallowed).
    """
    scripts: Dict[str, Dict[str, Any]] = {}

    for d in [_builtin_dir(), _user_dir()]:
        if not d.exists():
            continue
        for py_file in sorted(d.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            name = py_file.stem
            try:
                info = _static_script_info(py_file)
            except Exception:
                logger.warning("Skipping unparseable script %s", py_file, exc_info=True)
                continue
            if info is None:
                continue
            scripts[name] = info

    return scripts


def load_script(name: str):
    """Load a script module by name.

    This imports (executes) the module's top-level code. It is the only path
    that runs a script module; discovery/listing never does.

    Raises ValueError if script is not found.
    """
    scripts = discover_scripts()
    if name not in scripts:
        raise ValueError(f"Script '{name}' not found. Use 'llm-memex run --list' to see available scripts.")
    info = scripts[name]
    if info.get("module") is None:
        info["module"] = _load_module(name, info["path"])
    return info["module"]
