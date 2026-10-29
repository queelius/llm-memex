"""Writers for the longecho ``site/`` viewer folder.

Browsers block ``fetch`` (and so sql.js's wasm fetch) on ``file://``, but
they still run classic ``<script src>`` tags. These helpers emit the viewer's
binary inputs as scripts that hand base64 payloads to globals the page
defines (see ``get_template(..., db_chunks=...)`` in ``html_template``):

- ``sql-wasm-binary.js`` sets ``window.LLM_MEMEX_WASM_B64``;
- ``db/000.js`` ... each call ``llmMemexDbChunk(<index>, "<base64>")`` with a
  slice of the gzipped database.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path
from typing import List

VENDORED_DIR = Path(__file__).parent / "vendored"

WASM_SCRIPT_NAME = "sql-wasm-binary.js"
DB_CHUNK_DIR = "db"


def db_chunk_name(index: int) -> str:
    """File name of chunk ``index`` inside ``db/`` (zero-padded to 3 digits)."""
    return f"{index:03d}.js"


def copy_sql_js(site_dir: Path) -> List[Path]:
    """Copy the vendored sql.js loader into ``site_dir``; return the file written."""
    site_dir.mkdir(parents=True, exist_ok=True)
    target = site_dir / "sql-wasm.js"
    shutil.copyfile(VENDORED_DIR / "sql-wasm.js", target)
    return [target]


def write_wasm_script(site_dir: Path) -> List[Path]:
    """Write ``sql-wasm-binary.js``, the vendored wasm as a base64 global;
    return the file written."""
    site_dir.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode((VENDORED_DIR / "sql-wasm.wasm").read_bytes()).decode("ascii")
    script = f"window.LLM_MEMEX_WASM_B64 = {json.dumps(b64)};\n"
    target = site_dir / WASM_SCRIPT_NAME
    target.write_bytes(script.encode("ascii"))
    return [target]


def write_db_chunks(db_gz: Path, site_dir: Path, chunk_bytes: int) -> List[Path]:
    """Split the gzipped DB into ``db/NNN.js`` chunk scripts; return them in
    index order (the chunk count is the list's length).

    The gzip *bytes* are sliced into ``chunk_bytes`` pieces before base64
    encoding, so the viewer joins the decoded slices in index order and
    decompresses the result as one gzip stream.
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    out_dir = site_dir / DB_CHUNK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    with open(db_gz, "rb") as fh:
        while True:
            piece = fh.read(chunk_bytes)
            if not piece:
                break
            b64 = base64.b64encode(piece).decode("ascii")
            script = f"llmMemexDbChunk({len(written)},{json.dumps(b64)});\n"
            target = out_dir / db_chunk_name(len(written))
            target.write_bytes(script.encode("ascii"))
            written.append(target)
    if not written:
        # The viewer reads DB_CHUNKS = 0 as "archive too large"; an empty
        # input is a caller bug, not that case.
        raise ValueError(f"{db_gz} is empty; nothing to chunk")
    return written
