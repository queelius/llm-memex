"""Configuration and multi-database registry."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from llm_memex.db import Database


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load memex configuration from YAML file, env vars, or defaults.

    Priority: env vars override YAML values.
    If no config file exists, falls back to MEMEX_DATABASE_PATH env var.
    """
    config: Dict[str, Any] = {"databases": {}, "primary": None, "sql_write": False}

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path}: top-level YAML must be a mapping")
        config.update(loaded)
    elif env_path := os.environ.get("MEMEX_DATABASE_PATH"):
        config["databases"] = {"default": {"path": env_path}}
        config["primary"] = "default"

    # Env override for sql_write
    if os.environ.get("MEMEX_SQL_WRITE", "").lower() in ("true", "1", "yes"):
        config["sql_write"] = True

    return config


class DatabaseRegistry:
    """Manages multiple named Database instances."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.primary = config.get("primary")
        self.sql_write = config.get("sql_write", False)
        self._dbs: Dict[str, Database] = {}
        readonly = not self.sql_write
        for name, db_config in config.get("databases", {}).items():
            if not isinstance(db_config, dict):
                raise ValueError(
                    f"database '{name}' config must be a mapping, "
                    f"got {type(db_config).__name__}"
                )
            if "path" not in db_config:
                raise ValueError(f"database '{name}' is missing required 'path'")
            path = os.path.expanduser(db_config["path"])
            self._dbs[name] = Database(path, readonly=readonly)

    def get_db(self, name: str | None = None) -> Database:
        """Get a database by name, or the primary database if name is None."""
        if name is None:
            name = self.primary
            if name is None:
                available = ", ".join(sorted(self._dbs)) or "none"
                raise ValueError(
                    "No database specified and no primary database is "
                    f"configured (available: {available}). Pass a name or set "
                    "a primary in the config."
                )
        if name not in self._dbs:
            raise ValueError(f"Unknown database: {name}")
        return self._dbs[name]

    def all_dbs(self) -> Dict[str, Database]:
        """Return all registered databases."""
        return dict(self._dbs)

    def close(self):
        """Close all database connections."""
        for db in self._dbs.values():
            db.close()
        self._dbs.clear()
