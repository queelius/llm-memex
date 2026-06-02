"""Tests for memex configuration and database registry."""
import os

import pytest

from llm_memex.config import DatabaseRegistry, load_config


class TestLoadConfig:
    def test_from_yaml(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("databases:\n  main:\n    path: /tmp/main\nprimary: main\n")
        config = load_config(str(cfg))
        assert "main" in config["databases"]
        assert config["primary"] == "main"

    def test_single_db_env(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "single")
        monkeypatch.setenv("MEMEX_DATABASE_PATH", db_path)
        config = load_config(None)
        assert "default" in config["databases"]
        assert config["primary"] == "default"

    def test_missing_config_no_env(self):
        config = load_config("/nonexistent/config.yaml")
        assert config["databases"] == {}

    def test_sql_write_default_false(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("databases: {}\n")
        assert load_config(str(cfg))["sql_write"] is False

    def test_sql_write_env_override(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("databases: {}\nsql_write: false\n")
        monkeypatch.setenv("MEMEX_SQL_WRITE", "true")
        assert load_config(str(cfg))["sql_write"] is True

    def test_sql_write_yaml_true(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("databases: {}\nsql_write: true\n")
        config = load_config(str(cfg))
        assert config["sql_write"] is True

    def test_empty_yaml(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        config = load_config(str(cfg))
        assert config["databases"] == {}
        assert config["primary"] is None

    def test_none_path_no_env(self, monkeypatch):
        monkeypatch.delenv("MEMEX_DATABASE_PATH", raising=False)
        config = load_config(None)
        assert config["databases"] == {}

    def test_env_override_values(self, tmp_path, monkeypatch):
        """MEMEX_SQL_WRITE=1 and MEMEX_SQL_WRITE=yes also work."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("databases: {}\n")
        monkeypatch.setenv("MEMEX_SQL_WRITE", "1")
        assert load_config(str(cfg))["sql_write"] is True

        monkeypatch.setenv("MEMEX_SQL_WRITE", "yes")
        assert load_config(str(cfg))["sql_write"] is True

        monkeypatch.setenv("MEMEX_SQL_WRITE", "no")
        assert load_config(str(cfg))["sql_write"] is False

    def test_non_mapping_yaml_list_raises_clear_error(self, tmp_path):
        """A top-level YAML list (not a mapping) must raise a clear, file-named error.

        Regression for MCA-4: config.update(loaded) raised a cryptic
        ValueError/TypeError when safe_load returned a non-dict. The MCP server
        loads config at startup, so a malformed user config should fail with a
        message that names the file, not an opaque traceback.
        """
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- one\n- two\n")
        with pytest.raises(ValueError) as exc_info:
            load_config(str(cfg))
        msg = str(exc_info.value)
        assert str(cfg) in msg
        assert "mapping" in msg

    def test_non_mapping_yaml_scalar_raises_clear_error(self, tmp_path):
        """A top-level YAML scalar (e.g. truncated/garbage int) must also raise clearly."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("42\n")
        with pytest.raises(ValueError) as exc_info:
            load_config(str(cfg))
        msg = str(exc_info.value)
        assert str(cfg) in msg
        assert "mapping" in msg


class TestDatabaseRegistry:
    @staticmethod
    def _init_db(path):
        """Create an empty database so readonly opens succeed."""
        from llm_memex.db import Database
        Database(str(path)).close()

    def test_single_db(self, tmp_path):
        self._init_db(tmp_path / "main")
        reg = DatabaseRegistry({
            "databases": {"main": {"path": str(tmp_path / "main")}},
            "primary": "main",
            "sql_write": False,
        })
        db = reg.get_db("main")
        assert db is not None
        reg.close()

    def test_get_db_default(self, tmp_path):
        self._init_db(tmp_path / "main")
        reg = DatabaseRegistry({
            "databases": {"main": {"path": str(tmp_path / "main")}},
            "primary": "main",
            "sql_write": False,
        })
        assert reg.get_db(None) is reg.get_db("main")
        reg.close()

    def test_get_db_unknown(self, tmp_path):
        self._init_db(tmp_path / "main")
        reg = DatabaseRegistry({
            "databases": {"main": {"path": str(tmp_path / "main")}},
            "primary": "main",
            "sql_write": False,
        })
        with pytest.raises(ValueError, match="Unknown database"):
            reg.get_db("nope")
        reg.close()

    def test_all_dbs(self, tmp_path):
        self._init_db(tmp_path / "a")
        self._init_db(tmp_path / "b")
        reg = DatabaseRegistry({
            "databases": {
                "a": {"path": str(tmp_path / "a")},
                "b": {"path": str(tmp_path / "b")},
            },
            "primary": "a",
            "sql_write": False,
        })
        assert len(reg.all_dbs()) == 2
        reg.close()

    def test_close_clears_dbs(self, tmp_path):
        self._init_db(tmp_path / "main")
        reg = DatabaseRegistry({
            "databases": {"main": {"path": str(tmp_path / "main")}},
            "primary": "main",
            "sql_write": False,
        })
        reg.close()
        assert len(reg._dbs) == 0

    def test_expanduser_in_path(self, tmp_path):
        """Paths with ~ should be expanded."""
        self._init_db(tmp_path / "main")
        reg = DatabaseRegistry({
            "databases": {"main": {"path": str(tmp_path / "main")}},
            "primary": "main",
            "sql_write": False,
        })
        db = reg.get_db("main")
        assert db is not None
        reg.close()

    def test_sql_write_attribute(self, tmp_path):
        reg = DatabaseRegistry({
            "databases": {"main": {"path": str(tmp_path / "main")}},
            "primary": "main",
            "sql_write": True,
        })
        assert reg.sql_write is True
        reg.close()

    def test_empty_databases(self):
        reg = DatabaseRegistry({
            "databases": {},
            "primary": None,
            "sql_write": False,
        })
        assert len(reg.all_dbs()) == 0
        reg.close()

    def test_db_config_missing_path_raises_clear_error(self):
        """A databases entry without 'path' must raise a clear, name-bearing error.

        Regression for MCA-5: db_config["path"] raised a bare KeyError('path')
        that did not say which database was misconfigured.
        """
        with pytest.raises(ValueError) as exc_info:
            DatabaseRegistry({
                "databases": {"main": {"readonly": True}},
                "primary": "main",
                "sql_write": False,
            })
        msg = str(exc_info.value)
        assert "main" in msg
        assert "path" in msg

    def test_db_config_not_a_mapping_raises_clear_error(self):
        """A databases entry whose value is not a mapping must raise a clear error.

        Regression for MCA-5: a string/scalar db_config caused
        db_config["path"] to raise a confusing TypeError instead of naming
        the offending database.
        """
        with pytest.raises(ValueError) as exc_info:
            DatabaseRegistry({
                "databases": {"main": "/tmp/main"},
                "primary": "main",
                "sql_write": False,
            })
        msg = str(exc_info.value)
        assert "main" in msg
