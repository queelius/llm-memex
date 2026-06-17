"""Tests for the notes feature (schema v4 marginalia)."""
import os
import sqlite3
import subprocess
import sys

import pytest

from llm_memex.db import Database, SCHEMA_VERSION


def _get_tool_fn(server, name):
    """Extract a tool's underlying function from the FastMCP server."""
    tool = server._tool_manager._tools[name]
    return tool.fn


class TestSchemaV4:
    def test_schema_version_is_at_least_4(self):
        # Notes landed in v4; later versions must still include the table.
        assert SCHEMA_VERSION >= 4

    def test_fresh_database_has_notes_table(self, tmp_path):
        db = Database(str(tmp_path / "testdb"))
        tables = {
            r["name"]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "notes" in tables
        db.close()

    def test_fresh_database_has_notes_fts(self, tmp_path):
        db = Database(str(tmp_path / "testdb"))
        # FTS5 virtual tables appear in sqlite_master with type='table'
        tables = {
            r["name"]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "notes_fts" in tables
        db.close()

    def test_fresh_database_has_notes_indexes(self, tmp_path):
        db = Database(str(tmp_path / "testdb"))
        indexes = {
            r["name"]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_notes_target" in indexes
        assert "idx_notes_kind" in indexes
        db.close()

    def test_notes_column_shape(self, tmp_path):
        db = Database(str(tmp_path / "testdb"))
        cols = {
            r["name"]: r["type"]
            for r in db.conn.execute("PRAGMA table_info(notes)").fetchall()
        }
        assert "id" in cols
        assert "target_kind" in cols
        assert "conversation_id" in cols
        assert "message_id" in cols
        assert "text" in cols
        assert "created_at" in cols
        assert "updated_at" in cols
        db.close()


class TestSchemaV4EnrichmentMigration:
    def test_v3_enrichment_notes_move_to_notes_table(self, tmp_path):
        """Enrichment note rows at v3 must migrate into notes table at v4."""
        db_dir = tmp_path / "testdb"
        db_dir.mkdir()
        db_path = str(db_dir / "conversations.db")

        # Build a minimal v3-shaped database manually
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, title TEXT, source TEXT, model TEXT, summary TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
                starred_at DATETIME, pinned_at DATETIME, archived_at DATETIME,
                parent_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                sensitive BOOLEAN NOT NULL DEFAULT 0,
                metadata JSON NOT NULL DEFAULT '{}'
            );
            CREATE TABLE messages (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                id TEXT NOT NULL, role TEXT NOT NULL, parent_id TEXT, model TEXT,
                created_at DATETIME, sensitive BOOLEAN NOT NULL DEFAULT 0,
                content JSON NOT NULL, metadata JSON NOT NULL DEFAULT '{}',
                PRIMARY KEY (conversation_id, id)
            );
            CREATE TABLE tags (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                tag TEXT NOT NULL, PRIMARY KEY (conversation_id, tag)
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                conversation_id UNINDEXED, message_id UNINDEXED, text,
                tokenize = 'porter unicode61'
            );
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            CREATE TABLE enrichments (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                type TEXT NOT NULL, value TEXT NOT NULL, source TEXT NOT NULL,
                confidence REAL, created_at DATETIME NOT NULL,
                PRIMARY KEY (conversation_id, type, value)
            );
            CREATE TABLE provenance (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL, source_file TEXT, source_id TEXT,
                source_hash TEXT, imported_at DATETIME NOT NULL, importer_version TEXT,
                PRIMARY KEY (conversation_id, source_type)
            );
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (3)")
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES ('c1', 'Test', '2025-01-01 00:00:00', '2025-01-01 00:00:00')"
        )
        conn.execute(
            "INSERT INTO enrichments (conversation_id, type, value, source, created_at) "
            "VALUES ('c1', 'note', 'an important thought', 'user', '2025-01-01 00:00:00')"
        )
        conn.commit()
        conn.close()

        # Open via Database, triggering the v3 to v4 migration
        db = Database(str(db_dir))
        notes = db.conn.execute(
            "SELECT id, target_kind, conversation_id, text FROM notes "
            "WHERE conversation_id = 'c1'"
        ).fetchall()
        assert len(notes) == 1
        assert notes[0]["target_kind"] == "conversation"
        assert notes[0]["text"] == "an important thought"

        # Original enrichment note row should be gone
        remaining = db.conn.execute(
            "SELECT * FROM enrichments WHERE type = 'note'"
        ).fetchall()
        assert len(remaining) == 0

        # The note should be searchable via FTS5
        fts_rows = db.conn.execute(
            "SELECT text FROM notes_fts WHERE notes_fts MATCH 'important'"
        ).fetchall()
        assert len(fts_rows) == 1
        assert "important" in fts_rows[0]["text"]

        # Schema version should be at current (v4+)
        from llm_memex.db import SCHEMA_VERSION
        version_row = db.conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        assert version_row["version"] == SCHEMA_VERSION

        db.close()

    def test_v3_without_enrichment_notes_migrates_cleanly(self, tmp_path):
        """A v3 DB without any enrichment notes still migrates to v4 without errors."""
        db_dir = tmp_path / "testdb"
        db_dir.mkdir()
        db_path = str(db_dir / "conversations.db")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, title TEXT, source TEXT, model TEXT, summary TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
                starred_at DATETIME, pinned_at DATETIME, archived_at DATETIME,
                parent_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                sensitive BOOLEAN NOT NULL DEFAULT 0,
                metadata JSON NOT NULL DEFAULT '{}'
            );
            CREATE TABLE messages (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                id TEXT NOT NULL, role TEXT NOT NULL, parent_id TEXT, model TEXT,
                created_at DATETIME, sensitive BOOLEAN NOT NULL DEFAULT 0,
                content JSON NOT NULL, metadata JSON NOT NULL DEFAULT '{}',
                PRIMARY KEY (conversation_id, id)
            );
            CREATE TABLE tags (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                tag TEXT NOT NULL, PRIMARY KEY (conversation_id, tag)
            );
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                conversation_id UNINDEXED, message_id UNINDEXED, text,
                tokenize = 'porter unicode61'
            );
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            CREATE TABLE enrichments (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                type TEXT NOT NULL, value TEXT NOT NULL, source TEXT NOT NULL,
                confidence REAL, created_at DATETIME NOT NULL,
                PRIMARY KEY (conversation_id, type, value)
            );
            CREATE TABLE provenance (
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL, source_file TEXT, source_id TEXT,
                source_hash TEXT, imported_at DATETIME NOT NULL, importer_version TEXT,
                PRIMARY KEY (conversation_id, source_type)
            );
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (3)")
        conn.commit()
        conn.close()

        db = Database(str(db_dir))
        from llm_memex.db import SCHEMA_VERSION
        version_row = db.conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        assert version_row["version"] == SCHEMA_VERSION
        # notes table exists and is empty
        count = db.conn.execute("SELECT count(*) AS n FROM notes").fetchone()
        assert count["n"] == 0
        db.close()


@pytest.fixture
def db_with_conversation(tmp_path):
    """A database with one conversation and one message, ready for note CRUD."""
    db = Database(str(tmp_path / "testdb"))
    db.conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) "
        "VALUES ('c1', 'Test', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
    )
    db.conn.execute(
        "INSERT INTO messages (conversation_id, id, role, content, created_at) "
        "VALUES ('c1', 'm1', 'user', '[]', '2026-01-01 00:00:00')"
    )
    db.conn.commit()
    yield db
    db.close()


class TestDatabaseNotesCRUD:
    def test_add_conversation_note(self, db_with_conversation):
        db = db_with_conversation
        note_id = db.add_note(conversation_id="c1", text="conversation-level thought")
        assert note_id
        notes = db.get_notes(conversation_id="c1")
        assert len(notes) == 1
        assert notes[0]["text"] == "conversation-level thought"
        assert notes[0]["target_kind"] == "conversation"
        assert notes[0]["message_id"] is None
        assert notes[0]["conversation_id"] == "c1"

    def test_add_message_note(self, db_with_conversation):
        db = db_with_conversation
        note_id = db.add_note(
            conversation_id="c1", message_id="m1", text="a key moment"
        )
        notes = db.get_notes(conversation_id="c1", message_id="m1")
        assert len(notes) == 1
        assert notes[0]["target_kind"] == "message"
        assert notes[0]["message_id"] == "m1"
        assert notes[0]["id"] == note_id

    def test_add_note_supplied_id(self, db_with_conversation):
        db = db_with_conversation
        supplied = "my-custom-note-id"
        returned = db.add_note(
            conversation_id="c1", text="supplied id", note_id=supplied
        )
        assert returned == supplied
        notes = db.get_notes(conversation_id="c1")
        assert notes[0]["id"] == supplied

    def test_add_note_populates_fts(self, db_with_conversation):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="searchable quokka reference")
        fts = db.conn.execute(
            "SELECT text FROM notes_fts WHERE notes_fts MATCH 'quokka'"
        ).fetchall()
        assert len(fts) == 1
        assert "quokka" in fts[0]["text"]

    def test_update_note_changes_text(self, db_with_conversation):
        db = db_with_conversation
        note_id = db.add_note(conversation_id="c1", text="original")
        db.update_note(note_id, "revised")
        notes = db.get_notes(conversation_id="c1")
        assert notes[0]["text"] == "revised"

    def test_update_note_reindexes_fts(self, db_with_conversation):
        db = db_with_conversation
        note_id = db.add_note(conversation_id="c1", text="cat")
        db.update_note(note_id, "dog")
        cat_rows = db.conn.execute(
            "SELECT text FROM notes_fts WHERE notes_fts MATCH 'cat'"
        ).fetchall()
        dog_rows = db.conn.execute(
            "SELECT text FROM notes_fts WHERE notes_fts MATCH 'dog'"
        ).fetchall()
        assert cat_rows == []
        assert len(dog_rows) == 1

    def test_delete_note(self, db_with_conversation):
        db = db_with_conversation
        note_id = db.add_note(conversation_id="c1", text="to be deleted")
        db.delete_note(note_id)
        assert db.get_notes(conversation_id="c1") == []

    def test_delete_note_removes_from_fts(self, db_with_conversation):
        db = db_with_conversation
        note_id = db.add_note(conversation_id="c1", text="ephemeral")
        db.delete_note(note_id)
        fts = db.conn.execute(
            "SELECT text FROM notes_fts WHERE notes_fts MATCH 'ephemeral'"
        ).fetchall()
        assert fts == []

    def test_get_notes_by_target_kind(self, db_with_conversation):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="conv-level")
        db.add_note(conversation_id="c1", message_id="m1", text="msg-level")
        conv_notes = db.get_notes(conversation_id="c1", target_kind="conversation")
        msg_notes = db.get_notes(conversation_id="c1", target_kind="message")
        assert len(conv_notes) == 1
        assert len(msg_notes) == 1
        assert conv_notes[0]["text"] == "conv-level"
        assert msg_notes[0]["text"] == "msg-level"

    def test_search_notes_returns_matching(self, db_with_conversation):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="Kim was right about this")
        db.add_note(conversation_id="c1", text="completely unrelated thought")
        results = db.search_notes("Kim")
        assert len(results) == 1
        assert "Kim" in results[0]["text"]

    def test_search_notes_empty_query_returns_empty(self, db_with_conversation):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="whatever")
        results = db.search_notes("")
        assert results == []

    def test_search_notes_limit(self, db_with_conversation):
        db = db_with_conversation
        for i in range(10):
            db.add_note(conversation_id="c1", text=f"repeated keyword token {i}")
        results = db.search_notes("keyword", limit=3)
        assert len(results) == 3


class TestMCPAddNoteTool:
    def _setup_db(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES ('c1', 'Test', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
        )
        db.conn.execute(
            "INSERT INTO messages (conversation_id, id, role, content, created_at) "
            "VALUES ('c1', 'm1', 'user', '[]', '2026-01-01 00:00:00')"
        )
        db.conn.commit()
        return db

    def test_add_conversation_note_via_tool(self, tmp_db_path):
        from llm_memex.mcp import create_server
        db = self._setup_db(tmp_db_path)
        server = create_server(db=db, sql_write=True)
        add_note = _get_tool_fn(server, "add_note")
        result = add_note(conversation_id="c1", text="a thought")
        assert "note_id" in result
        assert result["target_kind"] == "conversation"
        notes = db.get_notes(conversation_id="c1")
        assert len(notes) == 1
        assert notes[0]["text"] == "a thought"
        db.close()

    def test_add_message_note_via_tool(self, tmp_db_path):
        from llm_memex.mcp import create_server
        db = self._setup_db(tmp_db_path)
        server = create_server(db=db, sql_write=True)
        add_note = _get_tool_fn(server, "add_note")
        result = add_note(conversation_id="c1", message_id="m1", text="key moment")
        assert result["target_kind"] == "message"
        notes = db.get_notes(conversation_id="c1", message_id="m1")
        assert len(notes) == 1
        db.close()

    def test_add_note_rejects_readonly(self, tmp_db_path):
        from llm_memex.mcp import create_server
        from fastmcp.exceptions import ToolError
        db = self._setup_db(tmp_db_path)
        db.close()
        db = Database(tmp_db_path, readonly=True)
        server = create_server(db=db, sql_write=False)
        add_note = _get_tool_fn(server, "add_note")
        with pytest.raises(ToolError, match="writes are disabled"):
            add_note(conversation_id="c1", text="will fail")
        db.close()

    def test_enrichment_note_type_rejected(self, tmp_db_path):
        from llm_memex.mcp import create_server
        from fastmcp.exceptions import ToolError
        db = self._setup_db(tmp_db_path)
        server = create_server(db=db, sql_write=True)
        update = _get_tool_fn(server, "update_conversations")
        with pytest.raises(ToolError, match="Invalid enrichment type"):
            update(
                ids=["c1"],
                add_enrichments=[{"type": "note", "value": "x", "source": "user"}],
            )
        db.close()

    def test_schema_resource_mentions_notes(self, tmp_db_path):
        db = Database(tmp_db_path)
        schema = db.get_schema()
        assert "notes" in schema
        assert "notes_fts" in schema
        db.close()

    def test_server_has_six_tools(self, tmp_db_path):
        from llm_memex.mcp import create_server
        db = Database(tmp_db_path)
        server = create_server(db=db, sql_write=True)
        # Verify add_note is discoverable
        add_note = _get_tool_fn(server, "add_note")
        assert add_note is not None
        db.close()


class TestNotesOrphanSurvive:
    def test_orphaned_note_set_null_on_conversation_delete(self, db_with_conversation):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="survive me")
        db.conn.execute("DELETE FROM conversations WHERE id = 'c1'")
        db.conn.commit()
        rows = db.conn.execute(
            "SELECT id, conversation_id, text FROM notes"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["text"] == "survive me"
        assert rows[0]["conversation_id"] is None


# ---------------------------------------------------------------------------
# CLI script tests for memex/scripts/note.py
# ---------------------------------------------------------------------------

class TestNoteScript:
    """Tests for the note CLI script using _cmd_run (same pattern as TestCLIRun)."""

    @staticmethod
    def _db_dir(db):
        """Get the directory path that _open_db / Database() expects."""
        return os.path.dirname(db.db_path)

    def _run(self, db, script_args, apply=False, capsys=None):
        """Run the note script via _cmd_run with the given arguments."""
        import argparse as _ap
        from llm_memex.cli import _cmd_run

        args = _ap.Namespace(
            name="note",
            list=False,
            apply=apply,
            verbose=False,
            db=self._db_dir(db),
        )
        _cmd_run(args, script_args)
        if capsys is not None:
            return capsys.readouterr().out
        return ""

    def test_note_script_discovered(self):
        """note script is discoverable via the framework."""
        from llm_memex.scripts import load_script
        mod = load_script("note")
        assert hasattr(mod, "register_args")
        assert hasattr(mod, "run")
        assert mod.__doc__

    def test_add_dry_run(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["add", "--conv", "c1", "my", "dry", "note"],
            apply=False, capsys=capsys,
        )
        assert "DRY" in out
        assert "my dry note" in out
        # Nothing written
        assert db.get_notes(conversation_id="c1") == []

    def test_add_apply(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["add", "--conv", "c1", "a", "real", "note"],
            apply=True, capsys=capsys,
        )
        assert "Added" in out
        notes = db.get_notes(conversation_id="c1")
        assert len(notes) == 1
        assert notes[0]["text"] == "a real note"
        assert notes[0]["target_kind"] == "conversation"

    def test_add_message_level(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["add", "--conv", "c1", "--msg", "m1", "msg", "note"],
            apply=True, capsys=capsys,
        )
        assert "message" in out
        notes = db.get_notes(conversation_id="c1", message_id="m1")
        assert len(notes) == 1
        assert notes[0]["target_kind"] == "message"

    def test_add_missing_conv(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["add", "orphan", "text"],
            apply=True, capsys=capsys,
        )
        assert "Error" in out

    def test_add_missing_text(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["add", "--conv", "c1"],
            apply=True, capsys=capsys,
        )
        assert "Error" in out

    def test_list(self, db_with_conversation, capsys):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="note one")
        db.add_note(conversation_id="c1", text="note two")
        out = self._run(
            db, ["list", "--conv", "c1"],
            apply=False, capsys=capsys,
        )
        assert "note one" in out
        assert "note two" in out
        assert "2 note(s)" in out

    def test_list_empty(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["list", "--conv", "c1"],
            apply=False, capsys=capsys,
        )
        assert "No notes found" in out

    def test_list_missing_conv(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["list"],
            apply=False, capsys=capsys,
        )
        assert "Error" in out

    def test_list_filters_by_msg(self, db_with_conversation, capsys):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="conv-level")
        db.add_note(conversation_id="c1", message_id="m1", text="msg-level")
        out = self._run(
            db, ["list", "--conv", "c1", "--msg", "m1"],
            apply=False, capsys=capsys,
        )
        assert "msg-level" in out
        assert "conv-level" not in out
        assert "1 note(s)" in out

    def test_search(self, db_with_conversation, capsys):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="quantum entanglement idea")
        db.add_note(conversation_id="c1", text="grocery list")
        out = self._run(
            db, ["search", "quantum"],
            apply=False, capsys=capsys,
        )
        assert "quantum" in out
        assert "1 result(s)" in out

    def test_search_no_results(self, db_with_conversation, capsys):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="something")
        out = self._run(
            db, ["search", "nonexistent"],
            apply=False, capsys=capsys,
        )
        assert "No matching notes found" in out

    def test_search_missing_query(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["search"],
            apply=False, capsys=capsys,
        )
        assert "Error" in out

    def test_search_multi_word_query(self, db_with_conversation, capsys):
        db = db_with_conversation
        db.add_note(conversation_id="c1", text="deep learning transformers note")
        out = self._run(
            db, ["search", "deep", "learning"],
            apply=False, capsys=capsys,
        )
        assert "deep learning" in out

    def test_delete_dry_run(self, db_with_conversation, capsys):
        db = db_with_conversation
        note_id = db.add_note(conversation_id="c1", text="ephemeral")
        out = self._run(
            db, ["delete", note_id],
            apply=False, capsys=capsys,
        )
        assert "DRY" in out
        # Not deleted
        assert len(db.get_notes(conversation_id="c1")) == 1

    def test_delete_apply(self, db_with_conversation, capsys):
        db = db_with_conversation
        note_id = db.add_note(conversation_id="c1", text="ephemeral")
        out = self._run(
            db, ["delete", note_id],
            apply=True, capsys=capsys,
        )
        assert "Deleted" in out
        assert db.get_notes(conversation_id="c1") == []

    def test_delete_missing_id(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["delete"],
            apply=True, capsys=capsys,
        )
        assert "Error" in out

    def test_delete_nonexistent(self, db_with_conversation, capsys):
        db = db_with_conversation
        out = self._run(
            db, ["delete", "bogus-id"],
            apply=True, capsys=capsys,
        )
        assert "not found" in out


# ---------------------------------------------------------------------------
# Exporter notes tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_notes(tmp_path):
    """A database with one conversation, one message, and notes on each."""
    db = Database(str(tmp_path / "export-notes-db"))
    db.conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) "
        "VALUES ('c1', 'Noted Conv', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
    )
    db.conn.execute(
        "INSERT INTO messages (conversation_id, id, role, content, created_at) "
        "VALUES ('c1', 'm1', 'user', '[{\"type\":\"text\",\"text\":\"hello\"}]', "
        "'2026-01-01 00:00:00')"
    )
    db.conn.commit()
    db.add_note(conversation_id="c1", text="conv-level annotation")
    db.add_note(conversation_id="c1", message_id="m1", text="msg-level annotation")
    yield db
    db.close()


def _load_conv(db):
    """Load the single test conversation from the db."""
    return db.load_conversation("c1")


class TestMarkdownExportNotes:
    def test_markdown_includes_conv_note(self, db_with_notes, tmp_path):
        from llm_memex.exporters.markdown import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.md")
        export([conv], out, db=db_with_notes, include_notes=True)
        text = open(out).read()
        assert "conv-level annotation" in text

    def test_markdown_includes_msg_note(self, db_with_notes, tmp_path):
        from llm_memex.exporters.markdown import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.md")
        export([conv], out, db=db_with_notes, include_notes=True)
        text = open(out).read()
        assert "msg-level annotation" in text

    def test_markdown_notes_are_blockquotes(self, db_with_notes, tmp_path):
        from llm_memex.exporters.markdown import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.md")
        export([conv], out, db=db_with_notes, include_notes=True)
        text = open(out).read()
        assert "> **Note:** conv-level annotation" in text
        assert "> **Note:** msg-level annotation" in text

    def test_markdown_no_notes_flag(self, db_with_notes, tmp_path):
        from llm_memex.exporters.markdown import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.md")
        export([conv], out, db=db_with_notes, include_notes=False)
        text = open(out).read()
        assert "conv-level annotation" not in text
        assert "msg-level annotation" not in text

    def test_markdown_no_db_still_works(self, db_with_notes, tmp_path):
        from llm_memex.exporters.markdown import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.md")
        export([conv], out)  # no db, no include_notes -- should not crash
        text = open(out).read()
        assert "Noted Conv" in text
        assert "annotation" not in text


class TestJSONExportNotes:
    def test_json_includes_conv_notes(self, db_with_notes, tmp_path):
        import json
        from llm_memex.exporters.json_export import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.json")
        export([conv], out, db=db_with_notes, include_notes=True)
        data = json.loads(open(out).read())
        assert len(data) == 1
        assert "notes" in data[0]
        conv_notes = data[0]["notes"]
        assert any(n["text"] == "conv-level annotation" for n in conv_notes)

    def test_json_includes_msg_notes(self, db_with_notes, tmp_path):
        import json
        from llm_memex.exporters.json_export import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.json")
        export([conv], out, db=db_with_notes, include_notes=True)
        data = json.loads(open(out).read())
        msg = data[0]["messages"][0]
        assert "notes" in msg
        assert any(n["text"] == "msg-level annotation" for n in msg["notes"])

    def test_json_no_notes_flag(self, db_with_notes, tmp_path):
        import json
        from llm_memex.exporters.json_export import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.json")
        export([conv], out, db=db_with_notes, include_notes=False)
        data = json.loads(open(out).read())
        assert "notes" not in data[0]
        assert "notes" not in data[0]["messages"][0]

    def test_json_no_db_still_works(self, db_with_notes, tmp_path):
        import json
        from llm_memex.exporters.json_export import export
        conv = _load_conv(db_with_notes)
        out = str(tmp_path / "out.json")
        export([conv], out)
        data = json.loads(open(out).read())
        assert data[0]["id"] == "c1"
        assert "notes" not in data[0]


class TestArkivExportNotes:
    """Test arkiv export notes via _build_records (avoids __version__ import
    shadowing issue when running under tests/llm_memex/)."""

    def _import_build_records(self):
        """Import _build_records from arkiv_export, working around the
        tests/llm_memex/ __init__.py shadowing the real llm_memex package."""
        import importlib.util
        mod_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "llm_memex", "exporters", "arkiv_export.py"
        )
        spec = importlib.util.spec_from_file_location("arkiv_export", mod_path,
                                                       submodule_search_locations=[])
        mod = importlib.util.module_from_spec(spec)
        # Patch llm_memex.__version__ so the module can load
        import llm_memex as _memex_pkg
        if not hasattr(_memex_pkg, "__version__"):
            _memex_pkg.__version__ = "0.0.0-test"
        spec.loader.exec_module(mod)
        return mod._build_records

    def test_arkiv_includes_msg_notes(self, db_with_notes):
        build = self._import_build_records()
        conv = _load_conv(db_with_notes)
        records = build([conv], include_notes=True, db=db_with_notes)
        # One message record + one conversation-level note carrier (LLM-3:
        # conv-level notes used to be silently dropped, leaving just the
        # message record).
        msg_records = [r for r in records if r["metadata"].get("message_id")]
        assert len(msg_records) == 1
        meta = msg_records[0]["metadata"]
        assert "notes" in meta
        assert any(n["text"] == "msg-level annotation" for n in meta["notes"])
        carriers = [
            r for r in records
            if r["metadata"].get("target_kind") == "conversation"
        ]
        assert len(carriers) == 1
        assert any(
            n["text"] == "conv-level annotation"
            for n in carriers[0]["metadata"]["conversation_notes"]
        )

    def test_arkiv_no_notes_flag(self, db_with_notes):
        build = self._import_build_records()
        conv = _load_conv(db_with_notes)
        records = build([conv], include_notes=False, db=db_with_notes)
        assert len(records) == 1
        assert "notes" not in records[0]["metadata"]

    def test_arkiv_no_db_still_works(self, db_with_notes):
        build = self._import_build_records()
        conv = _load_conv(db_with_notes)
        records = build([conv])
        assert len(records) == 1
        assert "notes" not in records[0]["metadata"]
