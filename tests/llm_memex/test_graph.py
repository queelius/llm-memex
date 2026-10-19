"""Tests for schema v5→v6→v7 migrations and marginalia v2.

v5 introduced edges, trails, and notes v2 columns.
v6 removed trails (moved to memex).
v7 removed edges (cross-record graphs belong to memex too).

Covers:
- v4 → v7 migration preserves existing notes, adds v2 columns with defaults
- Unversioned (pre-v1) → v7 migration works (the _create_missing_tables path)
- Fresh DBs at v7 have no trails, no trail_steps, and no edges tables
- Notes v2: anchor fields, parent-child cascade, kind field
"""
import sqlite3
from datetime import datetime

import pytest

from llm_memex.db import Database, SCHEMA_VERSION
from llm_memex.models import Conversation, Message, text_block


@pytest.fixture
def db(tmp_db_path):
    return Database(tmp_db_path)


@pytest.fixture
def seeded_db(db):
    """DB with one conversation and two messages, for target ids."""
    now = datetime.now()
    conv = Conversation(
        id="c1", created_at=now, updated_at=now, title="Test",
    )
    conv.add_message(Message(id="m1", role="user", content=[text_block("hi")]))
    conv.add_message(Message(
        id="m2", role="assistant", content=[text_block("hello")], parent_id="m1",
    ))
    db.save_conversation(conv)
    return db


class TestSchema:
    def test_schema_version(self):
        assert SCHEMA_VERSION == 8

    def test_fresh_db_has_no_edges_table(self, db):
        """v7 removed edges; cross-record graphs live in memex, not here."""
        tables = {
            r["name"]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "edges" not in tables

    def test_fresh_db_has_no_trails_tables(self, db):
        """v6 removed trails; fresh database should not have them."""
        tables = {
            r["name"]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "trails" not in tables
        assert "trail_steps" not in tables

    def test_fresh_db_has_notes_v2_columns(self, db):
        cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(notes)").fetchall()}
        for c in ("kind", "anchor_start", "anchor_end", "anchor_hash", "parent_note_id"):
            assert c in cols

    def test_edge_methods_removed(self):
        """The edge-manipulation methods are gone from the Database class."""
        for attr in ("add_edge", "get_edges", "delete_edge", "prune_stale_edges"):
            assert not hasattr(Database, attr), (
                f"Database.{attr} should have been removed in schema v7"
            )


class TestMigration:
    def test_preserves_existing_notes(self, tmp_db_path, tmp_path):
        """An existing v4 DB should migrate forward without losing notes data."""
        import os
        # Build a v4 database shape manually.
        db_file = os.path.join(tmp_db_path, "conversations.db")
        raw = sqlite3.connect(db_file)
        raw.executescript(_V4_SCHEMA)
        raw.execute(
            "INSERT INTO conversations (id, created_at, updated_at) "
            "VALUES ('c1', '2024-01-01', '2024-01-01')"
        )
        raw.execute(
            "INSERT INTO notes (id, target_kind, conversation_id, message_id, "
            "text, created_at, updated_at) "
            "VALUES ('n1', 'conversation', 'c1', NULL, 'old note', "
            "'2024-01-01', '2024-01-01')"
        )
        raw.commit()
        raw.close()

        # Re-open via Database: migrations should run.
        db = Database(tmp_db_path)
        assert db.conn.execute("SELECT version FROM schema_version").fetchone()["version"] == SCHEMA_VERSION

        row = db.conn.execute(
            "SELECT id, text, kind, anchor_start, parent_note_id FROM notes WHERE id='n1'"
        ).fetchone()
        assert row["text"] == "old note"
        # New columns populated with their defaults
        assert row["kind"] == "freeform"
        assert row["anchor_start"] is None
        assert row["parent_note_id"] is None

        # v6: trails tables should not exist (dropped by migration)
        tables = {
            r["name"]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "trails" not in tables
        assert "trail_steps" not in tables

    def test_migration_is_idempotent_on_preexisting_unversioned_db(self, tmp_path):
        """An un-versioned DB (no schema_version table) bootstraps cleanly."""
        import os
        db_dir = tmp_path / "bare"
        db_dir.mkdir()
        db_file = os.path.join(str(db_dir), "conversations.db")
        # Create a minimal conversations table with no schema_version, simulating
        # an old install. _create_missing_tables plus all migrations should run.
        raw = sqlite3.connect(db_file)
        raw.executescript("""
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, title TEXT, source TEXT, model TEXT,
                summary TEXT, message_count INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
                starred_at DATETIME, pinned_at DATETIME, archived_at DATETIME,
                sensitive BOOLEAN NOT NULL DEFAULT 0,
                metadata JSON NOT NULL DEFAULT '{}'
            );
        """)
        raw.commit()
        raw.close()
        # Opening the DB should drive it all the way to the current version.
        db = Database(str(db_dir))
        assert db.conn.execute("SELECT version FROM schema_version").fetchone()["version"] == SCHEMA_VERSION


class TestMarginaliaV2:
    def test_anchor_fields_round_trip(self, seeded_db):
        nid = seeded_db.add_note(
            conversation_id="c1",
            message_id="m1",
            text="important point",
            kind="highlight",
            anchor_start=5,
            anchor_end=20,
            anchor_hash="sha256:abc",
        )
        row = seeded_db.conn.execute(
            "SELECT kind, anchor_start, anchor_end, anchor_hash FROM notes WHERE id = ?",
            (nid,),
        ).fetchone()
        assert row["kind"] == "highlight"
        assert row["anchor_start"] == 5
        assert row["anchor_end"] == 20
        assert row["anchor_hash"] == "sha256:abc"

    def test_kind_defaults_to_freeform(self, seeded_db):
        nid = seeded_db.add_note(
            conversation_id="c1", message_id="m1", text="just thinking",
        )
        row = seeded_db.conn.execute(
            "SELECT kind FROM notes WHERE id = ?", (nid,),
        ).fetchone()
        assert row["kind"] == "freeform"

    def test_parent_note_cascade(self, seeded_db):
        parent = seeded_db.add_note(
            conversation_id="c1", message_id="m1", text="parent note",
        )
        child = seeded_db.add_note(
            conversation_id="c1", message_id="m1", text="reply",
            parent_note_id=parent,
        )
        # Delete the parent — child should cascade
        seeded_db.delete_note(parent)
        row = seeded_db.conn.execute(
            "SELECT COUNT(*) as n FROM notes WHERE id = ?", (child,),
        ).fetchone()
        assert row["n"] == 0


# A v4-era schema replayed manually to seed the migration test above.
_V4_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version VALUES (4);
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
    tokenize='porter unicode61'
);
CREATE TABLE enrichments (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    type TEXT NOT NULL, value TEXT NOT NULL, source TEXT NOT NULL, confidence REAL,
    created_at DATETIME NOT NULL, PRIMARY KEY (conversation_id, type, value)
);
CREATE TABLE provenance (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL, source_file TEXT, source_id TEXT, source_hash TEXT,
    imported_at DATETIME NOT NULL, importer_version TEXT,
    PRIMARY KEY (conversation_id, source_type)
);
CREATE TABLE notes (
    id TEXT PRIMARY KEY,
    target_kind TEXT NOT NULL CHECK (target_kind IN ('message', 'conversation')),
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    message_id TEXT, text TEXT NOT NULL,
    created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL
);
CREATE VIRTUAL TABLE notes_fts USING fts5(
    note_id UNINDEXED, conversation_id UNINDEXED, message_id UNINDEXED, text,
    tokenize='porter unicode61'
);
CREATE INDEX idx_notes_target ON notes(conversation_id, message_id);
CREATE INDEX idx_notes_kind ON notes(target_kind);
"""
