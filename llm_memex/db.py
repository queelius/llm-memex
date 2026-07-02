"""SQLite database layer for memex. Raw sqlite3 — no ORM."""
from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_memex.models import Conversation, Message

SCHEMA_VERSION = 7

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, title TEXT, source TEXT, model TEXT, summary TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
    starred_at DATETIME, pinned_at DATETIME, archived_at DATETIME,
    parent_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    sensitive BOOLEAN NOT NULL DEFAULT 0,
    metadata JSON NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS messages (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    id TEXT NOT NULL, role TEXT NOT NULL, parent_id TEXT, model TEXT,
    created_at DATETIME, sensitive BOOLEAN NOT NULL DEFAULT 0,
    content JSON NOT NULL, metadata JSON NOT NULL DEFAULT '{}',
    PRIMARY KEY (conversation_id, id)
);
CREATE TABLE IF NOT EXISTS tags (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tag TEXT NOT NULL, PRIMARY KEY (conversation_id, tag)
);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    conversation_id UNINDEXED, message_id UNINDEXED, text,
    tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS enrichments (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (conversation_id, type, value)
);
CREATE TABLE IF NOT EXISTS provenance (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_file TEXT,
    source_id TEXT,
    source_hash TEXT,
    imported_at DATETIME NOT NULL,
    importer_version TEXT,
    PRIMARY KEY (conversation_id, source_type)
);
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    target_kind TEXT NOT NULL CHECK (target_kind IN ('message', 'conversation')),
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    message_id TEXT,
    text TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    -- Marginalia v2 additions (schema v5):
    kind TEXT NOT NULL DEFAULT 'freeform',
    anchor_start INTEGER,
    anchor_end INTEGER,
    anchor_hash TEXT,
    parent_note_id TEXT REFERENCES notes(id) ON DELETE CASCADE
);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    note_id UNINDEXED, conversation_id UNINDEXED, message_id UNINDEXED, text,
    tokenize = 'porter unicode61'
);
CREATE INDEX IF NOT EXISTS idx_notes_target ON notes(conversation_id, message_id);
CREATE INDEX IF NOT EXISTS idx_notes_target_kind ON notes(target_kind);
CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(kind);
CREATE INDEX IF NOT EXISTS idx_notes_parent ON notes(parent_note_id) WHERE parent_note_id IS NOT NULL;

-- (edges table removed in schema v7: cross-record relationships belong to
-- the federation layer, not to individual archives. See memex/memex
-- docs for the new home.)

CREATE INDEX IF NOT EXISTS idx_conversations_source ON conversations(source);
CREATE INDEX IF NOT EXISTS idx_conversations_model ON conversations(model);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversations_starred ON conversations(starred_at) WHERE starred_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_conversations_pinned ON conversations(pinned_at) WHERE pinned_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_conversations_archived ON conversations(archived_at) WHERE archived_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(conversation_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_enrichments_type ON enrichments(type);
CREATE INDEX IF NOT EXISTS idx_enrichments_source ON enrichments(source);
CREATE INDEX IF NOT EXISTS idx_conversations_parent ON conversations(parent_conversation_id) WHERE parent_conversation_id IS NOT NULL;
"""


def _migrate_to_v2(conn):
    """Add enrichments and provenance tables, backfill provenance from conversations.source."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS enrichments (
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL,
            created_at DATETIME NOT NULL,
            PRIMARY KEY (conversation_id, type, value)
        );
        CREATE TABLE IF NOT EXISTS provenance (
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            source_type TEXT NOT NULL,
            source_file TEXT,
            source_id TEXT,
            source_hash TEXT,
            imported_at DATETIME NOT NULL,
            importer_version TEXT,
            PRIMARY KEY (conversation_id, source_type)
        );
        CREATE INDEX IF NOT EXISTS idx_enrichments_type ON enrichments(type);
        CREATE INDEX IF NOT EXISTS idx_enrichments_source ON enrichments(source);
    """)
    # Backfill provenance from conversations.source
    conn.execute(
        "INSERT OR IGNORE INTO provenance "
        "(conversation_id, source_type, imported_at) "
        "SELECT id, source, created_at FROM conversations "
        "WHERE source IS NOT NULL"
    )
    conn.commit()


def _migrate_to_v3(conn):
    """Add parent_conversation_id column to conversations."""
    # Guarded against duplicates: a pre-existing (un-versioned) DB may already
    # have this column from SCHEMA_SQL via _create_missing_tables, and the
    # guard also makes the migration idempotent if it is ever re-run.
    try:
        conn.execute(
            "ALTER TABLE conversations ADD COLUMN parent_conversation_id "
            "TEXT REFERENCES conversations(id) ON DELETE SET NULL"
        )
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e):
            raise
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_parent "
        "ON conversations(parent_conversation_id) "
        "WHERE parent_conversation_id IS NOT NULL"
    )
    conn.commit()


def _migrate_to_v4(conn):
    """Add notes and notes_fts tables, migrate existing enrichment 'note' entries."""
    import uuid as _uuid

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            target_kind TEXT NOT NULL CHECK (target_kind IN ('message', 'conversation')),
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            message_id TEXT,
            text TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
            note_id UNINDEXED, conversation_id UNINDEXED, message_id UNINDEXED, text,
            tokenize = 'porter unicode61'
        );
        CREATE INDEX IF NOT EXISTS idx_notes_target ON notes(conversation_id, message_id);
        CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(target_kind);
    """)

    # Migrate any existing conversation-level notes from enrichments into notes table
    rows = conn.execute(
        "SELECT conversation_id, value, created_at FROM enrichments WHERE type = 'note'"
    ).fetchall()
    for row in rows:
        note_id = str(_uuid.uuid4())
        conv_id = row["conversation_id"]
        value = row["value"]
        created_at = row["created_at"]
        conn.execute(
            "INSERT INTO notes (id, target_kind, conversation_id, message_id, text, "
            "created_at, updated_at) VALUES (?, 'conversation', ?, NULL, ?, ?, ?)",
            (note_id, conv_id, value, created_at, created_at),
        )
        conn.execute(
            "INSERT INTO notes_fts (note_id, conversation_id, message_id, text) "
            "VALUES (?, ?, NULL, ?)",
            (note_id, conv_id, value),
        )
    conn.execute("DELETE FROM enrichments WHERE type = 'note'")
    conn.commit()


def _migrate_to_v5(conn):
    """Graph spine + marginalia v2.

    Historical migration: v5 introduced the edges and trails tables and the
    notes v2 columns. Trails were removed in v6 and edges in v7 — both moved
    to the memex federation layer. This migration still creates edges
    (and trails) for databases following the 4→5→6→7 chain; those tables
    are dropped by subsequent migrations. Fresh DBs bootstrap directly to
    the current version and never materialize them.

    Additive to the live schema:
    - ALTER TABLE notes to add: kind, anchor_start, anchor_end, anchor_hash,
      parent_note_id. All are nullable or have defaults so existing rows are
      unaffected.
    """
    # Notes v2 columns. SQLite ALTER TABLE ADD COLUMN restrictions:
    #  - NOT NULL requires a DEFAULT (kind has 'freeform')
    #  - FK REFERENCES is allowed when existing rows get NULL
    # Guarded against duplicates because pre-existing (un-versioned) DBs
    # may have created the notes table directly from SCHEMA_SQL before
    # this migration runs — see _create_missing_tables.
    _new_note_columns = [
        "ALTER TABLE notes ADD COLUMN kind TEXT NOT NULL DEFAULT 'freeform'",
        "ALTER TABLE notes ADD COLUMN anchor_start INTEGER",
        "ALTER TABLE notes ADD COLUMN anchor_end INTEGER",
        "ALTER TABLE notes ADD COLUMN anchor_hash TEXT",
        "ALTER TABLE notes ADD COLUMN parent_note_id TEXT "
        "REFERENCES notes(id) ON DELETE CASCADE",
    ]
    for stmt in _new_note_columns:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e):
                raise

    # Rename/add indexes. idx_notes_kind previously indexed target_kind in
    # the v4 SCHEMA_SQL but that index name is now used for the new 'kind'
    # column. We preserve the old target_kind index under a new name.
    conn.executescript("""
        DROP INDEX IF EXISTS idx_notes_kind;
        CREATE INDEX IF NOT EXISTS idx_notes_target_kind
            ON notes(target_kind);
        CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(kind);
        CREATE INDEX IF NOT EXISTS idx_notes_parent
            ON notes(parent_note_id)
            WHERE parent_note_id IS NOT NULL;
    """)

    # (edges table was introduced in v5 but removed in v7; DBs that take
    # the 4 → 5 → 6 → 7 migration path transiently create it below and
    # _migrate_to_v7 drops it. Fresh DBs bootstrap directly to v7 and
    # never create it.)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            from_kind TEXT NOT NULL,
            from_id TEXT NOT NULL,
            to_kind TEXT NOT NULL,
            to_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            metadata JSON NOT NULL DEFAULT '{}',
            created_at DATETIME NOT NULL
        );
    """)

    # Trails and trail steps.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trails (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            metadata JSON NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS trail_steps (
            trail_id TEXT NOT NULL REFERENCES trails(id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            target_kind TEXT NOT NULL,
            target_id TEXT NOT NULL,
            annotation TEXT,
            PRIMARY KEY (trail_id, position)
        );
        CREATE INDEX IF NOT EXISTS idx_trail_steps_target
            ON trail_steps(target_kind, target_id);
    """)

    conn.commit()


def _migrate_to_v6(conn):
    """Drop trails and trail_steps.

    Trails moved to the memex layer (cross-archive coordinator).
    Each archive is now domain-focused; cross-archive operations like
    trails live above in memex. See ~/github/beta/memex/docs/
    for the new design.

    Existing trail data is discarded. This is acceptable because trails
    were brand new in v5 and not yet in production use.
    """
    conn.executescript("""
        DROP INDEX IF EXISTS idx_trail_steps_target;
        DROP TABLE IF EXISTS trail_steps;
        DROP TABLE IF EXISTS trails;
    """)
    conn.commit()


def _migrate_to_v7(conn):
    """Drop edges table and its indexes.

    The edges table was added in v5 alongside trails, intended as a generic
    polymorphic relationship store. Trails moved to memex in v6, but
    edges stayed — dormant. Nothing in llm-memex actually writes to it
    (no importer, MCP tool, or CLI), so it has been dead code.

    Per the ecosystem contract (see ~/github/memex/CLAUDE.md): archives do
    not build graphs; cross-record relationships belong in the federation
    layer. v7 aligns the archive with that contract by removing the unused
    infrastructure.

    Any existing edges are discarded. The feature has never been exposed
    through user-facing tools, so no production data is expected to exist.
    """
    conn.executescript("""
        DROP INDEX IF EXISTS idx_edges_from;
        DROP INDEX IF EXISTS idx_edges_to;
        DROP INDEX IF EXISTS idx_edges_type;
        DROP INDEX IF EXISTS idx_edges_unique;
        DROP TABLE IF EXISTS edges;
    """)
    conn.commit()


MIGRATIONS: Dict[int, callable] = {
    1: _migrate_to_v2,
    2: _migrate_to_v3,
    3: _migrate_to_v4,
    4: _migrate_to_v5,
    5: _migrate_to_v6,
    6: _migrate_to_v7,
}


def _dict_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def _fmt_dt(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards (%, _) and the escape char itself."""
    return (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a user query for FTS5 MATCH: strip quotes, quote each token, join with OR.

    Returns empty string if no usable tokens remain.
    """
    sanitized = query.replace('"', "").replace("'", "")
    tokens = sanitized.split()
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens if t)


def _encode_cursor(updated_at: str, id: str) -> str:
    return base64.b64encode(
        json.dumps({"u": updated_at, "id": id}).encode()
    ).decode()


def _decode_cursor(cursor: str) -> tuple[str, str]:
    d = json.loads(base64.b64decode(cursor.encode()).decode())
    return d["u"], d["id"]


def _message_row(conv_id: str, msg: Message) -> tuple:
    """Build the column-ordered value tuple for an INSERT into messages.

    Column order: conversation_id, id, role, parent_id, model, created_at,
    sensitive, content, metadata.
    """
    return (
        conv_id, msg.id, msg.role, msg.parent_id, msg.model,
        _fmt_dt(msg.created_at), int(msg.sensitive),
        json.dumps(msg.content), json.dumps(msg.metadata),
    )


# SQLite authorizer action codes that create, drop, alter, or modify data.
# Database._authorize denies these on read-only connections. An authorizer is a
# stronger guard than PRAGMA query_only, which agent-supplied SQL can flip off.
_WRITE_ACTIONS = frozenset(
    getattr(sqlite3, _name) for _name in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE",
        "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_VIEW", "SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_VTABLE",
        "SQLITE_DROP_TABLE", "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_INDEX", "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TRIGGER", "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_VIEW", "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_VTABLE",
        "SQLITE_ALTER_TABLE",
    ) if hasattr(sqlite3, _name)
)


class Database:
    def __init__(self, path: str, readonly: bool = False):
        if path == ":memory:":
            self.db_path = ":memory:"
        else:
            db_dir = Path(path)
            if readonly and not db_dir.exists():
                raise FileNotFoundError(f"Database not found: {path}")
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_dir / "conversations.db")
        self.readonly = readonly
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = _dict_factory
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()
        if readonly:
            self.conn.execute("PRAGMA query_only=ON")
        # Install last, so schema creation/migration above is not blocked. The
        # authorizer reads self.readonly live, so flipping it later (the MCP
        # server does this) immediately tightens enforcement.
        self.conn.set_authorizer(self._authorize)

    def _authorize(self, action, arg1, arg2, db_name, source):
        """SQLite authorizer. Denies ATTACH/DETACH on every connection (nothing
        in llm-memex needs them, and they would let agent-supplied SQL reach
        arbitrary database files). On a read-only connection it additionally
        denies all writes and any attempt to turn query_only back off, so the
        read-only contract cannot be lifted from within SQL."""
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
            return sqlite3.SQLITE_DENY
        if self.readonly:
            if action in _WRITE_ACTIONS:
                return sqlite3.SQLITE_DENY
            if (action == sqlite3.SQLITE_PRAGMA and isinstance(arg1, str)
                    and arg1.lower() == "query_only" and arg2 is not None):
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def _ensure_schema(self):
        # Check if this is a pre-existing DB (has conversations but no schema_version)
        tables = {
            r["name"]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        has_conversations = "conversations" in tables
        has_version_table = "schema_version" in tables

        if not has_conversations:
            # Fresh DB: create all tables and set to current version
            self.conn.executescript(SCHEMA_SQL)
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self.conn.commit()
        elif not has_version_table:
            # Pre-existing DB without versioning: create only new tables
            # (skip SCHEMA_SQL to avoid conflicts with existing tables),
            # then bootstrap at v1 and run all migrations.
            self._create_missing_tables()
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (1)"
            )
            self.conn.commit()
            self._apply_migrations()
        else:
            # DB with version table: apply any pending migrations
            self._apply_migrations()

    def _create_missing_tables(self):
        """Create tables that don't yet exist, without touching existing tables.

        Used for pre-existing databases without versioning — runs SCHEMA_SQL
        but suppresses errors from indexes referencing columns not yet added
        by migrations (e.g., parent_conversation_id index on an old conversations table).
        """
        for stmt in SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError as e:
                msg = str(e)
                if "no such column" in msg or "already exists" in msg:
                    pass  # Expected for indexes on columns not yet migrated
                else:
                    raise
        self.conn.commit()

    def _apply_migrations(self):
        row = self.conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        current = row["version"] if row else 1
        if row is None:
            # The schema_version table exists but holds no row (e.g. an init
            # interrupted between table creation and the version INSERT). Seed
            # one now so the UPDATE below actually persists the version;
            # otherwise migrations silently re-run on every open and crash.
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (current,)
            )
            self.conn.commit()
        while current < SCHEMA_VERSION:
            migrate_fn = MIGRATIONS.get(current)
            if migrate_fn is None:
                raise RuntimeError(
                    f"No migration from v{current} to v{current + 1}"
                )
            migrate_fn(self.conn)
            current += 1
            self.conn.execute(
                "UPDATE schema_version SET version=?", (current,)
            )
            self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def execute_sql(self, sql: str, params: tuple = (),
                    max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(sql, params)
        if cursor.description is None:
            self.conn.commit()
            return []
        if max_rows is not None:
            # Bounded fetch: stops after max_rows rows instead of materializing
            # an unbounded result set in memory.
            return cursor.fetchmany(max_rows)
        return cursor.fetchall()

    def get_schema(self) -> str:
        # Filter out FTS5 internal tables (shadow tables) -- they're
        # implementation details that confuse LLMs trying to write SQL.
        rows = self.execute_sql(
            "SELECT name, sql, type FROM sqlite_master "
            "WHERE sql IS NOT NULL ORDER BY type, name"
        )
        # Derive shadow-table prefixes structurally from every FTS5 virtual
        # table (e.g. messages_fts -> "messages_fts_"), so adding a new FTS5
        # table later cannot reintroduce a leak. The virtual tables themselves
        # do not match their own "<name>_" prefix, so their DDL is kept.
        fts_prefixes = tuple(
            r["name"] + "_" for r in rows
            if r["type"] == "table" and "using fts5" in (r["sql"] or "").lower()
        )
        skip_prefixes = fts_prefixes + ("schema_version",)
        ddl = "\n\n".join(
            r["sql"] for r in rows
            if not any(r["name"].startswith(p) for p in skip_prefixes)
        )
        # Append relationship and query documentation
        docs = """
-- ══ Relationships ══════════════════════════════════════════════
-- messages.conversation_id   → conversations.id  (CASCADE delete)
-- tags.conversation_id       → conversations.id  (CASCADE delete)
-- enrichments.conversation_id → conversations.id (CASCADE delete)
-- provenance.conversation_id → conversations.id  (CASCADE delete)
-- conversations.parent_conversation_id → conversations.id (SET NULL on delete)
-- messages.parent_id         → messages.id (same conversation, tree structure)

-- ══ FTS5 Full-Text Search ══════════════════════════════════════
-- messages_fts indexes message text with porter stemming + unicode61.
-- Columns: conversation_id (UNINDEXED), message_id (UNINDEXED), text
--
-- FTS search query pattern:
--   SELECT m.conversation_id, c.title, m.id, m.role, m.content
--   FROM messages_fts f
--   JOIN messages m ON m.conversation_id = f.conversation_id AND m.id = f.message_id
--   JOIN conversations c ON c.id = m.conversation_id
--   WHERE messages_fts MATCH 'search terms'
--   LIMIT 20
--
-- MATCH syntax: 'word1 word2' (OR), 'word1 AND word2', '"exact phrase"'

-- ══ Boolean Timestamp Columns ══════════════════════════════════
-- starred_at, pinned_at, archived_at are NULL (false) or DATETIME (true).
-- Filter: WHERE starred_at IS NOT NULL (starred conversations)"""
        return ddl + docs

    def conversation_unchanged(self, conv_id: str, updated_at, message_count: int) -> bool:
        """Fast check: does this conversation already exist with matching fields?"""
        row = self.conn.execute(
            "SELECT updated_at, message_count FROM conversations WHERE id=?",
            (conv_id,),
        ).fetchone()
        if row is None:
            return False
        return row["updated_at"] == _fmt_dt(updated_at) and row["message_count"] == message_count

    def save_conversation(self, conv: Conversation) -> None:
        c = self.conn.cursor()
        try:
            # Upsert the row in place. We deliberately avoid INSERT OR REPLACE:
            # REPLACE deletes the existing row first, which fires ON DELETE SET
            # NULL on notes.conversation_id (detaching a user's marginalia from a
            # conversation that still exists) and on child conversations'
            # parent_conversation_id. ON CONFLICT DO UPDATE mutates in place, so
            # those foreign keys stay intact.
            c.execute(
                "INSERT INTO conversations "
                "(id,title,source,model,summary,message_count,"
                "created_at,updated_at,starred_at,pinned_at,archived_at,"
                "parent_conversation_id,sensitive,metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  title=excluded.title, source=excluded.source,"
                "  model=excluded.model,"
                # Keep an existing summary when the importer supplies none:
                # platform exports carry no summary, so a routine re-import
                # of an active conversation would otherwise null out a
                # summary the user set via update_conversations.
                "  summary=COALESCE(excluded.summary, conversations.summary),"
                "  message_count=excluded.message_count,"
                "  created_at=excluded.created_at, updated_at=excluded.updated_at,"
                "  starred_at=excluded.starred_at, pinned_at=excluded.pinned_at,"
                "  archived_at=excluded.archived_at,"
                "  parent_conversation_id=excluded.parent_conversation_id,"
                "  sensitive=excluded.sensitive, metadata=excluded.metadata",
                (
                    conv.id, conv.title, conv.source, conv.model, conv.summary,
                    conv.message_count, _fmt_dt(conv.created_at),
                    _fmt_dt(conv.updated_at), _fmt_dt(conv.starred_at),
                    _fmt_dt(conv.pinned_at), _fmt_dt(conv.archived_at),
                    conv.parent_conversation_id,
                    int(conv.sensitive), json.dumps(conv.metadata),
                ),
            )
            # REPLACE used to CASCADE-clear child rows; replicate that explicitly
            # for everything EXCEPT notes (which must stay attached). messages_fts
            # is not FK-backed, so it was always cleaned by hand.
            c.execute("DELETE FROM messages WHERE conversation_id=?", (conv.id,))
            c.execute("DELETE FROM tags WHERE conversation_id=?", (conv.id,))
            c.execute("DELETE FROM enrichments WHERE conversation_id=?", (conv.id,))
            c.execute("DELETE FROM provenance WHERE conversation_id=?", (conv.id,))
            c.execute(
                "DELETE FROM messages_fts WHERE conversation_id=?", (conv.id,)
            )
            for msg in conv.messages.values():
                c.execute(
                    "INSERT INTO messages "
                    "(conversation_id,id,role,parent_id,model,created_at,"
                    "sensitive,content,metadata) VALUES (?,?,?,?,?,?,?,?,?)",
                    _message_row(conv.id, msg),
                )
                text = msg.get_text()
                if text:
                    c.execute(
                        "INSERT INTO messages_fts "
                        "(conversation_id,message_id,text) VALUES (?,?,?)",
                        (conv.id, msg.id, text),
                    )
            for tag in conv.tags:
                c.execute(
                    "INSERT OR IGNORE INTO tags (conversation_id,tag) "
                    "VALUES (?,?)",
                    (conv.id, tag),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def merge_conversation(self, conv: Conversation) -> Dict[str, int]:
        """Merge a conversation into the DB without clobbering existing state.

        If the conversation does not exist, this is equivalent to
        save_conversation. If it already exists, we:

        - leave its conversations-row metadata (title, model, timestamps,
          starred_at, pinned_at, archived_at, enrichments, provenance)
          untouched;
        - INSERT OR IGNORE each message, so new messages are added and
          existing ones by the same (conversation_id, id) PK are kept;
        - INSERT INTO messages_fts for each new message (tryExec-style
          tolerance: if the FTS table is absent, skip);
        - INSERT OR IGNORE each tag;
        - recompute conversations.message_count from live rows;
        - push conversations.updated_at forward to the newest inserted
          message's created_at (only moves forward).

        Returns a stats dict: {"added_messages": N, "added_tags": N,
        "created_conversation": 0 or 1}.

        This is the round-trip mode for merging annotations and chat
        continuations back from an exported HTML bundle into the primary
        DB. The assumption is that the bundle side may have added new
        messages and notes, but cannot have changed enrichments or
        provenance in ways we want to trust.
        """
        c = self.conn.cursor()
        stats = {"added_messages": 0, "added_tags": 0, "created_conversation": 0}
        try:
            existing = c.execute(
                "SELECT 1 FROM conversations WHERE id=?", (conv.id,)
            ).fetchone()
            if not existing:
                # Fall back to full save for new conversations.
                self.save_conversation(conv)
                stats["created_conversation"] = 1
                stats["added_messages"] = len(conv.messages)
                stats["added_tags"] = len(conv.tags)
                return stats

            # Existing conversation: merge messages + tags only.
            for msg in conv.messages.values():
                cur = c.execute(
                    "INSERT OR IGNORE INTO messages "
                    "(conversation_id,id,role,parent_id,model,created_at,"
                    "sensitive,content,metadata) VALUES (?,?,?,?,?,?,?,?,?)",
                    _message_row(conv.id, msg),
                )
                if cur.rowcount:
                    stats["added_messages"] += 1
                    text = msg.get_text()
                    if text:
                        # Tolerate missing messages_fts (stripped on export).
                        try:
                            c.execute(
                                "INSERT INTO messages_fts "
                                "(conversation_id,message_id,text) "
                                "VALUES (?,?,?)",
                                (conv.id, msg.id, text),
                            )
                        except sqlite3.Error:
                            pass

            for tag in conv.tags:
                cur = c.execute(
                    "INSERT OR IGNORE INTO tags (conversation_id,tag) "
                    "VALUES (?,?)",
                    (conv.id, tag),
                )
                if cur.rowcount:
                    stats["added_tags"] += 1

            if stats["added_messages"]:
                # Always recompute the denormalized count when messages were
                # added. This must NOT be coupled to the updated_at guard below:
                # a merged message with a NULL or non-increasing created_at would
                # otherwise leave message_count stale (and feed a wrong
                # conversation_unchanged skip decision on later imports).
                c.execute(
                    "UPDATE conversations SET message_count = "
                    "(SELECT COUNT(*) FROM messages WHERE conversation_id=?) "
                    "WHERE id=?",
                    (conv.id, conv.id),
                )
                # Move updated_at forward only (never backward).
                c.execute(
                    "UPDATE conversations SET updated_at = "
                    "  (SELECT MAX(created_at) FROM messages WHERE conversation_id=?) "
                    "WHERE id=? AND ("
                    "  updated_at IS NULL OR updated_at < "
                    "    (SELECT MAX(created_at) FROM messages WHERE conversation_id=?)"
                    ")",
                    (conv.id, conv.id, conv.id),
                )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return stats

    def merge_notes(self, notes: List[Dict[str, Any]]) -> int:
        """Merge notes with INSERT OR IGNORE semantics (keyed by note id).

        Each note dict must have at minimum: id, conversation_id, text.
        Optional: message_id, target_kind (defaults to "conversation" when
        message_id is None, "message" otherwise), kind (defaults to
        "freeform"), created_at, updated_at.

        Returns the number of notes newly inserted.
        """
        if not notes:
            return 0
        c = self.conn.cursor()
        added = 0
        now = _fmt_dt(datetime.now())
        try:
            for n in notes:
                if not n.get("id") or not n.get("conversation_id"):
                    continue
                text = n.get("text") or ""
                if not text:
                    continue
                message_id = n.get("message_id")
                target_kind = n.get("target_kind") or (
                    "message" if message_id else "conversation"
                )
                kind = n.get("kind") or "freeform"
                created = n.get("created_at") or now
                updated = n.get("updated_at") or created
                cur = c.execute(
                    "INSERT OR IGNORE INTO notes "
                    "(id,target_kind,conversation_id,message_id,kind,text,"
                    " created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        n["id"], target_kind, n["conversation_id"],
                        message_id, kind, text, created, updated,
                    ),
                )
                if cur.rowcount:
                    added += 1
                    try:
                        c.execute(
                            "INSERT INTO notes_fts "
                            "(note_id,conversation_id,message_id,text) "
                            "VALUES (?,?,?,?)",
                            (n["id"], n["conversation_id"], message_id, text),
                        )
                    except sqlite3.Error:
                        pass
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return added

    def load_conversation(self, id: str) -> Optional[Conversation]:
        rows = self.execute_sql(
            "SELECT * FROM conversations WHERE id=?", (id,)
        )
        if not rows:
            return None
        r = rows[0]
        conv = Conversation(
            id=r["id"],
            created_at=_parse_dt(r["created_at"]) or datetime.now(),
            updated_at=_parse_dt(r["updated_at"]) or datetime.now(),
            title=r["title"],
            source=r["source"],
            model=r["model"],
            summary=r["summary"],
            message_count=r["message_count"],
            starred_at=_parse_dt(r["starred_at"]),
            pinned_at=_parse_dt(r["pinned_at"]),
            archived_at=_parse_dt(r["archived_at"]),
            parent_conversation_id=r["parent_conversation_id"],
            sensitive=bool(r["sensitive"]),
            metadata=json.loads(r["metadata"]) if r["metadata"] else {},
        )
        conv.tags = [
            t["tag"]
            for t in self.execute_sql(
                "SELECT tag FROM tags WHERE conversation_id=?", (id,)
            )
        ]
        for mr in self.execute_sql(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at",
            (id,),
        ):
            conv.add_message(
                Message(
                    id=mr["id"],
                    role=mr["role"],
                    content=json.loads(mr["content"]) if mr["content"] else [],
                    parent_id=mr["parent_id"],
                    model=mr["model"],
                    created_at=_parse_dt(mr["created_at"]),
                    sensitive=bool(mr["sensitive"]),
                    metadata=(
                        json.loads(mr["metadata"]) if mr["metadata"] else {}
                    ),
                )
            )
        return conv

    def query_conversations(
        self,
        query=None,
        title=None,
        starred=None,
        pinned=None,
        archived=None,
        sensitive=None,
        source=None,
        model=None,
        tag=None,
        before=None,
        after=None,
        enrichment_type=None,
        enrichment_value=None,
        limit=20,
        cursor=None,
    ) -> Dict[str, Any]:
        conds: List[str] = []
        params: List[Any] = []
        if query:
            fts_q = _sanitize_fts_query(query)
            if not fts_q:
                return {"items": [], "next_cursor": None, "has_more": False}
            # Use a correlated subquery rather than materializing matching ids:
            # this applies the final ORDER BY + LIMIT across ALL matches (no
            # silent row cap) and avoids a host-parameter explosion when many
            # conversations match.
            conds.append(
                "c.id IN (SELECT DISTINCT conversation_id FROM messages_fts "
                "WHERE messages_fts MATCH ?)"
            )
            params.append(fts_q)
        if title:
            conds.append("c.title LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(title)}%")
        for flag, column in (
            (starred, "c.starred_at"),
            (pinned, "c.pinned_at"),
            (archived, "c.archived_at"),
        ):
            if flag is True:
                conds.append(f"{column} IS NOT NULL")
            elif flag is False:
                conds.append(f"{column} IS NULL")
        if sensitive is True:
            conds.append("c.sensitive=1")
        elif sensitive is False:
            conds.append("c.sensitive=0")
        if source:
            conds.append("c.source=?")
            params.append(source)
        if model:
            conds.append("c.model=?")
            params.append(model)
        if tag:
            conds.append(
                "EXISTS(SELECT 1 FROM tags t "
                "WHERE t.conversation_id=c.id AND t.tag=?)"
            )
            params.append(tag)
        if enrichment_type or enrichment_value:
            e_conds = ["e.conversation_id=c.id"]
            if enrichment_type:
                e_conds.append("e.type=?")
                params.append(enrichment_type)
            if enrichment_value:
                e_conds.append("e.value LIKE ? ESCAPE '\\'")
                params.append(f"%{_escape_like(enrichment_value)}%")
            conds.append(
                f"EXISTS(SELECT 1 FROM enrichments e "
                f"WHERE {' AND '.join(e_conds)})"
            )
        if before:
            conds.append("c.created_at<?")
            params.append(before)
        if after:
            conds.append("c.created_at>?")
            params.append(after)
        if cursor:
            cdt, cid = _decode_cursor(cursor)
            conds.append(
                "(c.updated_at<? OR (c.updated_at=? AND c.id<?))"
            )
            params.extend([cdt, cdt, cid])
        where = " AND ".join(conds) if conds else "1=1"
        params.append(limit + 1)
        rows = self.execute_sql(
            f"SELECT c.id,c.title,c.source,c.model,c.message_count,"
            f"c.created_at,c.updated_at,c.starred_at,c.pinned_at,"
            f"c.archived_at,c.sensitive,c.summary,"
            f"(SELECT GROUP_CONCAT(t.tag) FROM tags t"
            f" WHERE t.conversation_id=c.id) as tags_csv "
            f"FROM conversations c WHERE {where} "
            f"ORDER BY c.updated_at DESC,c.id DESC LIMIT ?",
            tuple(params),
        )
        has_more = len(rows) > limit
        items = rows[:limit]
        nc = (
            _encode_cursor(items[-1]["updated_at"], items[-1]["id"])
            if has_more and items
            else None
        )
        return {"items": items, "next_cursor": nc, "has_more": has_more}

    def _fts_search(self, query: str) -> List[str]:
        fts_q = _sanitize_fts_query(query)
        if not fts_q:
            return []
        try:
            rows = self.execute_sql(
                "SELECT DISTINCT conversation_id FROM messages_fts "
                "WHERE messages_fts MATCH ? LIMIT 1000",
                (fts_q,),
            )
        except sqlite3.OperationalError:
            rows = self.execute_sql(
                "SELECT DISTINCT conversation_id FROM messages "
                "WHERE content LIKE ? ESCAPE '\\' LIMIT 1000",
                (f"%{_escape_like(query)}%",),
            )
        return [r["conversation_id"] for r in rows]

    def search_messages(
        self,
        query: str,
        mode: str = "fts",
        conversation_id: str | None = None,
        role: str | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search within message content. Returns message-level results with conversation metadata.

        Modes:
            fts: FTS5 MATCH (token OR, porter stemming)
            phrase: exact substring via LIKE (escapes wildcards)
            like: raw LIKE pattern (caller provides wildcards)
        """
        conds: List[str] = []
        params: List[Any] = []

        if mode == "fts":
            fts_q = _sanitize_fts_query(query)
            if not fts_q:
                return []
            join_clause = (
                "INNER JOIN messages_fts f "
                "ON m.conversation_id=f.conversation_id AND m.id=f.message_id "
            )
            conds.append("f.messages_fts MATCH ?")
            params.append(fts_q)
        elif mode == "phrase":
            join_clause = ""
            conds.append("m.content LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(query)}%")
        elif mode == "like":
            join_clause = ""
            conds.append("m.content LIKE ?")
            params.append(query)
        else:
            raise ValueError(f"Invalid search mode: {mode}")

        if conversation_id:
            conds.append("m.conversation_id=?")
            params.append(conversation_id)
        if role:
            conds.append("m.role=?")
            params.append(role)

        where = " AND ".join(conds) if conds else "1=1"
        params.append(limit)

        try:
            rows = self.execute_sql(
                f"SELECT m.conversation_id, m.id as message_id, m.role,"
                f" m.content, m.parent_id, m.model, m.created_at,"
                f" c.title as conversation_title "
                f"FROM messages m "
                f"{join_clause}"
                f"INNER JOIN conversations c ON m.conversation_id=c.id "
                f"WHERE {where} "
                f"LIMIT ?",
                tuple(params),
            )
        except sqlite3.OperationalError:
            return []

        return rows

    def get_context_messages(
        self, conversation_id: str, message_id: str, context: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get surrounding messages for a matched message by walking the tree."""
        # Walk up to find ancestors
        ancestors: List[str] = []
        current_id = message_id
        for _ in range(context):
            rows = self.execute_sql(
                "SELECT parent_id FROM messages "
                "WHERE conversation_id=? AND id=?",
                (conversation_id, current_id),
            )
            if not rows or rows[0]["parent_id"] is None:
                break
            current_id = rows[0]["parent_id"]
            ancestors.append(current_id)
        ancestors.reverse()

        # Walk down to find descendants (BFS by depth)
        descendants: List[str] = []
        frontier = [message_id]
        for _ in range(context):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self.execute_sql(
                f"SELECT id FROM messages "
                f"WHERE conversation_id=? AND parent_id IN ({placeholders}) "
                f"ORDER BY created_at",
                (conversation_id, *frontier),
            )
            next_frontier = [r["id"] for r in rows]
            descendants.extend(next_frontier)
            frontier = next_frontier

        # Fetch all messages in order
        all_ids = ancestors + [message_id] + descendants
        if not all_ids:
            return []
        placeholders = ",".join("?" for _ in all_ids)
        rows = self.execute_sql(
            f"SELECT id, role, content, parent_id, model, created_at "
            f"FROM messages WHERE conversation_id=? "
            f"AND id IN ({placeholders}) ORDER BY created_at",
            (conversation_id, *all_ids),
        )
        return rows

    def update_conversation(
        self,
        id,
        title=None,
        summary=None,
        starred=None,
        pinned=None,
        archived=None,
        sensitive=None,
        add_tags=None,
        remove_tags=None,
        metadata=None,
    ):
        existing = self.execute_sql(
            "SELECT id,metadata FROM conversations WHERE id=?", (id,)
        )
        if not existing:
            raise ValueError(f"Conversation not found: {id}")
        try:
            sets: List[str] = []
            params: List[Any] = []
            now = _fmt_dt(datetime.now())
            if title is not None:
                sets.append("title=?")
                params.append(title)
            if summary is not None:
                sets.append("summary=?")
                params.append(summary)
            if starred is True:
                sets.append("starred_at=?")
                params.append(now)
            elif starred is False:
                sets.append("starred_at=NULL")
            if pinned is True:
                sets.append("pinned_at=?")
                params.append(now)
            elif pinned is False:
                sets.append("pinned_at=NULL")
            if archived is True:
                sets.append("archived_at=?")
                params.append(now)
            elif archived is False:
                sets.append("archived_at=NULL")
            if sensitive is not None:
                sets.append("sensitive=?")
                params.append(int(sensitive))
            if metadata is not None:
                m = json.loads(existing[0]["metadata"] or "{}")
                m.update(metadata)
                sets.append("metadata=?")
                params.append(json.dumps(m))
            if sets:
                sets.append("updated_at=?")
                params.append(now)
                params.append(id)
                self.conn.execute(
                    f"UPDATE conversations SET {','.join(sets)} WHERE id=?",
                    tuple(params),
                )
            if add_tags:
                for t in add_tags:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO tags (conversation_id,tag) "
                        "VALUES (?,?)",
                        (id, t),
                    )
            if remove_tags:
                for t in remove_tags:
                    self.conn.execute(
                        "DELETE FROM tags WHERE conversation_id=? AND tag=?",
                        (id, t),
                    )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def append_message(self, conversation_id, message):
        if not self.execute_sql(
            "SELECT id FROM conversations WHERE id=?", (conversation_id,)
        ):
            raise ValueError(f"Conversation not found: {conversation_id}")
        try:
            now = _fmt_dt(datetime.now())
            self.conn.execute(
                "INSERT INTO messages "
                "(conversation_id,id,role,parent_id,model,created_at,"
                "sensitive,content,metadata) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    conversation_id, message.id, message.role, message.parent_id,
                    message.model, _fmt_dt(message.created_at) or now,
                    int(message.sensitive), json.dumps(message.content),
                    json.dumps(message.metadata),
                ),
            )
            text = message.get_text()
            if text:
                self.conn.execute(
                    "INSERT INTO messages_fts "
                    "(conversation_id,message_id,text) VALUES (?,?,?)",
                    (conversation_id, message.id, text),
                )
            self.conn.execute(
                "UPDATE conversations SET message_count="
                "(SELECT COUNT(*) FROM messages WHERE conversation_id=?),"
                "updated_at=? WHERE id=?",
                (conversation_id, now, conversation_id),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def update_message_content(
        self, conversation_id: str, message_id: str, content: list
    ) -> None:
        """Update a message's content and re-index FTS5."""
        existing = self.execute_sql(
            "SELECT id FROM messages WHERE conversation_id=? AND id=?",
            (conversation_id, message_id),
        )
        if not existing:
            raise ValueError(
                f"Message not found: {message_id} in conversation {conversation_id}"
            )
        try:
            self.conn.execute(
                "UPDATE messages SET content=? WHERE conversation_id=? AND id=?",
                (json.dumps(content), conversation_id, message_id),
            )
            # Re-index FTS: delete old entry, insert new if text content exists
            self.conn.execute(
                "DELETE FROM messages_fts WHERE conversation_id=? AND message_id=?",
                (conversation_id, message_id),
            )
            text = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if text:
                self.conn.execute(
                    "INSERT INTO messages_fts "
                    "(conversation_id,message_id,text) VALUES (?,?,?)",
                    (conversation_id, message_id, text),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all related data. Returns True if found."""
        try:
            # Clean FTS first (not covered by CASCADE)
            self.conn.execute(
                "DELETE FROM messages_fts WHERE conversation_id=?",
                (conversation_id,),
            )
            # Notes survive via ON DELETE SET NULL (orphan survival), but their
            # notes_fts shadow rows still carry the now-deleted conversation_id.
            # Capture the surviving note ids before the delete, then re-sync
            # their FTS rows afterward so the shadow column matches the notes
            # table (conversation_id NULL), the same way update_note does.
            orphaned_note_ids = [
                r["id"] for r in self.conn.execute(
                    "SELECT id FROM notes WHERE conversation_id=?",
                    (conversation_id,),
                ).fetchall()
            ]
            # CASCADE handles messages, tags, enrichments, provenance
            cursor = self.conn.execute(
                "DELETE FROM conversations WHERE id=?",
                (conversation_id,),
            )
            for note_id in orphaned_note_ids:
                row = self.conn.execute(
                    "SELECT conversation_id, message_id, text FROM notes "
                    "WHERE id=?",
                    (note_id,),
                ).fetchone()
                self.conn.execute(
                    "DELETE FROM notes_fts WHERE note_id=?", (note_id,)
                )
                if row is not None:
                    self.conn.execute(
                        "INSERT INTO notes_fts "
                        "(note_id, conversation_id, message_id, text) "
                        "VALUES (?, ?, ?, ?)",
                        (note_id, row["conversation_id"], row["message_id"],
                         row["text"]),
                    )
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception:
            self.conn.rollback()
            raise

    # ── Enrichments ──────────────────────────────────────────────

    def save_enrichment(
        self,
        conversation_id: str,
        type: str,
        value: str,
        source: str,
        confidence: float | None = None,
    ) -> None:
        self.save_enrichments(conversation_id, [
            {"type": type, "value": value, "source": source, "confidence": confidence},
        ])

    def save_enrichments(
        self, conversation_id: str, enrichments: List[Dict[str, Any]]
    ) -> None:
        now = _fmt_dt(datetime.now())
        try:
            for e in enrichments:
                self.conn.execute(
                    "INSERT OR REPLACE INTO enrichments "
                    "(conversation_id,type,value,source,confidence,created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        conversation_id,
                        e["type"],
                        e["value"],
                        e["source"],
                        e.get("confidence"),
                        now,
                    ),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_enrichments(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self.execute_sql(
            "SELECT type,value,source,confidence,created_at "
            "FROM enrichments WHERE conversation_id=? "
            "ORDER BY type,value",
            (conversation_id,),
        )

    def query_enrichments(
        self,
        type: str | None = None,
        value: str | None = None,
        source: str | None = None,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        conds: List[str] = []
        params: List[Any] = []
        if type:
            conds.append("e.type=?")
            params.append(type)
        if value:
            conds.append("e.value LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(value)}%")
        if source:
            conds.append("e.source=?")
            params.append(source)
        if conversation_id:
            conds.append("e.conversation_id=?")
            params.append(conversation_id)
        where = " AND ".join(conds) if conds else "1=1"
        params.append(limit)
        return self.execute_sql(
            f"SELECT e.conversation_id,e.type,e.value,e.source,"
            f"e.confidence,e.created_at,"
            f"c.title as conversation_title "
            f"FROM enrichments e "
            f"INNER JOIN conversations c ON e.conversation_id=c.id "
            f"WHERE {where} "
            f"ORDER BY e.created_at DESC LIMIT ?",
            tuple(params),
        )

    def delete_enrichment(
        self, conversation_id: str, type: str, value: str
    ) -> bool:
        try:
            cursor = self.conn.execute(
                "DELETE FROM enrichments "
                "WHERE conversation_id=? AND type=? AND value=?",
                (conversation_id, type, value),
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception:
            self.conn.rollback()
            raise

    # ── Notes (marginalia) ──────────────────────────────────────

    def add_note(
        self,
        *,
        conversation_id: str,
        text: str,
        message_id: Optional[str] = None,
        note_id: Optional[str] = None,
        kind: str = "freeform",
        anchor_start: Optional[int] = None,
        anchor_end: Optional[int] = None,
        anchor_hash: Optional[str] = None,
        parent_note_id: Optional[str] = None,
    ) -> str:
        """Add a free-form text note to a conversation or message.

        If message_id is provided, creates a message-level note
        (target_kind='message'). Otherwise creates a conversation-level note
        (target_kind='conversation'). Returns the note id.

        Marginalia v2 (schema v5):
            kind: note classifier, e.g. 'freeform', 'highlight', 'question',
                'contradiction', 'synthesis', 'todo'. Applications define
                their own vocabulary.
            anchor_start, anchor_end: character offsets pinning the note
                to a range within the target message's text content.
            anchor_hash: SHA-256 of the anchored substring. If the source
                content changes, the hash mismatch lets validators detect
                the note has drifted off its anchor.
            parent_note_id: if given, threads this note as a reply to
                another note. Deleting the parent cascades to the reply.
        """
        import uuid as _uuid
        if note_id is None:
            note_id = str(_uuid.uuid4())
        target_kind = "message" if message_id else "conversation"
        # A message-level note must anchor to a real message in this
        # conversation; otherwise the note points at nothing.
        if message_id is not None:
            exists = self.conn.execute(
                "SELECT 1 FROM messages WHERE conversation_id = ? AND id = ?",
                (conversation_id, message_id),
            ).fetchone()
            if exists is None:
                raise ValueError(
                    f"message_id {message_id!r} not found in conversation "
                    f"{conversation_id!r}; cannot anchor a note to it"
                )
        now = _fmt_dt(datetime.now())
        try:
            self.conn.execute(
                "INSERT INTO notes "
                "(id, target_kind, conversation_id, message_id, text, "
                "created_at, updated_at, kind, anchor_start, anchor_end, "
                "anchor_hash, parent_note_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    note_id, target_kind, conversation_id, message_id, text,
                    now, now, kind, anchor_start, anchor_end, anchor_hash,
                    parent_note_id,
                ),
            )
            self.conn.execute(
                "INSERT INTO notes_fts "
                "(note_id, conversation_id, message_id, text) "
                "VALUES (?, ?, ?, ?)",
                (note_id, conversation_id, message_id, text),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return note_id

    def update_note(self, note_id: str, text: str) -> bool:
        """Update an existing note's text and bump updated_at. Returns True if a row was updated."""
        now = _fmt_dt(datetime.now())
        try:
            cursor = self.conn.execute(
                "UPDATE notes SET text = ?, updated_at = ? WHERE id = ?",
                (text, now, note_id),
            )
            if cursor.rowcount == 0:
                self.conn.commit()
                return False
            # Re-index FTS5: DELETE old row, INSERT new with fresh text
            row = self.conn.execute(
                "SELECT conversation_id, message_id FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            self.conn.execute(
                "DELETE FROM notes_fts WHERE note_id = ?", (note_id,)
            )
            if row is not None:
                self.conn.execute(
                    "INSERT INTO notes_fts "
                    "(note_id, conversation_id, message_id, text) "
                    "VALUES (?, ?, ?, ?)",
                    (note_id, row["conversation_id"], row["message_id"], text),
                )
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def delete_note(self, note_id: str) -> bool:
        """Delete a note and its FTS5 entry. Returns True if a row was deleted."""
        try:
            self.conn.execute(
                "DELETE FROM notes_fts WHERE note_id = ?", (note_id,)
            )
            cursor = self.conn.execute(
                "DELETE FROM notes WHERE id = ?", (note_id,)
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception:
            self.conn.rollback()
            raise

    def get_notes(
        self,
        *,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        target_kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query notes filtered by conversation, message, or target kind.

        With no filters, returns all notes in the database. Results are
        ordered by created_at ascending.
        """
        conds = []
        params: List[Any] = []
        if conversation_id is not None:
            conds.append("conversation_id = ?")
            params.append(conversation_id)
        if message_id is not None:
            conds.append("message_id = ?")
            params.append(message_id)
        if target_kind is not None:
            conds.append("target_kind = ?")
            params.append(target_kind)
        where = " AND ".join(conds) if conds else "1=1"
        rows = self.conn.execute(
            f"SELECT id, target_kind, conversation_id, message_id, text, "
            f"created_at, updated_at FROM notes WHERE {where} "
            f"ORDER BY created_at ASC",
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_notes(
        self, query: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """FTS5 search across note text. Returns note dicts in match order."""
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []
        fts_rows = self.conn.execute(
            "SELECT note_id FROM notes_fts WHERE notes_fts MATCH ? LIMIT ?",
            (sanitized, limit),
        ).fetchall()
        ids = [r["note_id"] for r in fts_rows]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT id, target_kind, conversation_id, message_id, text, "
            f"created_at, updated_at FROM notes WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        # Preserve FTS match order
        by_id = {r["id"]: dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    # ── Provenance ──────────────────────────────────────────────

    def save_provenance(
        self,
        conversation_id: str,
        source_type: str,
        imported_at: str | None = None,
        source_file: str | None = None,
        source_id: str | None = None,
        source_hash: str | None = None,
        importer_version: str | None = None,
    ) -> None:
        now = imported_at or _fmt_dt(datetime.now())
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO provenance "
                "(conversation_id,source_type,source_file,source_id,"
                "source_hash,imported_at,importer_version) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    conversation_id, source_type, source_file,
                    source_id, source_hash, now, importer_version,
                ),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_provenance(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self.execute_sql(
            "SELECT source_type,source_file,source_id,source_hash,"
            "imported_at,importer_version "
            "FROM provenance WHERE conversation_id=?",
            (conversation_id,),
        )

    # ── Statistics ──────────────────────────────────────────────

    def get_statistics(self):
        tc = self.execute_sql(
            "SELECT COUNT(*) as n FROM conversations"
        )[0]["n"]
        tm = self.execute_sql(
            "SELECT COUNT(*) as n FROM messages"
        )[0]["n"]
        sources = {
            r["source"]: r["n"]
            for r in self.execute_sql(
                "SELECT source,COUNT(*) as n FROM conversations "
                "WHERE source IS NOT NULL GROUP BY source"
            )
        }
        models = {
            r["model"]: r["n"]
            for r in self.execute_sql(
                "SELECT model,COUNT(*) as n FROM conversations "
                "WHERE model IS NOT NULL GROUP BY model"
            )
        }
        tags = {
            r["tag"]: r["n"]
            for r in self.execute_sql(
                "SELECT tag,COUNT(*) as n FROM tags GROUP BY tag"
            )
        }
        enrichment_types = {
            r["type"]: r["n"]
            for r in self.execute_sql(
                "SELECT type,COUNT(*) as n FROM enrichments GROUP BY type"
            )
        }
        provenance_tracked = self.execute_sql(
            "SELECT COUNT(DISTINCT conversation_id) as n FROM provenance"
        )[0]["n"]
        return {
            "total_conversations": tc,
            "total_messages": tm,
            "sources": sources,
            "models": models,
            "tags": tags,
            "enrichment_types": enrichment_types,
            "provenance_tracked": provenance_tracked,
        }

    def list_paths(self, conversation_id):
        conv = self.load_conversation(conversation_id)
        if conv is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        result = []
        for i, path in enumerate(conv.get_all_paths()):
            first = path[0] if path else None
            last = path[-1] if path else None
            result.append({
                "index": i,
                "message_count": len(path),
                "first_message": {
                    "id": first.id,
                    "role": first.role,
                    "preview": first.get_text()[:100],
                } if first else None,
                "last_message": {
                    "id": last.id,
                    "role": last.role,
                    "preview": last.get_text()[:100],
                } if last else None,
                "leaf_id": last.id if last else None,
            })
        return result

    def get_path_messages(
        self,
        conversation_id,
        path_index=None,
        leaf_message_id=None,
        offset=0,
        limit=None,
    ):
        conv = self.load_conversation(conversation_id)
        if conv is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        if leaf_message_id:
            path = conv.get_path(leaf_message_id)
            if path is None:
                raise ValueError(f"Message not found: {leaf_message_id}")
        elif path_index is not None:
            all_paths = conv.get_all_paths()
            if path_index < 0 or path_index >= len(all_paths):
                raise ValueError(f"Path index out of range: {path_index}")
            path = all_paths[path_index]
        else:
            all_paths = conv.get_all_paths()
            path = all_paths[0] if all_paths else []
        if offset:
            path = path[offset:]
        if limit is not None:
            path = path[:limit]
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "parent_id": m.parent_id,
                "model": m.model,
                "created_at": _fmt_dt(m.created_at),
                "sensitive": m.sensitive,
                "metadata": m.metadata,
            }
            for m in path
        ]

    # (edges methods removed in schema v7; see _migrate_to_v7 docstring.
    # Cross-record graph persistence belongs to the memex federation
    # layer, not to individual archives.)
