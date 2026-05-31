import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from llm_memex.db import Database
from llm_memex.models import (
    Conversation,
    Message,
    text_block,
    tool_result_block,
    tool_use_block,
)


def _make_conv(id="c1", title="Test"):
    now = datetime.now()
    conv = Conversation(
        id=id, created_at=now, updated_at=now, title=title,
        source="test", model="gpt-4", tags=["python", "testing"],
    )
    conv.add_message(Message(id="m1", role="user", content=[text_block("hello")]))
    conv.add_message(
        Message(
            id="m2", role="assistant", content=[text_block("hi")],
            parent_id="m1", model="gpt-4",
        )
    )
    return conv


class TestDatabaseSchema:
    def test_creates_tables(self, tmp_db_path):
        db = Database(tmp_db_path)
        tables = db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = [r["name"] for r in tables]
        assert "conversations" in names
        assert "messages" in names
        assert "tags" in names
        assert "messages_fts" in names

    def test_creates_indexes(self, tmp_db_path):
        db = Database(tmp_db_path)
        indexes = db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        names = [r["name"] for r in indexes]
        assert "idx_conversations_created" in names
        assert "idx_messages_parent" in names
        assert "idx_tags_tag" in names

    def test_wal_mode(self, tmp_db_path):
        db = Database(tmp_db_path)
        assert db.execute_sql("PRAGMA journal_mode")[0]["journal_mode"] == "wal"

    def test_foreign_keys(self, tmp_db_path):
        db = Database(tmp_db_path)
        assert db.execute_sql("PRAGMA foreign_keys")[0]["foreign_keys"] == 1

    def test_close_idempotent(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.close()
        db.close()  # no error

    def test_get_schema(self, tmp_db_path):
        db = Database(tmp_db_path)
        schema = db.get_schema()
        assert "CREATE TABLE conversations" in schema
        assert "CREATE TABLE messages" in schema
        # Internal tables (schema_version, FTS5 shadow tables) are filtered out
        assert "schema_version" not in schema
        assert "messages_fts_config" not in schema
        # Relationship and FTS5 docs are appended
        assert "Relationships" in schema
        assert "FTS5" in schema

    def test_memory_db(self):
        db = Database(":memory:")
        tables = db.execute_sql("SELECT name FROM sqlite_master WHERE type='table'")
        assert len(tables) >= 3
        db.close()

    def test_execute_sql_returns_dicts(self, tmp_db_path):
        db = Database(tmp_db_path)
        rows = db.execute_sql("SELECT 1 as a, 2 as b")
        assert rows == [{"a": 1, "b": 2}]


class TestSchemaVersioning:
    def test_fresh_db_has_current_version(self, tmp_db_path):
        from llm_memex.db import SCHEMA_VERSION
        db = Database(tmp_db_path)
        rows = db.execute_sql("SELECT version FROM schema_version")
        assert len(rows) == 1
        assert rows[0]["version"] == SCHEMA_VERSION

    def test_reopen_db_no_duplicate_rows(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.close()
        db2 = Database(tmp_db_path)
        rows = db2.execute_sql("SELECT version FROM schema_version")
        assert len(rows) == 1

    def test_pre_existing_db_bootstraps_at_v1(self, tmp_db_path):
        """A DB created without schema_version gets bootstrapped at v1."""
        import sqlite3 as _sqlite3
        from llm_memex.db import SCHEMA_VERSION
        # Manually create a pre-existing DB without schema_version
        db_file = str(Path(tmp_db_path) / "conversations.db")
        conn = _sqlite3.connect(db_file)
        conn.execute(
            "CREATE TABLE conversations ("
            "id TEXT PRIMARY KEY, title TEXT, source TEXT, model TEXT, summary TEXT,"
            "message_count INTEGER NOT NULL DEFAULT 0,"
            "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,"
            "starred_at DATETIME, pinned_at DATETIME, archived_at DATETIME,"
            "sensitive BOOLEAN NOT NULL DEFAULT 0,"
            "metadata JSON NOT NULL DEFAULT '{}'"
            ")"
        )
        conn.commit()
        conn.close()
        # Now open with Database — should detect pre-existing and bootstrap
        db = Database(tmp_db_path)
        rows = db.execute_sql("SELECT version FROM schema_version")
        assert len(rows) == 1
        assert rows[0]["version"] == SCHEMA_VERSION

    def test_schema_version_filtered_from_schema(self, tmp_db_path):
        """schema_version is internal bookkeeping, filtered from get_schema() output."""
        db = Database(tmp_db_path)
        schema = db.get_schema()
        assert "schema_version" not in schema


class TestSaveLoad:
    def test_roundtrip(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        loaded = db.load_conversation("c1")
        assert loaded.title == "Test"
        assert loaded.source == "test"
        assert loaded.message_count == 2
        assert set(loaded.tags) == {"python", "testing"}

    def test_messages_preserved(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        loaded = db.load_conversation("c1")
        assert loaded.messages["m1"].get_text() == "hello"
        assert loaded.messages["m2"].parent_id == "m1"

    def test_tree_structure(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        paths = db.load_conversation("c1").get_all_paths()
        assert [m.id for m in paths[0]] == ["m1", "m2"]

    def test_load_nonexistent(self, tmp_db_path):
        assert Database(tmp_db_path).load_conversation("nope") is None

    def test_save_overwrites(self, tmp_db_path):
        db = Database(tmp_db_path)
        conv = _make_conv()
        db.save_conversation(conv)
        conv.title = "Updated"
        db.save_conversation(conv)
        assert db.load_conversation("c1").title == "Updated"

    def test_metadata_roundtrip(self, tmp_db_path):
        db = Database(tmp_db_path)
        conv = _make_conv()
        conv.metadata = {"q": 0.9}
        db.save_conversation(conv)
        assert db.load_conversation("c1").metadata == {"q": 0.9}

    def test_sensitive_roundtrip(self, tmp_db_path):
        db = Database(tmp_db_path)
        conv = _make_conv()
        conv.sensitive = True
        db.save_conversation(conv)
        assert db.load_conversation("c1").sensitive is True

    def test_content_blocks_roundtrip(self, tmp_db_path):
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(id="c1", created_at=now, updated_at=now)
        conv.add_message(
            Message(
                id="m1", role="assistant",
                content=[
                    text_block("searching"),
                    tool_use_block("c1", "search", {"q": "test"}),
                ],
            )
        )
        conv.add_message(
            Message(
                id="m2", role="tool",
                content=[tool_result_block("c1", content="found 5")],
                parent_id="m1",
            )
        )
        db.save_conversation(conv)
        m1 = db.load_conversation("c1").messages["m1"]
        assert m1.content[1]["type"] == "tool_use"
        assert m1.content[1]["name"] == "search"


def _populate_db(db):
    for i in range(1, 6):
        now = datetime(2024, 1, i)
        conv = Conversation(
            id=f"c{i}", created_at=now, updated_at=now, title=f"Chat {i}",
            source="openai" if i <= 3 else "anthropic",
            model="gpt-4" if i <= 3 else "claude-3",
            tags=["python"] if i % 2 == 0 else ["rust"],
        )
        if i == 1:
            conv.starred_at = now
        if i == 2:
            conv.pinned_at = now
        if i == 3:
            conv.archived_at = now
        conv.add_message(
            Message(id="m1", role="user", content=[text_block(f"topic {i}")])
        )
        conv.add_message(
            Message(
                id="m2", role="assistant",
                content=[text_block(f"answer {i}")], parent_id="m1",
            )
        )
        db.save_conversation(conv)


class TestQuery:
    def test_all(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        assert len(db.query_conversations()["items"]) == 5

    def test_limit(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations(limit=2)
        assert len(r["items"]) == 2 and r["has_more"] is True

    def test_starred(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations(starred=True)
        assert len(r["items"]) == 1 and r["items"][0]["id"] == "c1"

    def test_source(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        assert len(db.query_conversations(source="anthropic")["items"]) == 2

    def test_tag(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        assert len(db.query_conversations(tag="python")["items"]) == 2

    def test_not_archived(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        assert len(db.query_conversations(archived=False)["items"]) == 4


class TestSearch:
    def test_fts(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations(query="topic 3")
        assert "c3" in [i["id"] for i in r["items"]]

    def test_no_results(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        assert len(db.query_conversations(query="nonexistent_xyz")["items"]) == 0

    def test_with_filter(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations(query="topic", source="anthropic")
        for item in r["items"]:
            assert item["id"] in ("c4", "c5")


class TestUpdate:
    def test_title(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.update_conversation("c1", title="New")
        assert db.load_conversation("c1").title == "New"

    def test_star_unstar(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.update_conversation("c1", starred=True)
        assert db.load_conversation("c1").starred_at is not None
        db.update_conversation("c1", starred=False)
        assert db.load_conversation("c1").starred_at is None

    def test_add_remove_tags(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.update_conversation("c1", add_tags=["new"])
        assert "new" in db.load_conversation("c1").tags
        db.update_conversation("c1", remove_tags=["python"])
        assert "python" not in db.load_conversation("c1").tags

    def test_metadata_merge(self, tmp_db_path):
        db = Database(tmp_db_path)
        conv = _make_conv()
        conv.metadata = {"a": 1}
        db.save_conversation(conv)
        db.update_conversation("c1", metadata={"b": 2})
        assert db.load_conversation("c1").metadata == {"a": 1, "b": 2}

    def test_summary(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.update_conversation("c1", summary="A test.")
        assert db.load_conversation("c1").summary == "A test."

    def test_nonexistent(self, tmp_db_path):
        with pytest.raises(ValueError, match="not found"):
            Database(tmp_db_path).update_conversation("nope", title="x")


class TestAppend:
    def test_append(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.append_message(
            "c1",
            Message(
                id="m3", role="user",
                content=[text_block("followup")], parent_id="m2",
            ),
        )
        loaded = db.load_conversation("c1")
        assert len(loaded.messages) == 3 and loaded.message_count == 3

    def test_branch(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.append_message(
            "c1",
            Message(
                id="m2b", role="assistant",
                content=[text_block("alt")], parent_id="m1",
            ),
        )
        assert len(db.load_conversation("c1").get_all_paths()) == 2

    def test_nonexistent(self, tmp_db_path):
        with pytest.raises(ValueError, match="not found"):
            Database(tmp_db_path).append_message(
                "nope",
                Message(id="m1", role="user", content=[text_block("x")]),
            )

    def test_updates_fts(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.append_message(
            "c1",
            Message(
                id="m3", role="user",
                content=[text_block("unique_xyz_term")], parent_id="m2",
            ),
        )
        assert len(db.query_conversations(query="unique_xyz_term")["items"]) == 1


class TestStatistics:
    def test_stats(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        s = db.get_statistics()
        assert s["total_conversations"] == 5 and s["total_messages"] == 10
        assert "openai" in s["sources"] and "anthropic" in s["sources"]

    def test_stats_empty(self, tmp_db_path):
        s = Database(tmp_db_path).get_statistics()
        assert s["total_conversations"] == 0


class TestPaths:
    def test_list_paths(self, tmp_db_path):
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(id="c1", created_at=now, updated_at=now)
        conv.add_message(
            Message(id="m1", role="user", content=[text_block("q")])
        )
        conv.add_message(
            Message(
                id="m2a", role="assistant",
                content=[text_block("a1")], parent_id="m1",
            )
        )
        conv.add_message(
            Message(
                id="m2b", role="assistant",
                content=[text_block("a2")], parent_id="m1",
            )
        )
        db.save_conversation(conv)
        paths = db.list_paths("c1")
        assert len(paths) == 2 and paths[0]["index"] == 0

    def test_get_path_messages(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        msgs = db.get_path_messages("c1", path_index=0)
        assert len(msgs) == 2 and msgs[0]["role"] == "user"

    def test_get_path_by_leaf(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        msgs = db.get_path_messages("c1", leaf_message_id="m2")
        assert len(msgs) == 2

    def test_get_path_offset_limit(self, tmp_db_path):
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(id="c1", created_at=now, updated_at=now)
        for i in range(1, 6):
            conv.add_message(
                Message(
                    id=f"m{i}",
                    role="user" if i % 2 else "assistant",
                    content=[text_block(f"msg{i}")],
                    parent_id=f"m{i-1}" if i > 1 else None,
                )
            )
        db.save_conversation(conv)
        msgs = db.get_path_messages("c1", path_index=0, offset=1, limit=2)
        assert len(msgs) == 2 and msgs[0]["id"] == "m2"

    def test_list_paths_not_found(self, tmp_db_path):
        with pytest.raises(ValueError):
            Database(tmp_db_path).list_paths("nope")


class TestFTSInjection:
    """Tests for FTS5 MATCH sanitization (issue #1)."""

    def test_double_quotes_stripped(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        # Should not raise — double quotes are stripped
        result = db.query_conversations(query='hello "world"')
        assert isinstance(result["items"], list)

    def test_single_quotes_stripped(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        result = db.query_conversations(query="it's a test")
        assert isinstance(result["items"], list)

    def test_empty_after_sanitize(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        # All characters stripped → empty query → no results
        result = db.query_conversations(query='""')
        assert result["items"] == []

    def test_like_wildcards_escaped(self, tmp_db_path):
        """LIKE fallback should escape % and _ wildcards."""
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(
            id="c1", created_at=now, updated_at=now, title="Test",
        )
        conv.add_message(Message(id="m1", role="user", content=[text_block("100% complete")]))
        db.save_conversation(conv)
        # The FTS query will work normally; this tests the sanitization path
        result = db.query_conversations(query="100% complete")
        assert len(result["items"]) == 1


class TestReadonlyMode:
    """Tests for read-only enforcement (issue #2). Writes are blocked by a
    SQLite authorizer (stronger than the flippable query_only PRAGMA)."""

    def test_readonly_blocks_insert(self, tmp_db_path):
        db = Database(tmp_db_path, readonly=True)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            db.execute_sql("INSERT INTO conversations (id, message_count, created_at, updated_at, sensitive) VALUES ('x', 0, '2024-01-01', '2024-01-01', 0)")

    def test_readonly_allows_select(self, tmp_db_path):
        # Write data with a writable connection first
        db_w = Database(tmp_db_path)
        db_w.save_conversation(_make_conv())
        db_w.close()
        # Now open readonly
        db_r = Database(tmp_db_path, readonly=True)
        result = db_r.execute_sql("SELECT COUNT(*) as n FROM conversations")
        assert result[0]["n"] == 1

    def test_readonly_blocks_delete(self, tmp_db_path):
        db = Database(tmp_db_path, readonly=True)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            db.execute_sql("DELETE FROM conversations")

    def test_readonly_blocks_drop(self, tmp_db_path):
        db = Database(tmp_db_path, readonly=True)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            db.execute_sql("DROP TABLE IF EXISTS conversations")


class TestContextManager:
    """Tests for Database __enter__/__exit__ (issue #8)."""

    def test_context_manager_basic(self, tmp_db_path):
        with Database(tmp_db_path) as db:
            db.save_conversation(_make_conv())
            assert db.conn is not None
        # After exit, connection should be closed
        assert db.conn is None

    def test_context_manager_on_exception(self, tmp_db_path):
        try:
            with Database(tmp_db_path) as db:
                db.save_conversation(_make_conv())
                raise ValueError("test error")
        except ValueError:
            pass
        # Connection should still be closed even on exception
        assert db.conn is None

    def test_context_manager_data_persists(self, tmp_db_path):
        with Database(tmp_db_path) as db:
            db.save_conversation(_make_conv())
        # Reopen and verify data survived
        with Database(tmp_db_path) as db2:
            conv = db2.load_conversation("c1")
            assert conv is not None
            assert conv.title == "Test"


class TestTransactionRollback:
    """Tests for transaction safety in append_message and update_conversation (issue #3)."""

    def test_append_duplicate_message_rolls_back(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        # m1 already exists; inserting again should fail and rollback
        dup_msg = Message(id="m1", role="user", content=[text_block("dup")])
        with pytest.raises(Exception):
            db.append_message("c1", dup_msg)
        # Original message should be unchanged
        conv = db.load_conversation("c1")
        assert conv.messages["m1"].get_text() == "hello"


class TestQueryTitle:
    """Tests for query_conversations title filter."""

    def test_title_filter(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations(title="Chat 3")
        assert len(r["items"]) == 1
        assert r["items"][0]["id"] == "c3"

    def test_title_substring(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations(title="Chat")
        assert len(r["items"]) == 5

    def test_title_no_match(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations(title="nonexistent")
        assert len(r["items"]) == 0

    def test_title_with_wildcards_escaped(self, tmp_db_path):
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(
            id="w1", created_at=now, updated_at=now,
            title="100% done",
        )
        conv.add_message(Message(id="m1", role="user", content=[text_block("hi")]))
        db.save_conversation(conv)
        r = db.query_conversations(title="100%")
        assert len(r["items"]) == 1
        # Make sure plain "100" doesn't match via unescaped %
        r2 = db.query_conversations(title="100% d")
        assert len(r2["items"]) == 1

    def test_title_combined_with_source(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations(title="Chat", source="anthropic")
        assert len(r["items"]) == 2
        for item in r["items"]:
            assert item["id"] in ("c4", "c5")


class TestTagsInQueryResults:
    """Tests for tags_csv in query_conversations results."""

    def test_tags_csv_present(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations()
        for item in r["items"]:
            assert "tags_csv" in item

    def test_tags_csv_values(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        r = db.query_conversations()
        # c2 has tag "python", c1 has tag "rust"
        items_by_id = {i["id"]: i for i in r["items"]}
        assert items_by_id["c2"]["tags_csv"] == "python"
        assert items_by_id["c1"]["tags_csv"] == "rust"

    def test_tags_csv_none_when_no_tags(self, tmp_db_path):
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(
            id="notags", created_at=now, updated_at=now, title="No Tags",
        )
        conv.add_message(Message(id="m1", role="user", content=[text_block("hi")]))
        db.save_conversation(conv)
        r = db.query_conversations()
        items_by_id = {i["id"]: i for i in r["items"]}
        assert items_by_id["notags"]["tags_csv"] is None


class TestSearchMessages:
    """Tests for the new search_messages method."""

    def test_fts_search(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages("topic")
        assert len(results) >= 1
        assert all("message_id" in r for r in results)
        assert all("conversation_title" in r for r in results)

    def test_fts_no_results(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages("zzz_nonexistent_zzz")
        assert len(results) == 0

    def test_phrase_search(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages("topic 3", mode="phrase")
        assert len(results) >= 1
        # Check the content contains the exact phrase
        found = False
        for r in results:
            if "topic 3" in r["content"]:
                found = True
        assert found

    def test_like_search(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages("%topic%", mode="like")
        assert len(results) >= 1

    def test_filter_by_conversation_id(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages("topic", conversation_id="c1")
        assert all(r["conversation_id"] == "c1" for r in results)

    def test_filter_by_role(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages("answer", role="assistant")
        assert len(results) >= 1
        assert all(r["role"] == "assistant" for r in results)

    def test_limit(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages("topic", limit=2)
        assert len(results) <= 2

    def test_invalid_mode(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        with pytest.raises(ValueError, match="Invalid search mode"):
            db.search_messages("test", mode="invalid")

    def test_empty_fts_query(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages('""', mode="fts")
        assert results == []

    def test_returns_conversation_title(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        results = db.search_messages("topic 1")
        matching = [r for r in results if r["conversation_id"] == "c1"]
        assert len(matching) >= 1
        assert matching[0]["conversation_title"] == "Chat 1"


class TestContextMessages:
    """Tests for get_context_messages."""

    def test_context_messages(self, tmp_db_path):
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(id="c1", created_at=now, updated_at=now)
        for i in range(1, 6):
            conv.add_message(
                Message(
                    id=f"m{i}",
                    role="user" if i % 2 else "assistant",
                    content=[text_block(f"msg{i}")],
                    parent_id=f"m{i-1}" if i > 1 else None,
                    created_at=datetime(2024, 1, 1, 0, i),
                )
            )
        db.save_conversation(conv)
        # Get context around m3 with 1 message of context
        ctx = db.get_context_messages("c1", "m3", context=1)
        ids = [r["id"] for r in ctx]
        assert "m2" in ids  # parent
        assert "m3" in ids  # match
        assert "m4" in ids  # child

    def test_context_at_root(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        ctx = db.get_context_messages("c1", "m1", context=1)
        ids = [r["id"] for r in ctx]
        assert "m1" in ids
        assert "m2" in ids  # child

    def test_context_at_leaf(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        ctx = db.get_context_messages("c1", "m2", context=1)
        ids = [r["id"] for r in ctx]
        assert "m1" in ids  # parent
        assert "m2" in ids


class TestEnrichments:
    """Tests for enrichment CRUD methods."""

    def test_save_and_get(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_enrichment("c1", "summary", "A test conversation", "claude", 0.9)
        enrichments = db.get_enrichments("c1")
        assert len(enrichments) == 1
        assert enrichments[0]["type"] == "summary"
        assert enrichments[0]["value"] == "A test conversation"
        assert enrichments[0]["source"] == "claude"
        assert enrichments[0]["confidence"] == 0.9

    def test_save_batch(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_enrichments("c1", [
            {"type": "topic", "value": "python", "source": "claude"},
            {"type": "topic", "value": "testing", "source": "claude"},
            {"type": "importance", "value": "high", "source": "heuristic", "confidence": 0.8},
        ])
        enrichments = db.get_enrichments("c1")
        assert len(enrichments) == 3

    def test_upsert(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_enrichment("c1", "summary", "old", "claude")
        db.save_enrichment("c1", "summary", "old", "user")  # same PK, different source
        enrichments = db.get_enrichments("c1")
        assert len(enrichments) == 1
        assert enrichments[0]["source"] == "user"

    def test_null_confidence(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_enrichment("c1", "topic", "greeting", "heuristic")
        e = db.get_enrichments("c1")
        assert e[0]["confidence"] is None

    def test_delete(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_enrichment("c1", "topic", "python", "claude")
        assert db.delete_enrichment("c1", "topic", "python") is True
        assert db.get_enrichments("c1") == []

    def test_delete_nonexistent(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        assert db.delete_enrichment("c1", "topic", "nope") is False

    def test_cascade_on_conversation_delete(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_enrichment("c1", "topic", "python", "claude")
        # Re-saving triggers INSERT OR REPLACE → CASCADE deletes enrichments
        db.save_conversation(_make_conv())
        assert db.get_enrichments("c1") == []

    def test_query_by_type(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        db.save_enrichment("c1", "topic", "python", "claude")
        db.save_enrichment("c1", "summary", "A chat", "claude")
        db.save_enrichment("c2", "topic", "rust", "claude")
        results = db.query_enrichments(type="topic")
        assert len(results) == 2
        assert all(r["type"] == "topic" for r in results)

    def test_query_by_value(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        db.save_enrichment("c1", "topic", "python programming", "claude")
        db.save_enrichment("c2", "topic", "rust systems", "claude")
        results = db.query_enrichments(value="python")
        assert len(results) == 1
        assert results[0]["conversation_id"] == "c1"

    def test_query_by_source(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        db.save_enrichment("c1", "topic", "python", "claude")
        db.save_enrichment("c2", "topic", "rust", "user")
        results = db.query_enrichments(source="user")
        assert len(results) == 1

    def test_query_by_conversation_id(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        db.save_enrichment("c1", "topic", "python", "claude")
        db.save_enrichment("c2", "topic", "rust", "claude")
        results = db.query_enrichments(conversation_id="c1")
        assert len(results) == 1
        assert results[0]["conversation_id"] == "c1"

    def test_query_includes_title(self, tmp_db_path):
        db = Database(tmp_db_path)
        _populate_db(db)
        db.save_enrichment("c1", "topic", "python", "claude")
        results = db.query_enrichments(type="topic")
        assert results[0]["conversation_title"] == "Chat 1"


class TestProvenance:
    """Tests for provenance CRUD methods."""

    def test_save_and_get(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_provenance(
            "c1", source_type="openai",
            source_file="/data/export.json",
            source_id="conv-abc",
        )
        prov = db.get_provenance("c1")
        assert len(prov) == 1
        assert prov[0]["source_type"] == "openai"
        assert prov[0]["source_file"] == "/data/export.json"
        assert prov[0]["source_id"] == "conv-abc"

    def test_upsert(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_provenance("c1", source_type="openai", source_file="old.json")
        db.save_provenance("c1", source_type="openai", source_file="new.json")
        prov = db.get_provenance("c1")
        assert len(prov) == 1
        assert prov[0]["source_file"] == "new.json"

    def test_multiple_sources(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_provenance("c1", source_type="openai")
        db.save_provenance("c1", source_type="anthropic")
        prov = db.get_provenance("c1")
        assert len(prov) == 2

    def test_cascade_on_conversation_delete(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_provenance("c1", source_type="openai")
        db.save_conversation(_make_conv())  # CASCADE
        assert db.get_provenance("c1") == []


class TestMigrationV1toV2:
    """Tests for v1→v2 migration (enrichments + provenance tables)."""

    def test_migration_creates_tables(self, tmp_db_path):
        """Simulate v1 DB and verify migration adds enrichments+provenance."""
        import sqlite3 as _sqlite3
        from llm_memex.db import SCHEMA_VERSION
        db_file = str(Path(tmp_db_path) / "conversations.db")
        conn = _sqlite3.connect(db_file)
        # Create v1 schema (no enrichments, no provenance)
        conn.executescript("""
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, title TEXT, source TEXT, model TEXT,
                summary TEXT, message_count INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
                starred_at DATETIME, pinned_at DATETIME, archived_at DATETIME,
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
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (1);
            INSERT INTO conversations (id, source, created_at, updated_at, message_count, sensitive)
            VALUES ('c1', 'openai', '2024-01-01', '2024-01-01', 0, 0);
        """)
        # FTS5 needs separate statement
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "conversation_id UNINDEXED, message_id UNINDEXED, text,"
            "tokenize = 'porter unicode61')"
        )
        conn.commit()
        conn.close()

        # Open with Database — should run migration
        db = Database(tmp_db_path)
        tables = [r["name"] for r in db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        assert "enrichments" in tables
        assert "provenance" in tables
        # Schema version should be current after all migrations
        assert db.execute_sql("SELECT version FROM schema_version")[0]["version"] == SCHEMA_VERSION

    def test_migration_backfills_provenance(self, tmp_db_path):
        """Source field from conversations should be backfilled into provenance."""
        import sqlite3 as _sqlite3
        db_file = str(Path(tmp_db_path) / "conversations.db")
        conn = _sqlite3.connect(db_file)
        conn.executescript("""
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, title TEXT, source TEXT, model TEXT,
                summary TEXT, message_count INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
                starred_at DATETIME, pinned_at DATETIME, archived_at DATETIME,
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
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (1);
            INSERT INTO conversations VALUES ('c1', 'Chat 1', 'openai', NULL, NULL, 0, '2024-01-01', '2024-01-01', NULL, NULL, NULL, 0, '{}');
            INSERT INTO conversations VALUES ('c2', 'Chat 2', 'anthropic', NULL, NULL, 0, '2024-01-02', '2024-01-02', NULL, NULL, NULL, 0, '{}');
        """)
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "conversation_id UNINDEXED, message_id UNINDEXED, text,"
            "tokenize = 'porter unicode61')"
        )
        conn.commit()
        conn.close()

        db = Database(tmp_db_path)
        p1 = db.get_provenance("c1")
        assert len(p1) == 1
        assert p1[0]["source_type"] == "openai"
        p2 = db.get_provenance("c2")
        assert len(p2) == 1
        assert p2[0]["source_type"] == "anthropic"

    def test_migration_idempotent(self, tmp_db_path):
        """Opening the DB multiple times shouldn't fail or duplicate data."""
        from llm_memex.db import SCHEMA_VERSION
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_enrichment("c1", "topic", "test", "claude")
        db.close()
        db2 = Database(tmp_db_path)
        assert len(db2.get_enrichments("c1")) == 1
        assert db2.execute_sql("SELECT version FROM schema_version")[0]["version"] == SCHEMA_VERSION


class TestUpdateMessageContent:
    def test_updates_content(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        new_content = [{"type": "text", "text": "replaced"}]
        db.update_message_content("c1", "m1", new_content)
        conv = db.load_conversation("c1")
        assert conv.messages["m1"].content == new_content

    def test_updates_fts(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())  # m1 has "hello"
        db.update_message_content("c1", "m1", [{"type": "text", "text": "replaced text"}])
        # New text findable
        results = db.search_messages("replaced")
        assert len(results) >= 1
        # Old text not findable
        results = db.search_messages("hello")
        assert len(results) == 0

    def test_nonexistent_raises(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        with pytest.raises(ValueError):
            db.update_message_content("c1", "no_such_msg", [])

    def test_readonly_fails(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.close()
        db = Database(tmp_db_path, readonly=True)
        with pytest.raises(Exception):
            db.update_message_content("c1", "m1", [{"type": "text", "text": "nope"}])

    def test_empty_text_removes_fts(self, tmp_db_path):
        """Updating to non-text content removes FTS entry."""
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.update_message_content("c1", "m1", [{"type": "tool_use", "id": "t1"}])
        results = db.search_messages("hello")
        assert len(results) == 0

    def test_preserves_other_messages(self, tmp_db_path):
        """Updating one message doesn't affect others."""
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.update_message_content("c1", "m1", [{"type": "text", "text": "new"}])
        conv = db.load_conversation("c1")
        assert conv.messages["m2"].content == [{"type": "text", "text": "hi"}]
        results = db.search_messages("hi")
        assert len(results) >= 1


class TestDeleteConversation:
    def test_deletes_conversation(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        assert db.delete_conversation("c1") is True
        assert db.load_conversation("c1") is None

    def test_cascades_fts(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.delete_conversation("c1")
        results = db.search_messages("hello")
        assert len(results) == 0

    def test_cascades_enrichments(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.save_enrichment("c1", "topic", "test", "heuristic")
        db.delete_conversation("c1")
        assert db.get_enrichments("c1") == []

    def test_cascades_tags(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())  # has tags ["python", "testing"]
        db.delete_conversation("c1")
        rows = db.execute_sql("SELECT * FROM tags WHERE conversation_id='c1'")
        assert len(rows) == 0

    def test_nonexistent_returns_false(self, tmp_db_path):
        db = Database(tmp_db_path)
        assert db.delete_conversation("nope") is False

    def test_readonly_fails(self, tmp_db_path):
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        db.close()
        db = Database(tmp_db_path, readonly=True)
        with pytest.raises(Exception):
            db.delete_conversation("c1")


class TestReimportIntegrity:
    def test_reimport_keeps_notes_attached(self, tmp_db_path):
        """Re-saving an existing conversation must not orphan its notes (DB-1)."""
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv())
        note_id = db.add_note(conversation_id="c1", text="my marginalia")
        # Re-import an updated export of the same conversation (gains a message).
        conv = _make_conv()
        conv.add_message(Message(id="m3", role="user",
                                 content=[text_block("more")], parent_id="m2"))
        db.save_conversation(conv)
        row = db.conn.execute(
            "SELECT conversation_id FROM notes WHERE id=?", (note_id,)
        ).fetchone()
        db.close()
        assert row is not None and row["conversation_id"] == "c1"

    def test_reopen_with_empty_schema_version(self, tmp_db_path):
        """A schema_version left empty (interrupted init) must not crash reopen (DB-2)."""
        from llm_memex.db import SCHEMA_VERSION
        db = Database(tmp_db_path)
        db.close()
        raw = sqlite3.connect(str(Path(tmp_db_path) / "conversations.db"))
        raw.execute("DELETE FROM schema_version")
        raw.commit()
        raw.close()
        db2 = Database(tmp_db_path)  # previously raised OperationalError: duplicate column
        row = db2.conn.execute("SELECT version FROM schema_version").fetchone()
        db2.close()
        assert row is not None and row["version"] == SCHEMA_VERSION
