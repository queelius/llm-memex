"""Tests for memex MCP server.

These tests exercise the database methods that back the MCP tools
directly, plus smoke-test that the server object can be created.
Full MCP protocol testing is deferred to integration tests.
"""
import json
from datetime import datetime

import pytest

from llm_memex.db import Database
from llm_memex.models import Conversation, Message, text_block
from llm_memex.mcp import create_server, _conv_metadata


def _get_tool_fn(server, name):
    """Extract a tool's underlying function from the FastMCP server."""
    tool = server._tool_manager._tools[name]
    return tool.fn


@pytest.fixture
def db(tmp_db_path):
    """Database with one conversation containing two messages."""
    d = Database(tmp_db_path)
    now = datetime.now()
    conv = Conversation(
        id="c1", created_at=now, updated_at=now,
        title="Test", source="openai", tags=["python"],
    )
    conv.add_message(Message(id="m1", role="user", content=[text_block("hello")]))
    conv.add_message(Message(
        id="m2", role="assistant", content=[text_block("hi")], parent_id="m1",
    ))
    d.save_conversation(conv)
    return d


@pytest.fixture
def multi_db(tmp_db_path):
    """Database with multiple conversations for bulk/search tests."""
    d = Database(tmp_db_path)
    for i in range(1, 4):
        now = datetime(2024, 1, i)
        conv = Conversation(
            id=f"c{i}", created_at=now, updated_at=now,
            title=f"Chat {i}", source="openai" if i <= 2 else "anthropic",
            model="gpt-4", tags=["python"] if i % 2 == 0 else ["rust"],
        )
        conv.add_message(
            Message(id="m1", role="user", content=[text_block(f"topic {i}")])
        )
        conv.add_message(
            Message(
                id="m2", role="assistant",
                content=[text_block(f"answer {i}")], parent_id="m1",
            )
        )
        d.save_conversation(conv)
    return d


class TestCreateServer:
    def test_creates_server(self, db):
        server = create_server(db=db)
        assert server is not None

    def test_creates_server_with_sql_write(self, db):
        server = create_server(db=db, sql_write=True)
        assert server._test_sql_write is True

    def test_server_has_test_db(self, db):
        server = create_server(db=db)
        assert server._test_db is db

    def test_server_has_six_tools(self, db):
        server = create_server(db=db)
        tool_names = [t.name for t in server._tool_manager._tools.values()]
        assert sorted(tool_names) == sorted([
            "execute_sql", "get_conversation", "get_conversations",
            "update_conversations", "append_message", "add_note",
        ])

    def test_server_has_two_resources(self, db):
        server = create_server(db=db)
        # Resources are registered — we check the count
        resource_names = list(server._resource_manager._resources.keys())
        assert len(resource_names) == 2


class TestExecuteSQL:
    def test_select(self, db):
        result = db.execute_sql("SELECT id, title FROM conversations")
        assert len(result) == 1
        assert result[0]["id"] == "c1"

    def test_select_messages(self, db):
        result = db.execute_sql(
            "SELECT id, role FROM messages WHERE conversation_id=?", ("c1",)
        )
        assert len(result) == 2

    def test_pragma(self, db):
        result = db.execute_sql("PRAGMA table_info(conversations)")
        assert len(result) > 0


class TestGetConversation:
    """Test get_conversation through the DB methods it calls."""

    def test_metadata_mode(self, db):
        """id only → metadata + path summaries."""
        conv = db.load_conversation("c1")
        assert conv is not None
        meta = _conv_metadata(conv, db)
        assert meta["id"] == "c1"
        assert meta["title"] == "Test"
        assert meta["starred"] is False
        assert meta["pinned"] is False
        assert meta["archived"] is False
        assert meta["tags"] == ["python"]

    def test_metadata_paths(self, db):
        paths = db.list_paths("c1")
        assert len(paths) >= 1
        assert paths[0]["message_count"] == 2

    def test_messages_mode_by_index(self, db):
        messages = db.get_path_messages("c1", path_index=0)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_messages_mode_by_leaf(self, db):
        messages = db.get_path_messages("c1", leaf_message_id="m2")
        assert len(messages) == 2

    def test_messages_mode_with_offset(self, db):
        messages = db.get_path_messages("c1", offset=1)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_messages_mode_with_limit(self, db):
        messages = db.get_path_messages("c1", limit=1)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_not_found(self, db):
        assert db.load_conversation("nonexistent") is None

    def test_invalid_path_index(self, db):
        with pytest.raises(ValueError, match="out of range"):
            db.get_path_messages("c1", path_index=99)

    def test_invalid_leaf_id(self, db):
        with pytest.raises(ValueError, match="not found"):
            db.get_path_messages("c1", leaf_message_id="nonexistent")

    def test_export_markdown(self, db):
        conv = db.load_conversation("c1")
        paths = conv.get_all_paths()
        lines = [f"# {conv.title or conv.id}\n"]
        for i, path in enumerate(paths):
            if len(paths) > 1:
                lines.append(f"\n## Path {i}\n")
            for msg in path:
                lines.append(f"**{msg.role}**: {msg.get_text()}\n")
        md = "\n".join(lines)
        assert "# Test" in md
        assert "**user**: hello" in md

    def test_export_json(self, db):
        conv = db.load_conversation("c1")
        exported = json.dumps({
            "id": conv.id, "title": conv.title,
            "messages": [
                {"id": m.id, "role": m.role, "content": m.content, "parent_id": m.parent_id}
                for m in conv.messages.values()
            ],
        }, indent=2)
        data = json.loads(exported)
        assert data["id"] == "c1"
        assert len(data["messages"]) == 2


class TestUpdateConversations:
    """Test bulk update (the DB method is per-id, MCP layer loops)."""

    def test_single_update(self, db):
        db.update_conversation("c1", starred=True)
        conv = db.load_conversation("c1")
        assert conv.starred_at is not None

    def test_update_returns_state(self, db):
        db.update_conversation("c1", title="New Title")
        conv = db.load_conversation("c1")
        meta = _conv_metadata(conv, db)
        assert meta["title"] == "New Title"
        assert meta["tags"] == ["python"]

    def test_bulk_tags(self, multi_db):
        """Simulate bulk tagging of multiple conversations."""
        for cid in ["c1", "c2", "c3"]:
            multi_db.update_conversation(cid, add_tags=["bulk-tagged"])
        for cid in ["c1", "c2", "c3"]:
            conv = multi_db.load_conversation(cid)
            assert "bulk-tagged" in conv.tags

    def test_update_not_found(self, db):
        with pytest.raises(ValueError, match="not found"):
            db.update_conversation("nonexistent", title="x")

    def test_pin_unpin(self, db):
        db.update_conversation("c1", pinned=True)
        assert db.load_conversation("c1").pinned_at is not None
        db.update_conversation("c1", pinned=False)
        assert db.load_conversation("c1").pinned_at is None

    def test_archive(self, db):
        db.update_conversation("c1", archived=True)
        assert db.load_conversation("c1").archived_at is not None

    def test_sensitive(self, db):
        db.update_conversation("c1", sensitive=True)
        assert db.load_conversation("c1").sensitive is True

    def test_summary(self, db):
        db.update_conversation("c1", summary="A summary")
        assert db.load_conversation("c1").summary == "A summary"

    def test_metadata_merge(self, db):
        db.update_conversation("c1", metadata={"key": "value"})
        conv = db.load_conversation("c1")
        assert conv.metadata["key"] == "value"


class TestUpdateConversationsEnrichments:
    """Tests for enrichment support in update_conversations MCP tool."""

    def test_add_enrichments(self, db):
        """Adding enrichments via update_conversations saves them."""
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        result = tool_fn(
            ids=["c1"],
            add_enrichments=[
                {"type": "topic", "value": "greetings", "source": "claude"},
            ],
        )
        assert len(result["updated"]) == 1
        enrichments = result["updated"][0]["enrichments"]
        assert len(enrichments) == 1
        assert enrichments[0]["type"] == "topic"
        assert enrichments[0]["value"] == "greetings"

    def test_add_multiple_enrichments(self, db):
        """Can add multiple enrichments in one call."""
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        result = tool_fn(
            ids=["c1"],
            add_enrichments=[
                {"type": "topic", "value": "greetings", "source": "claude"},
                {"type": "importance", "value": "low", "source": "heuristic", "confidence": 0.3},
            ],
        )
        enrichments = result["updated"][0]["enrichments"]
        assert len(enrichments) == 2

    def test_remove_enrichments(self, db):
        """Removing enrichments via update_conversations deletes them."""
        db.save_enrichment("c1", "topic", "greetings", "claude")
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        result = tool_fn(
            ids=["c1"],
            remove_enrichments=[{"type": "topic", "value": "greetings"}],
        )
        enrichments = result["updated"][0]["enrichments"]
        assert len(enrichments) == 0

    def test_enrichment_validation_invalid_type(self, db):
        """Invalid enrichment type raises ToolError."""
        from fastmcp.exceptions import ToolError
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        with pytest.raises(ToolError, match="Invalid enrichment type"):
            tool_fn(
                ids=["c1"],
                add_enrichments=[
                    {"type": "invalid_type", "value": "x", "source": "claude"},
                ],
            )

    def test_enrichment_validation_invalid_source(self, db):
        """Invalid enrichment source raises ToolError."""
        from fastmcp.exceptions import ToolError
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        with pytest.raises(ToolError, match="Invalid enrichment source"):
            tool_fn(
                ids=["c1"],
                add_enrichments=[
                    {"type": "topic", "value": "x", "source": "invalid_source"},
                ],
            )

    def test_enrichment_validation_confidence_range(self, db):
        """Confidence outside 0.0-1.0 raises ToolError."""
        from fastmcp.exceptions import ToolError
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        with pytest.raises(ToolError, match="Confidence must be"):
            tool_fn(
                ids=["c1"],
                add_enrichments=[
                    {"type": "topic", "value": "x", "source": "claude", "confidence": 1.5},
                ],
            )

    def test_add_and_remove_enrichments_same_call(self, db):
        """Can add and remove enrichments in same update call."""
        db.save_enrichment("c1", "topic", "old_topic", "claude")
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        result = tool_fn(
            ids=["c1"],
            add_enrichments=[
                {"type": "topic", "value": "new_topic", "source": "claude"},
            ],
            remove_enrichments=[{"type": "topic", "value": "old_topic"}],
        )
        enrichments = result["updated"][0]["enrichments"]
        values = [e["value"] for e in enrichments]
        assert "new_topic" in values
        assert "old_topic" not in values

    def test_remove_enrichments_missing_keys(self, db):
        """remove_enrichments with missing type/value raises ToolError."""
        from fastmcp.exceptions import ToolError
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        with pytest.raises(ToolError, match="must have 'type' and 'value'"):
            tool_fn(ids=["c1"], remove_enrichments=[{"type": "topic"}])

    def test_add_enrichments_missing_value(self, db):
        """add_enrichments with missing value raises ToolError."""
        from fastmcp.exceptions import ToolError
        server = create_server(db=db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        with pytest.raises(ToolError, match="non-empty 'value'"):
            tool_fn(
                ids=["c1"],
                add_enrichments=[{"type": "topic", "source": "claude"}],
            )

    def test_enrichments_applied_to_multiple_conversations(self, multi_db):
        """Enrichments are applied to all conversation IDs in the batch."""
        server = create_server(db=multi_db, sql_write=True)
        tool_fn = _get_tool_fn(server, "update_conversations")
        result = tool_fn(
            ids=["c1", "c2"],
            add_enrichments=[
                {"type": "topic", "value": "shared_topic", "source": "claude"},
            ],
        )
        assert len(result["updated"]) == 2
        for conv in result["updated"]:
            topics = [e["value"] for e in conv["enrichments"] if e["type"] == "topic"]
            assert "shared_topic" in topics


class TestGetConversations:
    """Test bulk conversation retrieval."""

    def test_by_tag(self, multi_db):
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(tag="python")
        assert len(result) >= 1
        for r in result:
            assert "python" in r["tags"]

    def test_by_source(self, multi_db):
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(source="openai")
        assert len(result) >= 1
        for r in result:
            assert r["source"] == "openai"

    def test_by_ids(self, multi_db):
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(ids=["c1", "c3"])
        assert len(result) == 2
        ids = {r["id"] for r in result}
        assert ids == {"c1", "c3"}

    def test_preview_mode(self, multi_db):
        """Default: includes first/last message preview, not full messages."""
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(tag="python")
        assert len(result) >= 1
        r = result[0]
        assert "first_message" in r
        assert "preview" in r["first_message"]
        assert "messages" not in r

    def test_include_messages(self, multi_db):
        """include_messages=True returns full message content."""
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(tag="python", include_messages=True)
        assert len(result) >= 1
        r = result[0]
        assert "messages" in r
        assert len(r["messages"]) >= 2

    def test_includes_enrichments(self, multi_db):
        multi_db.save_enrichment("c2", "topic", "coding", "claude")
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(ids=["c2"])
        assert len(result) == 1
        assert len(result[0]["enrichments"]) == 1
        assert result[0]["enrichments"][0]["value"] == "coding"

    def test_limit(self, multi_db):
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(source="openai", limit=1)
        assert len(result) == 1

    def test_requires_filter(self, multi_db):
        from fastmcp.exceptions import ToolError
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        with pytest.raises(ToolError, match="filter"):
            fn()

    def test_fts_search(self, multi_db):
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(search="topic 2")
        assert len(result) >= 1

    def test_starred_filter(self, multi_db):
        multi_db.update_conversation("c1", starred=True)
        server = create_server(db=multi_db)
        fn = _get_tool_fn(server, "get_conversations")
        result = fn(starred=True)
        assert len(result) == 1
        assert result[0]["id"] == "c1"


class TestAppendMessage:
    def test_append(self, db):
        db.append_message(
            "c1",
            Message(id="m3", role="user", content=[text_block("more")], parent_id="m2"),
        )
        conv = db.load_conversation("c1")
        assert conv.message_count == 3

    def test_append_returns_updated_state(self, db):
        """After append, we should be able to get updated conversation metadata."""
        db.append_message(
            "c1",
            Message(id="m3", role="user", content=[text_block("more")], parent_id="m2"),
        )
        conv = db.load_conversation("c1")
        meta = _conv_metadata(conv, db)
        assert meta["message_count"] == 3

    def test_append_updates_count(self, db):
        db.append_message(
            "c1",
            Message(id="m3", role="user", content=[text_block("q1")], parent_id="m2"),
        )
        db.append_message(
            "c1",
            Message(id="m4", role="assistant", content=[text_block("a1")], parent_id="m3"),
        )
        conv = db.load_conversation("c1")
        assert conv.message_count == 4

    def test_append_not_found(self, db):
        with pytest.raises(ValueError, match="not found"):
            db.append_message(
                "nonexistent",
                Message(id="m3", role="user", content=[text_block("x")]),
            )


class TestResources:
    def test_schema_resource(self, db):
        schema = db.get_schema()
        assert "CREATE TABLE conversations" in schema
        assert "CREATE TABLE messages" in schema
        assert "messages_fts" in schema

    def test_statistics(self, db):
        stats = db.get_statistics()
        assert stats["total_conversations"] == 1
        assert stats["total_messages"] == 2
        assert "openai" in stats["sources"]
        assert "python" in stats["tags"]


class TestConvMetadata:
    """Test the _conv_metadata helper used by all tools."""

    def test_fields(self, db):
        conv = db.load_conversation("c1")
        meta = _conv_metadata(conv, db)
        assert "id" in meta
        assert "title" in meta
        assert "tags" in meta
        assert "starred" in meta
        assert "pinned" in meta
        assert "archived" in meta
        assert isinstance(meta["starred"], bool)
        assert isinstance(meta["pinned"], bool)
        assert isinstance(meta["archived"], bool)

    def test_tags_list(self, db):
        conv = db.load_conversation("c1")
        meta = _conv_metadata(conv, db)
        assert meta["tags"] == ["python"]

    def test_boolean_flags_default_false(self, db):
        conv = db.load_conversation("c1")
        meta = _conv_metadata(conv, db)
        assert meta["starred"] is False
        assert meta["pinned"] is False
        assert meta["archived"] is False

    def test_enrichments_in_metadata(self, db):
        db.save_enrichment("c1", "topic", "greetings", "claude")
        conv = db.load_conversation("c1")
        meta = _conv_metadata(conv, db)
        assert "enrichments" in meta
        assert len(meta["enrichments"]) == 1
        assert meta["enrichments"][0]["type"] == "topic"

    def test_provenance_in_metadata(self, db):
        db.save_provenance("c1", source_type="openai", source_file="export.json")
        conv = db.load_conversation("c1")
        meta = _conv_metadata(conv, db)
        assert "provenance" in meta
        assert len(meta["provenance"]) == 1
        assert meta["provenance"][0]["source_type"] == "openai"

    def test_empty_enrichments_and_provenance(self, db):
        conv = db.load_conversation("c1")
        meta = _conv_metadata(conv, db)
        assert meta["enrichments"] == []
        assert meta["provenance"] == []


class TestBranchingConversation:
    """Test with a branching conversation tree."""

    @pytest.fixture
    def branching_db(self, tmp_db_path):
        d = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(
            id="c2", created_at=now, updated_at=now,
            title="Branching", source="test",
        )
        conv.add_message(Message(id="m1", role="user", content=[text_block("start")]))
        conv.add_message(Message(id="m2a", role="assistant", content=[text_block("branch a")], parent_id="m1"))
        conv.add_message(Message(id="m2b", role="assistant", content=[text_block("branch b")], parent_id="m1"))
        d.save_conversation(conv)
        return d

    def test_list_paths_branching(self, branching_db):
        paths = branching_db.list_paths("c2")
        assert len(paths) == 2

    def test_get_path_by_leaf(self, branching_db):
        msgs_a = branching_db.get_path_messages("c2", leaf_message_id="m2a")
        assert len(msgs_a) == 2
        assert msgs_a[1]["id"] == "m2a"

        msgs_b = branching_db.get_path_messages("c2", leaf_message_id="m2b")
        assert len(msgs_b) == 2
        assert msgs_b[1]["id"] == "m2b"


class TestSqlSecurity:
    def test_readonly_cannot_be_bypassed_via_pragma(self, db):
        """Agent SQL must not flip query_only off and then write (MCP-1)."""
        server = create_server(db=db, sql_write=False)
        fn = _get_tool_fn(server, "execute_sql")
        try:
            fn(sql="PRAGMA query_only=OFF")  # denial is fine; success must not unlock writes
        except Exception:
            pass
        with pytest.raises(Exception):
            fn(sql="UPDATE conversations SET title='HACKED' WHERE id='c1'")
        rows = db.execute_sql("SELECT title FROM conversations WHERE id='c1'")
        assert rows[0]["title"] == "Test", "readonly bypass mutated the database"

    def test_attach_database_is_denied(self, db, tmp_path):
        """ATTACH must be denied so agent SQL cannot reach other database files (MCP-2)."""
        import sqlite3
        other = tmp_path / "secret.db"
        oc = sqlite3.connect(str(other))
        oc.execute("CREATE TABLE s(v)")
        oc.execute("INSERT INTO s VALUES('SECRET')")
        oc.commit()
        oc.close()
        server = create_server(db=db, sql_write=False)
        fn = _get_tool_fn(server, "execute_sql")
        with pytest.raises(Exception):
            fn(sql=f"ATTACH DATABASE '{other}' AS x")

    def test_attach_denied_even_when_writable(self, db, tmp_path):
        """ATTACH is denied even on a writable server (no llm-memex query needs it)."""
        import sqlite3
        other = tmp_path / "secret.db"
        oc = sqlite3.connect(str(other))
        oc.execute("CREATE TABLE s(v)")
        oc.commit()
        oc.close()
        server = create_server(db=db, sql_write=True)
        fn = _get_tool_fn(server, "execute_sql")
        with pytest.raises(Exception):
            fn(sql=f"ATTACH DATABASE '{other}' AS x")

    def test_append_message_denied_on_readonly(self, db):
        """append_message must refuse cleanly on a readonly DB, like the other write tools (MCP-4)."""
        from fastmcp.exceptions import ToolError
        server = create_server(db=db, sql_write=False)
        fn = _get_tool_fn(server, "append_message")
        with pytest.raises(ToolError):
            fn(conversation_id="c1", role="user",
               content=[{"type": "text", "text": "x"}])


class TestSqlBoundsAndContract:
    def test_get_conversations_search_no_silent_cap(self, tmp_db_path):
        """get_conversations(search=...) must not silently cap the FTS pre-filter
        at 1000 rows and drop the newest match (ARCH-1)."""
        from datetime import timedelta
        d = Database(tmp_db_path)
        base = datetime(2026, 1, 1, 0, 0, 0)
        for i in range(1001):
            t = base + timedelta(minutes=i)  # i=1000 newest
            conv = Conversation(id=f"c{i:04d}", created_at=t, updated_at=t,
                                title=f"chat {i}", source="test")
            conv.add_message(Message(id="m1", role="user",
                                     content=[text_block("needle here")]))
            # keep updated_at strictly increasing
            conv.updated_at = t
            d.save_conversation(conv)
        server = create_server(db=d, sql_write=False)
        fn = _get_tool_fn(server, "get_conversations")
        rows = fn(search="needle", limit=5)
        ids = [r["id"] for r in rows]
        assert "c1000" in ids
        assert ids[0] == "c1000"

    def test_execute_sql_caps_huge_result(self, db):
        """execute_sql must not materialize an unbounded result set; a query
        exceeding the row cap is rejected with guidance (MCP-5)."""
        from fastmcp.exceptions import ToolError
        server = create_server(db=db, sql_write=False)
        fn = _get_tool_fn(server, "execute_sql")
        huge = ("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL "
                "SELECT n+1 FROM r WHERE n < 20000) SELECT n FROM r")
        with pytest.raises(ToolError):
            fn(sql=huge)
        # a small query still works
        assert fn(sql="SELECT 1 AS x")[0]["x"] == 1

    def test_unknown_db_name_raises_tool_error(self, monkeypatch):
        """An unknown db name must surface as a clean ToolError, not a raw
        ValueError, across the tools (ARCH-3)."""
        from fastmcp.exceptions import ToolError
        from llm_memex import mcp as mcp_mod

        class FakeReg:
            def get_db(self, name=None):
                raise ValueError(f"Unknown database: {name}")

        monkeypatch.setattr(mcp_mod, "_get_registry", lambda ctx: FakeReg())
        with pytest.raises(ToolError):
            mcp_mod._get_db_from_ctx(object(), object(), "does-not-exist")


