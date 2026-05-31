"""Memex MCP server -- the primary interface."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from llm_memex.config import load_config, DatabaseRegistry
from llm_memex.db import _fmt_dt
from llm_memex.models import Message

# Absolute filesystem paths, scrubbed from error strings before they cross the
# MCP boundary so the agent cannot use them to map the host.
_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}")
_READONLY_HINTS = ("not authorized", "readonly database", "read-only", "query_only")


def _sql_error_to_tool_error(e: Exception, readonly: bool) -> ToolError:
    """Map a database error to a ToolError. A write blocked on a read-only
    database (by the authorizer or query_only) becomes the standard
    'writes disabled' message; every other message has host paths scrubbed."""
    msg = str(e)
    if readonly and any(h in msg.lower() for h in _READONLY_HINTS):
        return ToolError(
            "SQL writes are disabled for this database. "
            "Set MEMEX_SQL_WRITE=true to enable."
        )
    return ToolError(_PATH_RE.sub("<path>", msg))


@asynccontextmanager
async def lifespan(server):
    """Initialize database registry from config on server startup.

    Resolution order matches the CLI:
      1. MEMEX_CONFIG env var (if set and file exists)
      2. ~/.memex/config.yaml (if exists)
      3. MEMEX_DATABASE_PATH env var (single-db fallback)
      4. ~/.memex/default (last-resort single-db fallback)
    """
    import os
    import logging
    from pathlib import Path

    log = logging.getLogger("llm_memex.mcp")
    config_path = os.environ.get("MEMEX_CONFIG")
    if not config_path:
        default_yaml = Path.home() / ".memex" / "config.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)
    config = load_config(config_path)
    if not config.get("databases"):
        fallback = Path.home() / ".memex" / "default"
        config["databases"] = {"default": {"path": str(fallback)}}
        config["primary"] = "default"
        log.warning(
            "No llm-memex config found. Using fallback database at %s. "
            "Set MEMEX_CONFIG or create ~/.memex/config.yaml to configure.",
            fallback,
        )
    registry = DatabaseRegistry(config)
    try:
        yield {"registry": registry}
    finally:
        registry.close()


def create_server(db=None, sql_write=False):
    """Create the MCP server.

    Pass db for testing (skips lifespan and uses injected database).
    """
    mcp = FastMCP("llm-memex", lifespan=lifespan if db is None else None)
    if db is not None:
        if not sql_write and not db.readonly:
            db.conn.execute("PRAGMA query_only=ON")
            db.readonly = True
        mcp._test_db = db
        mcp._test_sql_write = sql_write
    _register_tools(mcp)
    _register_resources(mcp)
    return mcp


def _get_registry(ctx) -> DatabaseRegistry | None:
    """Get the DatabaseRegistry from the lifespan context, or None for test injection."""
    try:
        return ctx.request_context.lifespan_context["registry"]
    except (AttributeError, TypeError, KeyError):
        return None


def _get_db_from_ctx(mcp, ctx, db_name=None):
    """Get database from either lifespan context or test injection."""
    registry = _get_registry(ctx)
    if registry is not None:
        return registry.get_db(db_name)
    return mcp._test_db


def _extract_text(content) -> str:
    """Extract text from a message content field (JSON string or list of blocks)."""
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (ValueError, TypeError):
            return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        )
    return str(content) if content else ""


def _conv_metadata(conv, db) -> dict:
    """Build conversation metadata dict with boolean flags, tags, enrichments, provenance."""
    tags = [
        t["tag"]
        for t in db.execute_sql(
            "SELECT tag FROM tags WHERE conversation_id=?", (conv.id,)
        )
    ]
    return {
        "id": conv.id,
        "title": conv.title,
        "source": conv.source,
        "model": conv.model,
        "summary": conv.summary,
        "message_count": conv.message_count,
        "created_at": _fmt_dt(conv.created_at),
        "updated_at": _fmt_dt(conv.updated_at),
        "parent_conversation_id": conv.parent_conversation_id,
        "starred": conv.starred_at is not None,
        "pinned": conv.pinned_at is not None,
        "archived": conv.archived_at is not None,
        "sensitive": conv.sensitive,
        "tags": tags,
        "metadata": conv.metadata,
        "enrichments": db.get_enrichments(conv.id),
        "provenance": db.get_provenance(conv.id),
    }


def _register_tools(mcp: FastMCP):
    """Register all MCP tools on the server."""

    @mcp.tool(annotations={"readOnlyHint": True})
    def execute_sql(
        sql: Annotated[str, Field(description="SQL query to execute")],
        params: Annotated[list | None, Field(description="Query parameters for ? placeholders")] = None,
        db: Annotated[str | None, Field(description="Target database name")] = None,
        ctx: Context = None,
    ) -> list[dict]:
        """Run a SQL query against the database. Read-only by default (writes and ATTACH are blocked by a SQLite authorizer; set MEMEX_SQL_WRITE=true to allow writes).

Use llm-memex://schema resource for full DDL. Common queries:

List conversations:
  SELECT id, title, source, model, message_count, created_at, updated_at
  FROM conversations ORDER BY updated_at DESC LIMIT 20

FTS message search:
  SELECT m.conversation_id, c.title, m.id, m.role, m.content
  FROM messages_fts f
  JOIN messages m ON m.conversation_id = f.conversation_id AND m.id = f.message_id
  JOIN conversations c ON c.id = m.conversation_id
  WHERE messages_fts MATCH 'search terms'
  LIMIT 20

Filter by tag:
  SELECT c.id, c.title FROM conversations c
  JOIN tags t ON c.id = t.conversation_id WHERE t.tag = 'python'

Enrichments:
  SELECT e.*, c.title FROM enrichments e
  JOIN conversations c ON c.id = e.conversation_id
  WHERE e.type = 'topic'

Starred/pinned (use IS NOT NULL for boolean timestamp columns):
  SELECT id, title FROM conversations WHERE starred_at IS NOT NULL
"""
        database = _get_db_from_ctx(mcp, ctx, db)
        try:
            return database.execute_sql(sql, tuple(params) if params else ())
        except Exception as e:
            raise _sql_error_to_tool_error(e, database.readonly)

    @mcp.tool(annotations={"readOnlyHint": True})
    def get_conversation(
        id: Annotated[str, Field(description="Conversation ID")],
        path_index: Annotated[int | None, Field(description="Path index to read messages from")] = None,
        leaf_message_id: Annotated[str | None, Field(description="Leaf message ID to trace path from")] = None,
        offset: Annotated[int, Field(description="Skip first N messages (only in messages mode)")] = 0,
        limit: Annotated[int | None, Field(description="Max messages to return (only in messages mode)")] = None,
        format: Annotated[str | None, Field(description="Export format: 'markdown' or 'json'. When set, returns formatted string.")] = None,
        db: Annotated[str | None, Field(description="Target database")] = None,
        ctx: Context = None,
    ) -> dict | str:
        """Get conversation metadata, messages along a path, or export. Three modes:
        - id only → metadata + all path summaries
        - id + path_index/leaf_message_id → messages along that path (with offset/limit)
        - id + format → exported string (markdown or json)
        """
        database = _get_db_from_ctx(mcp, ctx, db)
        conv = database.load_conversation(id)
        if conv is None:
            raise ToolError(f"Conversation not found: {id}")

        # Export mode
        if format is not None:
            if format == "json":
                return json.dumps({
                    "id": conv.id,
                    "title": conv.title,
                    "messages": [
                        {
                            "id": m.id, "role": m.role,
                            "content": m.content, "parent_id": m.parent_id,
                        }
                        for m in conv.messages.values()
                    ],
                }, indent=2)
            # Default export: markdown
            paths = conv.get_all_paths()
            if path_index is not None:
                if path_index < 0 or path_index >= len(paths):
                    raise ToolError(f"Path index out of range: {path_index}")
                paths = [paths[path_index]]
            lines = [f"# {conv.title or conv.id}\n"]
            for i, path in enumerate(paths):
                if len(paths) > 1:
                    lines.append(f"\n## Path {i}\n")
                for msg in path:
                    lines.append(f"**{msg.role}**: {msg.get_content_md()}\n")
            return "\n".join(lines)

        # Messages mode: path_index or leaf_message_id specified
        if path_index is not None or leaf_message_id is not None:
            try:
                messages = database.get_path_messages(
                    id, path_index=path_index,
                    leaf_message_id=leaf_message_id,
                    offset=offset, limit=limit,
                )
            except ValueError as e:
                raise ToolError(str(e))
            return {
                "conversation": _conv_metadata(conv, database),
                "messages": messages,
            }

        # Metadata mode: id only
        meta = _conv_metadata(conv, database)
        meta["paths"] = database.list_paths(id)
        return meta

    @mcp.tool(annotations={"readOnlyHint": True})
    def get_conversations(
        tag: Annotated[str | None, Field(description="Filter by tag")] = None,
        source: Annotated[str | None, Field(description="Filter by source (openai, anthropic, etc.)")] = None,
        model: Annotated[str | None, Field(description="Filter by model")] = None,
        search: Annotated[str | None, Field(description="FTS5 full-text search across messages")] = None,
        ids: Annotated[list[str] | None, Field(description="Specific conversation IDs")] = None,
        starred: Annotated[bool | None, Field(description="Filter by starred status")] = None,
        pinned: Annotated[bool | None, Field(description="Filter by pinned status")] = None,
        include_messages: Annotated[bool, Field(description="Include full message content (default false — returns metadata + preview)")] = False,
        limit: Annotated[int, Field(description="Max conversations to return (default 20)")] = 20,
        db: Annotated[str | None, Field(description="Target database")] = None,
        ctx: Context = None,
    ) -> list[dict]:
        """Retrieve multiple conversations with metadata, tags, enrichments, and optionally messages in a single call.

Use this instead of execute_sql + get_conversation × N for bulk retrieval.
Without include_messages, returns metadata + first/last message preview (fast orientation).
With include_messages=True, returns full message content for all conversations.

Filters are combined with AND. At least one filter or ids must be provided.

Examples:
  get_conversations(tag="python", limit=5)                  → 5 python-tagged conversations
  get_conversations(search="bayesian inference", limit=10)  → FTS search with previews
  get_conversations(source="anthropic", include_messages=True, limit=3) → full content
  get_conversations(starred=True)                           → all starred conversations
  get_conversations(ids=["abc", "def"])                     → specific conversations by ID
"""
        if not any([tag, source, model, search, ids, starred is not None, pinned is not None]):
            raise ToolError("Provide at least one filter: tag, source, model, search, ids, starred, or pinned")

        database = _get_db_from_ctx(mcp, ctx, db)

        conds = []
        params = []

        if ids:
            placeholders = ",".join("?" for _ in ids)
            conds.append(f"c.id IN ({placeholders})")
            params.extend(ids)
        if source:
            conds.append("c.source = ?")
            params.append(source)
        if model:
            conds.append("c.model = ?")
            params.append(model)
        if starred is True:
            conds.append("c.starred_at IS NOT NULL")
        elif starred is False:
            conds.append("c.starred_at IS NULL")
        if pinned is True:
            conds.append("c.pinned_at IS NOT NULL")
        elif pinned is False:
            conds.append("c.pinned_at IS NULL")
        if tag:
            conds.append(
                "EXISTS(SELECT 1 FROM tags t WHERE t.conversation_id = c.id AND t.tag = ?)"
            )
            params.append(tag)

        # FTS search: find matching conversation IDs first
        if search:
            from llm_memex.db import _sanitize_fts_query
            fts_q = _sanitize_fts_query(search)
            if not fts_q:
                # Query sanitized to empty (only stop-words, punctuation, etc.)
                # Return empty rather than silently ignoring the search filter.
                return []
            try:
                fts_rows = database.execute_sql(
                    "SELECT DISTINCT conversation_id FROM messages_fts "
                    "WHERE messages_fts MATCH ? LIMIT 1000",
                    (fts_q,),
                )
                fts_ids = [r["conversation_id"] for r in fts_rows]
            except Exception:
                fts_ids = []
            if not fts_ids:
                return []
            placeholders = ",".join("?" for _ in fts_ids)
            conds.append(f"c.id IN ({placeholders})")
            params.extend(fts_ids)

        where = " AND ".join(conds) if conds else "1=1"
        params.append(limit)

        try:
            rows = database.execute_sql(
                f"SELECT c.id, c.title, c.source, c.model, c.summary, "
                f"c.message_count, c.created_at, c.updated_at, "
                f"c.starred_at, c.pinned_at, c.archived_at, "
                f"c.parent_conversation_id, c.sensitive, c.metadata "
                f"FROM conversations c WHERE {where} "
                f"ORDER BY c.updated_at DESC LIMIT ?",
                tuple(params),
            )
        except Exception as e:
            raise ToolError(str(e))

        result = []
        for row in rows:
            conv_id = row["id"]

            # Tags
            tags = [
                r["tag"] for r in database.execute_sql(
                    "SELECT tag FROM tags WHERE conversation_id = ?", (conv_id,)
                )
            ]

            # Enrichments
            enrichments = database.get_enrichments(conv_id)

            entry = {
                "id": conv_id,
                "title": row["title"],
                "source": row["source"],
                "model": row["model"],
                "summary": row["summary"],
                "message_count": row["message_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "starred": row["starred_at"] is not None,
                "pinned": row["pinned_at"] is not None,
                "archived": row["archived_at"] is not None,
                "sensitive": bool(row["sensitive"]),
                "tags": tags,
                "enrichments": enrichments,
            }

            if include_messages:
                # Full message content
                messages = database.execute_sql(
                    "SELECT id, role, content, parent_id, model, created_at "
                    "FROM messages WHERE conversation_id = ? ORDER BY created_at",
                    (conv_id,),
                )
                entry["messages"] = messages
            else:
                # Preview: first and last message text
                messages = database.execute_sql(
                    "SELECT id, role, content, created_at "
                    "FROM messages WHERE conversation_id = ? ORDER BY created_at",
                    (conv_id,),
                )
                if messages:
                    first = messages[0]
                    last = messages[-1] if len(messages) > 1 else None
                    entry["first_message"] = {
                        "role": first["role"],
                        "preview": _extract_text(first["content"])[:500],
                    }
                    if last:
                        entry["last_message"] = {
                            "role": last["role"],
                            "preview": _extract_text(last["content"])[:500],
                        }

            result.append(entry)

        return result

    VALID_ENRICHMENT_TYPES = {"summary", "topic", "importance", "excerpt"}
    VALID_ENRICHMENT_SOURCES = {"user", "claude", "heuristic"}

    @mcp.tool(annotations={"idempotentHint": True})
    def update_conversations(
        ids: Annotated[list[str], Field(description="Conversation IDs to update (1..N)")],
        title: Annotated[str | None, Field(description="New title")] = None,
        summary: Annotated[str | None, Field(description="New summary")] = None,
        starred: Annotated[bool | None, Field(description="Star/unstar")] = None,
        pinned: Annotated[bool | None, Field(description="Pin/unpin")] = None,
        archived: Annotated[bool | None, Field(description="Archive/unarchive")] = None,
        sensitive: Annotated[bool | None, Field(description="Mark sensitive")] = None,
        add_tags: Annotated[list[str] | None, Field(description="Tags to add")] = None,
        remove_tags: Annotated[list[str] | None, Field(description="Tags to remove")] = None,
        metadata: Annotated[dict | None, Field(description="Metadata to merge")] = None,
        add_enrichments: Annotated[list[dict] | None, Field(description="Enrichments to add: [{type, value, source, confidence?}]")] = None,
        remove_enrichments: Annotated[list[dict] | None, Field(description="Enrichments to remove: [{type, value}]")] = None,
        db: Annotated[str | None, Field(description="Target database")] = None,
        ctx: Context = None,
    ) -> dict:
        """Update conversation properties. Only provided fields change."""
        # Validate all enrichments upfront to avoid partial updates
        if add_enrichments:
            for e in add_enrichments:
                if not e.get("value"):
                    raise ToolError("Enrichment must have a non-empty 'value'")
                if e.get("type") not in VALID_ENRICHMENT_TYPES:
                    raise ToolError(
                        f"Invalid enrichment type: {e.get('type')}. "
                        f"Must be one of: {', '.join(sorted(VALID_ENRICHMENT_TYPES))}"
                    )
                if e.get("source") not in VALID_ENRICHMENT_SOURCES:
                    raise ToolError(
                        f"Invalid enrichment source: {e.get('source')}. "
                        f"Must be one of: {', '.join(sorted(VALID_ENRICHMENT_SOURCES))}"
                    )
                conf = e.get("confidence")
                if conf is not None and (conf < 0.0 or conf > 1.0):
                    raise ToolError(f"Confidence must be 0.0-1.0, got: {conf}")
        if remove_enrichments:
            for e in remove_enrichments:
                if "type" not in e or "value" not in e:
                    raise ToolError("Each remove_enrichments entry must have 'type' and 'value'")

        database = _get_db_from_ctx(mcp, ctx, db)
        # Fail fast on readonly databases: the whole call is going to fail,
        # don't crash mid-loop leaving caller with a partial-update state.
        if database.readonly:
            raise ToolError(
                "SQL writes are disabled for this database. "
                "Set MEMEX_SQL_WRITE=true to enable."
            )
        updated = []
        errors = []
        for cid in ids:
            try:
                database.update_conversation(
                    cid, title=title, summary=summary, starred=starred,
                    pinned=pinned, archived=archived, sensitive=sensitive,
                    add_tags=add_tags, remove_tags=remove_tags,
                    metadata=metadata,
                )
                if remove_enrichments:
                    for e in remove_enrichments:
                        database.delete_enrichment(cid, e["type"], e["value"])
                if add_enrichments:
                    database.save_enrichments(cid, add_enrichments)
                conv = database.load_conversation(cid)
                updated.append(_conv_metadata(conv, database))
            except (ValueError, sqlite3.OperationalError) as e:
                errors.append({"id": cid, "error": str(e)})
        return {"updated": updated, "errors": errors}

    @mcp.tool()
    def append_message(
        conversation_id: Annotated[str, Field(description="Conversation ID")],
        role: Annotated[str, Field(description="Message role: user, assistant, system, tool")],
        content: Annotated[list[dict], Field(description="Content blocks array")],
        parent_message_id: Annotated[str | None, Field(description="Parent message ID")] = None,
        message_model: Annotated[str | None, Field(description="Model that generated this message")] = None,
        db: Annotated[str | None, Field(description="Target database")] = None,
        ctx: Context = None,
    ) -> dict:
        """Add a message to the conversation tree. Returns created message and updated conversation metadata."""
        database = _get_db_from_ctx(mcp, ctx, db)
        if database.readonly:
            raise ToolError(
                "SQL writes are disabled for this database. "
                "Set MEMEX_SQL_WRITE=true to enable."
            )
        msg_id = str(uuid.uuid4())
        msg = Message(
            id=msg_id, role=role, content=content,
            parent_id=parent_message_id, model=message_model,
        )
        try:
            database.append_message(conversation_id, msg)
            conv = database.load_conversation(conversation_id)
            return {
                "message_id": msg_id,
                "conversation": _conv_metadata(conv, database),
            }
        except ValueError as e:
            raise ToolError(str(e))

    @mcp.tool()
    def add_note(
        conversation_id: Annotated[str, Field(description="Conversation ID the note attaches to")],
        text: Annotated[str, Field(description="Free-form text content of the note")],
        message_id: Annotated[str | None, Field(description="Message ID for message-level note (omit for conversation-level)")] = None,
        db: Annotated[str | None, Field(description="Target database")] = None,
        ctx: Context = None,
    ) -> dict:
        """Add a free-form text note (marginalia) to a message or conversation.

Use this when the user wants to annotate something in their archive with
their own observations, thoughts, or corrections. If message_id is provided,
creates a message-level note. Otherwise creates a conversation-level note.

Notes are searchable via execute_sql against notes_fts (FTS5):
  SELECT n.* FROM notes_fts f JOIN notes n ON n.id = f.note_id WHERE notes_fts MATCH 'query'
"""
        database = _get_db_from_ctx(mcp, ctx, db)
        if database.readonly:
            raise ToolError(
                "SQL writes are disabled for this database. "
                "Set MEMEX_SQL_WRITE=true to enable."
            )
        try:
            note_id = database.add_note(
                conversation_id=conversation_id,
                message_id=message_id,
                text=text,
            )
        except (ValueError, sqlite3.IntegrityError) as e:
            raise ToolError(str(e))
        return {
            "note_id": note_id,
            "target_kind": "message" if message_id else "conversation",
        }


def _register_resources(mcp: FastMCP):
    """Register all MCP resources on the server."""

    @mcp.resource("llm-memex://schema")
    def schema_resource(ctx: Context = None) -> str:
        """Database schema: DDL, indexes, relationships, FTS5 docs, and query patterns. Read this before writing SQL."""
        db = _get_db_from_ctx(mcp, ctx)
        return db.get_schema()

    @mcp.resource("llm-memex://databases")
    def databases_resource(ctx: Context = None) -> str:
        """Registered databases with statistics."""
        registry = _get_registry(ctx)
        if registry is not None:
            result = {}
            for name, db in registry.all_dbs().items():
                result[name] = db.get_statistics()
                result[name]["primary"] = (name == registry.primary)
            return json.dumps(result, indent=2)
        stats = mcp._test_db.get_statistics()
        stats["primary"] = True
        return json.dumps({"default": stats}, indent=2)


def main():
    """Entry point for `memex mcp`."""
    create_server().run()
