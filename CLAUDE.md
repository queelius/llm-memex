# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ecosystem Context

This repo (`llm-memex/`) is one archive in the `*-memex` ecosystem at `~/github/memex/` (see `../CLAUDE.md`). It covers **AI chat conversations** (ChatGPT, Claude, Gemini, Claude Code). Trails and cross-archive features have moved to `memex/`. This archive is domain-focused: store, index, and expose conversations.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests (coverage is enabled by default via pytest.ini)
pytest tests/llm_memex/ -v

# Run a single test file or test
pytest tests/llm_memex/test_db.py -v
pytest tests/llm_memex/test_db.py::TestDatabase::test_save_and_load -v

# Run with explicit coverage report
pytest tests/llm_memex/ --cov=llm_memex --cov-report=term-missing

# CLI
llm-memex import conversations.json                   # auto-detect format
llm-memex import ~/.claude/projects/ -r               # recursive directory import
llm-memex import file.json --format openai            # force format
llm-memex show                                        # list conversations
llm-memex show <id>                                   # view conversation
llm-memex show --search "topic"                       # FTS search
llm-memex export output.md --format markdown
llm-memex export ./site --format html                 # self-contained SPA directory
llm-memex export ./archive --format arkiv             # JSONL + schema.yaml
llm-memex import --list-formats                       # discover importers
llm-memex export --list-formats                       # discover exporters

# Scripts
llm-memex run --list
llm-memex run redact --words "word1,word2" --level word --apply
llm-memex run redact --pattern-file api_keys.txt --level word --apply
llm-memex run enrich_trivial --apply
llm-memex run note add --conv <id> "annotation text" --apply
llm-memex run note list --conv <id>
llm-memex run note search "query"

# MCP server
llm-memex mcp
```

## Architecture

**Package structure:**
```
llm_memex/
  __init__.py          # version (single source of truth; pyproject.toml reads it dynamically)
  __main__.py          # python -m llm_memex
  models.py            # ContentBlock, Message, Conversation (tree)
  db.py                # Database (raw sqlite3, WAL, FTS5, migrations, schema v6)
  config.py            # YAML config, DatabaseRegistry (multi-database support)
  mcp.py               # FastMCP server (6 tools, 2 resources)
  assets.py            # Asset resolution, copying, media markdown rendering
  cli.py               # argparse CLI (import, export, show, mcp, run)
  scripts/
    __init__.py          # Discovery + runner utilities
    enrich_trivial.py    # Bulk-enrich trivial conversations
    redact.py            # Content redaction (word/message/conversation level)
    note.py              # Marginalia: add/list/search/delete notes
    patterns/            # Built-in regex pattern files
  importers/           # Convention-based: detect() + import_path()
    _claude_code_common.py  # Shared helpers for Claude Code importers
    openai.py, anthropic.py, gemini.py
    claude_code.py          # Claude Code JSONL (conversation_only mode)
    claude_code_full.py     # Claude Code JSONL (full fidelity: tool_use, thinking, tool_result)
  exporters/           # Convention-based: export()
    markdown.py, json_export.py, arkiv_export.py
    html.py              # HTML SPA exporter (directory: index.html + DB + assets)
    html_template.py     # Self-contained HTML5 SPA template (composable functions)
```

**Key design decisions:**
- Raw sqlite3 (no ORM): WAL mode, FTS5, PRAGMA query_only for read-only
- Content blocks as plain dicts with "type" key, not classes
- Conversation trees: Dict[str, Message] with parent_id links, _children index
- Convention-based plugins: importers have `detect()` + `import_path()`, exporters have `export()`, scripts have `register_args()` + `run()`. User overrides in `~/.memex/{importers,exporters,scripts}/`
- FastMCP 2.x (pinned `<3`) with lifespan pattern for DB lifecycle
- Cursor-based keyset pagination (base64-encoded updated_at + id)
- Schema versioning: `schema_version` table + `MIGRATIONS` dict for forward migrations (current: v6)
- Multi-database YAML config (`~/.memex/config.yaml`) with `sql_write: true` to enable writes via MCP
- pyproject.toml version is dynamic from `llm_memex.__version__`

## Database Schema (v7)

Schema v7 migrations: v1→v2 (enrichments, provenance) → v3 (parent_conversation_id) → v4 (notes, notes_fts) → v5 (edges, marginalia v2) → v6 (drop trails, moved to memex federation layer) → v7 (drop edges, also moved to memex federation layer).

**Core tables:**
- `conversations`: PK `id`, boolean timestamps `starred_at`/`pinned_at`/`archived_at` (NULL=false, DATETIME=true), self-referential `parent_conversation_id`
- `messages`: composite PK `(conversation_id, id)`, tree via `parent_id`, content as JSON array of content blocks
- `tags`: PK `(conversation_id, tag)`
- `enrichments`: PK `(conversation_id, type, value)`, types: summary|topic|importance|excerpt, sources: user|claude|heuristic
- `provenance`: PK `(conversation_id, source_type)`, tracks import origin
- `notes` (marginalia): polymorphic `target_kind` ('message' or 'conversation'), FK with `ON DELETE SET NULL` (orphan survival). Marginalia v2 fields: `kind` (default 'freeform'), `anchor_start`/`anchor_end` (character offsets), `anchor_hash` (drift detection), `parent_note_id` (threaded replies, CASCADE delete)

**FTS5 virtual tables:**
- `messages_fts`: columns `conversation_id` (UNINDEXED), `message_id` (UNINDEXED), `text`. Porter + unicode61 tokenizer
- `notes_fts`: columns `note_id` (UNINDEXED), `conversation_id` (UNINDEXED), `message_id` (UNINDEXED), `text`

## MCP Tools

| Tool | Purpose |
|---|---|
| `execute_sql` | Primary read interface for arbitrary SQL. Schema available via `llm-memex://schema`. |
| `get_conversation` | Tree-aware retrieval + export (3 modes: metadata, messages, export) |
| `get_conversations` | Bulk retrieval with filters (ids, tag, source, model, search, starred, pinned) and optional full messages. Collapses the N+1 of execute_sql + get_conversation x N. |
| `update_conversations` | Modify conversation properties, tags, enrichments. Bulk 1..N. |
| `append_message` | Add message to conversation tree with consistency guarantees. |
| `add_note` | Annotate a message or conversation with a free-form text note (marginalia). |

**Resources:** `llm-memex://schema` (DDL + relationships + FTS5 docs), `llm-memex://databases` (multi-db discovery + stats)

All tools accept an optional `db` parameter for multi-database targeting.

### Reading notes via execute_sql

```sql
-- Notes for a conversation
SELECT id, target_kind, message_id, text, kind, created_at FROM notes
WHERE conversation_id = ? ORDER BY created_at

-- FTS search across notes
SELECT n.* FROM notes_fts f JOIN notes n ON n.id = f.note_id
WHERE notes_fts MATCH 'query'
```

## Testing

- Tests in `tests/llm_memex/` organized by module: `test_db.py`, `test_mcp.py`, `test_cli.py`, `test_importers.py`, `test_exporters.py`, `test_models.py`, `test_notes.py`, `test_graph.py`, `test_redact.py`, `test_scripts.py`, `test_assets.py`, `test_config.py`, `test_html_export.py`, `test_integration.py`
- `conftest.py` provides `tmp_db_path` fixture
- Server tests exercise DB methods directly; `create_server(db=db)` sets PRAGMA query_only=ON; use `sql_write=True` for tests that call write tools
- `_get_tool_fn(server, name)` extracts the underlying function from FastMCP for direct invocation in tests
- Coverage is enabled by default in `pytest.ini`

## HTML SPA Export

The HTML exporter outputs a self-contained directory (`index.html` + `conversations.db` + `assets/`):

- **Template architecture**: `get_template(schema_ddl)` assembles from composable functions: `_css_variables()`, `_css_layout()`, `_css_components()`, `_html_structure()`, `_js_core()`, `_js_ui()`, `_js_notes()`, `_js_chat(schema_ddl)`, `_js_settings()`
- **Light/dark mode**: `data-theme` attribute on `<html>`, defaults to OS preference, manual toggle saved to localStorage
- **Librarian chat**: agentic tool-use loop (streaming first round, non-streaming for tool rounds, max 5 rounds). SQL guard rejects non-SELECT/EXPLAIN + `getRowsModified` runtime check
- **Notes UI**: pencil icons on messages/headers, inline composer, edit/delete, persisted in sql.js in-memory DB. Graceful degradation for pre-v4 archives
- **Security**: `escAttr()` for attribute escaping, `safeMediaUrl()` allowlist (blocks javascript:/vbscript:), URL validation in renderMarkdown
- **Edge proxy**: default endpoint `metafunctor-edge.queelius.workers.dev/v1/messages` (no API key needed; proxy injects server-side, locks model to haiku)

## Gotchas

- `Database(path)` takes a **directory** path (creates `conversations.db` inside it), except for `:memory:`
- `Database(path, readonly=True)` sets PRAGMA query_only, so no writes are possible at SQLite level
- `_auto_import` prefers `claude_code_full` over `claude_code` when both detect the same file
- `get_all_paths()` is iterative (not recursive) to handle 1000+ message chains
- FTS queries sanitize via `_sanitize_fts_query()` (strips quotes, OR-joins tokens); LIKE fallback uses `_escape_like()`
- `get_schema()` filters out FTS5 shadow tables and `schema_version` from output
- `save_conversation` uses INSERT OR REPLACE (triggers CASCADE delete on messages/tags/enrichments/provenance). Notes survive via `ON DELETE SET NULL`
- Importers set `conv.metadata["_provenance"]`. CLI pops it before save, writes to provenance table after (CASCADE-safe)
- Two Claude Code importers share `_claude_code_common.py`: `claude_code` (conversation_only, strips tool use/thinking) and `claude_code_full` (full fidelity). Both use `source="claude_code"`, differentiated by `importer_mode` metadata and provenance `source_type`
- `claude_code_full` imports subagent files from `subagents/` directories with `parent_conversation_id` links; subagent IDs are `{sessionId}:{agentId}` (deterministic); `conversation_only` skips subagents entirely
- FTS5 is NOT trigger-maintained: `messages_fts` is manually updated in `save_conversation`, `append_message`, `update_message_content`, `delete_conversation`. `notes_fts` is maintained by `add_note`, `update_note`, `delete_note`, and `delete_conversation` (which re-syncs the shadow rows of notes orphaned by `ON DELETE SET NULL` so they no longer point at the deleted conversation)
- HTML SPA search uses `LIKE '%term%'` on raw `messages.content` JSON (no FTS5 in sql.js)
- Redact script scans only `type="text"` content blocks; tool_use/thinking blocks in full-fidelity imports are not scanned
- MCP lifespan resolution: `MEMEX_CONFIG` env var, then `~/.memex/config.yaml`, then fallback to `~/.memex/default`
- Enrichment type `original_content` used by redact script is not validated by MCP (DB layer has no type constraints)
- `update_conversations` fails fast on readonly databases before entering the bulk loop
- **`archived_at` is a visible user flag here, not a default read filter (intentional C1 deviation).** Like `starred_at` and `pinned_at`, it is a boolean timestamp (NULL=false, DATETIME=true) the user toggles for organization. Default local reads (`query_conversations`, the `get_conversations` MCP tool, `show`) intentionally include archived conversations so the user can still see and manage them. The soft-delete privacy guarantee is enforced on the *publish/export* paths instead: the HTML and arkiv exporters default to `include_archived=False`, so archived conversations never travel in a published bundle unless the caller explicitly opts in. If a future change wants archived hidden from default reads too, add an `include_archived` flag rather than silently flipping the default.
