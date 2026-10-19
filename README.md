# llm-memex

Personal conversation knowledge base. Import, search, and analyze conversations from ChatGPT, Claude, Gemini, Claude Code, and Codex. MCP-first for LLM agent access.

## Install

```bash
pip install llm-memex
```

For development:
```bash
git clone https://github.com/queelius/llm-memex
cd llm-memex
pip install -e ".[dev]"
```

## Quick Start

**Import conversations:**
```bash
llm-memex import conversations.json          # auto-detects format
llm-memex import ~/.claude/projects/          # directory of Claude Code sessions
llm-memex import ~/.codex/sessions/           # directory of Codex CLI sessions
llm-memex import export.json --format openai  # force format
```

**Browse and search:**
```bash
llm-memex show                               # list conversations
llm-memex show <id>                          # view a conversation
llm-memex show --search "topic"              # full-text search
```

**Query the database:** use the MCP server (for LLM agents) or `sqlite3` directly:
```bash
sqlite3 ~/.memex/conversations/conversations.db "SELECT count(*) FROM conversations"
```

**Export:**
```bash
llm-memex export output.md --format markdown
llm-memex export output.json --format json
llm-memex export ./archive --format arkiv    # universal archive format
llm-memex export ./site --format html        # self-contained HTML SPA
```

**MCP server** (for Claude Desktop, agent SDKs, etc.):
```bash
llm-memex mcp
```

**Scripts:**
```bash
llm-memex run --list                                           # available scripts
llm-memex run redact --words "secret" --level word --apply
llm-memex run redact --pattern-file api_keys.txt --apply
llm-memex run enrich_trivial --apply
```

## Supported Formats

| Format                | Import | Export | Notes |
|-----------------------|--------|--------|-------|
| OpenAI (ChatGPT)      | Yes    | -      | JSON export |
| Anthropic (Claude)    | Yes    | -      | JSON export |
| Gemini                | Yes    | -      | JSON export |
| Claude Code           | Yes    | -      | JSONL, conversation-only mode |
| Claude Code (full)    | Yes    | -      | Full fidelity: tool_use, thinking, subagents |
| Codex CLI             | Yes    | -      | Rollout JSONL (both layouts), conversation-only; split threads become branches, subagents linked to parents |
| Markdown              | -      | Yes    | |
| JSON                  | -      | Yes    | |
| HTML (SPA)            | -      | Yes    | Self-contained, light/dark, librarian chat |
| Arkiv                 | -      | Yes    | Universal record format (JSONL + schema.yaml) |

## HTML Export

The HTML exporter builds a self-contained single-page app that loads the SQLite database client-side via sql.js (Wasm). Features:

- **Light/dark mode**: follows OS preference with a manual toggle
- **Full browser UI**: conversation list, search, filter by source/tag, timeline sparkline
- **Librarian chat**: ask questions about your archive. An LLM queries the database via Anthropic tool use, using the `metafunctor-edge` proxy by default (no API key required). You can also configure a direct Anthropic endpoint.
- **Per-conversation resume chat**: continue an existing conversation
- **Marginalia**: annotate messages and conversations with free-form notes, inline in the browser

## Notes (Marginalia)

Annotate messages and conversations with free-form text notes. Notes are stored in a dedicated `notes` table with FTS5 search, and appear across all surfaces (CLI, MCP, HTML SPA, exporters).

```bash
llm-memex run note add --conv <id> "this was a turning point" --apply
llm-memex run note add --conv <id> --msg <id> "key insight here" --apply
llm-memex run note list --conv <id>
llm-memex run note search "turning point"
llm-memex run note delete <note_id> --apply
```

Notes are included in exports by default. Use `--no-notes` to strip them:

```bash
llm-memex export ./public --format html              # includes notes
llm-memex export ./public --format html --no-notes   # strips notes
```

In the HTML SPA, click the pencil icon on any message or conversation header to add a note inline. Notes persist in the browser's sql.js copy and are included when you download the DB.

## Multi-Database Config

llm-memex supports multiple named databases via `~/.memex/config.yaml`:

```yaml
primary: conversations
databases:
  conversations:
    path: ~/.memex/conversations
  claude_code_full:
    path: ~/.memex/claude_code_full
  sandbox:
    path: ~/.memex/sandbox
```

All CLI commands and MCP tools accept `--db <name>` (CLI) or `db=<name>` (MCP) to target a specific database. The `primary` database is used when no name is specified.

## MCP Tools

When running as an MCP server, llm-memex exposes 6 tools:

- `execute_sql`: Primary read interface. All queries via SQL (read-only by default).
- `get_conversation`: Tree-aware retrieval + export (metadata, messages, markdown/JSON).
- `get_conversations`: Bulk retrieval with filters (tag, source, model, search, ids) and optional full messages.
- `update_conversations`: Modify properties, tags, and enrichments (bulk).
- `append_message`: Add messages to conversation trees.
- `add_note`: Annotate a message or conversation with a free-form text note.

Resources: `llm-memex://schema` (DDL + query patterns), `llm-memex://databases` (multi-db discovery + stats).

## Development

```bash
pytest tests/llm_memex/ -v             # run tests
pytest tests/llm_memex/ --cov=llm_memex    # with coverage
```

## License

MIT
