# llm-memex

Personal conversation knowledge base. Import, search, and analyze conversations from ChatGPT, Claude, Gemini, Claude Code, Codex, and GitHub Copilot Chat. MCP-first for LLM agent access.

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
llm-memex import ~/.claude/history.jsonl      # Claude Code prompt history (survives transcript pruning)
llm-memex import ~/.config/Code/User          # VS Code Copilot Chat sessions
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
llm-memex export ./archive --format longecho --owner "Your Name"   # resilient archive folder
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
| Claude Code history   | Yes    | -      | `~/.claude/history.jsonl`: the user's prompts only, grouped by session (or project and day before session ids existed) |
| GitHub Copilot Chat   | Yes    | -      | VS Code chat sessions (JSON snapshots and JSONL operation logs), conversation-only |
| Markdown              | -      | Yes    | |
| JSON                  | -      | Yes    | |
| HTML (SPA)            | -      | Yes    | Self-contained, light/dark, librarian chat |
| Arkiv                 | -      | Yes    | Universal record format (JSONL + schema.yaml) |
| longecho              | -      | Yes    | Self-explaining folder: README, data, Markdown text, offline viewer |

## HTML Export

The HTML exporter builds a self-contained single-page app that loads the SQLite database client-side via sql.js (Wasm). Features:

- **Light/dark mode**: follows OS preference with a manual toggle
- **Full browser UI**: conversation list, search, filter by source/tag, timeline sparkline
- **Librarian chat**: ask questions about your archive. An LLM queries the database via Anthropic tool use, using the `metafunctor-edge` proxy by default (no API key required). You can also configure a direct Anthropic endpoint.
- **Per-conversation resume chat**: continue an existing conversation
- **Marginalia**: annotate messages and conversations with free-form notes, inline in the browser

## longecho archive

`--format longecho` writes a folder meant to outlast the tools that made it. It is one database exported as a [longecho](https://github.com/queelius/longecho) archive:

```bash
llm-memex export ./archive --format longecho --owner "Your Name"
```

The folder holds:

- `README.md`: explains the archive to a reader with no context, including how to open each part with the plainest tool. `--owner` sets the name shown there.
- `SHA256SUMS`: checksums of everything the export wrote (`sha256sum -c SHA256SUMS`).
- `data/`: the SQLite database (`conversations.db`), the arkiv JSONL records (`arkiv/`), and the media (`assets/`).
- `text/`: every conversation as a Markdown file, plus an `index.md`.
- `site/`: the viewer. Double-click `site/index.html` to open it; it needs no server and no network. Media that lives on the web is shown as a link, not loaded.

The viewer loads the whole database into the browser, so it is limited to archives whose database is at most 1 GiB. A larger archive still gets `data/` and `text/`, just no viewer database.

Archived conversations (unless `--include-archived`) and notes (under `--no-notes`) are left out of every part.

Exporting again into the same folder is safe. Only the files the previous export listed in `SHA256SUMS` are replaced or removed; files you added by hand are kept, except under `data/arkiv/` and `data/assets/`, which the export rewrites whole: there it stops before changing anything and asks you to move them out. A folder that this command did not write (no `generator: llm-memex longecho` stamp in its `README.md`) is refused and left untouched. Every refusal happens before anything is deleted, so the previous export stays as it was. If an export fails partway (a full disk, Ctrl-C), run it again: it replaces what the failed run left.

The export needs free space of about three times the database: a temporary snapshot in `$TMPDIR`, then `data/` and `text/` in the destination. Point `TMPDIR` at another disk if the default one is short of space (`TMPDIR=/mnt/big llm-memex export ...`).

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
