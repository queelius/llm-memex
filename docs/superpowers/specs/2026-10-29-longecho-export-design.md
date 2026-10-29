# longecho export: a resilient archive folder

Date: 2026-10-29. Status: design, awaiting approval.

## Why

The HTML export is a site, not an archive. It opens only when served
over HTTP (double-clicking `index.html` fails, because browsers block
`fetch` of `file://` URLs for both the database and the sql.js wasm), it
carries no explanation for a reader who does not know llm-memex, and its
data lives in one gzipped file that only the viewer reads.

The user wants the export to be useful to themselves and to other people,
and to be a resilient digital artifact in the longecho sense: a folder
that still explains and opens itself when the tools that made it are gone
(`~/github/beta/longecho/docs/spec.md`). Each rung of the degradation
ladder should work on its own: a text editor, `sqlite3`, a browser opened
straight from the folder, an LLM reading the README.

This is step 1 of a larger HTML-export improvement. Later steps, in order:
agent sessions rendered as work (tool-call cards, diffs, thinking); a
dashboard front page; reading polish (highlighting, math, branches, real
search); publish safety (selection, secret scan, preview); then static
no-JavaScript pages, replay and timelines. Step 1 is the foundation the
others build on, and the user asked for the simpler things first.

## Goals

1. `llm-memex export <dir> --format longecho` writes a longecho-compliant
   folder from one database.
2. A reader with no context understands it from the root `README.md`
   alone, including how to open each part with the plainest tool.
3. The viewer opens by double-clicking `site/index.html`, with no server
   and no network, for archives up to the size cap below.
4. Every conversation is readable in a text editor as Markdown.
5. The data is in durable formats a future reader can open without
   llm-memex: SQLite, JSONL.
6. The folder can be verified (`sha256sum -c SHA256SUMS`).

## Non-goals (for step 1)

- Changing what `--format html` produces. Its output stays as is.
- Any visual redesign of the viewer, a dashboard, or agent-log rendering.
- Combining several databases into one archive. One export reads one
  database (`--db`). A parent folder of several exports can be made a
  longecho source later by writing a README above them.
- Making the full-fidelity Claude Code archive (4.1 GB without its search
  index) browsable. It exceeds the cap; it still gets `data/` and `text/`.

## Anti-goals (failure even if it works)

- Anything the HTML export keeps private leaking through the new parts:
  archived conversations (unless `--include-archived`) and notes under
  `--no-notes` must be absent from `data/`, `text/`, `site/`, and the
  README's counts alike. Every part is built from one projected snapshot.
- Any network request from the viewer, or any default API endpoint
  (workspace conventions C6e and #9 still hold).
- Overwriting a folder llm-memex did not write.
- A README that claims a fallback nobody tested.

## Measured sizes (2026-10-29, this machine)

Size of each live database without its FTS5 tables, which is roughly
what the export carries:

| Database | Conversations | Without FTS5 |
|----------|--------------:|-------------:|
| conversations (ChatGPT, Claude, Gemini) | 3,144 | 249 MB |
| claude_code | 2,881 | 226 MB |
| codex | 1,495 | 96 MB |
| claude_code_history | 1,910 | 28 MB |
| copilot | 130 | 20 MB |
| claude_code_full | 14,728 | 4,149 MB |

sql.js holds the whole database in wasm memory, so a few hundred MB is
fine in a desktop browser and several GB is not.

## The folder

```
archive/
  README.md          the interface; longecho frontmatter
  SHA256SUMS         sha256sum format, every file except itself
  data/
    conversations.db the projected snapshot (sqlite3 opens it)
    arkiv/           arkiv bundle: records JSONL + schema.yaml + README.md
    assets/          media referenced by retained messages
  text/
    index.md         every conversation by date, linking its file
    2025/
      2025-06-14-<slug>-<short id>.md
  site/
    README.md        viewer README, frontmatter generator: llm-memex <version>
    index.html
    sql-wasm.js
    sql-wasm-binary.js   the wasm as base64, for file://
    db/000.js, 001.js …  the gzipped DB as base64 chunks, for file://
```

Every durable file is within two levels of the root README, as longecho
requires. `site/` is a view: longecho ignores it for compliance, so the
folder is compliant through `data/` and `text/` alone.

### Root README.md

Frontmatter: `name`, `description`, `datetime` (export time, ISO 8601),
`generator: llm-memex longecho <version>` (distinct from the arkiv
exporter's `llm-memex` stamp, so neither replaces the other), and `contents` with file-style entries
for `data/`, `text/`, `site/` and their one-line descriptions. Body,
written for someone with none of the tools:

- what this is (an archive of AI conversations), whose (from an optional
  `--owner` value; without it the README says "a personal archive"),
  which sources, how many conversations and messages, the date range;
- how to open each part, plainest first: the `text/` files in any text
  editor; `site/index.html` by double-clicking; `data/conversations.db`
  with `sqlite3`, with two example queries that run against the schema;
  the arkiv JSONL with any JSON tool;
- what was left out (archived conversations, notes if `--no-notes`, the
  search index and why), so a reader knows the archive's boundary;
- how to verify (`sha256sum -c SHA256SUMS`).

The counts and dates come from the projected snapshot, never the live DB.

### data/

`conversations.db` is the same projection the HTML exporter makes today
(backup-API snapshot, archived and `--no-notes` data dropped, FTS5
removed, `journal_mode=DELETE`, vacuumed), uncompressed so `sqlite3`
opens it directly. `arkiv/` is the existing arkiv exporter's directory
output, built from the same snapshot.

`assets/` holds the media the retained messages reference (the existing
`_copy_retained_assets` logic), here rather than in `site/`, because
longecho ignores `site/` and the media is data. Messages keep their
`assets/...` references, which resolve relative to `data/`, as they do
relative to the live database. Media types longecho does not count as
durable (`.heic`, `.mov`) are copied as they are and show up as longecho
warnings; the export does not convert them.

### text/

One Markdown file per conversation, rendered with the existing Markdown
exporter's per-conversation logic (heading, source, messages per branch
path, notes as blockquotes when included). Files sit under the
conversation's creation year. Names are
`<YYYY-MM-DD>-<slug>-<first 8 chars of a hash of the id>.md`: readable,
and stable across re-exports. `text/index.md` lists every conversation by
date with title, source and a relative link.

Unlike the Markdown exporter, a branching conversation is not written
once per path: the main path comes first, then each other branch from its
fork point, so shared messages appear once (agent sessions branch a lot).
Media links point at `../../data/assets/...`. Media stored inline in the
database as `data:` URIs is not written into the Markdown; a one-line
placeholder names its type and size and says the site shows it.

### site/

Today's viewer, with two loader changes so it works from `file://`:

- **wasm**: `sql-wasm-binary.js` sets a global to the base64 of the
  vendored `sql-wasm.wasm`; when present the viewer passes the decoded
  bytes to `initSqlJs({wasmBinary})` instead of fetching the file.
- **database**: each `db/NNN.js` calls a global push function with one
  base64 slice of `conversations.db.gz` (16 MB of gzip per chunk); the
  viewer loads them in order with `<script>` tags, joins the bytes, and
  decompresses with `DecompressionStream('gzip')`. The chunk count is
  written into `index.html`, so a missing chunk is reported by name.

When the chunk globals are absent the viewer falls back to its current
`fetch` path, so `--format html` output behaves as before. The viewer's
drag-and-drop picker stays as the last resort.

- **media**: the viewer takes a media base path (`../data/assets/` here,
  `assets/` for `--format html`) and `safeMediaUrl` accepts that prefix,
  so the files exist once.
- **network**: nothing is fetched from the network by default. Remote
  `http(s)` media in messages renders as a link rather than loading,
  and the page's Content-Security-Policy adds `img-src` and `media-src`
  limited to `'self' data: blob:`. Chat stays as in the HTML export: off
  until the reader configures an endpoint, so `connect-src` is not
  locked. (The workspace convention forbids default endpoints, not
  reader-chosen ones.)

**Size cap.** When the projected DB exceeds 1 GiB uncompressed, `site/`
is written without the `db/` chunks and the viewer opens on its file
picker with a notice; the README says the archive is too large for the
browser viewer and points to `text/` and `data/`.

`site/README.md` carries `generator: llm-memex longecho <version>` so
longecho recognizes it as another tool's view.

## Behaviour

- **Destination.** An empty or missing directory is written. A non-empty
  directory is replaced only if its root `README.md` frontmatter has a
  `generator` beginning with `llm-memex longecho`; otherwise the export
  refuses. The existing alias checks (destination must not overlap the
  source DB or its assets) apply. A replacement deletes exactly the files
  listed in the previous `SHA256SUMS` (then any directories left empty)
  before writing, so a conversation dropped since the last export is not
  left behind and files a person added by hand are kept. `SHA256SUMS`
  lists what the export wrote.
- **Options** pass through as for `--format html`: `--no-notes`,
  `--include-archived`.
- **Composition.** A new `exporters/longecho.py` takes one snapshot and
  calls shared helpers; it does not shell out to other exporters. The
  snapshot and projection code currently inside `html.py` moves into
  helpers both exporters use.
- **Streaming.** Every part is built from the projected snapshot, one
  conversation at a time; the exporter ignores the list the CLI loaded.
  `_cmd_export` must therefore not load every conversation first for this
  format (an exporter declares that it reads the database itself), so the
  4.1 GB `claude_code_full` archive exports in bounded memory. That
  archive gets `data/` and `text/` and a site without chunks.

## Testing

- longecho compliance: `longecho` (PyPI 0.6.1) as an optional test
  dependency; the test runs its checker on the output, asserts compliant
  with no warnings, and skips when longecho is not installed.
- Privacy: an archived conversation and a note seeded in the source are
  absent from every part (`data/` DB and arkiv, `text/`, site chunks, the
  README counts) under the default and `--no-notes` options.
- Loader: reassembling the `db/` chunks yields exactly the gzip bytes, and
  `sql-wasm-binary.js` decodes to the vendored wasm.
- `SHA256SUMS` verifies, and covers every file but itself.
- Destination rules: refuses a foreign non-empty dir, replaces its own,
  removes stale files.
- Size cap: with a tiny cap injected, no chunks are written and the README
  says so.
- Branches: a forked conversation's shared messages appear once in
  `text/`.
- Network: the template for this format carries the stricter CSP, and a
  remote image URL renders as a link.
- Manual acceptance, recorded in the PR: export the copilot,
  conversations and claude_code_full databases (the last: peak memory
  and time noted), double-click `site/index.html` in Chrome and
  Firefox via `file://`, and run the README's `sqlite3` queries.

## Calls made by Claude (open to override)

- A new format rather than changing `--format html`.
- Script-chunk loading on every protocol, rather than also shipping a
  `.db.gz` for HTTP (one path, one test; costs about a third more bytes
  when served).
- The 1 GiB cap and the 16 MB chunk size.
- `data/conversations.db` without FTS5 (smaller, and a rebuilt index
  could carry rows of dropped conversations); `text/` grouped by year.
- `text/index.md` exists; no per-conversation HTML in step 1.
- Media in `data/assets/`, shared by the viewer; inline `data:` media is
  a placeholder in `text/`, not extracted to files.
- Remote media becomes a link; chat stays opt-in rather than removed.
- Owner only from a `--owner` flag (no config key, nothing by default).
- Replacement deletes only what the previous `SHA256SUMS` lists.
- The exporter streams from the snapshot, so even the largest archive
  exports; branches are written once from their fork point.

## Left to the builder

Slug rules, Markdown details beyond what the Markdown exporter does,
README wording (within the content list above), helper names and module
boundaries, and how the viewer shows load progress.
