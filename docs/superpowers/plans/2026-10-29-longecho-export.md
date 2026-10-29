# longecho Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `llm-memex export <dir> --format longecho` writes a longecho-compliant archive folder (README, `data/`, `text/`, `site/`, `SHA256SUMS`) whose viewer opens by double-click.

**Architecture:** One projected SQLite snapshot (the HTML exporter's existing pipeline, moved into a shared module) feeds every part. The exporter streams conversations from that snapshot one at a time, writing Markdown, arkiv JSONL, and the README summary; the viewer is today's template with options for a media base path, a stricter network policy, and loading the wasm and the gzipped DB from `<script>` files so `file://` works.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`/`gzip`/`hashlib`/`base64`, PyYAML (already a dependency), `arkiv>=0.4`, sql.js (vendored), `longecho` (test-only, optional).

**Spec:** `docs/superpowers/specs/2026-10-29-longecho-export-design.md`

## Global Constraints

- Branch `html-longecho`. Commits end with the two attribution lines from the session (`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and the `Claude-Session:` line).
- Test command (pytest is not in `~/.venvs/memex`): `TEST="uv run --with pytest --with pytest-cov --with pytest-asyncio --with pytest-timeout --with longecho --python $HOME/.venvs/memex/bin/python pytest"`; single files with `--no-cov`.
- `--format html` output must not change. Every existing test in `tests/llm_memex/` keeps passing.
- No new runtime dependencies. `longecho` goes only in the `dev` extra.
- Stamp: `generator: llm-memex longecho <version>` in both the root and `site/` READMEs; version from `importlib.metadata.version("llm-memex")` (C10).
- Constants: `SITE_MAX_DB_BYTES = 1 << 30` (uncompressed projected DB), `DB_CHUNK_BYTES = 16 * 1024 * 1024` (gzip bytes per chunk); both overridable by exporter kwargs `site_max_db_bytes`, `db_chunk_bytes` for tests.
- Every part is built from the projected snapshot, never the live DB and never the list the CLI passes.
- Never read `dev/` or `~/.memex` databases in tests; manual acceptance (Task 8) runs on `~/.memex` but writes only to the scratchpad.

## Review Focus

1. **Local media under the stricter CSP on `file://`**: `'self'` may not match a `file:` page in Chrome, so images from `../data/assets/` must still display. The CSP includes `file:` in `img-src`/`media-src`; Task 4 tests the header text, Task 8 checks it in a browser.
2. **A damaged previous export**: a stamped README without `SHA256SUMS` (interrupted run) must refuse with a message naming the folder; a `SHA256SUMS` line naming an absolute path or one with `..` must be ignored, never deleted. Tests in Task 5.
3. **Awkward titles**: `[`, `]`, `|`, newlines, only non-ASCII, empty. Filenames fall back to `untitled`, index links stay intact. Tests in Task 3.
4. **An empty database** (zero conversations): the export still succeeds, is compliant, and the README says 0. Test in Task 6.
5. **Destination overlapping the source**: the output dir equal to, or containing, the live DB's directory, or inside its `assets/`, is refused before any file is touched. Test in Task 1.

---

### Task 1: Shared snapshot module

**Files:**
- Create: `llm_memex/exporters/_snapshot.py`
- Modify: `llm_memex/exporters/html.py` (move helpers out, import them back)
- Test: `tests/llm_memex/test_snapshot.py`

**Interfaces:**
- Produces:
  - `strip_fts5_and_vacuum(db_path: Path, *, drop_notes: bool = False, include_archived: bool = False) -> None` (the current `_strip_fts5_and_vacuum`, renamed)
  - `gzip_file(src: Path, dst: Path, level: int = 6, *, keep_source: bool = False) -> None`
  - `media_urls(value) -> Iterator[str]`
  - `copy_retained_assets(db_path: Path, source: Path, destination: Path) -> None`
  - `check_destination(db_path: Path, out_dir: Path) -> None`, raises `ValueError` when `out_dir` resolves to the source DB's directory, an ancestor of it, or a path inside its `assets/`.
  - `take_snapshot(db_path: Path, dest_db: Path, *, include_notes: bool, include_archived: bool) -> None`: backup API over `mode=ro` into `dest_db`, then `strip_fts5_and_vacuum`.
- `html.py` keeps module-level aliases `_strip_fts5_and_vacuum = strip_fts5_and_vacuum` etc. so existing tests that reach into it still work. The module name starts with `_`, so `_discover_formats` skips it.

- [ ] **Step 1: Write failing tests** in `tests/llm_memex/test_snapshot.py`: `test_take_snapshot_drops_archived_and_fts` (seed one live and one archived conversation with `Database`; after `take_snapshot`, `SELECT id FROM conversations` == `[("live",)]` and `sqlite_master` has no `%_fts%` names and `PRAGMA journal_mode` is `delete`); `test_take_snapshot_no_notes_drops_table`; `test_check_destination_refuses_source_dir_ancestor_and_assets` (three `pytest.raises(ValueError)` cases plus one sibling dir that passes); `test_gzip_file_keep_source` (source still exists, `gzip.decompress` equals it).
- [ ] **Step 2: Run** `$TEST tests/llm_memex/test_snapshot.py --no-cov`; expect ImportError.
- [ ] **Step 3: Implement** `_snapshot.py` by moving the four helpers out of `html.py` and adding `check_destination` and `take_snapshot`; rewrite `html.export` to call them (its own alias checks become `check_destination` plus the existing DB-file alias check).
- [ ] **Step 4: Run** `$TEST tests/llm_memex/test_snapshot.py tests/llm_memex/test_html_export.py tests/llm_memex/test_html_snapshot.py --no-cov`; expect all pass.
- [ ] **Step 5: Commit** `refactor(export): move the publication snapshot into a shared module`.

### Task 2: Streaming arkiv directory writer

**Files:**
- Modify: `llm_memex/exporters/arkiv_export.py`
- Test: `tests/llm_memex/test_exporters.py` (new class `TestArkivStreaming`)

**Interfaces:**
- Produces: `write_directory(conversations: Iterable[Conversation], out_dir: Path, *, include_notes: bool = True, db=None) -> int` returning the conversation count. Writes `conversations.jsonl` incrementally (`records_to_jsonl` per conversation's records, appended), then `schema.yaml` and `README.md` with the existing helpers. `_compute_schema` accepts any iterable; feed it a generator that writes each record as it yields it, so records are never all held at once.

- [ ] **Step 1: Write failing test** `test_write_directory_matches_export`: for three conversations (one with a message note when `db` is given), `write_directory(iter(convs), a)` and `export(convs, str(b))` produce byte-identical files for all three names.
- [ ] **Step 2: Run** it; expect AttributeError.
- [ ] **Step 3: Implement** `write_directory`.
- [ ] **Step 4: Run** `$TEST tests/llm_memex/test_exporters.py tests/llm_memex/test_arkiv_roundtrip.py --no-cov`; expect pass.
- [ ] **Step 5: Commit** `feat(arkiv): stream a directory bundle conversation by conversation`.

### Task 3: Markdown text files

**Files:**
- Create: `llm_memex/exporters/_longecho_text.py`
- Test: `tests/llm_memex/test_longecho_text.py`

**Interfaces:**
- Produces:
  - `text_path(conv: Conversation) -> PurePosixPath`: `"<YYYY>/<YYYY-MM-DD>-<slug>-<sha256(conv.id)[:8]>.md"` from `conv.created_at`; slug is the title lowercased, runs of non-`[a-z0-9]` become `-`, trimmed, at most 50 chars, else `untitled`.
  - `render_conversation(conv: Conversation, notes: list[dict]) -> str`: `notes` are rows from `Database.get_notes(conversation_id=...)` (empty when notes are excluded).
  - `IndexEntry(path: PurePosixPath, title: str, created_at: datetime, source: str | None)` (dataclass) and `render_index(entries: Iterable[IndexEntry]) -> str`: `# Conversations`, then a `## <year>` heading per year, newest year first, each line `- <YYYY-MM-DD> [<title>](<path>) · <source>`, with `\`, `[`, `]` backslash-escaped and newlines in titles collapsed to spaces.
- Rendering rules: title as `#` heading, then source, created date, conversation-level notes as `> **Note:** …`. Messages via `Message.get_content_md()` after rewriting media: a block whose url starts with `assets/` links to `../../data/assets/<rest>`; a block with inline base64 `data` (or a `data:` url) becomes `[inline <media_type>, <N> KB, shown in site/]`. Branches: iterate `conv.get_all_paths()`, emit only messages not yet emitted; before the new messages of every path after the first, emit a heading `## Branch <k>` and a line naming the fork point (the role and first 60 chars of the parent message's text). Message notes follow their message as blockquotes.

- [ ] **Step 1: Write failing tests**: `test_text_path_stable_and_slugged` (title `"Hello, World! [draft]"` → `2025/2025-06-14-hello-world-draft-<8 hex>.md`, same result twice); `test_text_path_untitled_fallbacks` (titles `""`, `"日本語"`, `None`); `test_branch_shared_messages_once` (root → a → {b, c}: `a`'s text occurs once, `"## Branch 2"` occurs once and names `a`); `test_media_rewritten_and_inline_placeholder` (`assets/x.png` → `](../../data/assets/x.png)`; 3,000 bytes of base64 → `[inline image/png, 2 KB, shown in site/]` and no `base64,` in output); `test_notes_rendered`; `test_index_escapes_titles` (title `"a]b[c\nd"` renders `[a\]b\[c d](...)`).
- [ ] **Step 2: Run** `$TEST tests/llm_memex/test_longecho_text.py --no-cov`; expect ImportError.
- [ ] **Step 3: Implement** the module.
- [ ] **Step 4: Run** it; expect pass.
- [ ] **Step 5: Commit** `feat(longecho): render conversations as stable Markdown files`.

### Task 4: Viewer that opens from `file://`

**Files:**
- Modify: `llm_memex/exporters/html_template.py` (`get_template`, `_js_core`, `safeMediaUrl`/`renderMedia`, markdown image rule)
- Create: `llm_memex/exporters/_longecho_site.py`
- Create: `tests/llm_memex/data/template_default.html` (golden copy of today's output)
- Test: `tests/llm_memex/test_longecho_site.py`

**Interfaces:**
- Produces:
  - `get_template(schema_ddl: str = "", *, media_base: str = "assets/", strict_network: bool = False, db_chunks: int | None = None) -> str`. Defaults reproduce today's output exactly.
  - In the page: `var MEDIA_BASE`, `var STRICT_NETWORK`, `var DB_CHUNKS` (`null` = use the fetch cascade; `0` = the archive was too large: show the picker with a notice saying so and pointing at `text/` and `data/`; `n > 0` = load `db/000.js` … by inserting `<script>` tags in order).
  - Globals the generated scripts call: `window.LLM_MEMEX_WASM_B64 = "<base64>"` (set by `sql-wasm-binary.js`; when present, `initSqlJs({wasmBinary: <decoded Uint8Array>})`) and `window.llmMemexDbChunk(index, base64)` (stores a slice; after the last, concatenate, decompress with `DecompressionStream("gzip")`, open). A chunk script that fails to load reports its file name in the notice.
  - Media: a url starting with `assets/` is rewritten to `MEDIA_BASE + rest`, and `safeMediaUrl` accepts the `MEDIA_BASE` prefix. With `STRICT_NETWORK`, `http(s)` media and markdown images render as `<a … rel="noopener noreferrer">` links, and the CSP gains `img-src 'self' file: data: blob:; media-src 'self' file: data: blob:`.
  - `_longecho_site.py`: `write_wasm_script(site_dir: Path) -> None` and `write_db_chunks(db_gz: Path, site_dir: Path, chunk_bytes: int) -> int` (writes `db/000.js` … each as `llmMemexDbChunk(<i>,"<base64>");\n`, returns count), plus copying `sql-wasm.js`.

- [ ] **Step 1: Write failing tests**: first, before touching the template, save `get_template("SCHEMA")` to `tests/llm_memex/data/template_default.html`; then `test_default_template_unchanged` (`get_template("SCHEMA")` equals that file); `test_strict_template_has_csp_and_flags` (contains `img-src 'self' file: data: blob:`, `var STRICT_NETWORK = true`, `var MEDIA_BASE = "../data/assets/"`, `var DB_CHUNKS = 3`); `test_chunks_reassemble` (random 40 KB gzip file, `chunk_bytes=16384` → 3 files; parsing each file's base64 argument and joining equals the input); `test_wasm_script_decodes_to_vendored_wasm`.
- [ ] **Step 2: Run** `$TEST tests/llm_memex/test_longecho_site.py --no-cov`; expect failures.
- [ ] **Step 3: Implement** the template options and `_longecho_site.py`.
- [ ] **Step 4: Run** it plus `tests/llm_memex/test_html_export.py`; expect pass.
- [ ] **Step 5: Commit** `feat(html): viewer options for file:// loading, media base, strict network`.

### Task 5: READMEs, checksums, destination rules

**Files:**
- Create: `llm_memex/exporters/_longecho_docs.py`
- Test: `tests/llm_memex/test_longecho_docs.py`

**Interfaces:**
- Produces:
  - `ArchiveSummary` dataclass: `conversations: int, messages: int, sources: list[str], first: str | None, last: str | None` (ISO dates), `owner: str | None, include_notes: bool, include_archived: bool, site_has_data: bool, exported_at: str` (ISO 8601).
  - `summarize(db_path: Path) -> dict` returning the counts, sorted distinct sources and min/max `created_at` dates from a snapshot (the exporter adds the rest).
  - `README_QUERIES: list[tuple[str, str]]`: `(description, sql)` pairs shown in the README as `sqlite3 data/conversations.db "<sql>"`. Two entries: the 20 newest conversations (`created_at`, `title`, `source`), and messages whose content contains a word (`LIKE '%sqlite%'`) with their conversation title, `LIMIT 10`.
  - `render_root_readme(summary: ArchiveSummary) -> str`: frontmatter `name`, `description`, `datetime`, `generator: llm-memex longecho <version>`, `contents` as file entries (`text/index.md`, `site/index.html`, `data/conversations.db`, `data/arkiv/conversations.jsonl`, `SHA256SUMS`, each with a description). Body sections in this order: what this is (owner or "a personal archive", sources, counts, date range); how to open it, plainest first (text editor on `text/`, double-click `site/index.html`, or, when `not site_has_data`, a sentence saying the archive is too large for the browser viewer; `sqlite3` with `README_QUERIES`; the JSONL); what was left out (archived conversations unless included, notes when excluded, the search index and why); how to verify (`sha256sum -c SHA256SUMS`). Frontmatter via `yaml.safe_dump`.
  - `render_site_readme() -> str`: frontmatter `name: llm-memex viewer`, `description`, `generator: llm-memex longecho <version>`; body says to double-click `index.html` and that it reads a copy of `../data/conversations.db`.
  - `GENERATOR_PREFIX = "llm-memex longecho"`.
  - `prepare_destination(out_dir: Path) -> None`: missing or empty → create; otherwise require `README.md` frontmatter `generator` starting with `GENERATOR_PREFIX` (else `ValueError` naming the folder and the stamp found), require `SHA256SUMS` (else `ValueError` saying a previous export looks interrupted and to remove the folder), delete each listed file whose path is relative with no `..` part and resolves inside `out_dir`, then remove directories left empty (deepest first, never `out_dir`), then delete `SHA256SUMS`.
  - `write_sha256sums(out_dir: Path) -> None`: every regular file under `out_dir` except `SHA256SUMS`, sorted POSIX relative paths, `<hex>  <path>\n`, hashed in 1 MB reads.

- [ ] **Step 1: Write failing tests**: `test_readme_frontmatter_and_sections` (parse frontmatter with `yaml.safe_load`; generator prefix; the four sections in order; "a personal archive" without owner, the owner's name with one); `test_readme_too_large_sentence`; `test_readme_queries_run` (each `README_QUERIES` sql executes on a DB built with `Database` + `strip_fts5_and_vacuum`); `test_prepare_refuses_foreign` (an arkiv bundle README with `generator: llm-memex 0.x` is refused); `test_prepare_refuses_interrupted`; `test_prepare_deletes_listed_keeps_unlisted` (listed file and its emptied dir removed; a hand-added `notes.txt` kept); `test_prepare_ignores_escaping_paths` (lines `../outside.txt` and `/etc/x` leave both files intact); `test_sha256sums_verifies` (run `sha256sum -c` via `subprocess` with `cwd=out_dir` when `shutil.which("sha256sum")`, else recompute in Python).
- [ ] **Step 2: Run** `$TEST tests/llm_memex/test_longecho_docs.py --no-cov`; expect ImportError.
- [ ] **Step 3: Implement** the module.
- [ ] **Step 4: Run** it; expect pass.
- [ ] **Step 5: Commit** `feat(longecho): archive README, checksums, and safe re-export`.

### Task 6: The exporter

**Files:**
- Create: `llm_memex/exporters/longecho.py`
- Test: `tests/llm_memex/test_longecho_export.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `READS_DATABASE = True` and `export(conversations, path: str, **kwargs) -> int` (returns the exported conversation count; ignores `conversations`). Kwargs: `db_path` (required, else `ValueError`), `include_notes=True`, `include_archived=False`, `owner=None`, `site_max_db_bytes`, `db_chunk_bytes`.
- Order: `check_destination` → `prepare_destination` → `take_snapshot` into a `tempfile.TemporaryDirectory` → open it with `Database(tmp, readonly=True)` and, for each id from `SELECT id FROM conversations ORDER BY created_at, id`, `load_conversation`, write its `text/` file, collect an `IndexEntry`, and yield it to `arkiv_export.write_directory(..., out/"data/arkiv")` (one pass: drive the text writing from inside the generator `write_directory` consumes) → close; `PRAGMA journal_mode=DELETE` on the snapshot; copy it to `data/conversations.db` → `copy_retained_assets(..., out/"data/assets")` → `text/index.md` → `site/`: template with `media_base="../data/assets/"`, `strict_network=True`, and `db_chunks` = chunk count, or `0` when the snapshot exceeds `site_max_db_bytes` (then no gzip and no chunks) → `site/README.md` → root `README.md` → `write_sha256sums`. `get_notes` is called only when `include_notes`.

- [ ] **Step 1: Write failing tests** (fixture: a `Database` in `tmp_path/"src"` with a branching conversation, one with `assets/pic.png` on disk, one archived, one with a conversation note and a message note, two sources): `test_export_is_longecho_compliant` (`longecho = pytest.importorskip("longecho.checker")`; `check_compliance(out).compliant` and `warnings == []`); `test_layout` (the README, `SHA256SUMS`, `data/conversations.db`, `data/arkiv/conversations.jsonl`, `data/assets/pic.png`, `text/index.md`, one `.md` per retained conversation, `site/index.html`, `site/sql-wasm.js`, `site/sql-wasm-binary.js`, `site/db/000.js`, `site/README.md`; no `-wal`/`-shm` anywhere); `test_privacy_every_part` (parametrized over default and `include_notes=False`: the archived conversation's id and text, and under `include_notes=False` the note texts, are absent from `data/conversations.db`, the arkiv JSONL, every `text/` file, the DB rebuilt from `site/db/*.js`, and the README; the README's conversation count equals the snapshot's); `test_size_cap` (`site_max_db_bytes=1`: no `site/db/`, `var DB_CHUNKS = 0` in the page, README has the too-large sentence, still compliant); `test_reexport_removes_dropped_conversation` (export, archive one more conversation in the source, export again: its `.md` is gone and `sha256sum` verifies); `test_empty_database`; `test_requires_db_path`.
- [ ] **Step 2: Run** `$TEST tests/llm_memex/test_longecho_export.py --no-cov`; expect ImportError.
- [ ] **Step 3: Implement** `longecho.py`.
- [ ] **Step 4: Run** the whole suite `$TEST`; expect all pass. Note coverage for the new modules in the commit message if any are under 90%.
- [ ] **Step 5: Commit** `feat(export): longecho archive format`.

### Task 7: CLI, docs, dev extra

**Files:**
- Modify: `llm_memex/cli.py` (`export` subparser, `_cmd_export`), `pyproject.toml` (`dev` extra gains `longecho`), `README.md`, `CLAUDE.md` (commands list, package tree, an "longecho export" paragraph under HTML SPA Export)
- Test: `tests/llm_memex/test_cli.py` (class `TestCLIExportLongecho`)

**Interfaces:**
- `export` gains `--owner NAME` (passed as `owner=`; other exporters ignore it).
- `_cmd_export`: when `getattr(exporter_mod, "READS_DATABASE", False)`, skip the conversation-loading loop, call `exporter_mod.export([], args.output, db_path=..., db=db, include_notes=..., include_archived=..., owner=args.owner)`, and print `Exported {n} conversation(s) to {output}` from its return value. A `ValueError` from the exporter prints `Error: <message>` to stderr and exits 1.

- [ ] **Step 1: Write failing tests**: `test_export_longecho_end_to_end` (subprocess `python -m llm_memex export <out> --format longecho --db <src> --owner Ada`, like `TestCLIImportCopilotAndHistory._run`: exit 0, stdout has the count, `Ada` appears in the README); `test_export_longecho_does_not_load_conversations` (in process: `monkeypatch.setattr(sys, "argv", [...])`, monkeypatch `llm_memex.db.Database.query_conversations` to raise, call `llm_memex.cli.main()`; the export succeeds because only the CLI's loading loop calls that method); `test_export_longecho_refuses_foreign_dir` (non-empty dir with a plain `README.md` → exit 1, stderr mentions the stamp, the dir is unchanged).
- [ ] **Step 2: Run** them; expect failures.
- [ ] **Step 3: Implement** the CLI change, add the extra, update `README.md` and `CLAUDE.md`.
- [ ] **Step 4: Run** `$TEST`; expect all pass.
- [ ] **Step 5: Commit** `feat(cli): export --format longecho, --owner`.

### Task 8: Manual acceptance on real archives

No code. Results go in the PR description; nothing from the real archives is committed or sent to an external reviewer.

- [ ] **Step 1:** Export `copilot`, `conversations`, and `claude_code_full` into the scratchpad (`llm-memex export <scratch>/<name> --format longecho --db <name>`), recording wall time and peak RSS (`/usr/bin/time -v`). Expected: all three succeed; `claude_code_full` has `DB_CHUNKS = 0`.
- [ ] **Step 2:** `longecho check <scratch>/<name>` for each: compliant; list any warnings (expected only for non-durable media types).
- [ ] **Step 3:** `cd <scratch>/<name> && sha256sum -c --quiet SHA256SUMS`: no output.
- [ ] **Step 4:** Run both `README_QUERIES` with the `sqlite3` CLI against each `data/conversations.db`: rows come back.
- [ ] **Step 5:** Open `file://<scratch>/conversations/site/index.html` in Chrome (chrome-devtools MCP) and Firefox: the conversation list loads with no network requests in the network panel and no console errors; a conversation with an image shows it; for `claude_code_full`, the too-large notice shows.
- [ ] **Step 6:** Open one `text/` file and `text/index.md` in a text editor view (`less`): readable, links resolve to existing files.
