"""Export conversations as a self-contained HTML SPA directory."""
import contextlib
import gzip
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import List
from urllib.parse import unquote, urlsplit

from llm_memex.db import Database
from llm_memex.exporters.html_template import get_template
from llm_memex.models import Conversation


_VENDORED_DIR = Path(__file__).parent / "vendored"
_SQL_JS_FILES = ("sql-wasm.js", "sql-wasm.wasm")
_FTS5_TABLES = ("messages_fts", "notes_fts")
# gzip level 6 is the sweet spot: near-maximum ratio with modest CPU cost.
# On text-heavy llm-memex DBs, this yields ~65% transfer reduction.
_DB_GZIP_LEVEL = 6


def _strip_fts5_and_vacuum(
    db_path: Path, *, drop_notes: bool = False, include_archived: bool = False
) -> None:
    """Sanitize the exported DB copy: drop FTS5 tables, remove redaction-undo
    plaintext, drop soft-deleted conversations, optionally drop marginalia,
    and VACUUM.

    sql.js (used by the HTML SPA) cannot query FTS5 — it's not compiled in.
    The shadow tables are ~50% of a typical DB, so dropping them before
    export roughly halves bundle size. The SPA falls back to LIKE queries.

    The redact script stores the pre-redaction content in ``original_content``
    enrichments so redaction is reversible. That undo copy holds exactly the
    secrets the user asked to remove, so it must never travel into a published
    bundle (treat any exported HTML as potentially public). It is deleted here.

    When ``drop_notes`` is true (the ``--no-notes`` export path), the
    ``notes`` marginalia table is dropped entirely so private annotations
    never travel in a published bundle. ``notes_fts`` is already dropped via
    ``_FTS5_TABLES``; this removes the base table the SPA would otherwise
    read directly.

    Unless ``include_archived`` is true, soft-deleted conversations
    (``archived_at IS NOT NULL``) and all their child rows are physically
    removed from the copy. The SPA never references ``archived_at``, so a
    conversation the user "deleted" before publishing would otherwise appear
    in the home list, search, and librarian SQL of a public bundle. Treat
    any exported HTML as potentially public.

    Sets ``PRAGMA journal_mode=DELETE`` on the copy so no -wal/-shm sidecar
    files are left next to the exported database when the process is
    interrupted, and a subsequent VACUUM produces a fully-packed file.
    """
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        for fts in _FTS5_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {fts}")
        try:
            conn.execute("DELETE FROM enrichments WHERE type='original_content'")
        except sqlite3.OperationalError:
            pass  # no enrichments table (pre-v2 schema): nothing to strip
        if not include_archived:
            # Remove soft-deleted conversations and their child rows. Done
            # explicitly per table (rather than relying on FK cascade, which
            # needs PRAGMA foreign_keys=ON set before any transaction) so it
            # is robust across schema versions and ON DELETE SET NULL on
            # notes (which would otherwise orphan a note's text in the copy).
            archived = (
                "(SELECT id FROM conversations WHERE archived_at IS NOT NULL)"
            )
            for tbl in ("messages", "tags", "enrichments", "provenance", "notes"):
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(
                        f"DELETE FROM {tbl} WHERE conversation_id IN {archived}"
                    )
            conn.execute("DELETE FROM conversations WHERE archived_at IS NOT NULL")
        if drop_notes:
            conn.execute("DROP TABLE IF EXISTS notes")
        conn.commit()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("VACUUM")


def _gzip_file(src: Path, dst: Path, level: int = _DB_GZIP_LEVEL) -> None:
    """Stream src → gzip → dst, then delete src.

    Chunked to avoid loading the whole DB into memory for big archives.
    """
    with open(src, "rb") as fin, gzip.open(
        str(dst), "wb", compresslevel=level
    ) as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)
    src.unlink()


def _media_urls(value):
    """Walk structured content, including media nested in tool results."""
    if isinstance(value, dict):
        if value.get("type") == "media" and isinstance(value.get("url"), str):
            yield value["url"]
        for child in value.values():
            yield from _media_urls(child)
    elif isinstance(value, list):
        for child in value:
            yield from _media_urls(child)


def _copy_retained_assets(db_path: Path, source: Path, destination: Path) -> None:
    """Copy only media referenced by the sanitized snapshot, within assets/."""
    if destination.exists():
        shutil.rmtree(destination)
    root = source.resolve()
    with sqlite3.connect(db_path) as conn:
        for (content,) in conn.execute("SELECT content FROM messages"):
            for url in _media_urls(json.loads(content)):
                parsed = urlsplit(url)
                relative = Path(unquote(parsed.path))
                if parsed.scheme or parsed.netloc or relative.is_absolute():
                    continue
                if not relative.parts or relative.parts[0] != "assets" or ".." in relative.parts:
                    continue
                relative = Path(*relative.parts[1:])
                src = (root / relative).resolve()
                if not src.is_relative_to(root) or not src.is_file():
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)


def export(conversations: List[Conversation], path: str, **kwargs) -> None:
    """Export as HTML SPA directory.

    Emitted files:
    - ``index.html``                     the single-page application
    - ``sql-wasm.js``, ``sql-wasm.wasm`` vendored sql.js (no CDN dependency)
    - ``conversations.db.gz``            gzipped copy of the source database
      with FTS5 stripped (if ``db_path`` provided). The SPA fetches this
      transparently and decompresses via ``DecompressionStream('gzip')``
      — no library dependency on the reader side.
    - ``assets/``                        media referenced by retained messages

    Parameters
    ----------
    conversations : list[Conversation]
        Not used directly (the DB copy carries all data), but accepted for
        exporter API compatibility.
    path : str
        Destination directory to create/populate.
    **kwargs :
        db_path : str, optional
            Path to the source conversations.db file.  When provided (and not
            ``":memory:"``), take a consistent SQLite snapshot and copy only
            referenced assets from its sibling ``assets/`` directory.
        compress_db : bool, optional
            Whether to gzip the exported DB (default True). Set False for
            tooling that needs the raw .db file inline.
    """
    out_dir = Path(path)
    db_path = kwargs.get("db_path")
    has_db = db_path and db_path != ":memory:" and os.path.exists(db_path)
    if has_db:
        source_db = Path(db_path).resolve()
        target_db = out_dir / "conversations.db"
        source_assets = (Path(db_path).parent / "assets").resolve()
        target_assets = (out_dir / "assets").resolve()
        aliases_db = target_db.resolve() == source_db or (
            target_db.exists() and target_db.samefile(source_db)
        )
        overlaps_assets = source_assets.is_relative_to(target_assets) or target_assets.is_relative_to(source_assets)
        if aliases_db or overlaps_assets:
            raise ValueError("Export destination must not overwrite the source database or assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    compress_db = kwargs.get("compress_db", True)
    # The CLI passes include_notes=False for `--no-notes`. The DB copy is
    # what carries all SPA data, so honoring the flag means dropping the
    # notes table from that copy (the json/markdown exporters honor it by
    # filtering their record stream; this exporter must do it at the DB).
    include_notes = kwargs.get("include_notes", True)
    # Privacy default: soft-deleted conversations never travel in a published
    # bundle unless the caller explicitly opts in.
    include_archived = kwargs.get("include_archived", False)

    # Extract schema DDL from the database if available
    schema_ddl = ""
    if has_db:
        try:
            with Database(str(Path(db_path).parent), readonly=True) as db:
                schema_ddl = db.get_schema()
        except Exception:
            pass

    # Write index.html (utf-8 explicit: the template contains non-ASCII, so a
    # non-UTF-8 default locale would otherwise raise UnicodeEncodeError).
    (out_dir / "index.html").write_text(
        get_template(schema_ddl=schema_ddl), encoding="utf-8"
    )

    # Vendor sql.js (no CDN dependency)
    for filename in _SQL_JS_FILES:
        src = _VENDORED_DIR / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)

    # Copy DB (stripping FTS5 shadow tables), optionally gzip, and copy assets
    if has_db:
        dest_db = out_dir / "conversations.db"
        # A filesystem copy omits committed WAL frames, including an entire
        # newly-created schema. SQLite's backup API includes committed changes
        # without checkpointing or closing the user's live connection.
        with contextlib.closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as source:
            with contextlib.closing(sqlite3.connect(dest_db)) as destination:
                source.backup(destination)
        _strip_fts5_and_vacuum(
            dest_db, drop_notes=not include_notes, include_archived=include_archived
        )
        _copy_retained_assets(dest_db, Path(db_path).parent / "assets", out_dir / "assets")
        if compress_db:
            _gzip_file(dest_db, out_dir / "conversations.db.gz")
        else:
            # The browser tries gzip first; an older bundle must not mask the
            # new snapshot or keep excluded data available beside it.
            (out_dir / "conversations.db.gz").unlink(missing_ok=True)
