"""Export conversations as a self-contained HTML SPA directory."""
import gzip
import os
import shutil
import sqlite3
from pathlib import Path
from typing import List

from llm_memex.db import Database
from llm_memex.exporters.html_template import get_template
from llm_memex.models import Conversation


_VENDORED_DIR = Path(__file__).parent / "vendored"
_SQL_JS_FILES = ("sql-wasm.js", "sql-wasm.wasm")
_FTS5_TABLES = ("messages_fts", "notes_fts")
# gzip level 6 is the sweet spot: near-maximum ratio with modest CPU cost.
# On text-heavy llm-memex DBs, this yields ~65% transfer reduction.
_DB_GZIP_LEVEL = 6


def _strip_fts5_and_vacuum(db_path: Path, *, drop_notes: bool = False) -> None:
    """Sanitize the exported DB copy: drop FTS5 tables, remove redaction-undo
    plaintext, optionally drop marginalia, and VACUUM.

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


def export(conversations: List[Conversation], path: str, **kwargs) -> None:
    """Export as HTML SPA directory.

    Emitted files:
    - ``index.html``                     the single-page application
    - ``sql-wasm.js``, ``sql-wasm.wasm`` vendored sql.js (no CDN dependency)
    - ``conversations.db.gz``            gzipped copy of the source database
      with FTS5 stripped (if ``db_path`` provided). The SPA fetches this
      transparently and decompresses via ``DecompressionStream('gzip')``
      — no library dependency on the reader side.
    - ``assets/``                        copy of media assets directory

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
            ``":memory:"``), the DB and its sibling ``assets/`` directory are
            copied into the output.
        compress_db : bool, optional
            Whether to gzip the exported DB (default True). Set False for
            tooling that needs the raw .db file inline.
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = kwargs.get("db_path")
    has_db = db_path and db_path != ":memory:" and os.path.exists(db_path)
    compress_db = kwargs.get("compress_db", True)
    # The CLI passes include_notes=False for `--no-notes`. The DB copy is
    # what carries all SPA data, so honoring the flag means dropping the
    # notes table from that copy (the json/markdown exporters honor it by
    # filtering their record stream; this exporter must do it at the DB).
    include_notes = kwargs.get("include_notes", True)

    # Extract schema DDL from the database if available
    schema_ddl = ""
    if has_db:
        try:
            with Database(str(Path(db_path).parent), readonly=True) as db:
                schema_ddl = db.get_schema()
        except Exception:
            pass

    # Write index.html
    (out_dir / "index.html").write_text(get_template(schema_ddl=schema_ddl))

    # Vendor sql.js (no CDN dependency)
    for filename in _SQL_JS_FILES:
        src = _VENDORED_DIR / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)

    # Copy DB (stripping FTS5 shadow tables), optionally gzip, and copy assets
    if has_db:
        dest_db = out_dir / "conversations.db"
        shutil.copy2(db_path, dest_db)
        _strip_fts5_and_vacuum(dest_db, drop_notes=not include_notes)
        if compress_db:
            _gzip_file(dest_db, out_dir / "conversations.db.gz")
        assets_dir = Path(db_path).parent / "assets"
        if assets_dir.is_dir():
            dest_assets = out_dir / "assets"
            if dest_assets.exists():
                shutil.rmtree(dest_assets)
            shutil.copytree(assets_dir, dest_assets)
