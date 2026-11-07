"""Export conversations as a self-contained HTML SPA directory."""
import os
import shutil
from pathlib import Path
from typing import List

from llm_memex.db import Database
from llm_memex.exporters._snapshot import (
    check_destination,
    copy_retained_assets,
    gzip_file,
    take_snapshot,
)
from llm_memex.exporters.html_template import get_template
from llm_memex.models import Conversation


_VENDORED_DIR = Path(__file__).parent / "vendored"
_SQL_JS_FILES = ("sql-wasm.js", "sql-wasm.wasm")


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
        aliases_db = target_db.resolve() == source_db or (
            target_db.exists() and target_db.samefile(source_db)
        )
        if aliases_db:
            raise ValueError("Export destination must not overwrite the source database or assets")
        check_destination(Path(db_path), out_dir)
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
        take_snapshot(
            Path(db_path), dest_db, include_notes=include_notes, include_archived=include_archived
        )
        copy_retained_assets(dest_db, Path(db_path).parent / "assets", out_dir / "assets")
        if compress_db:
            gzip_file(dest_db, out_dir / "conversations.db.gz")
        else:
            # The browser tries gzip first; an older bundle must not mask the
            # new snapshot or keep excluded data available beside it.
            (out_dir / "conversations.db.gz").unlink(missing_ok=True)
