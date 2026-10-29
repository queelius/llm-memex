"""Publication snapshot helpers shared by the HTML and longecho exporters.

A published bundle must be built from a consistent, sanitized copy of the
database, never from the live file: these helpers take that copy, strip what
must not travel (FTS5 shadow tables, redaction-undo plaintext, archived
conversations, optionally notes), gzip it, and copy only the media it
references.
"""
import contextlib
import gzip
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlsplit

_FTS5_TABLES = ("messages_fts", "notes_fts")
# gzip level 6 is the sweet spot: near-maximum ratio with modest CPU cost.
# On text-heavy llm-memex DBs, this yields ~65% transfer reduction.
_DB_GZIP_LEVEL = 6


def strip_fts5_and_vacuum(
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
    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
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
    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("VACUUM")


def gzip_file(
    src: Path, dst: Path, level: int = _DB_GZIP_LEVEL, *, keep_source: bool = False
) -> None:
    """Stream src → gzip → dst, then delete src unless ``keep_source``.

    Chunked to avoid loading the whole DB into memory for big archives.
    """
    with open(src, "rb") as fin, gzip.open(
        str(dst), "wb", compresslevel=level
    ) as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)
    if not keep_source:
        src.unlink()


def media_urls(value) -> Iterator[str]:
    """Walk structured content, including media nested in tool results."""
    if isinstance(value, dict):
        if value.get("type") == "media" and isinstance(value.get("url"), str):
            yield value["url"]
        for child in value.values():
            yield from media_urls(child)
    elif isinstance(value, list):
        for child in value:
            yield from media_urls(child)


def copy_retained_assets(db_path: Path, source: Path, destination: Path) -> None:
    """Copy only media referenced by the sanitized snapshot, within assets/."""
    if destination.exists():
        shutil.rmtree(destination)
    root = source.resolve()
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        for (content,) in conn.execute("SELECT content FROM messages"):
            for url in media_urls(json.loads(content)):
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


def check_destination(db_path: Path, out_dir: Path) -> None:
    """Refuse an output directory that would overwrite the source archive.

    Raises ``ValueError`` when ``out_dir`` is the source database's directory,
    an ancestor of it, a path inside its ``assets/``, or when its own
    ``assets/`` (e.g. a symlink) resolves into the source ``assets/``, or when
    the source ``assets/`` (e.g. a symlink) resolves to somewhere inside
    ``out_dir``. Called before any file is touched.
    """
    source_dir = Path(db_path).resolve().parent
    # Resolved, so a symlinked source assets/ is judged by where it leads.
    source_assets = (source_dir / "assets").resolve()
    out = Path(out_dir).resolve()
    out_assets = (out / "assets").resolve()
    if (
        source_dir.is_relative_to(out)
        or out.is_relative_to(source_assets)
        or out_assets.is_relative_to(source_assets)
        or source_assets.is_relative_to(out_assets)
        or source_assets.is_relative_to(out)
    ):
        raise ValueError(
            "Export destination must not overwrite the source database or assets"
        )


def take_snapshot(
    db_path: Path, dest_db: Path, *, include_notes: bool, include_archived: bool
) -> None:
    """Copy the live database to ``dest_db`` and sanitize the copy.

    A filesystem copy omits committed WAL frames, including an entire
    newly-created schema. SQLite's backup API includes committed changes
    without checkpointing or closing the user's live connection.
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as source:
        with contextlib.closing(sqlite3.connect(dest_db)) as destination:
            source.backup(destination)
    strip_fts5_and_vacuum(
        dest_db, drop_notes=not include_notes, include_archived=include_archived
    )
