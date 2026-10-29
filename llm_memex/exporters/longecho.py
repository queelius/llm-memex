"""Export the archive as a longecho folder: README, data/, text/, site/.

Every part is built from one projected snapshot of the database (archived
conversations and, under ``--no-notes``, notes already removed), one
conversation at a time, so nothing the HTML export keeps private can leak
through a new part and a multi-gigabyte archive exports in bounded memory.
The ``conversations`` argument the CLI may pass is ignored for the same reason.

Layout and rules: ``docs/superpowers/specs/2026-10-29-longecho-export-design.md``.
"""
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List

from llm_memex.db import Database
from llm_memex.exporters import arkiv_export
from llm_memex.exporters._longecho_docs import (
    README_NAME,
    SUMS_NAME,
    ArchiveSummary,
    prepare_destination,
    render_root_readme,
    render_site_readme,
    summarize,
    write_sha256sums,
)
from llm_memex.exporters._longecho_site import (
    copy_sql_js,
    write_db_chunks,
    write_wasm_script,
)
from llm_memex.exporters._longecho_text import (
    IndexEntry,
    render_conversation,
    render_index,
    text_path,
)
from llm_memex.exporters._snapshot import (
    check_destination,
    copy_retained_assets,
    gzip_file,
    take_snapshot,
)
from llm_memex.exporters.html_template import get_template

# The CLI must not load every conversation for this exporter: it reads the
# database itself, one conversation at a time.
READS_DATABASE = True

SITE_MAX_DB_BYTES = 1 << 30  # uncompressed projected DB the viewer will load
DB_CHUNK_BYTES = 16 * 1024 * 1024  # gzip bytes per site/db/NNN.js

_SNAPSHOT_NAME = "conversations.db"
# Entries of the destination that every export writes into. A symlink in any
# of them would send a write somewhere else.
_TOP_LEVEL = (README_NAME, SUMS_NAME, "data", "text", "site")
# Directories a fresh export writes with no regard for what is there
# (write_directory overwrites, copy_retained_assets removes first). A previous
# export's copies are gone by now, so anything present is not ours to lose.
_WRITTEN_BLINDLY = ("data/arkiv", "data/assets")


def export(conversations: Any, path: str, **kwargs) -> int:
    """Write a longecho archive to the directory *path*; return its conversation count.

    Parameters
    ----------
    conversations :
        Ignored (accepted for exporter API compatibility).
    path :
        Destination directory: missing, empty, or a previous export by this tool.
    db_path : str
        The live ``conversations.db``. Required.
    include_notes : bool
        Keep notes (default True).
    include_archived : bool
        Keep archived conversations (default False).
    owner : str | None
        Name shown in the README ("an archive of <owner>").
    site_max_db_bytes, db_chunk_bytes : int
        Viewer size cap and gzip bytes per chunk (defaults above).
    """
    db_path = kwargs.get("db_path")
    if not db_path or db_path == ":memory:" or not os.path.isfile(db_path):
        raise ValueError(
            "longecho export needs db_path, the path of an existing conversations.db"
        )
    db_path = Path(db_path)
    include_notes = kwargs.get("include_notes", True)
    include_archived = kwargs.get("include_archived", False)
    owner = kwargs.get("owner")
    site_max_db_bytes = kwargs.get("site_max_db_bytes", SITE_MAX_DB_BYTES)
    db_chunk_bytes = kwargs.get("db_chunk_bytes", DB_CHUNK_BYTES)

    out = Path(path)
    check_destination(db_path, out)
    prepare_destination(out)
    _refuse_planted_paths(out)

    with tempfile.TemporaryDirectory(prefix="llm-memex-longecho-") as tmp:
        tmp_dir = Path(tmp)
        snapshot = tmp_dir / _SNAPSHOT_NAME
        take_snapshot(
            db_path, snapshot, include_notes=include_notes, include_archived=include_archived
        )

        entries: List[IndexEntry] = []
        with Database(str(tmp_dir), readonly=True) as snap_db:
            schema_ddl = snap_db.get_schema()
            ids = [
                row["id"]
                for row in snap_db.conn.execute(
                    "SELECT id FROM conversations ORDER BY created_at, id"
                )
            ]
            (out / "data").mkdir(exist_ok=True)
            count = arkiv_export.write_directory(
                _text_writing_stream(snap_db, ids, out / "text", entries, include_notes),
                out / "data" / "arkiv",
                include_notes=include_notes,
                db=snap_db if include_notes else None,
            )

        # Opening the snapshot through Database switched it to WAL.
        with sqlite3.connect(str(snapshot)) as conn:
            conn.execute("PRAGMA journal_mode=DELETE")
        shutil.copyfile(snapshot, out / "data" / _SNAPSHOT_NAME)
        copy_retained_assets(snapshot, db_path.parent / "assets", out / "data" / "assets")

        (out / "text").mkdir(exist_ok=True)
        (out / "text" / "index.md").write_text(
            render_index(entries), encoding="utf-8", errors="replace"
        )

        chunks = _write_site(
            out / "site", snapshot, tmp_dir, schema_ddl,
            site_max_db_bytes=site_max_db_bytes, db_chunk_bytes=db_chunk_bytes,
        )
        facts = summarize(snapshot)

    summary = ArchiveSummary(
        owner=owner,
        include_notes=include_notes,
        include_archived=include_archived,
        site_has_data=chunks > 0,
        exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **facts,
    )
    (out / README_NAME).write_text(render_root_readme(summary), encoding="utf-8")
    write_sha256sums(out)  # last: it lists every file written above
    return count


def _refuse_planted_paths(out: Path) -> None:
    """Refuse, before writing anything, a destination that would redirect a write."""
    for name in _TOP_LEVEL:
        if (out / name).is_symlink():
            raise ValueError(
                f"{out / name} is a symbolic link; refusing to write through it"
            )
    for name in _WRITTEN_BLINDLY:
        if (out / name).is_symlink() or (out / name).exists():
            raise ValueError(
                f"{out / name} already exists and was not written by a previous "
                "export; refusing to overwrite it. Move it away or choose another folder."
            )


def _text_writing_stream(
    snap_db: Database,
    ids: List[str],
    text_dir: Path,
    entries: List[IndexEntry],
    include_notes: bool,
) -> Iterator[Any]:
    """Yield each conversation of the snapshot, writing its text/ file on the way.

    ``write_directory`` consumes this generator, so the arkiv records and the
    Markdown come from one pass with one conversation in memory at a time.
    """
    for conversation_id in ids:
        conv = snap_db.load_conversation(conversation_id)
        if conv is None:
            continue
        notes = snap_db.get_notes(conversation_id=conv.id) if include_notes else []
        rel = text_path(conv)
        target = text_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # errors="replace": a lone surrogate in imported text must not abort
        # an export of thousands of conversations.
        target.write_text(
            render_conversation(conv, notes), encoding="utf-8", errors="replace"
        )
        entries.append(IndexEntry(rel, conv.title or "", conv.created_at, conv.source))
        yield conv


def _write_site(
    site_dir: Path,
    snapshot: Path,
    tmp_dir: Path,
    schema_ddl: str,
    *,
    site_max_db_bytes: int,
    db_chunk_bytes: int,
) -> int:
    """Write the viewer; return the number of database chunks (0 = none)."""
    site_dir.mkdir(exist_ok=True)
    chunks = 0
    if snapshot.stat().st_size <= site_max_db_bytes:
        db_gz = tmp_dir / f"{_SNAPSHOT_NAME}.gz"
        gzip_file(snapshot, db_gz, keep_source=True)
        chunks = write_db_chunks(db_gz, site_dir, db_chunk_bytes)
    (site_dir / "index.html").write_text(
        get_template(
            schema_ddl,
            media_base="../data/assets/",
            strict_network=True,
            db_chunks=chunks,
        ),
        encoding="utf-8",
    )
    copy_sql_js(site_dir)
    write_wasm_script(site_dir)
    (site_dir / README_NAME).write_text(render_site_readme(), encoding="utf-8")
    return chunks
