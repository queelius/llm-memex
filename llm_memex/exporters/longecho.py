"""Export the archive as a longecho folder: README, data/, text/, site/.

Every part is built from one projected snapshot of the database (archived
conversations and, under ``--no-notes``, notes already removed), one
conversation at a time, so nothing the HTML export keeps private can leak
through a new part and a multi-gigabyte archive exports in bounded memory.
The ``conversations`` argument the CLI may pass is ignored for the same reason.

Layout and rules: ``docs/superpowers/specs/2026-10-29-longecho-export-design.md``.
"""
import contextlib
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, List, Tuple

from llm_memex.db import Database
from llm_memex.exporters import arkiv_export
from llm_memex.exporters._longecho_docs import (
    README_NAME,
    SUMS_NAME,
    ArchiveSummary,
    listed_paths,
    prepare_destination,
    render_placeholder_readme,
    render_root_readme,
    render_site_readme,
    summarize,
    write_sha256sums,
)
from llm_memex.exporters._longecho_site import (
    DB_CHUNK_DIR,
    WASM_SCRIPT_NAME,
    copy_sql_js,
    db_chunk_name,
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
    _refuse_foreign_content(out)  # decided before anything is deleted
    prepare_destination(out)
    _refuse_symlinks(out)
    _refuse_leftovers(out)
    # Stamped stand-in, so an interruption from here on leaves a folder the
    # next run reports as interrupted rather than refuses as foreign.
    (out / README_NAME).write_text(render_placeholder_readme(), encoding="utf-8")
    written: List[PurePosixPath] = []  # every file this run creates, for SHA256SUMS

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
                _text_writing_stream(
                    snap_db, ids, out / "text", entries, written, include_notes
                ),
                out / "data" / "arkiv",
                include_notes=include_notes,
                db=snap_db if include_notes else None,
            )

        # Opening the snapshot through Database switched it to WAL.
        with contextlib.closing(sqlite3.connect(str(snapshot))) as conn:
            conn.execute("PRAGMA journal_mode=DELETE")
        shutil.copyfile(snapshot, out / "data" / _SNAPSHOT_NAME)
        written.append(PurePosixPath("data", _SNAPSHOT_NAME))
        copy_retained_assets(snapshot, db_path.parent / "assets", out / "data" / "assets")
        # data/arkiv and data/assets were absent before this run (checked above),
        # so everything in them is ours.
        written += _files_under(out, "data/arkiv") + _files_under(out, "data/assets")

        (out / "text").mkdir(exist_ok=True)
        (out / "text" / "index.md").write_text(render_index(entries), encoding="utf-8")
        written.append(PurePosixPath("text", "index.md"))

        chunks, site_files = _write_site(
            out / "site", snapshot, tmp_dir, schema_ddl,
            site_max_db_bytes=site_max_db_bytes, db_chunk_bytes=db_chunk_bytes,
        )
        written += site_files
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
    written.append(PurePosixPath(README_NAME))
    write_sha256sums(out, written)  # last: it lists what this run wrote, and only that
    return count


def _files_under(out: Path, rel: str) -> List[PurePosixPath]:
    """Every regular file below ``out/rel``, as paths relative to ``out``."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(out / rel, followlinks=False):
        for name in filenames:
            full = Path(dirpath) / name
            found.append(PurePosixPath(full.relative_to(out).as_posix()))
    return sorted(found)


def _leftover_message(path: Path) -> str:
    return (
        f"{path} was not written by a previous export; refusing to overwrite or "
        f"delete it. Move {path} away (or remove it) and export again."
    )


def _refuse_foreign_content(out: Path) -> None:
    """Before anything is deleted: refuse if data/arkiv or data/assets hold what
    the previous export's SHA256SUMS does not list.

    A fresh export rewrites those two directories wholesale, so hand-added
    content in them would be lost. When the folder is not a previous export
    (no SHA256SUMS) ``prepare_destination`` decides, and refuses or finds
    nothing there.
    """
    sums = out / SUMS_NAME
    if not sums.is_file():
        return
    listed = listed_paths(sums)
    listed_dirs = {parent for rel in listed for parent in rel.parents}
    for name in _WRITTEN_BLINDLY:
        top = out / name
        if top.is_symlink():
            raise ValueError(f"{top} is a symbolic link; refusing to write through it")
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            for entry in sorted(dirnames + filenames):
                full = Path(dirpath) / entry
                rel = PurePosixPath(full.relative_to(out).as_posix())
                if full.is_symlink():
                    raise ValueError(f"{full} is a symbolic link; refusing to write through it")
                if rel not in (listed if entry in filenames else listed_dirs):
                    raise ValueError(_leftover_message(full))


def _refuse_symlinks(out: Path) -> None:
    """Refuse, before writing anything, a destination that would redirect a write.

    The five names the export writes at the top, and everything below ``data/``,
    ``text/`` and ``site/``, must not be symbolic links: a write through one
    would land outside the folder. Nothing is followed while looking.
    """
    for name in _TOP_LEVEL:
        if (out / name).is_symlink():
            raise ValueError(
                f"{out / name} is a symbolic link; refusing to write through it"
            )
    for name in ("data", "text", "site"):
        top = out / name
        if top.exists() and not top.is_dir():
            raise ValueError(f"{top} exists and is not a directory")
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            for entry in sorted(dirnames + filenames):
                full = Path(dirpath) / entry
                if full.is_symlink():
                    raise ValueError(
                        f"{full} is a symbolic link; refusing to write through it"
                    )


def _refuse_leftovers(out: Path) -> None:
    """Backstop for ``_refuse_foreign_content``: nothing may remain where we write blindly."""
    for name in _WRITTEN_BLINDLY:
        if (out / name).is_symlink() or (out / name).exists():
            raise ValueError(_leftover_message(out / name))


def _text_writing_stream(
    snap_db: Database,
    ids: List[str],
    text_dir: Path,
    entries: List[IndexEntry],
    written: List[PurePosixPath],
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
        target.write_text(render_conversation(conv, notes), encoding="utf-8")
        written.append(PurePosixPath("text") / rel)
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
) -> Tuple[int, List[PurePosixPath]]:
    """Write the viewer; return the number of database chunks (0 = none) and
    the files written, relative to the export root."""
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
    names = ["index.html", "sql-wasm.js", WASM_SCRIPT_NAME, README_NAME]
    names += [f"{DB_CHUNK_DIR}/{db_chunk_name(i)}" for i in range(chunks)]
    return chunks, [PurePosixPath("site", name) for name in names]
