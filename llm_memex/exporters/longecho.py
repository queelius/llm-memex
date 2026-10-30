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
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, List, Set

from llm_memex.db import Database
from llm_memex.exporters import arkiv_export
from llm_memex.exporters._longecho_docs import (
    README_NAME,
    SUMS_NAME,
    ArchiveSummary,
    clear_destination,
    inspect_destination,
    remove_empty_directories,
    render_placeholder_readme,
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
    # Every refusal is decided here, before anything is deleted, so a refused
    # run leaves the previous export exactly as it was.
    listed = inspect_destination(out)
    _refuse_foreign_content(out, listed)
    _refuse_symlinks(out)

    run = _Run(out)
    try:
        # The temporary directory is removed before the handler below runs,
        # which frees its space when a full disk was the failure.
        with tempfile.TemporaryDirectory(prefix="llm-memex-longecho-") as tmp:
            tmp_dir = Path(tmp)
            snapshot = tmp_dir / _SNAPSHOT_NAME
            # First, while the destination is untouched: a full disk (the
            # likeliest failure, the snapshot being the biggest write) then
            # leaves the previous export intact.
            take_snapshot(
                db_path, snapshot, include_notes=include_notes,
                include_archived=include_archived,
            )
            clear_destination(out, listed)
            run.cleared = True
            # Stamped stand-in until the real README replaces it.
            run.will_write(README_NAME).write_text(
                render_placeholder_readme(), encoding="utf-8"
            )
            _refuse_leftovers(out)
            run.blind_dirs_ours = True

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
                    _text_writing_stream(snap_db, ids, run, entries, include_notes),
                    out / "data" / "arkiv",
                    include_notes=include_notes,
                    db=snap_db if include_notes else None,
                )

            # Opening the snapshot through Database switched it to WAL.
            with contextlib.closing(sqlite3.connect(str(snapshot))) as conn:
                conn.execute("PRAGMA journal_mode=DELETE")
            shutil.copyfile(snapshot, run.will_write("data", _SNAPSHOT_NAME))
            copy_retained_assets(snapshot, db_path.parent / "assets", out / "data" / "assets")

            (out / "text").mkdir(exist_ok=True)
            run.will_write("text", "index.md").write_text(
                render_index(entries), encoding="utf-8"
            )

            chunks = _write_site(
                run, snapshot, tmp_dir, schema_ddl,
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
        run.will_write(README_NAME).write_text(
            render_root_readme(summary), encoding="utf-8"
        )
        # Last: it lists what this run wrote, and only that.
        write_sha256sums(out, run.finished_files())
    except BaseException:
        run.record_failure()
        raise
    return count


class _Run:
    """What one export has written into the destination so far.

    A path is recorded before its file is written, so a file a failure cut
    short is still accounted for (``write_sha256sums`` skips paths that do not
    exist). After the destination was cleared, a failed run lists what it
    wrote in ``SHA256SUMS`` before the error propagates: the folder then holds
    the stamped stand-in README plus a listing, and the next run replaces it
    the way it replaces a finished export.
    """

    def __init__(self, out: Path) -> None:
        self.out = out
        self.written: List[PurePosixPath] = []
        self.cleared = False  # the previous export's files are gone
        # data/arkiv and data/assets were proved absent after clearing, so
        # every file in them is this run's.
        self.blind_dirs_ours = False

    def will_write(self, *parts: str) -> Path:
        """Record ``out/<parts>`` as written by this run; return its full path."""
        rel = PurePosixPath(*parts)
        self.written.append(rel)
        return self.out.joinpath(*rel.parts)

    def record(self, paths: List[Path]) -> None:
        """Record files a helper already wrote below ``out``."""
        self.written += [PurePosixPath(p.relative_to(self.out).as_posix()) for p in paths]

    def finished_files(self) -> List[PurePosixPath]:
        """Everything this run wrote, including the two blindly written trees."""
        files = list(self.written)
        if self.blind_dirs_ours:
            for name in _WRITTEN_BLINDLY:
                files += _files_under(self.out, name)
        return files

    def record_failure(self) -> None:
        """List what this run wrote, so the next run can replace it.

        Nothing to do before clearing: the previous export is still whole. If
        writing the listing fails too (the disk is still full), the stamped
        README without ``SHA256SUMS`` is reported as interrupted next time;
        the original error is the one that propagates.
        """
        if not self.cleared:
            return
        if self.blind_dirs_ours:
            # Empty directories this run made there would be refused as foreign.
            dirs = [PurePosixPath(name) for name in _WRITTEN_BLINDLY]
            for name in _WRITTEN_BLINDLY:
                dirs += _dirs_under(self.out, name)
            with contextlib.suppress(OSError):
                remove_empty_directories(self.out, dirs)
        try:
            write_sha256sums(self.out, self.finished_files())
        except Exception:
            pass


def _dirs_under(out: Path, rel: str) -> List[PurePosixPath]:
    """Every directory below ``out/rel``, as paths relative to ``out``."""
    found = []
    for dirpath, dirnames, _filenames in os.walk(out / rel, followlinks=False):
        for name in dirnames:
            full = Path(dirpath) / name
            found.append(PurePosixPath(full.relative_to(out).as_posix()))
    return found


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


def _refuse_foreign_content(out: Path, listed: Set[PurePosixPath]) -> None:
    """Before anything is deleted: refuse if data/arkiv or data/assets hold
    anything but regular files the previous export's SHA256SUMS lists (*listed*,
    from ``inspect_destination``) and the directories that hold them.

    A fresh export rewrites those two directories wholesale, so hand-added
    content in them would be lost; and once the listed files are deleted the
    two directories must be gone, or ``_refuse_leftovers`` would stop the run
    after the previous export was already cleared.
    """
    listed_dirs = {parent for rel in listed for parent in rel.parents}
    for name in _WRITTEN_BLINDLY:
        top = out / name
        if top.is_symlink():
            raise ValueError(f"{top} is a symbolic link; refusing to write through it")
        # The tree itself too: one holding no listed file (or not a directory)
        # would survive the clearing.
        if top.exists() and (not top.is_dir() or PurePosixPath(name) not in listed_dirs):
            raise ValueError(_leftover_message(top))
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            for entry in sorted(dirnames + filenames):
                full = Path(dirpath) / entry
                rel = PurePosixPath(full.relative_to(out).as_posix())
                if full.is_symlink():
                    raise ValueError(f"{full} is a symbolic link; refusing to write through it")
                is_file = entry in filenames
                if rel not in (listed if is_file else listed_dirs):
                    raise ValueError(_leftover_message(full))
                if is_file and not stat.S_ISREG(full.lstat().st_mode):
                    raise ValueError(f"{full} is not a regular file; refusing to delete it")


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
    run: _Run,
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
        target = run.will_write("text", *rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_conversation(conv, notes), encoding="utf-8")
        entries.append(IndexEntry(rel, conv.title or "", conv.created_at, conv.source))
        yield conv


def _write_site(
    run: _Run,
    snapshot: Path,
    tmp_dir: Path,
    schema_ddl: str,
    *,
    site_max_db_bytes: int,
    db_chunk_bytes: int,
) -> int:
    """Write the viewer into ``site/``, recording every file in *run*; return
    the number of database chunks (0 = left out, too large)."""
    site_dir = run.out / "site"
    site_dir.mkdir(exist_ok=True)
    chunks = 0
    if snapshot.stat().st_size <= site_max_db_bytes:
        db_gz = tmp_dir / f"{_SNAPSHOT_NAME}.gz"
        gzip_file(snapshot, db_gz, keep_source=True)
        chunk_files = write_db_chunks(db_gz, site_dir, db_chunk_bytes)
        run.record(chunk_files)
        chunks = len(chunk_files)
    run.will_write("site", "index.html").write_text(
        get_template(
            schema_ddl,
            media_base="../data/assets/",
            strict_network=True,
            db_chunks=chunks,
        ),
        encoding="utf-8",
    )
    run.record(copy_sql_js(site_dir))
    run.record(write_wasm_script(site_dir))
    run.will_write("site", README_NAME).write_text(render_site_readme(), encoding="utf-8")
    return chunks
