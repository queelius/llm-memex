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
    STAGING_MARKER,
    STAGING_PREFIX,
    SUMS_NAME,
    ArchiveSummary,
    clear_destination,
    inspect_destination,
    is_staging_leftover,
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
# Directories an export owns whole: anything in them the previous export did
# not list is refused, since the new copies replace them wholesale.
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
    # Every refusal is decided here, before anything is written or deleted,
    # so a refused run leaves the previous export exactly as it was.
    listed = inspect_destination(out)
    _refuse_foreign_content(out, listed)
    _refuse_symlinks(out)

    created = not out.exists()
    out.mkdir(parents=True, exist_ok=True)
    _remove_stale_staging(out)
    # Built inside the destination, so every move into place is a rename on
    # one filesystem; the previous export is untouched until the build is
    # complete, and a failed build costs nothing but the staging folder.
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out))
    try:
        (staging / STAGING_MARKER).write_text("llm-memex longecho staging\n", encoding="utf-8")
        count = _build(
            staging, db_path,
            include_notes=include_notes, include_archived=include_archived,
            owner=owner, site_max_db_bytes=site_max_db_bytes,
            db_chunk_bytes=db_chunk_bytes,
        )
        files = _export_files(staging)
        _refuse_collisions(out, files, listed)
        _swap(out, staging, files, listed)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if created:
            with contextlib.suppress(OSError):
                out.rmdir()  # only if the failed run left it empty
        raise
    shutil.rmtree(staging, ignore_errors=True)
    return count


def _build(
    dest: Path,
    db_path: Path,
    *,
    include_notes: bool,
    include_archived: bool,
    owner: Any,
    site_max_db_bytes: int,
    db_chunk_bytes: int,
) -> int:
    """Write a complete export into the empty directory *dest*; return its count."""
    # The temporary directory is removed before an error propagates, which
    # frees its space when a full disk was the failure.
    with tempfile.TemporaryDirectory(prefix="llm-memex-longecho-") as tmp:
        tmp_dir = Path(tmp)
        snapshot = tmp_dir / _SNAPSHOT_NAME
        take_snapshot(
            db_path, snapshot, include_notes=include_notes,
            include_archived=include_archived,
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
            (dest / "data").mkdir()
            count = arkiv_export.write_directory(
                _text_writing_stream(snap_db, ids, dest, entries, include_notes),
                dest / "data" / "arkiv",
                include_notes=include_notes,
                db=snap_db if include_notes else None,
            )

        # Opening the snapshot through Database switched it to WAL.
        with contextlib.closing(sqlite3.connect(str(snapshot))) as conn:
            conn.execute("PRAGMA journal_mode=DELETE")
        shutil.copyfile(snapshot, dest / "data" / _SNAPSHOT_NAME)
        copy_retained_assets(snapshot, db_path.parent / "assets", dest / "data" / "assets")

        (dest / "text").mkdir(exist_ok=True)
        (dest / "text" / "index.md").write_text(render_index(entries), encoding="utf-8")

        chunks = _write_site(
            dest, snapshot, tmp_dir, schema_ddl,
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
    (dest / README_NAME).write_text(render_root_readme(summary), encoding="utf-8")
    # Everything in dest is this run's, so the listing is the whole tree.
    write_sha256sums(dest, _export_files(dest))
    return count


def _swap(
    out: Path, staging: Path, files: List[PurePosixPath], listed: Set[PurePosixPath]
) -> None:
    """Replace the previous export in *out* with the finished one in *staging*.

    Only renames happen here, so the folder is in a mixed state for
    moments, not for the length of an export. The previous export's
    stamped README stays until the new one replaces it in one rename (the
    first move), and ``SHA256SUMS`` arrives last. If a move fails, what was
    moved is listed in ``SHA256SUMS`` before the error propagates, so the
    next run replaces it; only a hard kill in this window leaves the stamped
    README without a listing, which the next run reports as interrupted.
    """
    readme, sums = PurePosixPath(README_NAME), PurePosixPath(SUMS_NAME)
    clear_destination(out, set(listed) - {readme})
    moved: List[PurePosixPath] = []
    try:
        for rel in [readme] + [f for f in files if f not in (readme, sums)]:
            target = out.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Recorded first, so an interrupt just after the rename cannot
            # leave a moved file out of the recovery listing (missing paths
            # are skipped when it is written).
            moved.append(rel)
            os.replace(staging.joinpath(*rel.parts), target)
        os.replace(staging / SUMS_NAME, out / SUMS_NAME)
    except BaseException:
        # Only beside a stamped README (old or new, ours either way): a first
        # export that failed before its README arrived lists nothing, so the
        # caller can remove the folder it created.
        if (out / README_NAME).is_file():
            with contextlib.suppress(Exception):
                write_sha256sums(out, [readme, *moved])
        raise


def _remove_stale_staging(out: Path) -> None:
    """Remove staging folders a killed run left (marked as such, or empty)."""
    for entry in out.iterdir():
        if is_staging_leftover(entry):
            shutil.rmtree(entry)


def _files_under(root: Path, rel: str = ".") -> List[PurePosixPath]:
    """Every regular file below ``root/rel``, as sorted paths relative to *root*."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root / rel, followlinks=False):
        for name in filenames:
            full = Path(dirpath) / name
            found.append(PurePosixPath(full.relative_to(root).as_posix()))
    return sorted(found)


def _export_files(staging: Path) -> List[PurePosixPath]:
    """The export's files in *staging*: all of them but the staging marker."""
    return [f for f in _files_under(staging) if f != PurePosixPath(STAGING_MARKER)]


def _leftover_message(path: Path) -> str:
    return (
        f"{path} was not written by a previous export; refusing to overwrite or "
        f"delete it. Move {path} away (or remove it) and export again."
    )


def _refuse_collisions(
    out: Path, files: List[PurePosixPath], listed: Set[PurePosixPath]
) -> None:
    """Refuse, before anything is deleted, to move a new file onto one a person made.

    A path the new export writes may already exist in *out* only as a file
    the previous export listed (or as its README and listing); and no
    directory it needs may be a file there.
    """
    ours = set(listed) | {PurePosixPath(README_NAME), PurePosixPath(SUMS_NAME)}
    for rel in files:
        for parent in list(rel.parents)[:-1]:
            full = out.joinpath(*parent.parts)
            if full.is_symlink() or (full.exists() and not full.is_dir()):
                raise ValueError(_leftover_message(full))
        target = out.joinpath(*rel.parts)
        if (target.is_symlink() or target.exists()) and rel not in ours:
            raise ValueError(_leftover_message(target))


def _refuse_foreign_content(out: Path, listed: Set[PurePosixPath]) -> None:
    """Before anything is deleted: refuse if data/arkiv or data/assets hold
    anything but regular files the previous export's SHA256SUMS lists (*listed*,
    from ``inspect_destination``) and the directories that hold them.

    A fresh export rewrites those two directories wholesale, so hand-added
    content in them would be lost; and once the listed files are deleted the
    two directories must be gone before the new ones move in.
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


def _text_writing_stream(
    snap_db: Database,
    ids: List[str],
    dest: Path,
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
        target = dest.joinpath("text", *rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_conversation(conv, notes), encoding="utf-8")
        entries.append(IndexEntry(rel, conv.title or "", conv.created_at, conv.source))
        yield conv


def _write_site(
    dest: Path,
    snapshot: Path,
    tmp_dir: Path,
    schema_ddl: str,
    *,
    site_max_db_bytes: int,
    db_chunk_bytes: int,
) -> int:
    """Write the viewer into ``dest/site``; return the number of database
    chunks (0 = left out, too large)."""
    site_dir = dest / "site"
    site_dir.mkdir(exist_ok=True)
    chunks = 0
    if snapshot.stat().st_size <= site_max_db_bytes:
        db_gz = tmp_dir / f"{_SNAPSHOT_NAME}.gz"
        gzip_file(snapshot, db_gz, keep_source=True)
        chunks = len(write_db_chunks(db_gz, site_dir, db_chunk_bytes))
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
    (site_dir / README_NAME).write_text(
        render_site_readme(site_has_data=chunks > 0), encoding="utf-8"
    )
    return chunks
