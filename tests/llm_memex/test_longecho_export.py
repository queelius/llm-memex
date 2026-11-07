"""End-to-end tests for ``--format longecho`` (llm_memex.exporters.longecho)."""
import base64
import contextlib
import gzip
import os
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from llm_memex.db import Database
from llm_memex.exporters import longecho
from llm_memex.models import Conversation, Message, media_block, text_block

ARCHIVED_ID = "archived-conv-id-xyz"
ARCHIVED_TEXT = "ARCHIVED-ONLY-TEXT-QQQ"
CONV_NOTE = "CONVERSATION-NOTE-ZZZ"
MSG_NOTE = "MESSAGE-NOTE-ZZZ"
REDACTED_TEXT = "REDACTED-SECRET-UNDO-QQQ"

_ROOT_FILES = ("README.md", "SHA256SUMS")
_CHUNK_RE = re.compile(r'llmMemexDbChunk\((\d+),"([^"]*)"\);')


def _conv(cid, title, source, day, messages, archived=False):
    when = datetime(2025, 6, day, 8, 0)
    conv = Conversation(cid, when, when, title=title, source=source,
                        archived_at=when if archived else None)
    for message in messages:
        conv.add_message(message)
    return conv


def _msg(mid, role, content, parent=None):
    return Message(id=mid, role=role, parent_id=parent,
                   content=content if isinstance(content, list) else [text_block(content)])


@pytest.fixture
def source(tmp_path):
    """A live archive: branching, media, archived, and noted conversations."""
    directory = tmp_path / "src"
    (directory / "assets").mkdir(parents=True)
    (directory / "assets" / "pic.png").write_bytes(b"\x89PNG fake image")
    with Database(str(directory)) as db:
        db.save_conversation(_conv(
            "branchy", "Branching talk", "openai", 3,
            [_msg("m1", "user", "root question"),
             _msg("m2", "assistant", "first answer", "m1"),
             _msg("m3", "assistant", "second answer", "m1")]))
        db.save_conversation(_conv(
            "pictured", "With a picture", "claude_code", 9,
            [_msg("p1", "user", [text_block("look"),
                                 media_block("image/png", url="assets/pic.png")])]))
        db.save_conversation(_conv(
            ARCHIVED_ID, "Hidden away", "openai", 12,
            [_msg("a1", "user", ARCHIVED_TEXT)], archived=True))
        db.add_note(conversation_id="branchy", text=CONV_NOTE)
        db.add_note(conversation_id="branchy", message_id="m2", text=MSG_NOTE)
        # What the redact script keeps so a redaction can be undone: the
        # secret itself, which must never travel.
        db.save_enrichment("branchy", "original_content",
                           f'{{"message_id": "m2", "content": "{REDACTED_TEXT}"}}',
                           "redact")
    return directory / "conversations.db"


def _export(source_db, out, **kwargs):
    return longecho.export([], str(out), db_path=str(source_db), **kwargs)


def _all_files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file())


def _site_db_bytes(out):
    """The database rebuilt from site/db/*.js, as the viewer would."""
    chunks = {}
    for script in sorted((out / "site" / "db").glob("*.js")):
        index, b64 = _CHUNK_RE.fullmatch(script.read_text().strip()).groups()
        chunks[int(index)] = base64.b64decode(b64)
    return gzip.decompress(b"".join(chunks[i] for i in sorted(chunks)))


def _sha256_ok(out):
    return subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=out,
                          capture_output=True, text=True)


# ------------------------------------------------------------------ outcome

def test_export_returns_count_of_retained_conversations(source, tmp_path):
    assert _export(source, tmp_path / "out") == 2


def test_export_is_longecho_compliant(source, tmp_path):
    checker = pytest.importorskip("longecho.checker")
    out = tmp_path / "out"
    _export(source, out)
    result = checker.check_compliance(out)
    assert result.compliant, result
    assert result.warnings == []


def test_layout(source, tmp_path):
    out = tmp_path / "out"
    _export(source, out)
    for name in ("README.md", "SHA256SUMS", "data/conversations.db",
                 "data/arkiv/conversations.jsonl", "data/assets/pic.png",
                 "text/index.md", "site/index.html", "site/sql-wasm.js",
                 "site/sql-wasm-binary.js", "site/db/000.js", "site/README.md"):
        assert (out / name).is_file(), name
    texts = [p for p in (out / "text").rglob("*.md") if p.name != "index.md"]
    assert len(texts) == 2
    assert {p.parent.name for p in texts} == {"2025"}
    assert not [p for p in out.rglob("*") if p.name.endswith(("-wal", "-shm"))]
    assert not (out / "data" / "conversations.db-journal").exists()


def _sums_listing(out):
    return [line.split("  ", 1)[1] for line in (out / "SHA256SUMS").read_text().splitlines()]


def test_sums_list_exactly_what_the_export_wrote_and_verify(source, tmp_path):
    out = tmp_path / "out"
    _export(source, out)
    expected = [
        "README.md", "data/arkiv/README.md", "data/arkiv/conversations.jsonl",
        "data/arkiv/schema.yaml", "data/assets/pic.png", "data/conversations.db",
        "site/README.md", "site/db/000.js", "site/index.html",
        "site/sql-wasm-binary.js", "site/sql-wasm.js", "text/index.md",
    ] + sorted(p.relative_to(out).as_posix()
               for p in (out / "text" / "2025").glob("*.md"))
    assert _sums_listing(out) == sorted(expected)
    assert _sha256_ok(out).returncode == 0


def test_hand_added_files_survive_repeated_reexports(source, tmp_path):
    out = tmp_path / "out"
    _export(source, out)
    (out / "my-notes.txt").write_text("root")
    (out / "text" / "mine.md").write_text("in text")
    (out / "data" / "extra.txt").write_text("in data")
    for _ in range(2):
        _export(source, out)
        assert (out / "my-notes.txt").read_text() == "root"
        assert (out / "text" / "mine.md").read_text() == "in text"
        assert (out / "data" / "extra.txt").read_text() == "in data"
        listed = _sums_listing(out)
        assert not {"my-notes.txt", "text/mine.md", "data/extra.txt"} & set(listed)
        assert _sha256_ok(out).returncode == 0


def test_readme_stamped_and_site_uses_data_assets(source, tmp_path):
    out = tmp_path / "out"
    _export(source, out, owner="Ada")
    readme = (out / "README.md").read_text()
    assert "generator: llm-memex longecho" in readme
    assert "Ada" in readme
    assert "generator: llm-memex longecho" in (out / "site" / "README.md").read_text()
    page = (out / "site" / "index.html").read_text()
    assert "../data/assets/" in page


# ------------------------------------------------------------------ privacy

@pytest.mark.parametrize("include_notes", [True, False])
def test_privacy_every_part(source, tmp_path, include_notes):
    out = tmp_path / "out"
    _export(source, out, include_notes=include_notes)
    leaks = [ARCHIVED_ID.encode(), ARCHIVED_TEXT.encode(), REDACTED_TEXT.encode()]
    if not include_notes:
        leaks += [CONV_NOTE.encode(), MSG_NOTE.encode()]
    parts = {
        "data/conversations.db": (out / "data" / "conversations.db").read_bytes(),
        "site db chunks": _site_db_bytes(out),
    }
    for path in _all_files(out):
        rel = path.relative_to(out).as_posix()
        if rel.startswith(("data/arkiv/", "text/")) or rel in _ROOT_FILES:
            parts[rel] = path.read_bytes()
    assert len(parts) > 5
    for name, blob in parts.items():
        for leak in leaks:
            assert leak not in blob, f"{leak!r} leaked into {name}"

    with contextlib.closing(sqlite3.connect(out / "data" / "conversations.db")) as conn:
        snapshot_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    assert snapshot_count == 2
    assert f"{snapshot_count} conversations" in (out / "README.md").read_text()
    if include_notes:
        assert CONV_NOTE in "".join(
            p.read_text() for p in (out / "text").rglob("*.md"))


def test_include_archived_keeps_it_everywhere(source, tmp_path):
    out = tmp_path / "out"
    assert _export(source, out, include_archived=True) == 3
    assert ARCHIVED_TEXT.encode() in (out / "data" / "arkiv" / "conversations.jsonl").read_bytes()
    assert "3 conversations" in (out / "README.md").read_text()


# ---------------------------------------------------------------- size cap

def test_size_cap(source, tmp_path):
    checker = pytest.importorskip("longecho.checker")
    out = tmp_path / "out"
    _export(source, out, site_max_db_bytes=1)
    assert not (out / "site" / "db").exists()
    assert "var DB_CHUNKS = 0" in (out / "site" / "index.html").read_text()
    assert "too large for the browser viewer" in (out / "README.md").read_text()
    assert checker.check_compliance(out).compliant
    assert not list(out.rglob("*.gz"))


def test_chunk_size_is_respected(source, tmp_path):
    out = tmp_path / "out"
    _export(source, out, db_chunk_bytes=1024)
    chunks = sorted((out / "site" / "db").glob("*.js"))
    assert len(chunks) > 1
    assert f"var DB_CHUNKS = {len(chunks)}" in (out / "site" / "index.html").read_text()
    assert _site_db_bytes(out) == (out / "data" / "conversations.db").read_bytes()


# --------------------------------------------------------------- re-export

def test_reexport_removes_dropped_conversation(source, tmp_path):
    out = tmp_path / "out"
    _export(source, out)
    before = {p.name for p in (out / "text").rglob("*.md")}
    with contextlib.closing(sqlite3.connect(source)) as conn:
        conn.execute("UPDATE conversations SET archived_at = '2026-01-01T00:00:00' "
                     "WHERE id = 'pictured'")
        conn.commit()
    assert _export(source, out) == 1
    after = {p.name for p in (out / "text").rglob("*.md")}
    assert len(before) == 3 and len(after) == 2  # two conversations plus index.md, then one
    assert not (out / "data" / "assets" / "pic.png").exists()
    assert _sha256_ok(out).returncode == 0


def test_refuses_a_folder_it_did_not_write(source, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="refusing"):
        _export(source, out)
    assert (out / "precious.txt").read_text() == "keep"


# ------------------------------------------------------------ empty database

def test_empty_database(tmp_path):
    checker = pytest.importorskip("longecho.checker")
    with Database(str(tmp_path / "empty")):
        pass
    out = tmp_path / "out"
    assert _export(tmp_path / "empty" / "conversations.db", out) == 0
    assert checker.check_compliance(out).compliant
    assert "0 conversations" in (out / "README.md").read_text()
    assert (out / "text" / "index.md").is_file()
    assert (out / "data" / "arkiv" / "conversations.jsonl").read_bytes() == b""
    assert _sha256_ok(out).returncode == 0


# ------------------------------------------------------------------ arguments

def test_requires_db_path(tmp_path):
    with pytest.raises(ValueError, match="db_path"):
        longecho.export([], str(tmp_path / "out"))
    with pytest.raises(ValueError, match="db_path"):
        longecho.export([], str(tmp_path / "out"), db_path=":memory:")
    with pytest.raises(ValueError, match="db_path"):
        longecho.export([], str(tmp_path / "out"), db_path=str(tmp_path / "nope.db"))
    assert not (tmp_path / "out").exists()


def test_reads_database_flag():
    assert longecho.READS_DATABASE is True


# ------------------------------------------------------- destination overlap

def test_refuses_destination_overlapping_source(source, tmp_path):
    before = source.read_bytes()
    for bad in (source.parent, tmp_path, source.parent / "assets" / "deep"):
        with pytest.raises(ValueError, match="must not overwrite the source"):
            _export(source, bad)
    assert source.read_bytes() == before
    assert (source.parent / "assets" / "pic.png").is_file()
    assert not (source.parent / "README.md").exists()


@pytest.mark.parametrize("target", ["data/assets", ""])
def test_refuses_source_assets_symlinked_into_destination(source, tmp_path, target):
    """A symlinked source assets/ must not let the rmtree of data/assets reach it."""
    out = tmp_path / "out"
    media = out / "data" / "assets"
    media.mkdir(parents=True)
    (media / "keep.png").write_bytes(b"precious")
    shutil.rmtree(source.parent / "assets")
    (source.parent / "assets").symlink_to(out / target if target else out)
    with pytest.raises(ValueError, match="must not overwrite the source"):
        _export(source, out)
    assert (media / "keep.png").read_bytes() == b"precious"


# ------------------------------------------------------- planted symlinks

@pytest.mark.parametrize("name", ["README.md", "SHA256SUMS", "data", "text", "site"])
def test_refuses_symlinked_destination_entry(source, tmp_path, name):
    out = tmp_path / "out"
    _export(source, out)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("do not touch")
    planted = out / name
    if planted.is_dir():
        shutil.copytree(planted, outside / name)
        shutil.rmtree(planted)
    else:
        shutil.copy2(planted, outside / name)
        planted.unlink()
    planted.symlink_to(outside / name)
    snapshot = {p.relative_to(outside): p.read_bytes() for p in _all_files(outside)}
    with pytest.raises(ValueError, match=re.escape(f"{planted} is a symbolic link")):
        _export(source, out)
    assert planted.is_symlink()
    assert {p.relative_to(outside): p.read_bytes() for p in _all_files(outside)} == snapshot


@pytest.mark.parametrize("name", ["data/arkiv", "data/assets"])
def test_refuses_foreign_content_before_deleting_anything(source, tmp_path, name):
    out = tmp_path / "out"
    _export(source, out)
    foreign = out / name / "hand-made.txt"
    foreign.write_text("mine")
    with pytest.raises(ValueError, match=re.escape(f"{foreign} was not written")):
        _export(source, out)
    assert foreign.read_text() == "mine"
    # the previous export is still whole: nothing was deleted before refusing
    assert (out / "data" / "conversations.db").is_file()
    assert (out / "SHA256SUMS").is_file()
    assert len(list((out / "text").rglob("*.md"))) == 3


def test_damaged_listing_is_refused_before_deleting_anything(source, tmp_path):
    """A SHA256SUMS that no longer lists data/arkiv: refused up front, not
    after the rest of the export was cleared."""
    out = tmp_path / "out"
    _export(source, out)
    (out / "SHA256SUMS").write_text("")
    before = _tree(out)
    with pytest.raises(ValueError, match="was not written"):
        _export(source, out)
    assert _tree(out) == before


def test_listed_name_that_is_no_longer_a_regular_file_is_refused_up_front(source, tmp_path):
    """It could not be deleted, so data/assets would survive the clearing."""
    out = tmp_path / "out"
    _export(source, out)
    pic = out / "data" / "assets" / "pic.png"
    pic.unlink()
    os.mkfifo(pic)
    before = _tree(out)
    with pytest.raises(ValueError, match="not a regular file"):
        _export(source, out)
    assert _tree(out) == before


def test_refuses_foreign_empty_directory_where_it_would_write(source, tmp_path):
    out = tmp_path / "out"
    _export(source, out)
    (out / "data" / "assets" / "mine").mkdir()
    with pytest.raises(ValueError, match="was not written"):
        _export(source, out)
    assert (out / "data" / "conversations.db").is_file()


def test_file_where_a_blind_directory_goes_is_refused_before_deleting(source, tmp_path):
    """Re-review probe: a file at data/arkiv passed the up-front checks, so the
    previous export was cleared before the leftover backstop refused."""
    out = tmp_path / "out"
    _export(source, out)
    shutil.rmtree(out / "data" / "arkiv")
    (out / "data" / "arkiv").write_text("hand")
    before = _tree(out)
    with pytest.raises(ValueError, match="was not written"):
        _export(source, out)
    assert _tree(out) == before


@pytest.mark.parametrize("name", ["data/arkiv", "data/assets"])
def test_emptied_blind_directory_is_replaced(source, tmp_path, name):
    """Re-review probe: its listed files gone, the empty directory used to
    survive the clearing, so the backstop refused after the export was deleted."""
    out = tmp_path / "out"
    _export(source, out)
    _hand_added(out)
    shutil.rmtree(out / name)
    (out / name).mkdir()
    assert _export(source, out) == 2
    _assert_hand_added_kept(out)
    _assert_clean_export(out)


def test_unlisted_empty_blind_directory_is_refused_before_deleting(source, tmp_path):
    """data/assets that the previous export never wrote is not ours to remove."""
    out = tmp_path / "out"
    _export(source, out)
    shutil.rmtree(out / "data" / "assets")
    write = (out / "SHA256SUMS").read_text().splitlines()
    (out / "SHA256SUMS").write_text(
        "".join(l + "\n" for l in write if "data/assets/" not in l))
    (out / "data" / "assets").mkdir()
    before = _tree(out)
    with pytest.raises(ValueError, match="was not written"):
        _export(source, out)
    assert _tree(out) == before


# ----------------------------------------------- symlinks below the top level

def _plant(out, tmp_path, kind):
    """Replace a path below site/, text/ or data/ with a symlink to outside."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("precious")
    year = next((out / "text").iterdir())
    if kind == "site-index":
        (outside / "index.html").write_text("outside page")
        (out / "site" / "index.html").unlink()
        (out / "site" / "index.html").symlink_to(outside / "index.html")
        return outside, "site/index.html"
    if kind == "text-year":
        name = year.name
        shutil.rmtree(year)
        year.symlink_to(outside, target_is_directory=True)
        return outside, f"text/{name}"
    if kind == "site-db":
        shutil.rmtree(out / "site" / "db")
        (out / "site" / "db").symlink_to(outside, target_is_directory=True)
        return outside, "site/db"
    if kind == "data-db":
        (outside / "conversations.db").write_text("outside db")
        (out / "data" / "conversations.db").unlink()
        (out / "data" / "conversations.db").symlink_to(outside / "conversations.db")
        return outside, "data/conversations.db"
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["site-index", "text-year", "site-db", "data-db"])
def test_refuses_nested_symlink_and_touches_nothing_outside(source, tmp_path, kind):
    out = tmp_path / "out"
    _export(source, out)
    outside, rel = _plant(out, tmp_path, kind)
    before = {p.relative_to(outside): p.read_bytes() for p in _all_files(outside)}
    sums = (out / "SHA256SUMS").read_bytes()
    with pytest.raises(ValueError, match=re.escape(f"{out / rel} is a symbolic link")):
        _export(source, out)
    assert {p.relative_to(outside): p.read_bytes() for p in _all_files(outside)} == before
    assert (out / rel).is_symlink()
    # refused before anything was deleted: the previous export is still whole
    assert (out / "SHA256SUMS").read_bytes() == sums
    assert "generator: llm-memex longecho" in (out / "README.md").read_text()


# ------------------------------------------------- refusals and failed runs

def _tree(root):
    """Every entry below *root*: file bytes, symlink targets, and directories."""
    tree = {}
    for path in sorted(Path(root).rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            tree[rel] = ("link", str(path.readlink()))
        elif path.is_dir():
            tree[rel] = ("dir",)
        elif path.is_file():
            tree[rel] = path.read_bytes()
        else:
            tree[rel] = ("special",)
    return tree


def _hand_added(out):
    (out / "my-letter.md").write_text("hand-added")
    (out / "text" / "mine.md").write_text("in text")
    (out / "data" / "extra.txt").write_text("in data")


def _assert_hand_added_kept(out):
    assert (out / "my-letter.md").read_text() == "hand-added"
    assert (out / "text" / "mine.md").read_text() == "in text"
    assert (out / "data" / "extra.txt").read_text() == "in data"


def _assert_clean_export(out):
    """Every file is either listed in SHA256SUMS or one of the hand-added ones."""
    listed = set(_sums_listing(out))
    present = {p.relative_to(out).as_posix() for p in _all_files(out)}
    assert present - listed == {"SHA256SUMS", "my-letter.md", "text/mine.md",
                                "data/extra.txt"}
    assert _sha256_ok(out).returncode == 0


def test_symlink_refusal_keeps_previous_export_and_next_run_replaces_it(source, tmp_path):
    """Reviewer probe 1: a refusal used to delete the export first, after which
    the folder was refused for good as not written by llm-memex."""
    out = tmp_path / "out"
    _export(source, out)
    _hand_added(out)
    before = _tree(out)
    outside = tmp_path / "outside"
    outside.mkdir()
    (out / "text" / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        _export(source, out)
    (out / "text" / "link").unlink()
    assert _tree(out) == before
    assert _sha256_ok(out).returncode == 0
    assert _export(source, out) == 2
    _assert_hand_added_kept(out)
    _assert_clean_export(out)


def test_snapshot_failure_keeps_previous_export(source, tmp_path, monkeypatch):
    """Reviewer probe 2: the snapshot (the step a full disk breaks) runs before
    the destination is touched."""
    out = tmp_path / "out"
    _export(source, out)
    _hand_added(out)
    before = _tree(out)

    def disk_full(*args, **kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(longecho, "take_snapshot", disk_full)
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        _export(source, out)
    assert _tree(out) == before
    assert _sha256_ok(out).returncode == 0
    monkeypatch.undo()
    assert _export(source, out) == 2
    _assert_hand_added_kept(out)
    _assert_clean_export(out)


# Steps of the build: before data/assets is written (data/arkiv already is),
# inside site/, and at the very end.
_BUILD_STEPS = ["copy_retained_assets", "write_wasm_script", "render_root_readme"]


def _staging_dirs(out):
    return [p for p in out.iterdir() if p.name.startswith(".llm-memex-staging-")]


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"),
                                   KeyboardInterrupt()], ids=["oserror", "interrupt"])
@pytest.mark.parametrize("step", _BUILD_STEPS)
def test_failed_build_keeps_previous_export(source, tmp_path, monkeypatch, step, error):
    """The new export is built aside: a failure anywhere in it leaves the
    previous export byte for byte, with no staging debris."""
    out = tmp_path / "out"
    _export(source, out)
    _hand_added(out)
    before = _tree(out)

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(longecho, step, boom)
    with pytest.raises(type(error)):
        _export(source, out)
    assert _tree(out) == before
    assert _sha256_ok(out).returncode == 0

    monkeypatch.undo()
    assert _export(source, out) == 2
    _assert_hand_added_kept(out)
    _assert_clean_export(out)


def test_failed_first_export_leaves_nothing(source, tmp_path, monkeypatch):
    out = tmp_path / "new" / "out"

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(longecho, "write_wasm_script", boom)
    with pytest.raises(OSError):
        _export(source, out)
    assert not out.exists()
    monkeypatch.undo()
    assert _export(source, out) == 2
    assert _sha256_ok(out).returncode == 0
    listed = set(_sums_listing(out))
    assert {p.relative_to(out).as_posix() for p in _all_files(out)} - listed == {"SHA256SUMS"}


def _fail_after_moves(monkeypatch, n):
    """Make the swap's os.replace fail after *n* successful moves."""
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        # Moves out of staging only (writing SHA256SUMS renames within it).
        if ".llm-memex-staging-" in str(src) and ".llm-memex-staging-" not in str(dst):
            calls["n"] += 1
            if calls["n"] > n:
                raise OSError(5, "Input/output error")
        return real(src, dst)

    monkeypatch.setattr(longecho.os, "replace", flaky)


@pytest.mark.parametrize("n", [0, 3])
def test_failed_swap_is_replaced_by_the_next_run(source, tmp_path, monkeypatch, n):
    out = tmp_path / "out"
    _export(source, out)
    _hand_added(out)
    _fail_after_moves(monkeypatch, n)
    with pytest.raises(OSError, match="Input/output"):
        _export(source, out)
    monkeypatch.undo()
    # stamped, and what the run moved in is listed, so the next run replaces it
    assert "generator: llm-memex longecho" in (out / "README.md").read_text()
    assert _sha256_ok(out).returncode == 0
    assert not _staging_dirs(out)
    _assert_hand_added_kept(out)
    assert _export(source, out) == 2
    _assert_hand_added_kept(out)
    _assert_clean_export(out)


def test_killed_swap_is_reported_as_interrupted(source, tmp_path, monkeypatch):
    """A hard kill mid-swap (here: the recovery listing fails too) leaves the
    stamped README without SHA256SUMS, which is refused with advice that
    keeps hand-added files; the original error is the one raised."""
    out = tmp_path / "out"
    _export(source, out)
    _fail_after_moves(monkeypatch, 3)
    real_sums = longecho.write_sha256sums

    def sums_fail(dest, files):
        if Path(dest) == out:
            raise OSError(28, "No space left on device")
        return real_sums(dest, files)

    monkeypatch.setattr(longecho, "write_sha256sums", sums_fail)
    with pytest.raises(OSError, match="Input/output"):
        _export(source, out)
    assert "generator: llm-memex longecho" in (out / "README.md").read_text()
    assert not (out / "SHA256SUMS").exists()
    monkeypatch.undo()
    with pytest.raises(ValueError, match="interrupted") as err:
        _export(source, out)
    assert "Move out any files you added yourself" in str(err.value)


@pytest.mark.parametrize("first", [True, False], ids=["first-export", "re-export"])
def test_killed_build_leftovers_are_removed(source, tmp_path, first):
    """A staging folder a killed run left is the exporter's own: the next run
    removes it, and it does not make a fresh destination look foreign."""
    out = tmp_path / "out"
    if not first:
        _export(source, out)
    stale = out / ".llm-memex-staging-dead"
    (stale / "data").mkdir(parents=True)
    (stale / "data" / "half.db").write_bytes(b"partial")
    (stale / ".llm-memex-staging").write_text("marker")
    (out / ".llm-memex-staging-empty").mkdir()
    assert _export(source, out) == 2
    assert not _staging_dirs(out)
    listed = set(_sums_listing(out))
    assert {p.relative_to(out).as_posix() for p in _all_files(out)} - listed == {"SHA256SUMS"}


def test_unmarked_folder_with_staging_name_is_never_removed(source, tmp_path):
    """Review probe: cleanup used to delete any folder named like staging."""
    out = tmp_path / "out"
    _export(source, out)
    mine = out / ".llm-memex-staging-my-backup"
    mine.mkdir()
    (mine / "keep.txt").write_text("mine")
    assert _export(source, out) == 2
    assert (mine / "keep.txt").read_text() == "mine"
    fresh = tmp_path / "fresh"
    (fresh / ".llm-memex-staging-my-backup").mkdir(parents=True)
    (fresh / ".llm-memex-staging-my-backup" / "keep.txt").write_text("mine")
    with pytest.raises(ValueError, match="not written by"):
        _export(source, fresh)


def test_first_export_failing_on_first_move_leaves_nothing(source, tmp_path, monkeypatch):
    """Review probe: an empty SHA256SUMS used to remain, refused for good."""
    out = tmp_path / "out"
    _fail_after_moves(monkeypatch, 0)
    with pytest.raises(OSError, match="Input/output"):
        _export(source, out)
    monkeypatch.undo()
    assert not out.exists()
    assert _export(source, out) == 2


def test_interrupt_just_after_a_move_keeps_it_listed(source, tmp_path, monkeypatch):
    """Review probe: a file renamed in but not yet recorded was left out of the
    listing, and under data/arkiv blocked every later run."""
    out = tmp_path / "out"
    _export(source, out)
    _hand_added(out)
    real = os.replace

    def interrupted(src, dst):
        real(src, dst)
        if ".llm-memex-staging-" in str(src) and "/data/arkiv/" in str(dst):
            raise KeyboardInterrupt

    monkeypatch.setattr(longecho.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _export(source, out)
    monkeypatch.undo()
    assert any(p.startswith("data/arkiv/") for p in _sums_listing(out))
    assert _export(source, out) == 2
    _assert_hand_added_kept(out)
    _assert_clean_export(out)


def test_new_file_landing_on_a_hand_made_one_is_refused_before_touching(source, tmp_path):
    """The previous export used to overwrite an unlisted file at a path it
    writes; now that is refused while everything is still in place."""
    out = tmp_path / "out"
    _export(source, out)
    lines = (out / "SHA256SUMS").read_text().splitlines()
    (out / "SHA256SUMS").write_text(
        "".join(l + "\n" for l in lines if not l.endswith("  text/index.md")))
    (out / "text" / "index.md").write_text("my own index")
    before = _tree(out)
    with pytest.raises(ValueError, match="was not written"):
        _export(source, out)
    assert _tree(out) == before


def test_mixed_naive_and_aware_created_at(tmp_path):
    """Reviewer probe 3: one row with an offset used to crash text/index.md."""
    directory = tmp_path / "src"
    with Database(str(directory)) as db:
        for i in range(2):
            db.save_conversation(_conv(f"c{i}", f"T{i}", "openai", 14 + i,
                                       [_msg("m", "user", "x")]))
    with contextlib.closing(sqlite3.connect(directory / "conversations.db")) as conn:
        conn.execute("UPDATE conversations SET created_at='2025-06-14T08:00:00+00:00' "
                     "WHERE id='c0'")
        conn.commit()
    out = tmp_path / "out"
    assert _export(directory / "conversations.db", out) == 2
    index = (out / "text" / "index.md").read_text()
    assert index.index("[T1]") < index.index("[T0]")  # newest first


# -------------------------------------------------------------- live source

def test_source_untouched(source, tmp_path):
    with Database(str(source.parent)) as db:
        db.conn.execute("PRAGMA wal_checkpoint(FULL)")
    _export(source, tmp_path / "out")
    with contextlib.closing(sqlite3.connect(source)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    assert (count, notes) == (3, 2)
