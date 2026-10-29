"""Tests for the longecho README, checksum, and destination rules."""
import hashlib
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from llm_memex.db import Database
from llm_memex.exporters._longecho_docs import (
    GENERATOR_PREFIX,
    README_QUERIES,
    ArchiveSummary,
    prepare_destination,
    render_root_readme,
    render_site_readme,
    summarize,
    write_sha256sums,
)
from llm_memex.exporters._snapshot import strip_fts5_and_vacuum
from llm_memex.models import Conversation, Message, text_block


def _summary(**overrides):
    values = dict(
        conversations=12, messages=340, sources=["claude_code", "openai"],
        first="2024-01-05", last="2026-10-29", owner=None, include_notes=True,
        include_archived=False, site_has_data=True,
        exported_at="2026-10-29T12:00:00+00:00",
    )
    values.update(overrides)
    return ArchiveSummary(**values)


def _split(readme):
    """Return (frontmatter dict, body) of a README."""
    assert readme.startswith("---\n")
    head, _, body = readme[4:].partition("\n---\n")
    return yaml.safe_load(head), body


def _seed_db(tmp_path):
    source = tmp_path / "source"
    with Database(str(source)) as db:
        for cid, src, day in (("a", "openai", 3), ("b", "claude_code", 9)):
            when = datetime(2025, 6, day, 8, 0)
            conv = Conversation(cid, when, when, title=f"About sqlite {cid}", source=src)
            conv.add_message(Message(id="m1", role="user",
                                     content=[text_block("I like sqlite")]))
            conv.add_message(Message(id="m2", role="assistant", parent_id="m1",
                                     content=[text_block("so do I")]))
            db.save_conversation(conv)
    return source / "conversations.db"


# ---------------------------------------------------------------- summarize

def test_summarize_counts_sources_and_dates(tmp_path):
    db_path = _seed_db(tmp_path)
    assert summarize(db_path) == {
        "conversations": 2, "messages": 4,
        "sources": ["claude_code", "openai"],
        "first": "2025-06-03", "last": "2025-06-09",
    }


def test_summarize_empty_database(tmp_path):
    with Database(str(tmp_path / "empty")):
        pass
    assert summarize(tmp_path / "empty" / "conversations.db") == {
        "conversations": 0, "messages": 0, "sources": [], "first": None, "last": None,
    }


def test_summarize_does_not_switch_to_wal(tmp_path):
    db_path = _seed_db(tmp_path)
    strip_fts5_and_vacuum(db_path)  # leaves journal_mode=DELETE
    summarize(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not list(db_path.parent.glob("*-wal"))


# ------------------------------------------------------------- root README

def test_readme_frontmatter_and_sections():
    text = render_root_readme(_summary())
    meta, body = _split(text)
    assert meta["name"] == "llm-memex archive"
    assert "12 conversations" in meta["description"]
    assert "2024-01-05" in meta["description"] and "2026-10-29" in meta["description"]
    assert meta["datetime"] == "2026-10-29T12:00:00+00:00"
    assert meta["generator"].startswith(GENERATOR_PREFIX + " ")
    assert GENERATOR_PREFIX == "llm-memex longecho"
    assert [entry["path"] for entry in meta["contents"]] == [
        "text/index.md", "site/index.html", "data/conversations.db",
        "data/arkiv/conversations.jsonl", "SHA256SUMS",
    ]
    assert all(entry["description"] for entry in meta["contents"])

    order = [body.index(heading) for heading in (
        "## What this is", "## How to open it", "## What was left out",
        "## How to verify",
    )]
    assert order == sorted(order)
    assert "a personal archive" in body
    assert "claude_code, openai" in body
    assert "12 conversations" in body and "340 messages" in body
    assert "sha256sum -c SHA256SUMS" in body
    # plainest first: text editor, then the browser viewer, then sqlite3, then JSONL
    opening = body.split("## How to open it")[1].split("## What was left out")[0]
    positions = [opening.index(marker) for marker in (
        "text/index.md", "site/index.html", "sqlite3", "conversations.jsonl")]
    assert positions == sorted(positions)
    # nothing about the tools before the reader knows what this is
    assert "sqlite3" not in body.split("## How to open it")[0]


def test_readme_names_the_owner():
    text = render_root_readme(_summary(owner="Ada Lovelace"))
    meta, body = _split(text)
    assert meta["name"] == "llm-memex archive of Ada Lovelace"
    assert "Ada Lovelace" in body
    assert "a personal archive" not in body


def test_readme_owner_with_yaml_specials_round_trips():
    meta, _ = _split(render_root_readme(_summary(owner="Smith: the #1 [author]")))
    assert meta["name"] == "llm-memex archive of Smith: the #1 [author]"


def test_readme_empty_archive():
    meta, body = _split(render_root_readme(_summary(
        conversations=0, messages=0, sources=[], first=None, last=None)))
    assert "0 conversations" in meta["description"]
    assert "0 conversations" in body


def test_readme_too_large_sentence():
    large = render_root_readme(_summary(site_has_data=False))
    assert "too large for the browser viewer" in large
    assert "too large" not in render_root_readme(_summary(site_has_data=True))


def test_readme_left_out_reflects_options():
    def left_out(**overrides):
        text = render_root_readme(_summary(**overrides))
        return text.split("## What was left out")[1].split("## How to verify")[0]

    default = left_out()
    assert "Archived conversations (ones the owner had set aside) are left out" in default
    assert "Notes the owner attached to conversations are included" in default
    assert "search index" in default
    assert "Notes the owner attached to conversations are left out" in left_out(
        include_notes=False)
    full = left_out(include_archived=True, include_notes=True)
    assert "Archived conversations are included" in full
    assert "search index" in full


def test_readme_queries_run(tmp_path):
    db_path = _seed_db(tmp_path)
    strip_fts5_and_vacuum(db_path)
    assert len(README_QUERIES) == 2
    rendered = render_root_readme(_summary())
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for description, sql in README_QUERIES:
            rows = conn.execute(sql).fetchall()
            assert rows, description
            assert f'sqlite3 data/conversations.db "{sql}"' in rendered
            assert description in rendered
    # the 20-newest query is ordered, the word query is limited
    assert "LIMIT 20" in README_QUERIES[0][1] and "LIMIT 10" in README_QUERIES[1][1]
    assert "LIKE '%sqlite%'" in README_QUERIES[1][1]


def test_site_readme():
    meta, body = _split(render_site_readme())
    assert meta["name"] == "llm-memex viewer"
    assert meta["description"]
    assert meta["generator"].startswith(GENERATOR_PREFIX + " ")
    assert "index.html" in body and "../data/conversations.db" in body
    assert "double-click" in body.lower()


# --------------------------------------------------------- prepare_destination

def _stamped_export(out_dir, files, *, sums=True, generator=None):
    """Lay out a previous export: *files* maps relative paths to text."""
    generator = generator or f"{GENERATOR_PREFIX} 0.1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(f"---\nname: x\ngenerator: {generator}\n---\nbody\n")
    for rel, content in files.items():
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    if sums:
        write_sha256sums(out_dir)


def test_prepare_creates_missing_and_accepts_empty(tmp_path):
    prepare_destination(tmp_path / "new" / "nested")
    assert (tmp_path / "new" / "nested").is_dir()
    empty = tmp_path / "empty"
    empty.mkdir()
    prepare_destination(empty)
    assert empty.is_dir()


def test_prepare_refuses_a_file(tmp_path):
    target = tmp_path / "file"
    target.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        prepare_destination(target)


def test_prepare_refuses_foreign(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x"}, generator="llm-memex 0.9")  # arkiv bundle
    with pytest.raises(ValueError) as err:
        prepare_destination(out)
    assert str(out) in str(err.value)
    assert "llm-memex 0.9" in str(err.value)
    assert (out / "data" / "x.txt").exists()


@pytest.mark.parametrize("readme", [None, "no frontmatter at all\n",
                                    "---\nname: x\n---\nno generator\n",
                                    "---\n: [unclosed\n---\n"])
def test_prepare_refuses_missing_or_unstamped_readme(tmp_path, readme):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    if readme is not None:
        (out / "README.md").write_text(readme)
    with pytest.raises(ValueError, match=str(out)):
        prepare_destination(out)
    assert (out / "keep.txt").exists()


def test_prepare_refuses_interrupted(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x"}, sums=False)
    with pytest.raises(ValueError) as err:
        prepare_destination(out)
    assert str(out) in str(err.value)
    assert "interrupted" in str(err.value)
    assert "emove the folder" in str(err.value)
    assert (out / "data" / "x.txt").exists()


def test_prepare_deletes_listed_keeps_unlisted(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"text/2025/old.md": "old", "data/conversations.db": "db",
                          "data/assets/a/b.png": "png"})
    (out / "notes.txt").write_text("hand added")
    (out / "data" / "mine.txt").write_text("also mine")
    prepare_destination(out)
    assert not (out / "text").exists()  # emptied directories are removed
    assert not (out / "data" / "assets").exists()
    assert not (out / "data" / "conversations.db").exists()
    assert not (out / "SHA256SUMS").exists()
    assert not (out / "README.md").exists()
    assert (out / "notes.txt").read_text() == "hand added"
    assert (out / "data" / "mine.txt").exists()  # its directory is not empty
    assert out.is_dir()


def test_prepare_keeps_unrelated_empty_directories(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x"})
    (out / "mine").mkdir()
    prepare_destination(out)
    assert (out / "mine").is_dir()


def test_prepare_ignores_escaping_paths(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x"})
    digest = "0" * 64
    with open(out / "SHA256SUMS", "a") as fh:
        fh.write(f"{digest}  ../outside.txt\n")
        fh.write(f"{digest}  data/../../outside.txt\n")
        fh.write(f"{digest}  {outside}\n")
        fh.write(f"{digest}  /etc/hostname\n")
        fh.write("not a checksum line\n")
        fh.write(f"{digest[:-1]}  data/short.txt\n")
    (out / "data" / "short.txt").write_text("keep")
    prepare_destination(out)
    assert outside.read_text() == "precious"
    assert (out / "data" / "short.txt").exists()  # malformed line skipped
    assert not (out / "data" / "x.txt").exists()  # the valid line still applies
    assert not (out / "SHA256SUMS").exists()


def test_prepare_never_follows_symlinked_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("precious")
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x"})
    (out / "link").symlink_to(outside, target_is_directory=True)
    with open(out / "SHA256SUMS", "a") as fh:
        fh.write(f"{'0' * 64}  link/secret.txt\n")
    prepare_destination(out)
    assert (outside / "secret.txt").read_text() == "precious"
    assert (out / "link").is_symlink()


def test_prepare_never_deletes_a_listed_directory(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x"})
    (out / "keepdir").mkdir()
    (out / "keepdir" / "f.txt").write_text("f")
    with open(out / "SHA256SUMS", "a") as fh:
        fh.write(f"{'0' * 64}  keepdir\n")
    prepare_destination(out)
    assert (out / "keepdir" / "f.txt").exists()


def test_prepare_handles_missing_listed_file(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x", "data/y.txt": "y"})
    (out / "data" / "x.txt").unlink()  # a rerun after a partial cleanup
    prepare_destination(out)
    assert not (out / "data").exists()


def test_prepare_is_resumable_after_partial_cleanup(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x"})
    (out / "data" / "x.txt").unlink()
    (out / "data").rmdir()
    prepare_destination(out)  # README and SHA256SUMS still present: not refused
    assert not (out / "SHA256SUMS").exists()


# ------------------------------------------------------------ write_sha256sums

def _expected_sums(out_dir, skip=("SHA256SUMS",)):
    lines = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name not in skip):
        rel = path.relative_to(out_dir).as_posix()
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}\n")
    return "".join(lines)


def test_sha256sums_format_and_coverage(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"b/two.txt": "two", "a/one.txt": "one", "zero.bin": ""},
                    sums=False)
    (out / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024 + 17))  # > one 1 MB read
    write_sha256sums(out)
    text = (out / "SHA256SUMS").read_text()
    assert text == _expected_sums(out)
    paths = [line[66:] for line in text.splitlines()]
    assert paths == sorted(paths)
    assert "SHA256SUMS" not in paths
    # idempotent: a second run does not list SHA256SUMS or change the result
    write_sha256sums(out)
    assert (out / "SHA256SUMS").read_text() == text


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="sha256sum not installed")
def test_sha256sums_verifies(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"text/a.md": "a", "data/b.db": "b", "odd name \\ x.txt": "c"},
                    sums=False)
    (out / "new\nline.txt").write_text("nl")
    write_sha256sums(out)
    result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=out,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    (out / "text" / "a.md").write_text("tampered")
    result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=out,
                            capture_output=True, text=True)
    assert result.returncode != 0


def test_sha256sums_skips_symlinks(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("precious")
    other = tmp_path / "other.txt"
    other.write_text("other")
    out = tmp_path / "out"
    _stamped_export(out, {"data/x.txt": "x"}, sums=False)
    (out / "linkdir").symlink_to(outside, target_is_directory=True)
    (out / "linkfile").symlink_to(other)
    write_sha256sums(out)
    paths = [line[66:] for line in (out / "SHA256SUMS").read_text().splitlines()]
    assert paths == ["README.md", "data/x.txt"]


def test_escaped_names_round_trip_through_prepare(tmp_path):
    out = tmp_path / "out"
    _stamped_export(out, {"data/back\\slash.txt": "b"}, sums=False)
    (out / "data" / "new\nline.txt").write_text("n")
    write_sha256sums(out)
    prepare_destination(out)
    assert not (out / "data").exists()
