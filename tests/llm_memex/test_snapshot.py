"""The shared publication snapshot used by the HTML and longecho exporters."""
import gzip
import sqlite3
from datetime import datetime

import pytest

from llm_memex.db import Database
from llm_memex.exporters._snapshot import (
    check_destination,
    gzip_file,
    take_snapshot,
)
from llm_memex.models import Conversation, Message


def _conversation(identifier, *, archived=False):
    now = datetime(2026, 10, 29)
    conv = Conversation(identifier, now, now, archived_at=now if archived else None)
    conv.add_message(Message(id="m", role="user", content=[{"type": "text", "text": "hi"}]))
    return conv


def _seed(tmp_path):
    source = tmp_path / "source"
    with Database(str(source)) as db:
        db.save_conversation(_conversation("live"))
        db.save_conversation(_conversation("gone", archived=True))
    return source / "conversations.db"


def test_take_snapshot_drops_archived_and_fts(tmp_path):
    db_path = _seed(tmp_path)
    dest = tmp_path / "snap.db"
    take_snapshot(db_path, dest, include_notes=True, include_archived=False)
    with sqlite3.connect(dest) as conn:
        assert conn.execute("SELECT id FROM conversations").fetchall() == [("live",)]
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master")]
        assert not [n for n in names if "_fts" in n]
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_take_snapshot_no_notes_drops_table(tmp_path):
    db_path = _seed(tmp_path)
    kept = tmp_path / "kept.db"
    dropped = tmp_path / "dropped.db"
    take_snapshot(db_path, kept, include_notes=True, include_archived=False)
    take_snapshot(db_path, dropped, include_notes=False, include_archived=False)
    with sqlite3.connect(kept) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='notes'").fetchone()
    with sqlite3.connect(dropped) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='notes'").fetchone() is None


def test_check_destination_refuses_source_dir_ancestor_and_assets(tmp_path):
    db_path = _seed(tmp_path)
    source = db_path.parent
    (source / "assets").mkdir()
    for bad in (source, tmp_path, source / "assets", source / "assets" / "deep"):
        with pytest.raises(ValueError, match="source"):
            check_destination(db_path, bad)
    check_destination(db_path, tmp_path / "sibling")


def test_gzip_file_keep_source(tmp_path):
    src = tmp_path / "data.bin"
    src.write_bytes(b"payload" * 1000)
    dst = tmp_path / "data.bin.gz"
    gzip_file(src, dst, keep_source=True)
    assert src.exists()
    assert gzip.decompress(dst.read_bytes()) == src.read_bytes()


def test_gzip_file_deletes_source_by_default(tmp_path):
    src = tmp_path / "data.bin"
    src.write_bytes(b"payload")
    gzip_file(src, tmp_path / "data.bin.gz")
    assert not src.exists()


def test_check_destination_resolves_symlinked_source_assets(tmp_path):
    """A source assets/ that leads into the destination is judged by its target."""
    db_path = _seed(tmp_path)
    out = tmp_path / "out"
    (out / "data" / "assets").mkdir(parents=True)
    (db_path.parent / "assets").symlink_to(out / "data" / "assets")
    with pytest.raises(ValueError, match="must not overwrite the source"):
        check_destination(db_path, out)
