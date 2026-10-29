"""A published snapshot must include WAL data and only reachable media."""
from datetime import datetime
import sqlite3
import pytest

from llm_memex.db import Database
from llm_memex.exporters.html import export
from llm_memex.models import Conversation, Message, media_block


def conversation(identifier, url, *, archived=False):
    now = datetime(2026, 10, 28)
    conv = Conversation(identifier, now, now, archived_at=now if archived else None)
    conv.add_message(Message(id="m", role="user", content=[media_block("image/png", url=url)]))
    return conv


def test_export_includes_committed_wal_without_closing_writer(tmp_path):
    with Database(str(tmp_path / "source")) as db:
        db.conn.execute("PRAGMA wal_autocheckpoint=0")
        db.save_conversation(conversation("live", "assets/live.png"))
        output = tmp_path / "output"
        export([], str(output), db_path=str(tmp_path / "source/conversations.db"), compress_db=False)
        with sqlite3.connect(output / "conversations.db") as copy:
            assert copy.execute("SELECT id FROM conversations").fetchall() == [("live",)]
        assert db.load_conversation("live") is not None


def test_assets_follow_retained_content_not_sibling_directory(tmp_path):
    source = tmp_path / "source"
    with Database(str(source)) as db:
        db.save_conversation(conversation("live", "assets/nested/live.png"))
        db.save_conversation(conversation("private", "assets/private.png", archived=True))
    assets = source / "assets"
    (assets / "nested").mkdir(parents=True)
    (assets / "nested/live.png").write_bytes(b"public")
    (assets / "private.png").write_bytes(b"private")
    (assets / "orphan.png").write_bytes(b"orphan")
    output = tmp_path / "output"
    (output / "assets").mkdir(parents=True)
    (output / "assets/stale.png").write_bytes(b"previous export")
    export([], str(output), db_path=str(source / "conversations.db"), compress_db=False)
    files = {p.relative_to(output / "assets").as_posix() for p in (output / "assets").rglob("*") if p.is_file()}
    assert files == {"nested/live.png"}
    assert (assets / "private.png").exists()


def test_asset_symlink_cannot_export_outside_source(tmp_path):
    source = tmp_path / "source"
    with Database(str(source)) as db:
        db.save_conversation(conversation("live", "assets/leak.png"))
    (source / "assets").mkdir()
    secret = tmp_path / "secret"
    secret.write_bytes(b"private")
    (source / "assets/leak.png").symlink_to(secret)
    output = tmp_path / "output"
    export([], str(output), db_path=str(source / "conversations.db"), compress_db=False)
    assert not (output / "assets/leak.png").exists()


def test_switching_to_uncompressed_export_removes_stale_gzip(tmp_path):
    source = tmp_path / "source"
    with Database(str(source)) as db:
        db.save_conversation(conversation("live", "assets/live.png"))
    output = tmp_path / "output"
    export([], str(output), db_path=str(source / "conversations.db"))
    export([], str(output), db_path=str(source / "conversations.db"), compress_db=False)
    assert (output / "conversations.db").exists()
    assert not (output / "conversations.db.gz").exists()


def test_export_refuses_source_directory_before_opening_connections(tmp_path, monkeypatch):
    source = tmp_path / "source"
    with Database(str(source)):
        pass
    before = (source / "conversations.db").read_bytes()

    def unexpected_connection(*args, **kwargs):
        pytest.fail("source/output alias must be rejected before opening SQLite")

    monkeypatch.setattr(sqlite3, "connect", unexpected_connection)
    with pytest.raises(ValueError, match="source"):
        export([], str(source), db_path=str(source / "conversations.db"))
    assert (source / "conversations.db").read_bytes() == before
    assert not (source / "index.html").exists()


def test_custom_database_name_cannot_alias_source_assets(tmp_path):
    source = tmp_path / "source"
    with Database(str(source)):
        pass
    (source / "conversations.db").rename(source / "custom.db")
    (source / "assets").mkdir()
    (source / "assets/original.png").write_bytes(b"irreplaceable")
    with pytest.raises(ValueError, match="source"):
        export([], str(source), db_path=str(source / "custom.db"))
    assert (source / "assets/original.png").read_bytes() == b"irreplaceable"
