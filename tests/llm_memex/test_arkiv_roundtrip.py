"""Round-trip tests: Database -> arkiv bundle -> Database.

These are the durability contract tests for the arkiv format: if a seeded DB
round-trips losslessly through an arkiv bundle (directory, zip, or tar.gz),
the archive is provably durable through that interchange format.

Note: the arkiv export is intentionally text-centric — it does NOT round-trip
every llm-memex field (enrichments, provenance, starred/pinned flags,
conversation-level notes, tool_use blocks). These tests assert what IS
preserved; see ``llm_memex/importers/arkiv.py`` docstring for the full list.
"""
from __future__ import annotations

import json
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from llm_memex.db import Database
from llm_memex.exporters.arkiv_export import export as arkiv_export
from llm_memex.importers.arkiv import detect as arkiv_detect
from llm_memex.importers.arkiv import import_path as arkiv_import
from llm_memex.models import Conversation, Message, text_block


def _seed_conversation() -> Conversation:
    now = datetime(2026, 4, 22, 9, 0, 0)
    conv = Conversation(
        id="conv-1",
        created_at=now,
        updated_at=now,
        title="Bernoulli sets and you",
        source="openai",
        model="gpt-4o",
        tags=["math", "draft"],
    )
    conv.add_message(Message(
        id="m-1", role="user",
        content=[text_block("can you define a bernoulli set?")],
        created_at=now,
    ))
    conv.add_message(Message(
        id="m-2", role="assistant",
        content=[text_block("A Bernoulli set is a random approximate set...")],
        parent_id="m-1",
        created_at=datetime(2026, 4, 22, 9, 0, 1),
    ))
    return conv


def _seed_db(db_dir: Path, *, with_notes: bool = False) -> Database:
    db = Database(str(db_dir))
    conv = _seed_conversation()
    db.save_conversation(conv)
    if with_notes:
        db.add_note(
            conversation_id="conv-1",
            message_id="m-2",
            text="Good framing. Revisit for the paper.",
        )
    return db


# ── exporter output shape ──────────────────────────────────────

class TestExportShape:
    def test_export_directory_has_three_files(self, tmp_path):
        out = tmp_path / "archive"
        arkiv_export([_seed_conversation()], str(out))
        assert (out / "conversations.jsonl").is_file()
        assert (out / "schema.yaml").is_file()
        assert (out / "README.md").is_file()

    def test_export_zip_has_three_entries(self, tmp_path):
        out = tmp_path / "archive.zip"
        arkiv_export([_seed_conversation()], str(out))
        assert out.is_file()
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert {"conversations.jsonl", "schema.yaml", "README.md"} <= names

    def test_export_tar_gz_has_three_entries(self, tmp_path):
        out = tmp_path / "archive.tar.gz"
        arkiv_export([_seed_conversation()], str(out))
        assert out.is_file()
        with tarfile.open(out, "r:gz") as tf:
            names = {m.name for m in tf.getmembers()}
        assert {"conversations.jsonl", "schema.yaml", "README.md"} <= names

    def test_zip_is_deflated(self, tmp_path):
        """Our zip uses DEFLATE so it actually compresses text content."""
        out = tmp_path / "archive.zip"
        arkiv_export([_seed_conversation()], str(out))
        with zipfile.ZipFile(out) as zf:
            for info in zf.infolist():
                # Any method other than STORED; DEFLATE is 8.
                assert info.compress_type != zipfile.ZIP_STORED

    def test_extension_insensitive(self, tmp_path):
        """Both .tgz and .tar.gz produce tarballs."""
        out = tmp_path / "archive.tgz"
        arkiv_export([_seed_conversation()], str(out))
        with tarfile.open(out, "r:gz") as tf:
            names = {m.name for m in tf.getmembers()}
        assert "conversations.jsonl" in names


# ── round-trip: directory layout ───────────────────────────────

class TestDirectoryRoundTrip:
    def test_detect_directory(self, tmp_path):
        out = tmp_path / "archive"
        arkiv_export([_seed_conversation()], str(out))
        assert arkiv_detect(str(out))

    def test_conversation_fields_preserved(self, tmp_path):
        out = tmp_path / "archive"
        arkiv_export([_seed_conversation()], str(out))
        convs = arkiv_import(str(out))
        assert len(convs) == 1
        c = convs[0]
        assert c.id == "conv-1"
        assert c.title == "Bernoulli sets and you"
        assert c.source == "openai"
        assert c.model == "gpt-4o"
        assert set(c.tags) == {"math", "draft"}

    def test_messages_preserved_in_order(self, tmp_path):
        out = tmp_path / "archive"
        arkiv_export([_seed_conversation()], str(out))
        convs = arkiv_import(str(out))
        c = convs[0]
        # Messages sorted by timestamp and re-linked
        msg_list = sorted(c.messages.values(), key=lambda m: m.created_at)
        assert len(msg_list) == 2
        assert msg_list[0].role == "user"
        assert msg_list[0].content[0]["text"].startswith("can you define")
        assert msg_list[1].role == "assistant"
        assert msg_list[1].content[0]["text"].startswith("A Bernoulli set")

    def test_full_roundtrip_to_fresh_db(self, tmp_path):
        """DB → arkiv → fresh DB preserves the conversation + messages."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        with _seed_db(src_dir) as src_db:
            convs = list(src_db.query_conversations()["items"])
        assert len(convs) == 1

        arkiv_dir = tmp_path / "arkiv"
        # Export from the source db via the canonical pipeline
        arkiv_export(
            [_seed_conversation()],
            str(arkiv_dir),
        )

        # Import into a fresh DB
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        reimported = arkiv_import(str(arkiv_dir))
        with Database(str(dst_dir)) as dst_db:
            for conv in reimported:
                dst_db.save_conversation(conv)
            items = dst_db.query_conversations()["items"]
            assert len(items) == 1
            row = items[0]
            assert row["id"] == "conv-1"
            assert row["title"] == "Bernoulli sets and you"
            # Both messages round-tripped
            msg_rows = dst_db.execute_sql(
                "SELECT id, role FROM messages WHERE conversation_id = ?",
                ("conv-1",),
            )
            assert len(msg_rows) == 2


# ── round-trip: zip ────────────────────────────────────────────

class TestZipRoundTrip:
    def test_detect_zip(self, tmp_path):
        out = tmp_path / "archive.zip"
        arkiv_export([_seed_conversation()], str(out))
        assert arkiv_detect(str(out))

    def test_zip_roundtrip_preserves_conversation(self, tmp_path):
        out = tmp_path / "archive.zip"
        arkiv_export([_seed_conversation()], str(out))
        convs = arkiv_import(str(out))
        assert len(convs) == 1
        c = convs[0]
        assert c.id == "conv-1"
        assert c.title == "Bernoulli sets and you"
        msg_list = sorted(c.messages.values(), key=lambda m: m.created_at)
        assert len(msg_list) == 2
        assert msg_list[0].role == "user"

    def test_zip_not_detected_when_missing_jsonl(self, tmp_path):
        out = tmp_path / "empty.zip"
        with zipfile.ZipFile(out, "w") as zf:
            zf.writestr("README.md", "not arkiv")
        assert not arkiv_detect(str(out))


# ── round-trip: tar.gz ─────────────────────────────────────────

class TestTarGzRoundTrip:
    def test_detect_tar_gz(self, tmp_path):
        out = tmp_path / "archive.tar.gz"
        arkiv_export([_seed_conversation()], str(out))
        assert arkiv_detect(str(out))

    def test_tar_gz_roundtrip(self, tmp_path):
        out = tmp_path / "archive.tar.gz"
        arkiv_export([_seed_conversation()], str(out))
        convs = arkiv_import(str(out))
        assert len(convs) == 1
        c = convs[0]
        assert c.title == "Bernoulli sets and you"


# ── negative cases ─────────────────────────────────────────────

class TestBareJsonlInput:
    """Single-file arkiv inputs — what the browser SPA emits for round-trip."""

    def _write_bundle_and_extract_jsonl(self, tmp_path):
        bundle = tmp_path / "bundle"
        arkiv_export([_seed_conversation()], str(bundle))
        return (bundle / "conversations.jsonl").read_bytes()

    def test_detect_bare_jsonl(self, tmp_path):
        jsonl = self._write_bundle_and_extract_jsonl(tmp_path)
        target = tmp_path / "annotations.jsonl"
        target.write_bytes(jsonl)
        assert arkiv_detect(str(target))

    def test_roundtrip_bare_jsonl(self, tmp_path):
        jsonl = self._write_bundle_and_extract_jsonl(tmp_path)
        target = tmp_path / "annotations.jsonl"
        target.write_bytes(jsonl)
        convs = arkiv_import(str(target))
        assert len(convs) == 1
        assert convs[0].title == "Bernoulli sets and you"

    def test_detect_bare_jsonl_gz(self, tmp_path):
        import gzip as _gz
        jsonl = self._write_bundle_and_extract_jsonl(tmp_path)
        target = tmp_path / "annotations.jsonl.gz"
        target.write_bytes(_gz.compress(jsonl))
        assert arkiv_detect(str(target))

    def test_roundtrip_bare_jsonl_gz(self, tmp_path):
        """The exact path the SPA uses: user downloads an .jsonl.gz from the
        exported HTML, then imports it back into the primary DB."""
        import gzip as _gz
        jsonl = self._write_bundle_and_extract_jsonl(tmp_path)
        target = tmp_path / "annotations.jsonl.gz"
        target.write_bytes(_gz.compress(jsonl))
        convs = arkiv_import(str(target))
        assert len(convs) == 1
        c = convs[0]
        assert c.title == "Bernoulli sets and you"
        assert len(c.messages) == 2


class TestDetectRejects:
    def test_nonexistent_path(self, tmp_path):
        assert not arkiv_detect(str(tmp_path / "nope"))

    def test_directory_without_jsonl(self, tmp_path):
        d = tmp_path / "d"
        d.mkdir()
        assert not arkiv_detect(str(d))

    def test_random_json_file(self, tmp_path):
        f = tmp_path / "random.zip"
        with zipfile.ZipFile(f, "w") as zf:
            zf.writestr("conversations.jsonl", json.dumps({"mimetype": "text/plain"}))
        # No conversation_id/role metadata; should NOT be detected as ours.
        assert not arkiv_detect(str(f))


# ── notes round-trip ───────────────────────────────────────────

class TestMergeMode:
    """--merge preserves existing conversation state and adds only new
    messages + tags. This is the round-trip mode for HTML-SPA annotations."""

    def test_merge_preserves_enrichments_on_existing_conv(self, tmp_path):
        """Merging new messages must not wipe an enrichment the primary DB
        already has."""
        db_dir = tmp_path / "primary"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            conv = _seed_conversation()
            db.save_conversation(conv)
            # Mark the conversation with an enrichment (summary)
            db.save_enrichments(conv.id, [
                {"type": "summary", "value": "precious insight", "source": "user"}
            ])

            # Merge in "the same conversation plus one new message"
            new_conv = _seed_conversation()
            new_conv.add_message(Message(
                id="m-3", role="user",
                content=[text_block("a later question, added after the fact")],
                parent_id="m-2",
                created_at=datetime(2026, 4, 22, 10, 0, 0),
            ))
            stats = db.merge_conversation(new_conv)
            assert stats["added_messages"] == 1  # m-3 is new; m-1, m-2 already there
            assert stats["created_conversation"] == 0

            # Enrichment survives
            rows = db.execute_sql(
                "SELECT type, value, source FROM enrichments WHERE conversation_id=?",
                (new_conv.id,),
            )
            assert any(r["type"] == "summary" and r["value"] == "precious insight" for r in rows)

            # Message count reflects the new total
            msgs = db.execute_sql(
                "SELECT COUNT(*) as n FROM messages WHERE conversation_id=?",
                (new_conv.id,),
            )
            assert msgs[0]["n"] == 3

    def test_merge_creates_new_conv_when_not_present(self, tmp_path):
        db_dir = tmp_path / "primary"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            stats = db.merge_conversation(_seed_conversation())
            assert stats["created_conversation"] == 1
            assert stats["added_messages"] == 2

    def test_merge_is_idempotent(self, tmp_path):
        """Re-merging the same conversation should yield zero new messages."""
        db_dir = tmp_path / "primary"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.merge_conversation(_seed_conversation())
            stats2 = db.merge_conversation(_seed_conversation())
            assert stats2["created_conversation"] == 0
            assert stats2["added_messages"] == 0
            assert stats2["added_tags"] == 0

    def test_merge_notes_inserts_new_preserves_existing(self, tmp_path):
        db_dir = tmp_path / "primary"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_seed_conversation())
            db.add_note(
                conversation_id="conv-1", message_id="m-2",
                text="original primary-DB note",
            )
            orig_notes = db.execute_sql(
                "SELECT id, text FROM notes WHERE conversation_id='conv-1'"
            )
            assert len(orig_notes) == 1
            orig_id = orig_notes[0]["id"]

            # New batch: one new note + one that coincidentally has the same
            # id as the existing one (INSERT OR IGNORE should keep the original
            # text rather than overwriting it).
            added = db.merge_notes([
                {"id": "new-1", "conversation_id": "conv-1", "message_id": "m-1",
                 "text": "freshly-imported annotation"},
                {"id": orig_id, "conversation_id": "conv-1", "message_id": "m-2",
                 "text": "different text, should be ignored"},
            ])
            assert added == 1
            all_notes = {
                r["id"]: r["text"] for r in db.execute_sql(
                    "SELECT id, text FROM notes WHERE conversation_id='conv-1'"
                )
            }
            assert all_notes[orig_id] == "original primary-DB note"
            assert all_notes["new-1"] == "freshly-imported annotation"


class TestNotesRoundTrip:
    def test_message_level_notes_preserved_in_metadata(self, tmp_path):
        """Message-level notes ride in arkiv metadata and land in
        msg.metadata['_arkiv_notes'] on import so the CLI can materialize
        them after save_conversation."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        db = _seed_db(src_dir, with_notes=True)
        try:
            convs = db.query_conversations()["items"]
            full_conv = db.load_conversation(convs[0]["id"])
            arkiv_out = tmp_path / "archive"
            # Pass db while it's still open so the exporter can read notes.
            arkiv_export([full_conv], str(arkiv_out), db=db)
        finally:
            db.close()

        reimported = arkiv_import(str(arkiv_out))
        assert len(reimported) == 1
        c = reimported[0]
        # Find the message that had a note
        annotated = [m for m in c.messages.values()
                     if m.metadata.get("_arkiv_notes")]
        assert len(annotated) == 1
        notes = annotated[0].metadata["_arkiv_notes"]
        assert any("Revisit for the paper" in (n.get("text") or "") for n in notes)


# ── IMP-9: malformed tar member must not abort detection ───────

class TestTarGzMalformedDetect:
    def _make_malformed_tar_gz(self, path: Path) -> None:
        """Craft a .tar.gz whose conversations.jsonl member is a symlink with
        an unresolvable linkname. tarfile.extractfile() raises KeyError (not a
        TarError) for such a member; detect() must catch it and return False
        rather than letting the exception abort auto-detection."""
        with tarfile.open(path, "w:gz") as tf:
            info = tarfile.TarInfo(name="conversations.jsonl")
            info.type = tarfile.SYMTYPE
            info.linkname = "does/not/exist/anywhere.jsonl"
            info.size = 0
            tf.addfile(info)

    def test_detect_returns_false_not_raises_on_symlink_member(self, tmp_path):
        target = tmp_path / "evil.tar.gz"
        self._make_malformed_tar_gz(target)
        # Must NOT raise (KeyError from extractfile would abort the whole
        # auto-detection sweep); must return False instead.
        assert arkiv_detect(str(target)) is False


# ── EXP-2: schema example must be deterministic + faithful ─────

class TestSchemaExampleDeterminism:
    def _records_with_capped_list_key(self):
        """Build >20 records whose metadata carries a list-valued key ('tags')
        with >20 distinct values, forcing the >20-unique 'example' branch."""
        from llm_memex.exporters.arkiv_export import _build_records

        convs = []
        for i in range(25):
            now = datetime(2026, 4, 22, 9, 0, 0)
            conv = Conversation(
                id=f"conv-{i}",
                created_at=now,
                updated_at=now,
                title=f"title-{i}",
                source="openai",
                model="gpt-4o",
                tags=[f"tag-{i}-a", f"tag-{i}-b"],
            )
            conv.add_message(Message(
                id=f"m-{i}", role="user",
                content=[text_block(f"hello {i}")],
                created_at=now,
            ))
            convs.append(conv)
        return _build_records(convs)

    def test_schema_example_is_deterministic_across_runs(self):
        from llm_memex.exporters.arkiv_export import _compute_schema

        records = self._records_with_capped_list_key()
        s1 = _compute_schema(records)
        s2 = _compute_schema(records)
        # 'tags' has >20 unique list values -> uses 'example', not 'values'.
        assert "example" in s1["tags"]
        assert "values" not in s1["tags"]
        # Deterministic across two independent computations.
        assert s1["tags"]["example"] == s2["tags"]["example"]

    def test_schema_example_is_faithful_not_tuple_repr(self):
        from llm_memex.exporters.arkiv_export import _compute_schema

        records = self._records_with_capped_list_key()
        schema = _compute_schema(records)
        example = schema["tags"]["example"]
        # A faithful JSON-serializable value (the first real tags list),
        # NOT Python tuple-repr garbage like "('tag-0-a', 'tag-0-b')".
        assert example == ["tag-0-a", "tag-0-b"]
        json.dumps(example)  # must be JSON-serializable, raises otherwise


# ── EXP-3: tar.gz export must be byte-reproducible ─────────────

class TestTarGzReproducible:
    def test_two_tar_gz_exports_are_byte_identical(self, tmp_path):
        out1 = tmp_path / "a.tar.gz"
        out2 = tmp_path / "b.tar.gz"
        arkiv_export([_seed_conversation()], str(out1))
        arkiv_export([_seed_conversation()], str(out2))
        assert out1.read_bytes() == out2.read_bytes()

    def test_reproducible_tar_gz_still_imports(self, tmp_path):
        out = tmp_path / "repro.tar.gz"
        arkiv_export([_seed_conversation()], str(out))
        convs = arkiv_import(str(out))
        assert len(convs) == 1
        assert convs[0].title == "Bernoulli sets and you"
