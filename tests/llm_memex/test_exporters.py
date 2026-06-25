"""Tests for memex exporters: Markdown, JSON."""
import json
from datetime import datetime

from llm_memex.models import Conversation, Message, text_block, media_block
from llm_memex.exporters.markdown import export as md_export
from llm_memex.exporters.json_export import export as json_export


def _make_conv(id="c1", title="Test Chat"):
    now = datetime(2024, 6, 15)
    conv = Conversation(id=id, created_at=now, updated_at=now, title=title, source="test")
    conv.add_message(Message(id="m1", role="user", content=[text_block("hello")]))
    conv.add_message(Message(id="m2", role="assistant", content=[text_block("hi there")], parent_id="m1"))
    return conv


# ---------- Markdown ----------

class TestMarkdownExporter:
    def test_single_conversation(self, tmp_path):
        out = tmp_path / "out.md"
        md_export([_make_conv()], str(out))
        content = out.read_text()
        assert "# Test Chat" in content
        assert "hello" in content
        assert "hi there" in content

    def test_multiple_conversations(self, tmp_path):
        out = tmp_path / "out.md"
        md_export([_make_conv("c1", "First"), _make_conv("c2", "Second")], str(out))
        content = out.read_text()
        assert "# First" in content
        assert "# Second" in content

    def test_source_included(self, tmp_path):
        out = tmp_path / "out.md"
        md_export([_make_conv()], str(out))
        content = out.read_text()
        assert "*Source: test*" in content

    def test_roles_in_output(self, tmp_path):
        out = tmp_path / "out.md"
        md_export([_make_conv()], str(out))
        content = out.read_text()
        assert "**user**:" in content
        assert "**assistant**:" in content

    def test_no_title_uses_id(self, tmp_path):
        conv = _make_conv()
        conv.title = None
        out = tmp_path / "out.md"
        md_export([conv], str(out))
        content = out.read_text()
        assert "# c1" in content

    def test_empty_list(self, tmp_path):
        out = tmp_path / "out.md"
        md_export([], str(out))
        assert out.read_text() == ""

    def test_branching_conversation(self, tmp_path):
        now = datetime(2024, 6, 15)
        conv = Conversation(id="c1", created_at=now, updated_at=now, title="Branch")
        conv.add_message(Message(id="m1", role="user", content=[text_block("start")]))
        conv.add_message(Message(id="m2a", role="assistant", content=[text_block("reply A")], parent_id="m1"))
        conv.add_message(Message(id="m2b", role="assistant", content=[text_block("reply B")], parent_id="m1"))
        out = tmp_path / "out.md"
        md_export([conv], str(out))
        content = out.read_text()
        assert "reply A" in content
        assert "reply B" in content
        # Should have separator between paths
        assert content.count("---") >= 2


# ---------- JSON ----------

class TestJSONExporter:
    def test_single_conversation(self, tmp_path):
        out = tmp_path / "out.json"
        json_export([_make_conv()], str(out))
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "c1"
        assert data[0]["title"] == "Test Chat"
        assert len(data[0]["messages"]) == 2

    def test_messages_structure(self, tmp_path):
        out = tmp_path / "out.json"
        json_export([_make_conv()], str(out))
        data = json.loads(out.read_text())
        msg = data[0]["messages"][0]
        assert "id" in msg
        assert "role" in msg
        assert "content" in msg
        assert "parent_id" in msg

    def test_content_blocks_preserved(self, tmp_path):
        out = tmp_path / "out.json"
        json_export([_make_conv()], str(out))
        data = json.loads(out.read_text())
        msg = data[0]["messages"][0]
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "hello"

    def test_metadata_fields(self, tmp_path):
        out = tmp_path / "out.json"
        json_export([_make_conv()], str(out))
        data = json.loads(out.read_text())
        assert data[0]["source"] == "test"
        assert "created_at" in data[0]
        assert "updated_at" in data[0]
        assert "tags" in data[0]

    def test_multiple_conversations(self, tmp_path):
        out = tmp_path / "out.json"
        json_export([_make_conv("c1"), _make_conv("c2")], str(out))
        data = json.loads(out.read_text())
        assert len(data) == 2

    def test_empty_list(self, tmp_path):
        out = tmp_path / "out.json"
        json_export([], str(out))
        data = json.loads(out.read_text())
        assert data == []

    def test_roundtrip_content(self, tmp_path):
        """Content blocks should be valid JSON after export."""
        now = datetime(2024, 6, 15)
        conv = Conversation(id="c1", created_at=now, updated_at=now, title="Multi")
        conv.add_message(Message(id="m1", role="user", content=[
            text_block("hello"),
            media_block("image/png", url="http://example.com/img.png"),
        ]))
        out = tmp_path / "out.json"
        json_export([conv], str(out))
        data = json.loads(out.read_text())
        msg = data[0]["messages"][0]
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "media"
        assert msg["content"][1]["url"] == "http://example.com/img.png"


class TestArkivReadme:
    def test_readme_links_correct_arkiv_org(self):
        """LLM-8: the generated bundle README must point at the real arkiv
        repository (queelius/arkiv), not a nonexistent org."""
        from llm_memex.exporters.arkiv_export import _readme_bytes

        readme = _readme_bytes(num_conversations=3).decode("utf-8")
        assert "https://github.com/queelius/arkiv" in readme
        assert "alonzo-church" not in readme


class TestExportEncoding:
    """R2: text exporters must write UTF-8 regardless of the locale default.

    On a non-UTF-8 default locale, ``open(path, "w")`` (with no explicit
    ``encoding``) uses the locale codec; exporting non-ASCII content then
    raises UnicodeEncodeError and aborts. We simulate that locale by wrapping
    ``builtins.open`` so any text-mode call left at ``encoding=None`` falls
    back to ASCII (as a non-UTF-8 locale would). With the fix the exporters
    pass ``encoding='utf-8'`` explicitly, so the wrapper never substitutes.
    """

    @staticmethod
    def _force_ascii_locale_open(monkeypatch):
        import builtins

        real_open = builtins.open

        def patched_open(file, mode="r", *args, **kwargs):
            if "b" not in mode and kwargs.get("encoding") is None and len(args) < 1:
                kwargs["encoding"] = "ascii"
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", patched_open)

    def _make_unicode_conv(self):
        now = datetime(2024, 6, 15)
        conv = Conversation(
            id="c1", created_at=now, updated_at=now,
            title="Cafe resume naive", source="test",
        )
        # Non-ASCII content: accented Latin, em dash, CJK, emoji.
        unicode_text = "Cafe é — 你好 \U0001f600"
        conv.add_message(Message(id="m1", role="user", content=[text_block(unicode_text)]))
        return conv, unicode_text

    def test_markdown_exports_non_ascii_under_non_utf8_locale(self, tmp_path, monkeypatch):
        conv, unicode_text = self._make_unicode_conv()
        out = tmp_path / "out.md"
        self._force_ascii_locale_open(monkeypatch)
        md_export([conv], str(out))
        content = out.read_bytes().decode("utf-8")
        assert unicode_text in content

    def test_json_opens_output_with_explicit_utf8(self, tmp_path, monkeypatch):
        """The JSON exporter does not crash today only because json.dump
        defaults to ensure_ascii=True (ASCII-safe bytes). The open() call is
        still locale-dependent, so for robustness it must request UTF-8
        explicitly. Assert the encoding argument rather than content, since
        content alone cannot distinguish the bug from the fix here."""
        import builtins

        conv, _ = self._make_unicode_conv()
        out = tmp_path / "out.json"
        real_open = builtins.open
        captured = {}

        def spy_open(file, mode="r", *args, **kwargs):
            if str(file) == str(out) and "w" in mode:
                captured["encoding"] = kwargs.get("encoding")
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy_open)
        json_export([conv], str(out))
        assert captured.get("encoding") == "utf-8"
        data = json.loads(out.read_bytes().decode("utf-8"))
        assert data[0]["id"] == "c1"


class TestHtmlExportEncoding:
    def test_index_html_written_as_utf8(self, tmp_path, monkeypatch):
        """LLM-9: index.html must be written with encoding='utf-8'. The
        template carries non-ASCII characters, so relying on the locale
        default breaks export under a non-UTF-8 locale."""
        import pathlib

        from llm_memex.exporters import html as html_exporter

        captured = {}
        orig_write_text = pathlib.Path.write_text

        def spy_write_text(self, data, encoding=None, *args, **kwargs):
            if self.name == "index.html":
                captured["encoding"] = encoding
            return orig_write_text(self, data, encoding=encoding, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "write_text", spy_write_text)
        html_exporter.export([], str(tmp_path / "site"))
        assert captured.get("encoding") == "utf-8"
