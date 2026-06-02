"""Tests for media rendering, asset resolution, and asset copying."""
import base64
from datetime import datetime
from pathlib import Path

from llm_memex.assets import (
    copy_assets,
    resolve_openai_assets,
    resolve_source_assets,
    _media_type_to_ext,
    _safe_filename,
)
from llm_memex.models import (
    Conversation,
    Message,
    _render_media_md,
    media_block,
    text_block,
    tool_use_block,
)


# ── _render_media_md ────────────────────────────────────────────

class TestRenderMediaMd:
    def test_image_with_url(self):
        block = media_block("image/png", url="https://example.com/photo.png")
        assert _render_media_md(block) == "![image](https://example.com/photo.png)"

    def test_image_with_filename(self):
        block = media_block("image/jpeg", url="assets/pic.jpg", filename="sunset.jpg")
        assert _render_media_md(block) == "![sunset.jpg](assets/pic.jpg)"

    def test_audio(self):
        block = media_block("audio/mpeg", url="assets/song.mp3", filename="song.mp3")
        assert _render_media_md(block) == "[audio: song.mp3](assets/song.mp3)"

    def test_video(self):
        block = media_block("video/mp4", url="assets/clip.mp4", filename="clip.mp4")
        assert _render_media_md(block) == "[video: clip.mp4](assets/clip.mp4)"

    def test_pdf(self):
        block = media_block("application/pdf", url="assets/doc.pdf", filename="doc.pdf")
        assert _render_media_md(block) == "[pdf: doc.pdf](assets/doc.pdf)"

    def test_base64_image_data_uri(self):
        data = base64.b64encode(b"fake-png-data").decode()
        block = media_block("image/png", data=data)
        result = _render_media_md(block)
        assert result.startswith("![image](data:image/png;base64,")
        assert data in result

    def test_no_url_no_data_with_filename(self):
        block = {"type": "media", "media_type": "image/png", "filename": "lost.png"}
        assert _render_media_md(block) == "[lost.png]"

    def test_no_url_no_data_no_filename(self):
        block = {"type": "media", "media_type": "image/png"}
        assert _render_media_md(block) == ""

    def test_unknown_type_fallback(self):
        block = media_block("application/zip", url="assets/archive.zip", filename="archive.zip")
        assert _render_media_md(block) == "[attachment: archive.zip](assets/archive.zip)"


# ── Message.get_content_md ──────────────────────────────────────

class TestGetContentMd:
    def test_mixed_text_and_image(self):
        msg = Message(
            id="m1", role="assistant",
            content=[
                text_block("Here is the image:"),
                media_block("image/png", url="assets/photo.png"),
            ],
        )
        result = msg.get_content_md()
        assert "Here is the image:" in result
        assert "![image](assets/photo.png)" in result
        # Paragraph spacing between blocks
        assert "\n\n" in result

    def test_skips_tool_use(self):
        msg = Message(
            id="m1", role="assistant",
            content=[
                text_block("Calling a tool"),
                tool_use_block("t1", "search", {"q": "test"}),
                text_block("Got results"),
            ],
        )
        result = msg.get_content_md()
        assert "Calling a tool" in result
        assert "Got results" in result
        assert "search" not in result

    def test_text_only_matches_get_text(self):
        msg = Message(
            id="m1", role="user",
            content=[text_block("Hello"), text_block("World")],
        )
        # get_content_md uses \n\n, get_text uses \n
        assert msg.get_content_md() == "Hello\n\nWorld"
        assert msg.get_text() == "Hello\nWorld"

    def test_empty_content(self):
        msg = Message(id="m1", role="user", content=[])
        assert msg.get_content_md() == ""


# ── Helpers ─────────────────────────────────────────────────────

class TestHelpers:
    def test_media_type_to_ext_known(self):
        assert _media_type_to_ext("image/png") == ".png"
        assert _media_type_to_ext("audio/mpeg") == ".mp3"
        assert _media_type_to_ext("application/pdf") == ".pdf"

    def test_media_type_to_ext_fallback(self):
        assert _media_type_to_ext("image/tiff") == ".tiff"

    def test_safe_filename_with_name(self):
        result = _safe_filename("photo.png", "msg123", 0, "image/png")
        assert result == "photo.png"

    def test_safe_filename_without_name(self):
        result = _safe_filename(None, "abcdefghij", 2, "image/jpeg")
        assert result == "abcdefgh_2.jpg"

    def test_safe_filename_sanitizes(self):
        result = _safe_filename("my file (1).png", "msg1", 0, "image/png")
        assert " " not in result
        assert "(" not in result
        assert result.endswith(".png")

    def test_media_type_to_ext_junk_falls_back_to_bin(self):
        """A junk/attacker-controlled media_type must not produce a junk
        extension (MCA-6). Shell-injection-looking and traversal-looking
        subtypes fall back to a safe ".bin"."""
        # Shell metacharacters in the subtype must not leak into the ext.
        assert _media_type_to_ext("image/png; rm -rf") == ".bin"
        # Path-traversal-looking subtype must not become ".." .
        assert _media_type_to_ext("x/..") == ".bin"
        # Slashes / spaces / nonsense fall back too.
        assert _media_type_to_ext("foo/bar baz") == ".bin"

    def test_media_type_to_ext_clean_subtype_still_works(self):
        """A clean (alphanumeric) unknown subtype is still honored."""
        assert _media_type_to_ext("image/tiff") == ".tiff"
        assert _media_type_to_ext("application/x-tar") == ".x-tar"

    def test_safe_filename_with_junk_media_type_has_safe_ext(self):
        """When no name is supplied, the generated filename uses a safe ext
        even if media_type is junk (MCA-6)."""
        result = _safe_filename(None, "abcdefghij", 0, "image/png; rm -rf")
        assert result == "abcdefgh_0.bin"
        # No shell metacharacters or spaces leaked into the filename.
        assert ";" not in result
        assert " " not in result


# ── resolve_openai_assets ───────────────────────────────────────

class TestResolveOpenaiAssets:
    def test_resolves_file_service_url(self, tmp_path):
        # Create a fake exported file
        fake_file = tmp_path / "file-abc123-photo.png"
        fake_file.write_bytes(b"PNG data")

        conv = _make_conv([
            media_block("image/png", url="file-service://file-abc123"),
        ])
        count = resolve_openai_assets(conv, tmp_path)
        assert count == 1
        block = conv.messages["m1"].content[0]
        assert block["url"] == str(fake_file.resolve())

    def test_resolves_dalle_subdir(self, tmp_path):
        dalle_dir = tmp_path / "dalle-generations"
        dalle_dir.mkdir()
        fake_file = dalle_dir / "file-xyz789-art.png"
        fake_file.write_bytes(b"DALLE data")

        conv = _make_conv([
            media_block("image/png", url="file-service://file-xyz789"),
        ])
        count = resolve_openai_assets(conv, tmp_path)
        assert count == 1
        assert "dalle-generations" in conv.messages["m1"].content[0]["url"]

    def test_unresolvable_leaves_url_unchanged(self, tmp_path):
        conv = _make_conv([
            media_block("image/png", url="file-service://file-missing"),
        ])
        count = resolve_openai_assets(conv, tmp_path)
        assert count == 0
        assert conv.messages["m1"].content[0]["url"] == "file-service://file-missing"

    def test_skips_non_file_service_urls(self, tmp_path):
        conv = _make_conv([
            media_block("image/png", url="https://example.com/img.png"),
        ])
        count = resolve_openai_assets(conv, tmp_path)
        assert count == 0

    def test_resolves_sediment_url_in_conv_subdir(self, tmp_path):
        """Post-2024 OpenAI exports use sediment:// URLs with files under
        {source_dir}/{conv.id}/{kind}/file_{ID}-{assetID}.{ext}."""
        conv = _make_conv([
            media_block("image/png", url="sediment://file_abc123sha"),
        ])
        image_dir = tmp_path / conv.id / "image"
        image_dir.mkdir(parents=True)
        fake_file = image_dir / "file_abc123sha-assetuuid.png"
        fake_file.write_bytes(b"PNG")

        count = resolve_openai_assets(conv, tmp_path)
        assert count == 1
        assert conv.messages["m1"].content[0]["url"] == str(fake_file.resolve())

    def test_sediment_unresolvable_when_no_conv_dir(self, tmp_path):
        conv = _make_conv([
            media_block("image/png", url="sediment://file_nomatch"),
        ])
        count = resolve_openai_assets(conv, tmp_path)
        assert count == 0
        assert conv.messages["m1"].content[0]["url"] == "sediment://file_nomatch"

    def test_corrects_mislabeled_media_type(self, tmp_path):
        """When the declared media_type disagrees with the actual file
        extension, trust the extension. OpenAI sometimes marks .wav as
        image/png in the export metadata."""
        conv = _make_conv([
            media_block("image/png", url="sediment://file_audioactual"),
        ])
        audio_dir = tmp_path / conv.id / "audio"
        audio_dir.mkdir(parents=True)
        fake_file = audio_dir / "file_audioactual-x.wav"
        fake_file.write_bytes(b"RIFF....WAVE")

        resolve_openai_assets(conv, tmp_path)
        assert conv.messages["m1"].content[0]["media_type"] == "audio/wav"


# ── copy_assets ─────────────────────────────────────────────────

class TestCopyAssets:
    def test_copy_from_absolute_path(self, tmp_path):
        src_file = tmp_path / "source" / "photo.png"
        src_file.parent.mkdir()
        src_file.write_bytes(b"PNG image data")

        asset_dir = tmp_path / "db" / "assets"
        conv = _make_conv([
            media_block("image/png", url=str(src_file)),
        ])
        count = copy_assets(conv, asset_dir)
        assert count == 1
        block = conv.messages["m1"].content[0]
        assert block["url"].startswith("assets/")
        # Verify file was actually copied
        copied = asset_dir / block["url"].removeprefix("assets/")
        assert copied.exists()
        assert copied.read_bytes() == b"PNG image data"

    def test_copy_from_base64(self, tmp_path):
        raw = b"decoded image bytes"
        b64 = base64.b64encode(raw).decode()
        asset_dir = tmp_path / "assets"
        conv = _make_conv([
            media_block("image/png", data=b64),
        ])
        count = copy_assets(conv, asset_dir)
        assert count == 1
        block = conv.messages["m1"].content[0]
        assert block["url"].startswith("assets/")
        assert "data" not in block  # data key removed
        # Verify file written correctly
        written = asset_dir / block["url"].removeprefix("assets/")
        assert written.read_bytes() == raw

    def test_idempotent_skips_relative_urls(self, tmp_path):
        asset_dir = tmp_path / "assets"
        conv = _make_conv([
            media_block("image/png", url="assets/already-there.png"),
        ])
        count = copy_assets(conv, asset_dir)
        assert count == 0
        assert conv.messages["m1"].content[0]["url"] == "assets/already-there.png"

    def test_collision_rename(self, tmp_path):
        asset_dir = tmp_path / "assets"
        asset_dir.mkdir(parents=True)
        # Pre-create a file with the same name
        (asset_dir / "photo.png").write_bytes(b"existing")

        src_file = tmp_path / "photo.png"
        src_file.write_bytes(b"new data")
        conv = _make_conv([
            media_block("image/png", url=str(src_file), filename="photo.png"),
        ])
        count = copy_assets(conv, asset_dir)
        assert count == 1
        # Should have been renamed to avoid collision
        url = conv.messages["m1"].content[0]["url"]
        assert url.startswith("assets/photo_")
        assert url.endswith(".png")

    def test_skips_non_media_blocks(self, tmp_path):
        asset_dir = tmp_path / "assets"
        conv = _make_conv([text_block("just text")])
        count = copy_assets(conv, asset_dir)
        assert count == 0

    def test_malformed_base64_does_not_raise(self, tmp_path):
        """MCA-1: a media block with malformed base64 must not abort the
        import. copy_assets should skip the bad block (leaving its data
        intact, writing no file) and still process the other blocks."""
        raw_good = b"good image bytes"
        good_b64 = base64.b64encode(raw_good).decode()
        # "Incorrect padding": length not a multiple of 4 and bogus chars.
        bad_b64 = "not-valid-base64!!!"
        asset_dir = tmp_path / "assets"
        conv = _make_conv([
            media_block("image/png", data=bad_b64, filename="bad.png"),
            media_block("image/png", data=good_b64, filename="good.png"),
        ])
        # Must not raise binascii.Error / ValueError.
        count = copy_assets(conv, asset_dir)
        # Only the good block was written.
        assert count == 1
        bad_block = conv.messages["m1"].content[0]
        good_block = conv.messages["m1"].content[1]
        # Bad block: data preserved, no url rewrite, no file written.
        assert bad_block.get("data") == bad_b64
        assert "url" not in bad_block or not bad_block.get("url", "").startswith("assets/")
        # Good block: written and rewritten.
        assert good_block["url"].startswith("assets/")
        assert "data" not in good_block
        written = asset_dir / good_block["url"].removeprefix("assets/")
        assert written.read_bytes() == raw_good

    def test_copy_from_absolute_path_drops_data_key(self, tmp_path):
        """assets minor: Case 1 (file-copy branch) must also drop a
        co-present stale base64 ``data`` key so it does not re-serialize."""
        src_file = tmp_path / "source" / "photo.png"
        src_file.parent.mkdir()
        src_file.write_bytes(b"PNG image data")

        stale_b64 = base64.b64encode(b"stale embedded copy").decode()
        asset_dir = tmp_path / "db" / "assets"
        conv = _make_conv([
            media_block("image/png", url=str(src_file), data=stale_b64),
        ])
        count = copy_assets(conv, asset_dir)
        assert count == 1
        block = conv.messages["m1"].content[0]
        assert block["url"].startswith("assets/")
        # Stale base64 must be gone for consistency with the base64 branch.
        assert "data" not in block
        copied = asset_dir / block["url"].removeprefix("assets/")
        assert copied.read_bytes() == b"PNG image data"


# ── resolve_source_assets ───────────────────────────────────────

class TestResolveSourceAssets:
    def test_dispatches_openai(self, tmp_path):
        fake_file = tmp_path / "file-test1-img.png"
        fake_file.write_bytes(b"data")
        conv = _make_conv([
            media_block("image/png", url="file-service://file-test1"),
        ])
        count = resolve_source_assets(conv, tmp_path, "openai")
        assert count == 1

    def test_noop_for_anthropic(self, tmp_path):
        conv = _make_conv([
            media_block("image/png", data=base64.b64encode(b"data").decode()),
        ])
        count = resolve_source_assets(conv, tmp_path, "anthropic")
        assert count == 0

    def test_noop_for_unknown(self, tmp_path):
        conv = _make_conv([media_block("image/png", url="http://x.com/a.png")])
        count = resolve_source_assets(conv, tmp_path, "gemini")
        assert count == 0


# ── Test helper ─────────────────────────────────────────────────

def _make_conv(content_blocks):
    """Create a minimal conversation with one message containing given blocks."""
    now = datetime(2024, 1, 1)
    conv = Conversation(id="test-conv", created_at=now, updated_at=now)
    conv.add_message(Message(id="m1", role="user", content=content_blocks))
    return conv
