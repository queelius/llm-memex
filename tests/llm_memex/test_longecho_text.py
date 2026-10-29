"""Tests for the longecho text/ renderer (pure rendering, no I/O)."""
import base64
import hashlib
import re
from datetime import datetime
from pathlib import PurePosixPath

import pytest

from llm_memex.exporters._longecho_text import (
    IndexEntry,
    render_conversation,
    render_index,
    text_path,
)
from llm_memex.models import Conversation, Message, media_block, text_block


def _conv(title="Hello, World! [draft]", cid="conv-1", created=datetime(2025, 6, 14, 9, 30)):
    return Conversation(id=cid, created_at=created, updated_at=created, title=title,
                        source="openai")


def _msg(conv, mid, role, text, parent=None):
    conv.add_message(Message(id=mid, role=role, content=[text_block(text)], parent_id=parent))


def _sha8(cid):
    return hashlib.sha256(cid.encode()).hexdigest()[:8]


def test_text_path_stable_and_slugged():
    conv = _conv()
    p = text_path(conv)
    assert isinstance(p, PurePosixPath)
    assert re.fullmatch(r"2025/2025-06-14-hello-world-draft-[0-9a-f]{8}\.md", str(p))
    assert str(p).endswith(f"-{_sha8('conv-1')}.md")
    assert text_path(conv) == p


@pytest.mark.parametrize("title", ["", "日本語", None, "   ", "[]|\n"])
def test_text_path_untitled_fallbacks(title):
    conv = _conv(title=title)
    assert str(text_path(conv)) == f"2025/2025-06-14-untitled-{_sha8('conv-1')}.md"


def test_text_path_slug_capped_at_50_chars():
    p = text_path(_conv(title="word " * 40))
    slug = p.name[len("2025-06-14-"):-len(f"-{_sha8('conv-1')}.md")]
    assert 0 < len(slug) <= 50
    assert not slug.endswith("-") and not slug.startswith("-")


def test_text_path_distinct_ids_distinct_paths():
    assert text_path(_conv(cid="a")) != text_path(_conv(cid="b"))


def test_branch_shared_messages_once():
    conv = _conv()
    _msg(conv, "r", "user", "root question")
    _msg(conv, "a", "assistant", "shared answer alpha", parent="r")
    _msg(conv, "b", "user", "branch one text", parent="a")
    _msg(conv, "c", "user", "branch two text", parent="a")
    out = render_conversation(conv, [])
    assert out.count("**assistant**: shared answer alpha") == 1  # emitted once
    assert out.count("root question") == 1
    assert out.count("## Branch 2") == 1
    assert "## Branch 1" not in out
    branch = out.split("## Branch 2")[1]
    fork = [l for l in branch.splitlines() if l.strip()][0]
    assert "assistant" in fork and "shared answer alpha" in fork  # names the fork point
    assert out.index("branch one text") < out.index("## Branch 2") < out.index("branch two text")


def test_branch_fork_line_truncated_to_60_chars():
    conv = _conv()
    _msg(conv, "r", "user", "q")
    _msg(conv, "a", "assistant", "x" * 100, parent="r")
    _msg(conv, "b", "user", "one", parent="a")
    _msg(conv, "c", "user", "two", parent="a")
    out = render_conversation(conv, [])
    branch = out.split("## Branch 2")[1]
    assert "x" * 60 in branch and "x" * 61 not in branch.split("two")[0]


def test_media_rewritten_and_inline_placeholder():
    conv = _conv()
    data = base64.b64encode(bytes(2250)).decode()
    assert len(data) == 3000
    conv.add_message(Message(id="m", role="user", content=[
        text_block("look"),
        media_block("image/png", url="assets/x.png"),
        media_block("image/png", data=data),
        media_block("image/png", url="https://example.com/y.png"),
    ]))
    out = render_conversation(conv, [])
    assert "](../../data/assets/x.png)" in out
    assert "[inline image/png, 2 KB, shown in site/]" in out
    assert "base64," not in out and data not in out
    assert "](https://example.com/y.png)" in out


def test_inline_data_url_and_small_size():
    conv = _conv()
    tiny = base64.b64encode(b"hi").decode()
    conv.add_message(Message(id="m", role="user", content=[
        media_block("image/gif", url=f"data:image/gif;base64,{tiny}"),
    ]))
    out = render_conversation(conv, [])
    assert "[inline image/gif, <1 KB, shown in site/]" in out
    assert "base64," not in out


def test_media_rewrite_does_not_mutate_conversation():
    conv = _conv()
    conv.add_message(Message(id="m", role="user", content=[
        media_block("image/png", url="assets/x.png")]))
    render_conversation(conv, [])
    assert conv.messages["m"].content[0]["url"] == "assets/x.png"


def test_notes_rendered():
    conv = _conv()
    _msg(conv, "r", "user", "first")
    _msg(conv, "a", "assistant", "second", parent="r")
    notes = [
        {"target_kind": "conversation", "message_id": None, "text": "about the whole thing"},
        {"target_kind": "message", "message_id": "r", "text": "about the first"},
    ]
    out = render_conversation(conv, notes)
    assert "> **Note:** about the whole thing" in out
    assert "> **Note:** about the first" in out
    assert out.index("about the whole thing") < out.index("first")
    assert out.index("first") < out.index("about the first") < out.index("second")


def test_no_notes_no_note_markup():
    conv = _conv()
    _msg(conv, "r", "user", "first")
    assert "Note" not in render_conversation(conv, [])


def test_header_has_title_source_and_date():
    conv = _conv()
    _msg(conv, "r", "user", "first")
    out = render_conversation(conv, [])
    assert out.startswith("# Hello, World! [draft]\n")
    assert "openai" in out and "2025-06-14" in out


def test_header_awkward_titles():
    for title in ("", None, "line one\nline two"):
        conv = _conv(title=title)
        out = render_conversation(conv, [])
        first = out.splitlines()[0]
        assert first.startswith("# ") and len(first) > 2
    assert render_conversation(_conv(title="a\nb"), []).startswith("# a b\n")


def test_index_escapes_titles():
    e = IndexEntry(PurePosixPath("2025/2025-06-14-x-deadbeef.md"), "a]b[c\nd",
                   datetime(2025, 6, 14), "openai")
    out = render_index([e])
    assert "- 2025-06-14 [a\\]b\\[c d](2025/2025-06-14-x-deadbeef.md) · openai" in out


def test_index_awkward_titles_keep_links_intact():
    path = PurePosixPath("2025/2025-06-14-untitled-deadbeef.md")
    for title in ("", "日本語", "a|b", "back\\slash", "[x](http://evil)"):
        out = render_index([IndexEntry(path, title, datetime(2025, 6, 14), None)])
        line = [l for l in out.splitlines() if l.startswith("- ")][0]
        assert line.endswith(f"]({path})")
        assert len(line.splitlines()) == 1
        assert "[x](http" not in line


def test_index_groups_by_year_newest_first():
    entries = [
        IndexEntry(PurePosixPath("2023/a.md"), "old", datetime(2023, 1, 2), "claude"),
        IndexEntry(PurePosixPath("2025/b.md"), "new", datetime(2025, 3, 4), None),
        IndexEntry(PurePosixPath("2025/c.md"), "newer", datetime(2025, 5, 6), "openai"),
    ]
    out = render_index(entries)
    assert out.startswith("# Conversations\n")
    assert out.index("## 2025") < out.index("## 2023")
    assert out.index("[newer]") < out.index("[new]")
    assert "- 2025-03-04 [new](2025/b.md)\n" in out + "\n"
    assert "· claude" in out


def test_index_empty():
    assert render_index([]).startswith("# Conversations")


def test_index_sorts_naive_and_aware_times_together():
    """Reviewer probe 3: a row stored with an offset sits beside naive ones."""
    from datetime import timedelta, timezone

    entries = [
        IndexEntry(PurePosixPath("2025/a.md"), "naive noon", datetime(2025, 6, 14, 12), None),
        # 08:00 at UTC-6 is 14:00 UTC: newer than the naive (UTC) noon
        IndexEntry(PurePosixPath("2025/b.md"), "aware two pm",
                   datetime(2025, 6, 14, 8, tzinfo=timezone(timedelta(hours=-6))), None),
        IndexEntry(PurePosixPath("2025/c.md"), "aware ten am",
                   datetime(2025, 6, 14, 10, tzinfo=timezone.utc), None),
    ]
    out = render_index(entries)
    assert out.index("aware two pm") < out.index("naive noon") < out.index("aware ten am")
