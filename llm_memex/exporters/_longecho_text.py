"""Render conversations as the Markdown files of a longecho export.

Pure rendering: this module turns a ``Conversation`` into the text a reader
opens in an editor (``text/<YYYY>/<file>.md``) and builds ``text/index.md``.
It returns paths and strings only; writing them is the exporter's job.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

from llm_memex.models import Conversation, ContentBlock, Message, text_block

_SLUG_MAX = 50
_FORK_TEXT_MAX = 60
_ASSETS_PREFIX = "assets/"
# The rendered file lives at text/<YYYY>/<file>.md; assets live in data/assets/.
_ASSETS_FROM_TEXT = "../../data/assets/"


# -- helpers ---------------------------------------------------------------

def _one_line(text: Optional[str]) -> str:
    """Collapse every run of whitespace (newlines included) to a single space."""
    return " ".join((text or "").split())


def _slug(title: Optional[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").lower())[:_SLUG_MAX].strip("-")
    return slug or "untitled"


def text_path(conv: Conversation) -> PurePosixPath:
    """Stable path of a conversation's file, relative to ``text/``."""
    digest = hashlib.sha256(conv.id.encode("utf-8")).hexdigest()[:8]
    day = conv.created_at.strftime("%Y-%m-%d")
    return PurePosixPath(f"{conv.created_at.year:04d}") / f"{day}-{_slug(conv.title)}-{digest}.md"


def _decoded_size(b64: str) -> int:
    """Byte size of a base64 payload, without decoding it."""
    s = "".join(b64.split())
    return max(len(s) * 3 // 4 - (len(s) - len(s.rstrip("="))), 0)


def _size_label(nbytes: int) -> str:
    return "<1 KB" if nbytes < 1024 else f"{nbytes // 1024} KB"


def _inline_payload(block: ContentBlock) -> Optional[tuple[str, int]]:
    """(media type, decoded bytes) when the block embeds its data, else None."""
    url = block.get("url") or ""
    media_type = block.get("media_type") or ""
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if not media_type:
            media_type = header[len("data:"):].split(";")[0]
        if "base64" in header.split(";"):
            return media_type, _decoded_size(payload)
        return media_type, len(payload.encode("utf-8"))
    if not url and block.get("data"):
        return media_type, _decoded_size(str(block["data"]))
    return None


def _rewrite_block(block: ContentBlock) -> ContentBlock:
    """Return the block as the text/ file should show it (input is not mutated)."""
    if block.get("type") != "media":
        return block
    inline = _inline_payload(block)
    if inline is not None:
        media_type, nbytes = inline
        return text_block(f"[inline {media_type or 'media'}, {_size_label(nbytes)}, shown in site/]")
    url = block.get("url") or ""
    if url.startswith(_ASSETS_PREFIX):
        return {**block, "url": _ASSETS_FROM_TEXT + url[len(_ASSETS_PREFIX):]}
    return block


def _message_md(msg: Message) -> str:
    rewritten = Message(id=msg.id, role=msg.role,
                        content=[_rewrite_block(b) for b in msg.content])
    return rewritten.get_content_md()


def _note_md(note: Dict[str, Any]) -> str:
    text = str(note.get("text") or "")
    lines = text.splitlines() or [""]
    first = f"> **Note:** {lines[0]}"
    return "\n".join([first] + [f"> {l}" if l else ">" for l in lines[1:]])


def _fork_line(parent: Optional[Message]) -> str:
    if parent is None:
        return "*Forks from: a separate start of the conversation*"
    text = _one_line(parent.get_text())
    shown = text[:_FORK_TEXT_MAX] + ("..." if len(text) > _FORK_TEXT_MAX else "")
    return f"*Forks after {parent.role}: {shown}*"


# -- conversation ----------------------------------------------------------

def render_conversation(conv: Conversation, notes: List[Dict[str, Any]]) -> str:
    """The Markdown file for one conversation.

    ``notes`` are rows from ``Database.get_notes(conversation_id=...)`` (empty
    when notes are excluded). Messages shared by several branches appear once,
    in the first path that reaches them.
    """
    conv_notes = [n for n in notes if n.get("target_kind") == "conversation"]
    msg_notes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for n in notes:
        if n.get("target_kind") == "message" and n.get("message_id"):
            msg_notes[n["message_id"]].append(n)

    blocks: List[str] = [f"# {_one_line(conv.title) or 'Untitled'}"]
    meta = []
    if conv.source:
        meta.append(f"*Source: {_one_line(conv.source)}*")
    meta.append(f"*Created: {conv.created_at.strftime('%Y-%m-%d')}*")
    blocks.append("  \n".join(meta))
    blocks.extend(_note_md(n) for n in conv_notes)

    emitted: set[str] = set()
    branch = 0
    for path in conv.get_all_paths():
        fresh = [m for m in path if m.id not in emitted]
        if not fresh:
            continue
        branch += 1
        if branch > 1:
            blocks.append(f"## Branch {branch}")
            blocks.append(_fork_line(conv.messages.get(fresh[0].parent_id)))
        for msg in fresh:
            emitted.add(msg.id)
            blocks.append(f"**{msg.role}**: {_message_md(msg)}")
            blocks.extend(_note_md(n) for n in msg_notes.get(msg.id, []))
    return "\n\n".join(blocks) + "\n"


# -- index -----------------------------------------------------------------

@dataclass
class IndexEntry:
    path: PurePosixPath  # relative to text/, e.g. 2025/2025-06-14-x-1a2b3c4d.md
    title: str
    created_at: datetime
    source: Optional[str]


def _escape_link_text(text: str) -> str:
    return re.sub(r"([\\\[\]])", r"\\\1", _one_line(text))


def _sort_time(when: datetime) -> datetime:
    """A key that orders naive and aware times together: aware times become
    naive UTC, naive ones are taken as UTC as they are. (The database stores
    times without an offset; a row written with one by other tools still sorts.)"""
    if when.tzinfo is None:
        return when
    return when.astimezone(timezone.utc).replace(tzinfo=None)


def render_index(entries: Iterable[IndexEntry]) -> str:
    """``text/index.md``: one section per year, newest first."""
    by_year: Dict[int, List[IndexEntry]] = defaultdict(list)
    for e in entries:
        by_year[e.created_at.year].append(e)
    blocks = ["# Conversations"]
    for year in sorted(by_year, reverse=True):
        lines = [f"## {year:04d}", ""]
        for e in sorted(by_year[year], key=lambda e: (_sort_time(e.created_at), str(e.path)), reverse=True):
            line = (f"- {e.created_at.strftime('%Y-%m-%d')} "
                    f"[{_escape_link_text(e.title) or 'Untitled'}]({e.path.as_posix()})")
            if e.source:
                line += f" · {_one_line(e.source)}"
            lines.append(line)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
