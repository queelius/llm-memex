"""Data model for memex conversations."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

ContentBlock = Dict[str, Any]  # Always has "type" key

def text_block(text: str) -> ContentBlock:
    return {"type": "text", "text": text}

def media_block(media_type: str, *, url: str | None = None, data: str | None = None, filename: str | None = None) -> ContentBlock:
    block: ContentBlock = {"type": "media", "media_type": media_type}
    if url is not None: block["url"] = url
    if data is not None: block["data"] = data
    if filename is not None: block["filename"] = filename
    return block

def tool_use_block(id: str, name: str, input: Dict[str, Any]) -> ContentBlock:
    return {"type": "tool_use", "id": id, "name": name, "input": input}

def tool_result_block(tool_use_id: str, content: Any = None, is_error: bool = False) -> ContentBlock:
    block: ContentBlock = {"type": "tool_result", "tool_use_id": tool_use_id}
    if content is not None: block["content"] = content
    if is_error: block["is_error"] = True
    return block

def thinking_block(text: str) -> ContentBlock:
    return {"type": "thinking", "text": text}


def _render_media_md(block: ContentBlock) -> str:
    """Render a media content block as markdown."""
    media_type = block.get("media_type", "")
    url = block.get("url", "")
    filename = block.get("filename", "")
    data = block.get("data", "")

    # Build a data URI if we have base64 data but no URL
    if not url and data:
        url = f"data:{media_type};base64,{data}"

    if not url:
        return f"[{filename}]" if filename else ""

    if media_type.startswith("image/"):
        alt = filename or "image"
        return f"![{alt}]({url})"
    elif media_type.startswith("audio/"):
        label = filename or "audio"
        return f"[audio: {label}]({url})"
    elif media_type.startswith("video/"):
        label = filename or "video"
        return f"[video: {label}]({url})"
    elif media_type == "application/pdf":
        label = filename or "document"
        return f"[pdf: {label}]({url})"
    else:
        label = filename or "attachment"
        return f"[attachment: {label}]({url})"


@dataclass
class Message:
    id: str
    role: str
    content: List[ContentBlock]
    parent_id: Optional[str] = None
    model: Optional[str] = None
    created_at: Optional[datetime] = None
    sensitive: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_text(self) -> str:
        return "\n".join(
            block["text"] for block in self.content
            if block.get("type") == "text" and block.get("text")
        )

    def get_content_md(self) -> str:
        """Render all content blocks as markdown, including media."""
        parts = []
        for block in self.content:
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                parts.append(block["text"])
            elif btype == "media":
                rendered = _render_media_md(block)
                if rendered:
                    parts.append(rendered)
            # Skip tool_use, tool_result, thinking
        return "\n\n".join(parts)

@dataclass
class Conversation:
    id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None
    source: Optional[str] = None
    model: Optional[str] = None
    summary: Optional[str] = None
    message_count: int = 0
    starred_at: Optional[datetime] = None
    pinned_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    parent_conversation_id: Optional[str] = None
    sensitive: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: Dict[str, Message] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)
    _children: Dict[Optional[str], List[str]] = field(default_factory=dict, repr=False)

    def add_message(self, message: Message) -> None:
        self.messages[message.id] = message
        self.message_count = len(self.messages)
        if message.parent_id is None and message.id not in self.root_ids:
            self.root_ids.append(message.id)
        self._children.setdefault(message.parent_id, [])
        if message.id not in self._children[message.parent_id]:
            self._children[message.parent_id].append(message.id)

    def get_children(self, message_id: Optional[str]) -> List[Message]:
        return [self.messages[cid] for cid in self._children.get(message_id, []) if cid in self.messages]

    def _effective_roots(self) -> List[str]:
        """Root ids for traversal: the real roots, plus any message whose parent_id
        dangles (references a message not present), so orphans are never silently
        dropped from paths."""
        roots = list(self.root_ids)
        seen = set(roots)
        for mid, msg in self.messages.items():
            if (msg.parent_id is not None and msg.parent_id not in self.messages
                    and mid not in seen):
                roots.append(mid)
                seen.add(mid)
        return roots

    def get_all_paths(self) -> List[List[Message]]:
        """Get all root-to-leaf paths. Uses iterative DFS to avoid recursion limits.
        A per-path visited set breaks cycles (a malformed import can repeat a message
        id and re-parent it into an ancestor) so traversal always terminates."""
        paths: List[List[Message]] = []
        # Stack entries: (message_id, path_so_far, ids_on_path)
        stack: List[tuple[str, List[Message], frozenset]] = [
            (rid, [], frozenset()) for rid in reversed(self._effective_roots())
        ]
        while stack:
            msg_id, current, on_path = stack.pop()
            if msg_id not in self.messages:
                continue
            path = current + [self.messages[msg_id]]
            on_path = on_path | {msg_id}
            children = [c for c in self._children.get(msg_id, []) if c not in on_path]
            if not children:
                paths.append(path)
            else:
                for cid in reversed(children):
                    stack.append((cid, path, on_path))
        return paths

    def get_path(self, leaf_id: str) -> Optional[List[Message]]:
        if leaf_id not in self.messages:
            return None
        path = []
        current = leaf_id
        seen: set[str] = set()
        while current is not None and current not in seen:
            msg = self.messages.get(current)
            if msg is None: break
            seen.add(current)
            path.append(msg)
            current = msg.parent_id
        path.reverse()
        return path

    def get_leaf_ids(self) -> List[str]:
        has_children = {pid for pid, kids in self._children.items() if kids and pid is not None}
        return [mid for mid in self.messages if mid not in has_children]
