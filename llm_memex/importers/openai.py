"""Import OpenAI conversation exports (conversations.json)."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_memex.importers import parse_timestamp
from llm_memex.models import (
    Conversation,
    Message,
    media_block,
    text_block,
    tool_result_block,
    tool_use_block,
)


def _detect_file(path: str) -> bool:
    """Check if a single file is an OpenAI conversations.json export."""
    try:
        # utf-8-sig tolerates a leading BOM (otherwise json.load raises).
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "mapping" in data[0]:
            return True
        return False
    except (json.JSONDecodeError, IOError, OSError, KeyError, IndexError, ValueError):
        return False


def detect(path: str) -> bool:
    """Check if path is an OpenAI export file or directory containing one."""
    p = Path(path)
    if p.is_dir():
        candidate = p / "conversations.json"
        return candidate.exists() and _detect_file(str(candidate))
    return _detect_file(path)


def import_path(path: str) -> List[Conversation]:
    """Import conversations from an OpenAI export file or directory."""
    p = Path(path)
    if p.is_dir():
        return _import_file(str(p / "conversations.json"))
    return _import_file(path)


def _import_file(path: str) -> List[Conversation]:
    """Import conversations from a single OpenAI export file."""
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    conversations = []
    for item in data:
        conv = _import_conversation(item, path)
        if conv:
            conversations.append(conv)
    return conversations


def _import_conversation(data: dict, source_path: str = None) -> Optional[Conversation]:
    conv_id = data.get("id") or data.get("conversation_id", "")
    mapping = data.get("mapping", {})
    if not isinstance(mapping, dict):
        return None
    if not mapping:
        return None
    # create_time==0 is a valid epoch (1970), so test "is not None", not truthiness.
    created = (
        parse_timestamp(data.get("create_time"))
        if data.get("create_time") is not None
        else None
    ) or datetime.now()
    updated = (
        parse_timestamp(data.get("update_time"))
        if data.get("update_time") is not None
        else None
    ) or created
    conv = Conversation(
        id=conv_id,
        title=data.get("title") or "Untitled Conversation",
        source="openai",
        created_at=created,
        updated_at=updated,
    )
    model = None
    for node_id, node in mapping.items():
        if not isinstance(node, dict):
            continue
        msg_data = node.get("message")
        if not msg_data:
            continue
        if not isinstance(msg_data, dict):
            continue
        author = msg_data.get("author") or {}
        role = author.get("role", "unknown") if isinstance(author, dict) else "unknown"
        content_obj = msg_data.get("content") or {}
        if role == "system" and not (
            isinstance(content_obj, dict) and content_obj.get("parts")
        ):
            continue
        content = _extract_content(msg_data)
        if not content:
            content = [text_block("")]
        metadata = msg_data.get("metadata") or {}
        msg_model = metadata.get("model_slug") if isinstance(metadata, dict) else None
        if msg_model and role == "assistant":
            model = msg_model
        parent_id = node.get("parent")
        # Skip virtual root nodes (nodes without messages)
        if parent_id and isinstance(mapping.get(parent_id), dict) and \
                mapping.get(parent_id, {}).get("message") is None:
            parent_id = None
        msg = Message(
            id=node_id,
            role=role,
            content=content,
            parent_id=parent_id,
            model=msg_model,
            created_at=(
                parse_timestamp(msg_data.get("create_time"))
                if msg_data.get("create_time") is not None
                else None
            ),
        )
        conv.add_message(msg)
    conv.model = model
    conv.metadata["_provenance"] = {
        "source_type": "openai",
        "source_file": source_path,
        "source_id": conv_id,
    }
    return conv


def _extract_content(msg_data: dict) -> List[Dict[str, Any]]:
    """Extract content blocks from an OpenAI message."""
    content_obj = msg_data.get("content") or {}
    if not isinstance(content_obj, dict):
        content_obj = {}
    parts = content_obj.get("parts") or []
    if not isinstance(parts, list):
        parts = []
    content_type = content_obj.get("content_type", "text")
    blocks = []

    # Handle tool calls / tool results via metadata
    author = msg_data.get("author") or {}
    author_role = author.get("role", "") if isinstance(author, dict) else ""
    if content_type == "tether_browsing_display":
        text = "\n".join(str(p) for p in parts if isinstance(p, str))
        if text:
            blocks.append(text_block(text))
        return blocks

    for part in parts:
        if isinstance(part, str):
            blocks.append(text_block(part))
        elif isinstance(part, dict):
            if "asset_pointer" in part or part.get("content_type") == "image_asset_pointer":
                blocks.append(
                    media_block("image/png", url=part.get("asset_pointer", ""))
                )
            elif part.get("type") == "tool_use":
                blocks.append(
                    tool_use_block(
                        id=part.get("id", ""),
                        name=part.get("name", ""),
                        input=part.get("input", {}),
                    )
                )
            elif part.get("type") == "tool_result":
                blocks.append(
                    tool_result_block(
                        tool_use_id=part.get("tool_use_id", ""),
                        content=part.get("content"),
                        is_error=part.get("is_error", False),
                    )
                )
            else:
                blocks.append(text_block(str(part)))
    return blocks
