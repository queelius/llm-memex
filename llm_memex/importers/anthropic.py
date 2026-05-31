"""Import Anthropic/Claude conversation exports."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_memex.importers import detect_model, parse_timestamp
from llm_memex.models import (
    Conversation,
    Message,
    media_block,
    text_block,
    tool_result_block,
    tool_use_block,
)


def detect(path: str) -> bool:
    """Check if file is an Anthropic conversation export."""
    if Path(path).is_dir():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            sample = data[0]
        elif isinstance(data, dict):
            sample = data
        else:
            return False
        # Primary: chat_messages field
        if "chat_messages" in sample:
            return True
        # Secondary: uuid + messages with sender pattern
        if "uuid" in sample and "name" in sample:
            return True
        return False
    except (json.JSONDecodeError, IOError, KeyError, IndexError, ValueError):
        return False


def import_path(path: str) -> List[Conversation]:
    """Import conversations from an Anthropic export file."""
    with open(path) as f:
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
    conv_id = data.get("uuid") or data.get("id", str(uuid.uuid4()))
    title = data.get("name") or data.get("title", "Untitled Conversation")
    model = detect_model(data, ["chat_messages", "messages"], "claude")

    created = parse_timestamp(data.get("created_at")) or datetime.now()
    updated = parse_timestamp(data.get("updated_at")) or created

    conv = Conversation(
        id=conv_id,
        title=title,
        source="anthropic",
        model=model,
        created_at=created,
        updated_at=updated,
        tags=["anthropic", "claude"],
    )

    messages = data.get("chat_messages", data.get("messages", []))
    parent_id = None

    for idx, msg_data in enumerate(messages):
        msg_id = msg_data.get("uuid") or msg_data.get("id", f"msg_{idx}")
        sender = msg_data.get("sender", msg_data.get("role", "user"))
        role = "assistant" if sender in ("assistant", "model") else "user"

        content = _extract_content(msg_data)
        if not content:
            content = [text_block("")]

        msg = Message(
            id=msg_id,
            role=role,
            content=content,
            parent_id=parent_id,
            created_at=parse_timestamp(msg_data.get("created_at")),
        )
        conv.add_message(msg)
        parent_id = msg_id

    conv.metadata["_provenance"] = {
        "source_type": "anthropic",
        "source_file": source_path,
        "source_id": conv_id,
    }
    return conv


def _extract_content(msg_data: dict) -> List[Dict[str, Any]]:
    """Extract content blocks from an Anthropic message."""
    blocks: List[Dict[str, Any]] = []

    # Simple text field
    if "text" in msg_data:
        blocks.append(text_block(msg_data["text"]))
        # Handle attachments
        for att in msg_data.get("attachments", []):
            if isinstance(att, dict) and att.get("file_name"):
                blocks.append(
                    media_block(
                        att.get("file_type", "application/octet-stream"),
                        filename=att["file_name"],
                    )
                )
        return blocks

    # Content field (string or list of parts)
    raw = msg_data.get("content")
    if isinstance(raw, str):
        blocks.append(text_block(raw))
        return blocks

    if isinstance(raw, list):
        for part in raw:
            if isinstance(part, str):
                blocks.append(text_block(part))
            elif isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype == "text":
                    blocks.append(text_block(part.get("text", "")))
                elif ptype == "image":
                    source = part.get("source", {})
                    if isinstance(source, dict):
                        if source.get("type") == "base64":
                            blocks.append(
                                media_block(
                                    source.get("media_type", "image/png"),
                                    data=source.get("data"),
                                )
                            )
                        elif "url" in source:
                            blocks.append(
                                media_block("image/png", url=source["url"])
                            )
                elif ptype == "tool_use":
                    blocks.append(
                        tool_use_block(
                            id=part.get("id", ""),
                            name=part.get("name", ""),
                            input=part.get("input", {}),
                        )
                    )
                elif ptype == "tool_result":
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
