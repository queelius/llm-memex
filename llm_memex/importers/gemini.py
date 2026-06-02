"""Import Google Gemini conversation exports."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_memex.importers import detect_model, parse_timestamp
from llm_memex.models import Conversation, Message, media_block, text_block


def detect(path: str) -> bool:
    """Check if file is a Gemini conversation export."""
    if Path(path).is_dir():
        return False
    try:
        # utf-8-sig tolerates a leading BOM (otherwise json.load raises).
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if any(k in data for k in ("conversations", "turns", "conversation_id")):
                return True
        if isinstance(data, list) and data:
            first = data[0]
            # Do not claim other providers' exports: an OpenAI conversation dict
            # carries "mapping" and an Anthropic one carries "chat_messages".
            # Matching a "gemini"/"bard" substring anywhere in the stringified
            # record is not a format signature (it collides with any chat that
            # merely discusses those products), so structurally exclude them
            # before falling back to the substring heuristic.
            if isinstance(first, dict) and ("mapping" in first or "chat_messages" in first):
                return False
            if isinstance(first, dict) and any(
                k in first for k in ("turns", "conversation_id", "parts")
            ):
                return True
            sample = str(first).lower()
            if "gemini" in sample or "bard" in sample:
                return True
        return False
    except (json.JSONDecodeError, IOError, OSError, KeyError, IndexError, ValueError):
        return False


def import_path(path: str) -> List[Conversation]:
    """Import conversations from a Gemini export file."""
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    # Normalize to list of conversation dicts
    if isinstance(data, dict):
        if "conversations" in data:
            conv_list = data["conversations"]
        else:
            conv_list = [data]
    elif isinstance(data, list):
        conv_list = data
    else:
        conv_list = [data]

    conversations = []
    for item in conv_list:
        conv = _import_conversation(item, path)
        if conv:
            conversations.append(conv)
    return conversations


def _import_conversation(data: dict, source_path: str = None) -> Optional[Conversation]:
    conv_id = data.get("id") or data.get("conversation_id", str(uuid.uuid4()))
    title = data.get("title", "Untitled Conversation")
    model = detect_model(data, ["turns", "messages"], "gemini")

    created = parse_timestamp(data.get("created_at")) or datetime.now()
    updated = parse_timestamp(data.get("updated_at")) or created

    conv = Conversation(
        id=conv_id,
        title=title,
        source="gemini",
        model=model,
        created_at=created,
        updated_at=updated,
        tags=["google", "gemini"],
    )

    messages = data.get("turns")
    if messages is None:
        messages = data.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    parent_id = None

    for idx, msg_data in enumerate(messages):
        if not isinstance(msg_data, dict):
            continue
        msg_id = msg_data.get("id", f"msg_{idx}")
        role_str = str(msg_data.get("author", msg_data.get("role", "user")))
        role = "assistant" if role_str.lower() in ("model", "gemini", "bard") else "user"

        content = _extract_content(msg_data)
        if not content:
            content = [text_block("")]

        msg = Message(
            id=msg_id,
            role=role,
            content=content,
            parent_id=parent_id,
            created_at=parse_timestamp(msg_data.get("timestamp")),
        )
        conv.add_message(msg)
        parent_id = msg_id

    conv.metadata["_provenance"] = {
        "source_type": "gemini",
        "source_file": source_path,
        "source_id": conv_id,
    }
    return conv


def _extract_content(msg_data: dict) -> List[Dict[str, Any]]:
    """Extract content blocks from a Gemini message."""
    blocks: List[Dict[str, Any]] = []

    # Gemini uses "parts" for multimodal content. A non-list "parts" (e.g. a
    # plain string) must not be iterated character-by-character.
    parts = msg_data.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, str):
                blocks.append(text_block(part))
            elif isinstance(part, dict):
                if "text" in part:
                    blocks.append(text_block(part["text"]))
                elif "inline_data" in part:
                    inline = part["inline_data"]
                    blocks.append(
                        media_block(
                            inline.get("mime_type", "image/png"),
                            data=inline.get("data"),
                        )
                    )
        return blocks

    # Fallback: simple text/content field
    text = msg_data.get("content", msg_data.get("text", ""))
    if text:
        blocks.append(text_block(text))
    return blocks
