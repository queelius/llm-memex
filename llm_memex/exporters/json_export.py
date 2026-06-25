"""Export conversations as JSON."""
import json
from typing import Any, Dict, List

from llm_memex.models import Conversation


def export(conversations: List[Conversation], path: str, **kwargs) -> None:
    """Export conversations to a JSON file.

    Output is a JSON array of conversation objects, each containing
    full message data including content blocks.

    Parameters
    ----------
    include_notes : bool
        Include notes arrays on conversations and messages (default True).
    db : Database | None
        Database instance for querying notes.
    """
    include_notes = kwargs.get("include_notes", True)
    db = kwargs.get("db")

    data = []
    for conv in conversations:
        conv_notes: List[Dict[str, Any]] = []
        msg_notes: Dict[str, List[Dict[str, Any]]] = {}

        if include_notes and db:
            all_notes = db.get_notes(conversation_id=conv.id)
            for note in all_notes:
                entry = {
                    "id": note["id"],
                    "text": note["text"],
                    "created_at": str(note["created_at"]),
                }
                if note["target_kind"] == "conversation":
                    conv_notes.append(entry)
                elif note["target_kind"] == "message" and note["message_id"]:
                    msg_notes.setdefault(note["message_id"], []).append(entry)

        messages = []
        for m in conv.messages.values():
            msg_dict: Dict[str, Any] = {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "parent_id": m.parent_id,
                "model": m.model,
            }
            if include_notes and db:
                msg_dict["notes"] = msg_notes.get(m.id, [])
            messages.append(msg_dict)

        conv_dict: Dict[str, Any] = {
            "id": conv.id,
            "title": conv.title,
            "source": conv.source,
            "model": conv.model,
            "tags": conv.tags,
            "created_at": str(conv.created_at),
            "updated_at": str(conv.updated_at),
            "messages": messages,
        }
        if include_notes and db:
            conv_dict["notes"] = conv_notes
        data.append(conv_dict)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
