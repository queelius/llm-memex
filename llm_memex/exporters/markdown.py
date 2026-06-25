"""Export conversations as Markdown."""
from typing import Dict, List

from llm_memex.models import Conversation


def export(conversations: List[Conversation], path: str, **kwargs) -> None:
    """Export conversations to a Markdown file.

    Each conversation becomes a top-level heading with messages listed below.
    For branching conversations, each path is separated by a horizontal rule.

    Parameters
    ----------
    include_notes : bool
        Include notes as blockquotes (default True).
    db : Database | None
        Database instance for querying notes.
    """
    include_notes = kwargs.get("include_notes", True)
    db = kwargs.get("db")

    lines: List[str] = []
    for conv in conversations:
        lines.append(f"# {conv.title or conv.id}\n")
        if conv.source:
            lines.append(f"*Source: {conv.source}*\n")

        # Conversation-level notes
        conv_notes: List[Dict] = []
        msg_notes: Dict[str, List[Dict]] = {}
        if include_notes and db:
            all_notes = db.get_notes(conversation_id=conv.id)
            for note in all_notes:
                if note["target_kind"] == "conversation":
                    conv_notes.append(note)
                elif note["target_kind"] == "message" and note["message_id"]:
                    msg_notes.setdefault(note["message_id"], []).append(note)

        for note in conv_notes:
            lines.append(f"> **Note:** {note['text']}\n")

        for path_msgs in conv.get_all_paths():
            for msg in path_msgs:
                text = msg.get_content_md()
                lines.append(f"**{msg.role}**: {text}\n")
                for note in msg_notes.get(msg.id, []):
                    lines.append(f"> **Note:** {note['text']}\n")
            lines.append("---\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
