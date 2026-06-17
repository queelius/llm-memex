"""Import Claude Code conversation transcripts (JSONL session files).

Claude Code stores sessions as JSONL at ~/.claude/projects/<path>/<uuid>.jsonl.
Each line is a JSON event (user, assistant, progress, file-history-snapshot, etc.).

This importer captures the **conversation skeleton** only: real user requests and
assistant text responses. Tool use, thinking blocks, progress events, and file
snapshots are stripped. Metadata records the mode as "conversation_only" so a
future full-fidelity importer (claude_code_full) can coexist.
"""
from pathlib import Path
from typing import List

from llm_memex.models import Conversation, Message, text_block
from llm_memex.importers._claude_code_common import (
    detect,  # re-exported as public API
    parse_iso as _parse_iso,
    parse_records,
    extract_session_metadata,
    build_conversation,
    import_directory,
    MessageTreeBuilder,
)


def import_path(path: str) -> List[Conversation]:
    """Import a Claude Code JSONL session file or directory of sessions.

    If path is a directory, finds all .jsonl files recursively and imports each.
    """
    p = Path(path)
    if p.is_dir():
        return import_directory(path, _import_single)
    return _import_single(path)


def _import_single(path: str) -> List[Conversation]:
    """Import a single Claude Code JSONL session as a conversation.

    Extracts only the conversation skeleton:
    - User messages: type="user", userType="external", not sidechain, plain text
    - Assistant messages: type="assistant", not sidechain, text blocks only
    """
    records = parse_records(path)
    if not records:
        return []

    meta = extract_session_metadata(records)
    session_id = meta["session_id"]
    if session_id is None:
        return []

    # Filter to conversation turns
    messages = []
    # Reconstruct the real message tree from parentUuid rather than linear
    # chaining, so rewind/edit branches are not mis-parented (LLM-5). This
    # importer skips even more records, so the walk-to-nearest-imported-
    # ancestor logic matters more here.
    tree = MessageTreeBuilder(records)

    for rec in records:
        event_type = rec.get("type")

        # Skip sidechain messages
        if rec.get("isSidechain"):
            continue

        msg = rec.get("message") or {}
        if not isinstance(msg, dict):
            msg = {}

        if event_type == "user" and rec.get("userType") == "external":
            content = msg.get("content")
            # Only import plain text user messages (not tool_result arrays)
            if isinstance(content, str) and content.strip():
                msg_id = rec.get("uuid", f"user_{len(messages)}")
                messages.append(Message(
                    id=msg_id,
                    role="user",
                    content=[text_block(content)],
                    parent_id=tree.parent_for(rec, msg_id),
                    created_at=_parse_iso(rec["timestamp"]) if rec.get("timestamp") else None,
                ))
                tree.mark(msg_id)

        elif event_type == "assistant":
            content_blocks = msg.get("content", [])
            if not isinstance(content_blocks, list):
                continue
            # Extract only text blocks
            texts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        texts.append(text)
            # Skip assistant turns with no text (pure tool_use)
            if not texts:
                continue
            joined = "\n\n".join(texts)
            msg_id = rec.get("uuid", f"asst_{len(messages)}")
            messages.append(Message(
                id=msg_id,
                role="assistant",
                content=[text_block(joined)],
                parent_id=tree.parent_for(rec, msg_id),
                model=msg.get("model"),
                created_at=_parse_iso(rec["timestamp"]) if rec.get("timestamp") else None,
            ))
            tree.mark(msg_id)

    # If no actual messages were extracted, don't create a conversation
    if not messages:
        return []

    conv = build_conversation(
        messages, meta, path,
        source_type="claude_code",
        importer_mode="conversation_only",
    )
    return [conv]
