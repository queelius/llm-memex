"""Import Claude Code conversation transcripts with full fidelity (JSONL session files).

Claude Code stores sessions as JSONL at ~/.claude/projects/<path>/<uuid>.jsonl.
Each line is a JSON event (user, assistant, progress, file-history-snapshot, etc.).

Unlike the conversation_only importer (claude_code), this preserves **all** content
blocks: tool_use, tool_result, and thinking blocks are kept alongside text. This
enables auditing tool usage, replaying sessions, and analyzing thinking patterns.
"""
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from llm_memex.models import (
    Conversation, Message,
    text_block, thinking_block, tool_use_block, tool_result_block,
)
from llm_memex.importers._claude_code_common import (
    detect,  # re-exported as public API
    parse_iso as _parse_iso,
    parse_records,
    extract_session_metadata,
    build_conversation,
    import_directory,
    find_subagent_files,
    extract_agent_id,
)


def import_path(path: str) -> List[Conversation]:
    """Import a Claude Code JSONL session file or directory of sessions.

    If path is a directory, finds all .jsonl files recursively and imports each.
    For each parent session, also imports its subagent files with parent links.
    Parents are returned before children to satisfy FK ordering.
    """
    p = Path(path)
    if p.is_dir():
        return _import_directory_with_subagents(path)
    return _import_single_with_subagents(path)


def _convert_assistant_block(block):
    """Convert a Claude Code assistant content block to a memex content block.

    Returns a content block dict, or None if the block should be skipped.
    """
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype == "text":
        text = block.get("text", "").strip()
        return text_block(text) if text else None
    elif btype == "thinking":
        # Claude Code uses {"type": "thinking", "thinking": "..."} but
        # memex convention is {"type": "thinking", "text": "..."}
        text = block.get("thinking", "") or block.get("text", "")
        return thinking_block(text) if text else None
    elif btype == "tool_use":
        return tool_use_block(
            id=block.get("id", ""),
            name=block.get("name", ""),
            input=block.get("input", {}),
        )
    return None


def _convert_tool_result_block(block):
    """Convert a Claude Code tool_result block to a memex tool_result block.

    Returns a content block dict, or None if the block should be skipped.
    """
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return None
    return tool_result_block(
        tool_use_id=block.get("tool_use_id", ""),
        content=block.get("content"),
        is_error=block.get("is_error", False),
    )


def _import_single(path: str, ignore_sidechain: bool = False) -> List[Conversation]:
    """Import a single Claude Code JSONL session with full content fidelity.

    Preserves all content block types:
    - User messages: text (external) and tool_result (internal) messages
    - Assistant messages: text, thinking, and tool_use blocks
    Skips: progress, file-history-snapshot, queue-operation events.

    Args:
        path: Path to JSONL file.
        ignore_sidechain: If True, don't skip isSidechain records. Used for
            subagent files where all records are sidechain.
    """
    records = parse_records(path)
    if not records:
        return []

    meta = extract_session_metadata(records)
    session_id = meta["session_id"]
    if session_id is None:
        return []

    messages = []
    parent_id = None

    for rec in records:
        event_type = rec.get("type")

        if not ignore_sidechain and rec.get("isSidechain"):
            continue

        msg = rec.get("message") or {}
        if not isinstance(msg, dict):
            msg = {}

        if event_type == "user":
            content = msg.get("content")
            blocks = []

            if isinstance(content, str):
                # External user message with plain text
                if not content.strip():
                    continue
                blocks = [text_block(content)]
            elif isinstance(content, list):
                # Internal user message with tool_result blocks
                for block in content:
                    converted = _convert_tool_result_block(block)
                    if converted is not None:
                        blocks.append(converted)

            if not blocks:
                continue

            msg_id = rec.get("uuid", f"user_{len(messages)}")
            messages.append(Message(
                id=msg_id,
                role="user",
                content=blocks,
                parent_id=parent_id,
                created_at=_parse_iso(rec["timestamp"]) if rec.get("timestamp") else None,
            ))
            parent_id = msg_id

        elif event_type == "assistant":
            content_blocks = msg.get("content", [])
            if not isinstance(content_blocks, list):
                continue

            blocks = []
            for block in content_blocks:
                converted = _convert_assistant_block(block)
                if converted is not None:
                    blocks.append(converted)

            if not blocks:
                continue

            msg_id = rec.get("uuid", f"asst_{len(messages)}")
            messages.append(Message(
                id=msg_id,
                role="assistant",
                content=blocks,
                parent_id=parent_id,
                model=msg.get("model"),
                created_at=_parse_iso(rec["timestamp"]) if rec.get("timestamp") else None,
            ))
            parent_id = msg_id

    if not messages:
        return []

    conv = build_conversation(
        messages, meta, path,
        source_type="claude_code_full",
        importer_mode="full",
    )
    return [conv]


# Agent type → extra tags mapping
_AGENT_TYPE_TAGS = {
    "compact": "claude-code-compact",
    "acompact": "claude-code-compact",
    "prompt_suggestion": "claude-code-prompt-suggestion",
    "aprompt_suggestion": "claude-code-prompt-suggestion",
}


def _import_subagent(subagent_path: Path, parent_session_id: str) -> Optional[Conversation]:
    """Import a single subagent JSONL file with parent link.

    Returns None if the file yields no messages.
    """
    agent_id = extract_agent_id(subagent_path)
    convs = _import_single(str(subagent_path), ignore_sidechain=True)
    if not convs:
        return None

    conv = convs[0]
    # Deterministic ID encoding the relationship
    conv.id = f"{parent_session_id}:{agent_id}"
    conv.parent_conversation_id = parent_session_id
    conv.metadata["agent_id"] = agent_id

    # Update provenance
    conv.metadata["_provenance"]["source_id"] = conv.id

    # Tag by agent type
    conv.tags.append("claude-code-agent")
    extra_tag = _AGENT_TYPE_TAGS.get(agent_id)
    if extra_tag and extra_tag not in conv.tags:
        conv.tags.append(extra_tag)

    return conv


def _attach_subagents(result: List[Conversation], parent: Conversation, source_file: str) -> None:
    """Import subagent files for a parent session and append to result list."""
    for sub_path in find_subagent_files(source_file):
        try:
            child = _import_subagent(sub_path, parent.id)
            if child:
                result.append(child)
        except Exception as e:
            logger.warning("Skipping subagent %s: %s", sub_path, e)


def _import_single_with_subagents(path: str) -> List[Conversation]:
    """Import a single session file plus its subagents."""
    parents = _import_single(path)
    if not parents:
        return []

    parent = parents[0]
    result = [parent]
    _attach_subagents(result, parent, path)
    return result


def _import_directory_with_subagents(path: str) -> List[Conversation]:
    """Import a directory of sessions, including subagents for each parent.

    Parents are returned before children to satisfy FK ordering.
    """
    parents = import_directory(path, _import_single, skip_subagents=True)

    result = []
    for parent in parents:
        result.append(parent)
        source_file = parent.metadata.get("_provenance", {}).get("source_file")
        if source_file:
            _attach_subagents(result, parent, source_file)

    return result
