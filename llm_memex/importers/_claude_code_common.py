"""Shared helpers for Claude Code importers (conversation_only and full).

Claude Code stores sessions as JSONL at ~/.claude/projects/<path>/<uuid>.jsonl.
Each line is a JSON event (user, assistant, progress, file-history-snapshot, etc.).
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from llm_memex.models import Conversation, Message

logger = logging.getLogger(__name__)

# Event types that Claude Code uses (for detection)
KNOWN_EVENT_TYPES = {
    "user", "assistant", "system", "progress",
    "file-history-snapshot", "queue-operation",
}


def detect_file(path: str) -> bool:
    """Check if a single file is a Claude Code JSONL session transcript.

    Checks the first few records for a sessionId and a known event type.
    Some sessions start with file-history-snapshot records before the first
    conversation event, so we scan up to 10 lines.
    """
    try:
        if not path.endswith(".jsonl"):
            return False
        has_known_type = False
        has_session_id = False
        # errors="replace" tolerates corrupt (non-UTF8) bytes; the per-line JSON
        # try/except then skips any line the replacement made unparseable.
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                line = line.strip()
                if not line or line[0] == '\x00':
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if record.get("type") in KNOWN_EVENT_TYPES:
                    has_known_type = True
                if record.get("sessionId"):
                    has_session_id = True
                if has_known_type and has_session_id:
                    return True
        return False
    except (IOError, OSError):
        return False


def detect(path: str) -> bool:
    """Check if path is a Claude Code JSONL session or directory of sessions."""
    p = Path(path)
    if p.is_dir():
        return any(detect_file(str(jsonl)) for jsonl in p.rglob("*.jsonl"))
    return detect_file(path)


def parse_iso(ts: Any) -> Optional[datetime]:
    """Parse ISO 8601 timestamp, handling trailing Z.

    Tolerates non-string input (a malformed export may carry a numeric or null
    timestamp): returns None instead of raising AttributeError / ValueError.
    """
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def slug_to_title(slug: str) -> str:
    """Convert a slug like 'immutable-splashing-thompson' to title case."""
    return slug.replace("-", " ").title()


def parse_records(path: str) -> List[Dict[str, Any]]:
    """Parse a JSONL file into a list of records, skipping corrupted lines."""
    records = []
    # errors="replace" tolerates corrupt (non-UTF8) bytes; the per-line JSON
    # try/except then skips any line the replacement made unparseable.
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '\x00':
                continue
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return records


def extract_session_metadata(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract session-level metadata from records.

    Returns dict with keys: session_id, slug, first_ts, last_ts, model.
    """
    session_id = None
    slug = None
    first_ts = None
    last_ts = None
    model = None

    for rec in records:
        if session_id is None and rec.get("sessionId"):
            session_id = rec["sessionId"]
        if slug is None and rec.get("slug"):
            slug = rec["slug"]
        if rec.get("timestamp"):
            ts = rec["timestamp"]
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        if model is None and rec.get("type") == "assistant":
            msg = rec.get("message")
            if isinstance(msg, dict):
                model = msg.get("model")

    return {
        "session_id": session_id,
        "slug": slug,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "model": model,
    }


def build_conversation(
    messages: List[Message],
    meta: Dict[str, Any],
    path: str,
    source_type: str,
    importer_mode: str,
) -> Conversation:
    """Assemble a Conversation from extracted messages and session metadata.

    Shared by both Claude Code importers. The only per-importer differences are
    source_type (provenance) and importer_mode (metadata); the conversation
    fields, title-from-slug rule, and provenance shape are identical.
    """
    session_id = meta["session_id"]
    title = slug_to_title(meta["slug"]) if meta["slug"] else "Untitled Session"

    now = datetime.now(timezone.utc)
    # parse_iso returns None for a non-string / malformed timestamp; fall back to now.
    created_at = (parse_iso(meta["first_ts"]) if meta["first_ts"] else None) or now
    updated_at = (parse_iso(meta["last_ts"]) if meta["last_ts"] else None) or now
    conv = Conversation(
        id=session_id,
        title=title,
        source="claude_code",
        model=meta["model"],
        created_at=created_at,
        updated_at=updated_at,
        tags=["claude-code"],
    )

    for msg in messages:
        conv.add_message(msg)

    conv.metadata["_provenance"] = {
        "source_type": source_type,
        "source_file": path,
        "source_id": session_id,
    }
    conv.metadata["importer_mode"] = importer_mode

    return conv


def import_directory(
    path: str,
    import_fn: Callable[[str], List[Conversation]],
    skip_subagents: bool = True,
) -> List[Conversation]:
    """Walk a directory, detect Claude Code JSONL files, and import each.

    Args:
        path: Directory path to scan.
        import_fn: Function that imports a single JSONL file, returning List[Conversation].
        skip_subagents: If True (default), skip files in subagents/ directories.
    """
    p = Path(path)
    convs = []
    n_detected = 0
    n_subagent = 0
    n_empty = 0
    n_errored = 0
    for jsonl in sorted(p.rglob("*.jsonl")):
        if skip_subagents and any(part == "subagents" for part in jsonl.parts):
            n_subagent += 1
            continue
        if not detect_file(str(jsonl)):
            continue
        n_detected += 1
        try:
            result = import_fn(str(jsonl))
        except Exception as e:
            n_errored += 1
            logger.warning("Skipping %s: %s", jsonl, e)
            continue
        if result:
            convs.extend(result)
        else:
            n_empty += 1
    logger.info(
        "Directory import: %d detected, %d with messages, "
        "%d empty, %d subagent skipped, %d errored",
        n_detected, len(convs), n_empty, n_subagent, n_errored,
    )
    return convs


def find_subagent_files(session_path: str) -> List[Path]:
    """Find subagent JSONL files for a session.

    Claude Code stores subagent files at:
        <project>/<uuid>/subagents/agent-*.jsonl
    where the session file is:
        <project>/<uuid>.jsonl
    """
    p = Path(session_path)
    subagents_dir = p.parent / p.stem / "subagents"
    if not subagents_dir.is_dir():
        return []
    return sorted(subagents_dir.glob("*.jsonl"))


def extract_agent_id(subagent_path: Path) -> str:
    """Extract the agent ID from a subagent file path (stem of the filename)."""
    return subagent_path.stem
