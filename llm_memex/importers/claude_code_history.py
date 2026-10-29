"""Import Claude Code's prompt history (``~/.claude/history.jsonl``).

Claude Code appends every prompt the user submits to ``history.jsonl``:
``{"display", "pastedContents", "timestamp" (epoch ms), "project",
"sessionId"?}``. Unlike the session transcripts under ``~/.claude/projects``
(which Claude Code prunes after ``cleanupPeriodDays``), this file is not
pruned, so it is the only surviving record of sessions whose transcripts
are gone. It holds the user's side only: no responses, no tool use.

Two record shapes exist, told apart by structure:

- ``claude-code-history/1``: no ``sessionId`` (Claude Code before about
  November 2025). Prompts cannot be tied to a session, so they are grouped
  by project and UTC calendar day.
- ``claude-code-history/2``: with ``sessionId``. Prompts are grouped by
  session.

Each group becomes one conversation of user messages, in time order, tagged
``prompt-history``, with ``metadata.importer_mode = "prompt_history"``.
Conversation ids are ``history:<session id>`` and
``history:<project hash>:<YYYY-MM-DD>``, distinct from the transcript
importers' ids (the bare session id), so a prompt-only record never
replaces a full transcript; ``metadata.session_id`` links the two.

Pasted text is kept: ``display`` shows ``[Pasted text #N ...]`` and the
pasted content follows as its own text block, inline (``content``) or from
``<claude home>/paste-cache/<contentHash>.txt``. A paste whose cache file is
gone is counted, not fatal.
"""
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from llm_memex.importers import parse_timestamp
from llm_memex.models import Conversation, Message, text_block

logger = logging.getLogger(__name__)

# Bump when a change makes this importer read the same source differently;
# conversations stamped with an older revision are re-imported (cli.py).
IMPORTER_REVISION = 1

LAYOUT_PROJECT_DAY = "claude-code-history/1"
LAYOUT_SESSION = "claude-code-history/2"

_TITLE_MAX = 80


def _is_entry(rec: Any) -> bool:
    return (
        isinstance(rec, dict)
        and isinstance(rec.get("display"), str)
        and isinstance(rec.get("project"), str)
        and isinstance(rec.get("timestamp"), (int, float))
        and not isinstance(rec.get("timestamp"), bool)
    )


_DETECT_LINES = 10


def detect_file(path: Path) -> bool:
    """A JSONL file with a history entry among its first few records.

    Not just the first: a leading record of a kind this importer does not
    know (counted and skipped on import) must not hide the file.
    """
    if path.suffix != ".jsonl":
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            seen = 0
            for line in f:
                if not line.strip():
                    continue
                seen += 1
                if seen > _DETECT_LINES:
                    return False
                try:
                    if _is_entry(json.loads(line)):
                        return True
                except ValueError:
                    continue
    except OSError:
        return False
    return False


def detect(path: str) -> bool:
    """A prompt history file, or a directory with one at its top level.

    Only the top level: a recursive scan of ``~/.claude`` would read the
    first line of every session transcript under ``projects/``.
    """
    p = Path(path)
    if p.is_dir():
        return any(detect_file(f) for f in p.glob("*.jsonl"))
    return detect_file(p)


def _entries(path: Path, skipped: Counter) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                skipped["<unparseable line>"] += 1
                continue
            if _is_entry(rec):
                yield rec
            else:
                skipped["<not a history entry>"] += 1


def _when(rec: Dict[str, Any]) -> Optional[datetime]:
    return parse_timestamp(rec["timestamp"] / 1000)  # epoch milliseconds


def _group(rec: Dict[str, Any], when: datetime) -> Tuple[str, str]:
    """``(conversation id, layout)`` for an entry."""
    session = rec.get("sessionId")
    if isinstance(session, str) and session:
        return f"history:{session}", LAYOUT_SESSION
    project_hash = hashlib.sha256(rec["project"].encode("utf-8")).hexdigest()[:12]
    return f"history:{project_hash}:{when.date().isoformat()}", LAYOUT_PROJECT_DAY


def _pasted_blocks(rec: Dict[str, Any], paste_cache: Path,
                   skipped: Counter) -> List[Dict[str, Any]]:
    pasted = rec.get("pastedContents")
    if not isinstance(pasted, dict):
        return []
    blocks = []
    # Keys are paste numbers ("1", "2", ...), matching "[Pasted text #N]".
    for key in sorted(pasted, key=lambda k: (0, int(k), k) if k.isdigit() else (1, 0, k)):
        item = pasted[key]
        if not isinstance(item, dict) or item.get("type") != "text":
            skipped[f"paste/{item.get('type') if isinstance(item, dict) else '<not an object>'}"] += 1
            continue
        content = item.get("content")
        if not isinstance(content, str):
            digest = item.get("contentHash")
            content = None
            if isinstance(digest, str) and digest and "/" not in digest and "\\" not in digest:
                try:
                    content = (paste_cache / f"{digest}.txt").read_text(encoding="utf-8", errors="replace")
                except OSError:
                    content = None
            if content is None:
                skipped["paste/<cache file missing>"] += 1
                continue
        if content.strip():
            blocks.append(text_block(content))
    return blocks


def _title(prompts: List[str]) -> str:
    """First prompt that is not a slash command, else the first prompt."""
    chosen = next((p for p in prompts if p.strip() and not p.lstrip().startswith("/")),
                  next((p for p in prompts if p.strip()), ""))
    if not chosen.strip():
        return "Claude Code prompts"
    line = chosen.strip().splitlines()[0]
    return line[:_TITLE_MAX].rstrip() + ("…" if len(line) > _TITLE_MAX else "")


def import_path(path: str) -> List[Conversation]:
    """Import a ``history.jsonl`` file (or the one at a directory's top)."""
    p = Path(path)
    files = sorted(f for f in p.glob("*.jsonl") if detect_file(f)) if p.is_dir() else [p]
    skipped: Counter = Counter()
    convs = _import_files(files, skipped)
    if skipped:
        logger.info(
            "Claude Code history import: %d conversation(s); skipped: %s",
            len(convs), dict(skipped.most_common()),
        )
    return convs


def _import_files(paths: List[Path], skipped: Counter) -> List[Conversation]:
    """Group the entries of every file together.

    Several history files (a copy beside the live one) hold overlapping
    entries; grouping each file on its own would save the same conversation
    once per file, the last one winning. Entries are a union, and an entry
    present in several files is kept once.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    seen_entries: set = set()
    for path in paths:
        paste_cache = path.parent / "paste-cache"
        for rec in _entries(path, skipped):
            when = _when(rec)
            if when is None:
                skipped["<bad timestamp>"] += 1
                continue
            ident = json.dumps(rec, sort_keys=True)
            if ident in seen_entries:
                skipped["<duplicate entry>"] += 1
                continue
            seen_entries.add(ident)
            conv_id, layout = _group(rec, when)
            g = groups.setdefault(conv_id, {"layout": layout, "project": rec["project"],
                                            "session": rec.get("sessionId"),
                                            "source": path, "items": []})
            g["items"].append((when, rec, paste_cache))

    convs = []
    for conv_id, g in groups.items():
        items = sorted(g["items"], key=lambda item: item[0])  # stable: file order on ties
        conv = Conversation(
            id=conv_id,
            title=_title([rec["display"] for _, rec, _ in items]),
            source="claude_code",
            created_at=items[0][0],
            updated_at=items[-1][0],
            tags=["claude-code", "prompt-history"],
        )
        parent: Optional[str] = None
        seen: set = set()
        for when, rec, paste_cache in items:
            blocks = ([text_block(rec["display"])] if rec["display"].strip() else [])
            blocks += _pasted_blocks(rec, paste_cache, skipped)
            if not blocks:
                skipped["<empty prompt>"] += 1
                continue
            msg_id = f"{conv_id}:{rec['timestamp']}"
            if msg_id in seen:
                n = 2
                while f"{msg_id}~{n}" in seen:
                    n += 1
                msg_id = f"{msg_id}~{n}"
            seen.add(msg_id)
            conv.add_message(Message(id=msg_id, role="user", content=blocks,
                                     parent_id=parent, created_at=when))
            parent = msg_id
        if not conv.messages:
            continue
        conv.metadata["importer_mode"] = "prompt_history"
        conv.metadata["cwd"] = g["project"]
        if g["layout"] == LAYOUT_SESSION:
            conv.metadata["session_id"] = g["session"]
            conv.metadata["grouping"] = "session"
        else:
            conv.metadata["grouping"] = "project-day"
        conv.metadata["_provenance"] = {
            "source_type": "claude_code_history",
            "source_file": str(g["source"]),
            "source_id": g["session"] if g["layout"] == LAYOUT_SESSION else conv_id,
            "source_format": g["layout"],
        }
        convs.append(conv)
    return convs
