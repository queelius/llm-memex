"""Import OpenAI Codex CLI sessions (rollout JSONL files).

Codex writes session threads to
``<codex home>/sessions/YYYY/MM/DD/rollout-<timestamp>-<thread id>.jsonl``.
Two on-disk layouts exist, told apart by the records' structure (never by
the CLI version, which changes every few days and does not line up with
layout changes):

- ``codex-rollout/1`` (2025, before the envelope): a header line
  ``{"id", "timestamp", "instructions", "git"?}`` followed by bare response
  items (``{"type": "message", "role", "content"}``, ``function_call``,
  ``reasoning``, ...) and ``{"record_type": "state"}`` markers.
- ``codex-rollout/2``: every line is an envelope
  ``{"timestamp", "type", "payload", "ordinal"?}``; the first is
  ``session_meta`` (thread id, cwd, git, cli_version, source).

This importer keeps the **conversation skeleton**, like ``claude_code``:
user and assistant messages (text and images). Tool calls, reasoning
(encrypted), developer instructions, inter-agent messages, token counts and
UI events are skipped. ``event_msg`` records in particular are a UI event
stream that repeats the transcript (``item_completed`` carries every agent
and user message again), so only ``response_item`` messages are read.

Every record type the importer does not read is counted rather than
rejected, so a new Codex release adds to the "skipped" tally in the log
instead of failing the import.

**Split threads are trees.** A thread can span several rollout files
(segments), each opening with its own ``session_meta`` for the same thread
id. A later segment resumes the thread from an earlier point: its
``session_meta`` ordinal is the fork point, and the earlier segment's
records past it (which differ) are a branch the thread left behind. So each
segment is a linear chain, and a later segment's first message is parented
to the last message before the fork in the segment it resumed from. Both
branches are kept, as for Claude Code rewinds. Message ids are
``<thread>:<segment>:<ordinal>``; segment numbers follow the files'
chronological names, so ids stay stable as segments are added.

Subagent threads (``session_meta.source.subagent``) are imported as their
own conversations, tagged ``codex-subagent``. When their parent thread is
part of the same import they get ``parent_conversation_id``; the parent's
thread id is always kept in ``metadata.parent_thread_id``.
"""
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from llm_memex.importers import parse_timestamp
from llm_memex.models import Conversation, Message, media_block, text_block

logger = logging.getLogger(__name__)

# Bump when a change makes this importer read the same source differently;
# conversations stamped with an older revision are re-imported (cli.py).
IMPORTER_REVISION = 1

LAYOUT_1 = "codex-rollout/1"
LAYOUT_2 = "codex-rollout/2"

_TITLE_MAX = 80


# ── Reading ────────────────────────────────────────────────────


def _iter_records(path: str, skipped: Counter) -> Iterator[Dict[str, Any]]:
    """Stream a rollout file's JSON objects, one line at a time.

    Rollout files reach gigabytes (a long thread's tool output), so nothing
    holds a whole file in memory. Codex writes some strings with raw control
    characters, which strict JSON rejects, so lines are parsed with
    ``strict=False``. A line that still fails (e.g. the half-written last
    line of a live session) is counted and skipped.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line, strict=False)
            except ValueError:
                skipped["<unparseable line>"] += 1
                continue
            if isinstance(rec, dict):
                yield rec
            else:
                skipped["<unparseable line>"] += 1


def _first_record(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line, strict=False)
                except ValueError:
                    return None
                return rec if isinstance(rec, dict) else None
    except OSError:
        return None
    return None


def _layout(first: Dict[str, Any]) -> Optional[str]:
    """Recognize the layout from the first record's structure."""
    if first.get("type") == "session_meta" and isinstance(first.get("payload"), dict):
        return LAYOUT_2
    if (
        "type" not in first
        and "payload" not in first
        and isinstance(first.get("id"), str)
        and "timestamp" in first
        and "instructions" in first
    ):
        return LAYOUT_1
    return None


def _header_meta(first: Dict[str, Any]) -> Dict[str, Any]:
    """Thread metadata from a file's first record (either layout)."""
    return first["payload"] if _layout(first) == LAYOUT_2 else first


def detect_file(path: str) -> bool:
    if not path.endswith(".jsonl"):
        return False
    first = _first_record(path)
    return first is not None and _layout(first) is not None


def detect(path: str) -> bool:
    """A Codex rollout file, or a directory containing any."""
    p = Path(path)
    if p.is_dir():
        return any(detect_file(str(f)) for f in p.rglob("*.jsonl"))
    return detect_file(path)


# ── Thread titles (session_index.jsonl) ────────────────────────


def _codex_home(path: Path) -> Optional[Path]:
    """The directory holding ``sessions/`` for a rollout file, if any."""
    for parent in path.resolve().parents:
        if parent.name == "sessions":
            return parent.parent
    return None


@lru_cache(maxsize=8)
def _thread_names(index_path: str, mtime_ns: int) -> Dict[str, str]:
    """``{thread id: name}`` from session_index.jsonl; later lines win.

    Cached per (path, mtime), so a long-lived process sees renames.
    """
    names: Dict[str, str] = {}
    for rec in _iter_records(index_path, Counter()):
        tid, name = rec.get("id"), rec.get("thread_name")
        if isinstance(tid, str) and isinstance(name, str) and name.strip():
            names[tid] = name.strip()
    return names


def _thread_name(path: str, thread_id: str) -> Optional[str]:
    home = _codex_home(Path(path))
    if home is None:
        return None
    index = home / "session_index.jsonl"
    try:
        mtime = index.stat().st_mtime_ns
    except OSError:
        return None
    return _thread_names(str(index), mtime).get(thread_id)


# ── Content ────────────────────────────────────────────────────


def _is_injected_context(text: str) -> bool:
    """True when a user text part is one client-injected element.

    Codex sends context it generates itself (environment, AGENTS.md
    instructions, plugin hints, heartbeats, ...) as user-role text wrapped
    in a single XML-style element. The set of tag names grows release by
    release, so this checks the shape (the part is exactly one
    ``<name ...>...</name>`` element), not a list of names.
    """
    t = text.strip()
    if not (t.startswith("<") and t.endswith(">")):
        return False
    name_len = 0
    for ch in t[1:]:
        if ch.isalnum() or ch in "_-":
            name_len += 1
        else:
            break
    if name_len == 0:
        return False
    return t.endswith(f"</{t[1:1 + name_len]}>")


def _data_url_block(url: str) -> Optional[Dict[str, Any]]:
    """A media block from an RFC 2397 ``data:<type>;base64,<data>`` URL."""
    if not url.startswith("data:"):
        return None
    header, sep, data = url[len("data:"):].partition(",")
    if not sep or not data:
        return None
    params = header.split(";")
    if "base64" not in params[1:]:
        return None
    return media_block(params[0] or "application/octet-stream", data=data)


def _content_blocks(role: str, parts: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not isinstance(parts, list):
        return blocks
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text"):
            text = part.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if role == "user" and _is_injected_context(text):
                continue
            blocks.append(text_block(text))
        elif ptype == "input_image":
            url = part.get("image_url")
            block = _data_url_block(url) if isinstance(url, str) else None
            if block is not None:
                blocks.append(block)
    return blocks


# ── Segments ───────────────────────────────────────────────────


@dataclass
class _Segment:
    """One rollout file of a thread."""

    index: int
    path: str
    layout: str
    meta: Dict[str, Any]
    # Ordinal of the opening session_meta: where this segment resumed the
    # thread. None for layout 1 (no ordinals) and the opening segment.
    fork: Optional[int] = None
    first_ordinal: Optional[int] = None
    # (ordinal or None, message id) for each message imported from it.
    messages: List[Tuple[Optional[int], str]] = field(default_factory=list)


def _load_segments(paths: List[str], skipped: Counter) -> List[_Segment]:
    segments: List[_Segment] = []
    layout: Optional[str] = None
    for path in paths:
        first = _first_record(path)
        if first is None:
            continue
        file_layout = _layout(first)
        if file_layout is None or (layout is not None and file_layout != layout):
            skipped["<file in another layout>"] += 1
            continue
        layout = file_layout
        ordinal = first.get("ordinal")
        ordinal = ordinal if isinstance(ordinal, int) else None
        segments.append(_Segment(
            index=len(segments),
            path=path,
            layout=file_layout,
            meta=_header_meta(first),
            fork=ordinal if segments else None,
            first_ordinal=ordinal,
        ))
    return segments


def _fork_parent(segments: List[_Segment], k: int) -> Optional[str]:
    """Parent message id for the first message of segment *k*.

    The last message before segment k's fork point, in the latest earlier
    segment that reaches back that far. Without ordinals (layout 1), or with
    nothing before the fork in that segment, fall back along the chain.
    """
    if k == 0:
        return None
    seg = segments[k]
    if seg.fork is None:
        prev = segments[k - 1]
        return prev.messages[-1][1] if prev.messages else _fork_parent(segments, k - 1)
    for j in range(k - 1, -1, -1):
        cand = segments[j]
        if cand.first_ordinal is not None and cand.first_ordinal >= seg.fork:
            continue  # starts at or after the fork; cannot hold its parent
        before = [mid for ordinal, mid in cand.messages
                  if ordinal is not None and ordinal < seg.fork]
        if before:
            return before[-1]
        return _fork_parent(segments, j)
    return None


def _items(seg: _Segment, skipped: Counter, context: Dict[str, Any]) -> Iterator[
    Tuple[Dict[str, Any], Optional[str], Optional[int], str]
]:
    """(response item, timestamp, ordinal, key) for each item in a segment."""
    records = _iter_records(seg.path, skipped)
    next(records, None)  # the header, already read into seg.meta
    if seg.layout == LAYOUT_1:
        for i, rec in enumerate(records, start=1):
            if "record_type" in rec:
                skipped[f"record_type:{rec.get('record_type')}"] += 1
                continue
            yield rec, None, None, str(i)
        return
    for i, rec in enumerate(records, start=1):
        rtype = rec.get("type")
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            skipped[str(rtype)] += 1
            continue
        if rtype == "session_meta":
            continue
        if rtype == "turn_context":
            # Carries the model in effect for the turns that follow.
            if isinstance(payload.get("model"), str):
                context["model"] = payload["model"]
                context.setdefault("first_model", payload["model"])
            if isinstance(payload.get("cwd"), str):
                context.setdefault("cwd", payload["cwd"])
            continue
        if rtype != "response_item":
            skipped[str(rtype)] += 1
            continue
        ordinal = rec.get("ordinal")
        ordinal = ordinal if isinstance(ordinal, int) else None
        yield payload, rec.get("timestamp"), ordinal, str(ordinal if ordinal is not None else i)


# ── Import ─────────────────────────────────────────────────────


def _subagent(source: Any) -> Optional[Dict[str, Any]]:
    """The ``source.subagent`` description, or None for a top-level thread."""
    if isinstance(source, dict) and isinstance(source.get("subagent"), dict):
        return source["subagent"]
    return None


def _subagent_title(sub: Optional[Dict[str, Any]]) -> Optional[str]:
    """"Subagent Ada (explorer)" for a spawned thread with no human turn.

    A spawned subagent receives its task as an inter-agent message, not a
    user message, so the first-human-message fallback finds nothing.
    """
    spawn = sub.get("thread_spawn") if sub else None
    if not isinstance(spawn, dict):
        return None
    nick, role = spawn.get("agent_nickname"), spawn.get("agent_role")
    if not isinstance(nick, str) or not nick:
        return None
    return f"Subagent {nick}" + (f" ({role})" if isinstance(role, str) and role else "")


def _import_thread(paths: List[str], skipped: Counter) -> Optional[Conversation]:
    segments = _load_segments(paths, skipped)
    if not segments:
        return None
    meta = segments[0].meta
    thread_id = meta.get("id")
    if not isinstance(thread_id, str) or not thread_id:
        return None

    context: Dict[str, Any] = {}
    messages: List[Message] = []
    first_user_text: Optional[str] = None
    last_ts = meta.get("timestamp")
    for seg in segments:
        parent: Optional[str] = _fork_parent(segments, seg.index)
        for item, ts, ordinal, key in _items(seg, skipped, context):
            if ts:
                last_ts = ts
            role = item.get("role")
            if item.get("type") != "message":
                skipped[f"response_item/{item.get('type')}"] += 1
                continue
            if role not in ("user", "assistant"):
                skipped[f"message/{role}"] += 1
                continue
            blocks = _content_blocks(role, item.get("content"))
            if not blocks:
                skipped[f"message/{role}:no content"] += 1
                continue
            if role == "user" and first_user_text is None:
                first_user_text = next(
                    (b["text"] for b in blocks if b.get("type") == "text"), None
                )
            msg_id = f"{thread_id}:{seg.index}:{key}"
            messages.append(Message(
                id=msg_id,
                role=role,
                content=blocks,
                parent_id=parent,
                model=context.get("model") if role == "assistant" else None,
                created_at=parse_timestamp(ts) if ts else None,
            ))
            seg.messages.append((ordinal, msg_id))
            parent = msg_id

    if not messages:
        return None

    created = parse_timestamp(meta.get("timestamp"))
    updated = parse_timestamp(last_ts) or created
    if created is None:
        created = updated
    if created is None:
        return None

    title = _thread_name(segments[0].path, thread_id)
    if title is None and first_user_text:
        first_line = first_user_text.strip().splitlines()[0]
        title = first_line[:_TITLE_MAX].rstrip() + ("…" if len(first_line) > _TITLE_MAX else "")
    if title is None:
        title = _subagent_title(_subagent(meta.get("source")))
    conv = Conversation(
        id=thread_id,
        title=title or "Untitled Codex session",
        source="codex",
        model=context.get("first_model"),
        created_at=created,
        updated_at=updated,
        tags=["codex"],
    )
    for msg in messages:
        conv.add_message(msg)

    conv.metadata["importer_mode"] = "conversation_only"
    git = meta.get("git") if isinstance(meta.get("git"), dict) else {}
    # cwd/branch: where the thread started; client_version: the newest
    # client that wrote to it (the latest segment's header).
    context_fields = {
        "cwd": meta.get("cwd") or context.get("cwd"),
        "git_branch": git.get("branch"),
        "client_version": segments[-1].meta.get("cli_version"),
        "originator": meta.get("originator"),
    }
    for k, v in context_fields.items():
        if isinstance(v, str) and v:
            conv.metadata[k] = v

    sub = _subagent(meta.get("source"))
    if sub is not None:
        conv.tags.append("codex-subagent")
        spawn = sub.get("thread_spawn")
        if isinstance(spawn, dict):
            conv.metadata["subagent_kind"] = "thread_spawn"
            for key in ("parent_thread_id", "agent_nickname", "agent_role"):
                if isinstance(spawn.get(key), str) and spawn[key]:
                    conv.metadata[key] = spawn[key]
        else:
            # e.g. {"other": "guardian"}: a reviewer thread with no parent link.
            kind = next(iter(sub.values()), None)
            conv.metadata["subagent_kind"] = kind if isinstance(kind, str) else "other"

    if len(segments) > 1:
        conv.metadata["source_files"] = [s.path for s in segments]
    conv.metadata["_provenance"] = {
        "source_type": "codex",
        "source_file": segments[0].path,
        "source_id": thread_id,
        "source_format": segments[0].layout,
    }
    return conv


def _parents_first(convs: List[Conversation]) -> List[Conversation]:
    """Link subagents to parents present in this import; parents come first.

    ``parent_conversation_id`` is a foreign key, so a child may only point at
    a conversation saved before it. A parent missing from the import (its
    file was pruned, or a single child file was imported) leaves the link
    unset; ``metadata.parent_thread_id`` still records it.
    """
    by_id = {c.id: c for c in convs}
    for c in convs:
        parent = c.metadata.get("parent_thread_id")
        if parent in by_id and parent != c.id:
            c.parent_conversation_id = parent
    ordered: List[Conversation] = []
    placed: set = set()

    def place(c: Conversation, trail: frozenset) -> None:
        if c.id in placed:
            return
        parent = by_id.get(c.parent_conversation_id or "")
        if parent is not None and parent.id in trail:
            c.parent_conversation_id = None  # cycle: drop the link, keep the row
        elif parent is not None:
            place(parent, trail | {c.id})
        placed.add(c.id)
        ordered.append(c)

    for c in convs:
        place(c, frozenset())
    return ordered


def _thread_id(path: str) -> Optional[str]:
    first = _first_record(path)
    if first is None or _layout(first) is None:
        return None
    tid = _header_meta(first).get("id")
    return tid if isinstance(tid, str) and tid else None


def import_path(path: str) -> List[Conversation]:
    """Import one rollout file, or every rollout file under a directory.

    In a directory, a thread's segment files are assembled into one
    conversation. A single file is imported on its own (one segment of a
    split thread yields the part of the thread that file holds).
    """
    p = Path(path)
    skipped: Counter = Counter()
    # Rollout names start with the creation timestamp, so name order is
    # chronological across the YYYY/MM/DD directories too.
    files = sorted(p.rglob("*.jsonl"), key=lambda f: f.name) if p.is_dir() else [p]
    threads: Dict[str, List[str]] = {}
    for f in files:
        tid = _thread_id(str(f))
        if tid is not None:
            threads.setdefault(tid, []).append(str(f))
    convs: List[Conversation] = []
    for tid, paths in threads.items():
        try:
            conv = _import_thread(paths, skipped)
        except (OSError, ValueError) as e:
            logger.warning("Skipping thread %s (%s): %s", tid, paths[0], e)
            continue
        if conv is not None:
            convs.append(conv)
    if skipped:
        logger.info(
            "Codex import: %d conversation(s); skipped records by type: %s",
            len(convs), dict(skipped.most_common()),
        )
    return _parents_first(convs)
