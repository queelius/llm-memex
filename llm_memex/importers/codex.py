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
UI events are skipped. ``event_msg`` records are a UI event stream that
repeats the transcript, so only ``response_item`` messages are imported;
the stream's user messages are still read, as the record of what was
actually submitted as a user turn (see :func:`_is_client_context`).

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
``<thread>:<segment>:<key>``, where ``<segment>`` is the segment's own
``session_meta`` timestamp, so ids do not depend on which files an import
happened to read. Importing one file, or a subdirectory, still assembles
every segment of its threads from the Codex home, so a thread is never
replaced by a part of itself.

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
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

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


def _thread_id(path: str) -> Optional[str]:
    first = _first_record(path)
    if first is None or _layout(first) is None:
        return None
    tid = _header_meta(first).get("id")
    return tid if isinstance(tid, str) and tid else None


# ── Codex home: thread titles and sibling segments ─────────────


def _codex_home(path: Path) -> Optional[Path]:
    """The directory holding ``sessions/`` for a rollout path, if any."""
    p = path.resolve()
    for candidate in (p, *p.parents):
        if candidate.name == "sessions":
            return candidate.parent
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


def _rollouts_by_thread(files: List[Path]) -> Dict[str, List[str]]:
    """Group rollout files by thread id, each list in chronological order.

    Rollout names start with the creation timestamp, so name order is
    chronological across the YYYY/MM/DD directories.
    """
    threads: Dict[str, List[str]] = {}
    for f in sorted(files, key=lambda f: f.name):
        tid = _thread_id(str(f))
        if tid is not None:
            threads.setdefault(tid, []).append(str(f))
    return threads


# ── Content ────────────────────────────────────────────────────


def _block_element_name(text: str) -> Optional[str]:
    """The tag name if *text* has the shape of a client context block.

    Codex wraps context it generates itself in a block element: optionally
    one markdown heading line, then ``<name ...>`` followed immediately by a
    newline, and ``</name>`` closing at the very end with same-name tags
    balanced in between (measured on every such part in 1,031 real rollout
    files). Inline elements, such as a pasted ``<div>...</div>`` snippet or
    the ``<image ...>`` wrappers around an image, do not have this shape.
    """
    t = text.strip()
    if t.startswith("#"):
        heading, sep, rest = t.partition("\n")
        if not sep:
            return None
        t = rest.lstrip()
    if not t.startswith("<"):
        return None
    name_len = 0
    for ch in t[1:]:
        if ch.isalnum() or ch in "_-":
            name_len += 1
        else:
            break
    if name_len == 0:
        return None
    name = t[1:1 + name_len]
    close = f"</{name}>"
    open_end = t.find(">")
    if open_end < 0 or t[open_end - 1] == "/" or not t.endswith(close):
        return None
    if t[open_end + 1:open_end + 2] not in ("\n", "\r"):
        return None
    # The first <name> must be closed by the final </name>, not earlier.
    depth = 0
    i = 0
    while i < len(t):
        if t.startswith(close, i):
            depth -= 1
            if depth == 0 and i + len(close) != len(t):
                return None
            i += len(close)
            continue
        if t.startswith(f"<{name}", i) and (
            t[i + 1 + name_len:i + 2 + name_len] in (">", " ", "\n", "\t", "\r")
        ):
            depth += 1
        i += 1
    return name if depth == 0 else None


def _is_client_context(text: str, submitted: Set[str]) -> bool:
    """True when a user-role text part is context the client generated.

    Requires the block-element shape; and when the segment carries the UI
    stream's record of submitted user messages (*submitted*, from newer
    CLIs), the part must also be absent from it. Codex's own record wins:
    an answer to the agent's question is wrapped in such a block but is
    something the user submitted, and stays.
    """
    if _block_element_name(text) is None:
        return False
    stripped = text.strip()
    return not any(stripped == s or stripped in s for s in submitted)


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


def _submitted_texts(payload: Dict[str, Any]) -> List[str]:
    """User-message texts recorded by the UI event stream, if *payload* is one."""
    ptype = payload.get("type")
    if ptype == "user_message" and isinstance(payload.get("message"), str):
        return [payload["message"].strip()]
    if ptype == "item_completed":
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "UserMessage":
            return [
                c["text"].strip() for c in item.get("content") or []
                if isinstance(c, dict) and isinstance(c.get("text"), str)
            ]
    return []


# ── Segments ───────────────────────────────────────────────────


@dataclass
class _Entry:
    """A user/assistant message as read, before client context is removed."""

    role: str
    parts: Any
    ts: Optional[str]
    ordinal: Optional[int]
    key: str
    model: Optional[str]


@dataclass
class _Segment:
    """One rollout file of a thread."""

    path: str
    layout: str
    meta: Dict[str, Any]
    # Stable segment key for message ids: its own session_meta timestamp.
    key: str
    # Ordinal of the opening session_meta: where this segment resumed the
    # thread. None for the opening segment and for layout 1 (no ordinals).
    fork: Optional[int] = None
    first_ordinal: Optional[int] = None
    entries: List[_Entry] = field(default_factory=list)
    submitted: Set[str] = field(default_factory=set)
    # (ordinal or None, message id) for each message imported from it.
    messages: List[Tuple[Optional[int], str]] = field(default_factory=list)


def _load_segments(paths: List[str], skipped: Counter) -> List[_Segment]:
    segments: List[_Segment] = []
    keys: Counter = Counter()
    for path in paths:
        first = _first_record(path)
        layout = _layout(first) if first is not None else None
        if first is None or layout is None:
            skipped["<unrecognized file>"] += 1
            continue
        meta = _header_meta(first)
        ordinal = first.get("ordinal")
        ordinal = ordinal if isinstance(ordinal, int) else None
        ts = meta.get("timestamp") if isinstance(meta.get("timestamp"), str) else None
        key = ts or Path(path).stem
        keys[key] += 1
        if keys[key] > 1:  # two segments opened in the same instant
            key = f"{key}~{keys[key]}"
        segments.append(_Segment(
            path=path, layout=layout, meta=meta, key=key,
            fork=ordinal if segments else None, first_ordinal=ordinal,
        ))
    return segments


def _read_segment(seg: _Segment, skipped: Counter, context: Dict[str, Any]) -> None:
    """Stream one segment into ``seg.entries`` and ``seg.submitted``."""
    records = _iter_records(seg.path, skipped)
    next(records, None)  # the header, already read into seg.meta
    for i, rec in enumerate(records, start=1):
        if seg.layout == LAYOUT_1:
            if "record_type" in rec:
                skipped[f"record_type:{rec.get('record_type')}"] += 1
                continue
            item, ts, ordinal = rec, None, None
        else:
            rtype, payload = rec.get("type"), rec.get("payload")
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
            if rtype == "event_msg":
                seg.submitted.update(_submitted_texts(payload))
                skipped["event_msg"] += 1
                continue
            if rtype != "response_item":
                skipped[str(rtype)] += 1
                continue
            item, ts = payload, rec.get("timestamp")
            ordinal = rec.get("ordinal")
            ordinal = ordinal if isinstance(ordinal, int) else None
        if ts:
            context["last_ts"] = ts
        if item.get("type") != "message":
            skipped[f"response_item/{item.get('type')}"] += 1
            continue
        role = item.get("role")
        if role not in ("user", "assistant"):
            skipped[f"message/{role}"] += 1
            continue
        seg.entries.append(_Entry(
            role=role,
            parts=item.get("content"),
            ts=ts if isinstance(ts, str) else None,
            ordinal=ordinal,
            # Distinct prefixes: an ordinal and a line number never collide.
            key=f"o{ordinal}" if ordinal is not None else f"l{i}",
            model=context.get("model") if role == "assistant" else None,
        ))


def _content_blocks(entry: _Entry, submitted: Set[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not isinstance(entry.parts, list):
        return blocks
    for part in entry.parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text"):
            text = part.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if entry.role == "user" and _is_client_context(text, submitted):
                continue
            blocks.append(text_block(text))
        elif ptype == "input_image":
            url = part.get("image_url")
            block = _data_url_block(url) if isinstance(url, str) else None
            if block is not None:
                blocks.append(block)
    return blocks


def _fork_parent(segments: List[_Segment], k: int) -> Optional[str]:
    """Parent message id for the first message of segment *k*.

    The last message before segment k's fork point, in the latest earlier
    segment that reaches back that far. Without ordinals (layout 1, or a
    segment whose records carry none), the previous segment's last message;
    with nothing before the fork in that segment, fall back along the chain.
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
        if cand.messages and all(o is None for o, _ in cand.messages):
            return cand.messages[-1][1]
        before = [mid for o, mid in cand.messages if o is not None and o < seg.fork]
        if before:
            return before[-1]
        return _fork_parent(segments, j)
    return None


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


def _title_text(blocks: List[Dict[str, Any]]) -> Optional[str]:
    """First text block that reads as prose (not a tag fragment)."""
    for b in blocks:
        text = b.get("text") if b.get("type") == "text" else None
        if isinstance(text, str) and not text.lstrip().startswith("<"):
            return text
    return None


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
    seen_ids: Set[str] = set()
    first_user_text: Optional[str] = None
    for k, seg in enumerate(segments):
        _read_segment(seg, skipped, context)
        parent = _fork_parent(segments, k)
        for entry in seg.entries:
            blocks = _content_blocks(entry, seg.submitted)
            if not blocks:
                skipped[f"message/{entry.role}:no content"] += 1
                continue
            if entry.role == "user" and first_user_text is None:
                first_user_text = _title_text(blocks)
            msg_id = f"{thread_id}:{seg.key}:{entry.key}"
            if msg_id in seen_ids:  # a repeated ordinal; never overwrite
                n = 2
                while f"{msg_id}~{n}" in seen_ids:
                    n += 1
                msg_id = f"{msg_id}~{n}"
                skipped["<duplicate message key>"] += 1
            seen_ids.add(msg_id)
            messages.append(Message(
                id=msg_id,
                role=entry.role,
                content=blocks,
                parent_id=parent if parent != msg_id else None,
                model=entry.model,
                created_at=parse_timestamp(entry.ts) if entry.ts else None,
            ))
            seg.messages.append((entry.ordinal, msg_id))
            parent = msg_id
        seg.entries = []  # release; only message ids are needed from here on

    if not messages:
        return None

    created = parse_timestamp(meta.get("timestamp"))
    updated = parse_timestamp(context.get("last_ts")) or created
    if created is None:
        created = updated
    if created is None:
        return None

    title = _thread_name(segments[0].path, thread_id)
    if title is None and first_user_text:
        first_line = first_user_text.strip().splitlines()[0]
        title = first_line[:_TITLE_MAX].rstrip() + ("…" if len(first_line) > _TITLE_MAX else "")
    sub = _subagent(meta.get("source"))
    if title is None:
        title = _subagent_title(sub)
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
    for key, value in context_fields.items():
        if isinstance(value, str) and value:
            conv.metadata[key] = value

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
        # One layout per thread in practice; a thread resumed across a
        # layout change records both ("codex-rollout/1+codex-rollout/2").
        "source_format": "+".join(sorted({s.layout for s in segments})),
    }
    return conv


def _parents_first(convs: List[Conversation]) -> List[Conversation]:
    """Link subagents to parents present in this import; parents come first.

    ``parent_conversation_id`` is a foreign key, so a child may only point at
    a conversation saved before it. A parent missing from the import (its
    file was pruned) leaves the link unset here; ``metadata.parent_thread_id``
    still records it, and a link made by an earlier import is kept (the DB
    upsert does not clear it).
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


def import_path(path: str) -> List[Conversation]:
    """Import one rollout file, or every rollout file under a directory.

    Threads are always assembled whole: when *path* lies inside a Codex
    home (``<home>/sessions/...``), the other segments of each thread found
    under *path* are gathered from the whole ``sessions`` tree.
    """
    p = Path(path)
    skipped: Counter = Counter()
    files = list(p.rglob("*.jsonl")) if p.is_dir() else [p]
    threads = _rollouts_by_thread(files)
    home = _codex_home(p)
    if home is not None and threads:
        sessions = (home / "sessions").resolve()
        if p.resolve() != sessions:
            everywhere = _rollouts_by_thread(list(sessions.rglob("*.jsonl")))
            threads = {tid: everywhere.get(tid, segs) for tid, segs in threads.items()}
    convs: List[Conversation] = []
    for tid, paths in threads.items():
        try:
            conv = _import_thread(paths, skipped)
        except Exception as e:  # one malformed thread must not sink the batch
            skipped["<thread failed>"] += 1
            logger.warning("Skipping Codex thread %s (%s): %s", tid, paths[0], e)
            continue
        if conv is not None:
            convs.append(conv)
    if skipped:
        logger.info(
            "Codex import: %d conversation(s); skipped records by type: %s",
            len(convs), dict(skipped.most_common()),
        )
    return _parents_first(convs)
