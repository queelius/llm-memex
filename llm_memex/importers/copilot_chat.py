"""Import GitHub Copilot Chat sessions from VS Code.

VS Code keeps one file per chat session under its user data directory:
``<User>/workspaceStorage/<hash>/chatSessions/<session id>.json[l]`` for
sessions opened in a workspace, ``<User>/globalStorage/
emptyWindowChatSessions/`` for sessions opened in an empty window. The
workspace a session belongs to is named by the sibling
``<hash>/workspace.json`` (``{"folder": <uri>}`` or ``{"workspace": <uri>}``).

Two on-disk layouts exist, told apart by structure (never by VS Code
version):

- ``vscode-chat-session/1``: a ``.json`` file holding the whole serialized
  session ``{"version", "sessionId", "requests": [...], ...}``.
- ``vscode-chat-session/2``: a ``.jsonl`` operation log. The first line is
  ``{"kind": 0, "v": <session>}`` (a full snapshot, the same shape as
  layout 1); each later line edits it at the key path ``k``: ``kind`` 1 sets
  ``v``, 2 pushes the items ``v`` onto an array (truncating it to length
  ``i`` first when given), 3 deletes. These are the replay rules of VS
  Code's own reader. An unknown operation kind makes the session
  unreadable, as it does for VS Code, rather than replaying it wrong.

VS Code may keep both layouts of one session (``<id>.json`` and
``<id>.jsonl``), and its storage migration copies files between
directories, so sessions are deduplicated by id: the copy with the latest
``lastMessageDate`` wins, then the most recently written file. Importing
part of the storage (one file, one workspace) still reads every copy of
its sessions under the ``User`` directory.

This importer keeps the **conversation skeleton**, like ``codex`` and
``claude_code``: each request becomes a user message and its response an
assistant message. A response is a list of parts. Markdown, inline file
references and vulnerability-annotated markdown are inline text and are
joined as written; every other part (tool invocations, edits, thinking,
progress, confirmations, ...) is skipped and counted, and marks a
paragraph break, since the response's prose continues after it.
"""
import json
import logging
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from llm_memex.importers import parse_timestamp
from llm_memex.models import Conversation, Message, text_block

logger = logging.getLogger(__name__)

# Bump when a change makes this importer read the same source differently;
# conversations stamped with an older revision are re-imported (cli.py).
IMPORTER_REVISION = 1

LAYOUT_SNAPSHOT = "vscode-chat-session/1"
LAYOUT_LOG = "vscode-chat-session/2"

# The directories VS Code stores chat session files in.
SESSION_DIRS = ("chatSessions", "emptyWindowChatSessions")

_TITLE_MAX = 80

# Operation kinds of the layout-2 log.
_OP_SNAPSHOT, _OP_SET, _OP_PUSH, _OP_DELETE = 0, 1, 2, 3

# Response parts that are running text. A plain markdown part is serialized
# without a kind (``{"value": ...}``).
_INLINE_KINDS = {None, "markdownContent", "inlineReference", "markdownVuln"}


# ── Reading ────────────────────────────────────────────────────


def _is_session(obj: Any) -> bool:
    """A serialized chat session (the layout-1 file, the layout-2 snapshot)."""
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("sessionId"), str)
        and isinstance(obj.get("requests"), list)
        and "version" in obj
    )


def _first_line(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    return json.loads(line)
    except (OSError, ValueError):
        return None
    return None


def _layout(path: Path) -> Optional[str]:
    """Recognize a session file's layout from its content."""
    if path.suffix == ".jsonl":
        first = _first_line(path)
        if (isinstance(first, dict) and first.get("kind") == _OP_SNAPSHOT
                and _is_session(first.get("v"))):
            return LAYOUT_LOG
        return None
    if path.suffix == ".json":
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                obj = json.load(f)
        except (OSError, ValueError):
            return None
        return LAYOUT_SNAPSHOT if _is_session(obj) else None
    return None


def _session_files(root: Path) -> Iterator[Path]:
    """Candidate session files: ``*.json[l]`` inside VS Code session dirs."""
    if root.is_file():
        yield root
        return
    for name in SESSION_DIRS:
        dirs = [root] if root.name == name else root.rglob(name)
        for d in dirs:
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.is_file() and f.suffix in (".json", ".jsonl"):
                        yield f


def detect(path: str) -> bool:
    """A Copilot Chat session file, or a directory holding any."""
    return any(_layout(f) is not None for f in _session_files(Path(path)))


def _key(container: Any, key: Any) -> Any:
    """A JavaScript property key as Python indexes *container* with it.

    JS reads ``arr["0"]`` as ``arr[0]``; anything else on an array is not
    an element, so it is rejected rather than guessed.
    """
    if isinstance(container, list):
        if isinstance(key, str) and key.isdigit():
            return int(key)
        if isinstance(key, int) and not isinstance(key, bool) and key >= 0:
            return key
        raise ValueError(f"not an array index: {key!r}")
    if isinstance(container, dict):
        return key if isinstance(key, str) else str(key)
    raise ValueError(f"cannot index {type(container).__name__} with {key!r}")


def _get(container: Any, key: Any) -> Any:
    key = _key(container, key)
    if isinstance(container, list):
        return container[key] if key < len(container) else None
    return container.get(key)


def _put(container: Any, key: Any, value: Any) -> None:
    """``container[key] = value``; an array grows to fit, as in JS."""
    key = _key(container, key)
    if isinstance(container, list) and key >= len(container):
        container.extend([None] * (key + 1 - len(container)))
    container[key] = value


def _length(value: Any) -> int:
    """A valid JS array length (a non-negative integral number)."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value < 0 or value != int(value)):
        raise ValueError(f"invalid array length {value!r}")
    return int(value)


def _apply(state: Any, op: Dict[str, Any]) -> None:
    """Apply one layout-2 operation to *state* in place (VS Code's rules)."""
    kind, keys = op.get("kind"), op.get("k")
    if kind not in (_OP_SET, _OP_PUSH, _OP_DELETE):
        raise ValueError(f"unknown log operation kind {kind!r}")
    if not isinstance(keys, list):
        raise ValueError(f"log operation without a key path: {op!r}")
    if not keys:
        return
    parent = state
    for key in keys[:-1]:
        parent = parent[_key(parent, key)]
    last = keys[-1]
    if kind == _OP_SET:
        _put(parent, last, op.get("v"))
    elif kind == _OP_DELETE:
        if isinstance(parent, dict):
            parent.pop(_key(parent, last), None)
        else:
            _put(parent, last, None)
    else:  # _OP_PUSH
        items = list(_get(parent, last) or [])
        if op.get("i") is not None:
            cut = _length(op["i"])
            del items[cut:]
            items.extend([None] * (cut - len(items)))
        items.extend(op.get("v") or [])
        _put(parent, last, items)


def _replay(path: Path, skipped: Counter) -> Dict[str, Any]:
    """Rebuild a session from its layout-2 operation log.

    A half-written last line (VS Code was mid-append) is counted and
    dropped. An unreadable line *followed by* more operations means the
    log is corrupt, so the session is rejected rather than replayed wrong.
    """
    state: Any = None
    torn = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                op = json.loads(line)
            except ValueError:
                torn = True
                continue
            if torn:
                raise ValueError("unreadable line inside the operation log")
            if not isinstance(op, dict):
                raise ValueError("operation is not an object")
            if op.get("kind") == _OP_SNAPSHOT:
                state = op.get("v")
            elif state is None:
                raise ValueError("operation log has no initial snapshot")
            else:
                try:
                    _apply(state, op)
                except (KeyError, IndexError, TypeError, AttributeError) as e:
                    raise ValueError(f"operation does not fit the session: {e}") from e
    if torn:
        skipped["<truncated last line>"] += 1
    if not _is_session(state):
        raise ValueError("operation log does not rebuild a session")
    return state


def _load(path: Path, skipped: Counter) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``(layout, session)`` for a session file, None for any other file.

    Raises ValueError for a session log that cannot be replayed.
    """
    if path.suffix == ".jsonl":
        return (LAYOUT_LOG, _replay(path, skipped)) if _layout(path) else None
    if path.suffix != ".json":
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    return (LAYOUT_SNAPSHOT, obj) if _is_session(obj) else None


# ── Session context ────────────────────────────────────────────


def _ms(value: Any) -> Optional[Any]:
    """VS Code stores epoch milliseconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return parse_timestamp(value / 1000)


def _workspace(path: Path) -> Dict[str, str]:
    """``workspace`` (URI) and, for a local folder, ``cwd`` (path).

    Read from ``<hash>/workspace.json`` beside the ``chatSessions`` dir.
    """
    ws_file = path.parent.parent / "workspace.json"
    try:
        with open(ws_file, encoding="utf-8") as f:
            ws = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(ws, dict):
        return {}
    uri = ws.get("folder") or ws.get("workspace")
    if not isinstance(uri, str) or not uri:
        return {}
    out = {"workspace": uri}
    parsed = urlparse(uri)
    if parsed.scheme == "file" and "folder" in ws:
        out["cwd"] = unquote(parsed.path)
    return out


# ── Responses ──────────────────────────────────────────────────


def _uri_path(obj: Any) -> Optional[str]:
    """The path of a serialized URI, a Location (``{uri, range}``) or a
    symbol (``{name, location}``)."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("path"), str):
        return obj["path"]
    for key in ("uri", "location"):
        inner = _uri_path(obj.get(key))
        if inner:
            return inner
    return None


def _reference_text(part: Dict[str, Any]) -> str:
    ref = part.get("inlineReference")
    name = part.get("name")
    if not isinstance(name, str) or not name:
        name = ref.get("name") if isinstance(ref, dict) else None
    if not isinstance(name, str) or not name:
        path = _uri_path(ref)
        name = PurePosixPath(path).name if path else None
    return f"`{name}`" if name else ""


def _markdown_value(value: Any) -> str:
    """A MarkdownString (``{"value": ...}``) or a bare string."""
    if isinstance(value, dict):
        value = value.get("value")
    return value if isinstance(value, str) else ""


def _response_text(parts: Any, skipped: Counter) -> str:
    """Join a response's inline parts; any other part breaks the paragraph."""
    paragraphs: List[str] = []
    current: List[str] = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            skipped["response/<not an object>"] += 1
            continue
        kind = part.get("kind")
        if kind not in _INLINE_KINDS:
            skipped[f"response/{kind}"] += 1
            if current:
                paragraphs.append("".join(current))
                current = []
            continue
        if kind == "inlineReference":
            current.append(_reference_text(part))
        elif kind is None:
            current.append(_markdown_value(part))
        else:  # markdownContent, markdownVuln
            current.append(_markdown_value(part.get("content")))
    if current:
        paragraphs.append("".join(current))
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


# ── Assembly ───────────────────────────────────────────────────


def _title(session: Dict[str, Any], first_prompt: Optional[str]) -> str:
    custom = session.get("customTitle")
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    if first_prompt:
        line = first_prompt.strip().splitlines()[0]
        return line[:_TITLE_MAX].rstrip() + ("…" if len(line) > _TITLE_MAX else "")
    return "Untitled Copilot session"


def _model(request: Dict[str, Any]) -> Optional[str]:
    model = request.get("modelId")
    return model if isinstance(model, str) and model else None


def _build(session: Dict[str, Any], path: Path, layout: str,
           skipped: Counter) -> Optional[Conversation]:
    session_id = session["sessionId"]
    messages: List[Message] = []
    parent: Optional[str] = None
    first_prompt: Optional[str] = None
    first_model: Optional[str] = None
    last_ts = None
    for n, req in enumerate(session["requests"]):
        if not isinstance(req, dict):
            skipped["request/<not an object>"] += 1
            continue
        request_id = req.get("requestId")
        if not isinstance(request_id, str) or not request_id:
            ts = req.get("timestamp")
            ok = isinstance(ts, (int, float)) and not isinstance(ts, bool)
            request_id = f"t{ts}" if ok else f"r{n}"
        asked = _ms(req.get("timestamp"))
        agent = req.get("agent") if isinstance(req.get("agent"), dict) else {}
        msg_meta = {"agent": agent["id"]} if isinstance(agent.get("id"), str) else {}

        prompt = (req.get("message") or {}).get("text") if isinstance(req.get("message"), dict) else None
        if isinstance(prompt, str) and prompt.strip():
            msg_id = f"{session_id}:{request_id}"
            messages.append(Message(
                id=msg_id, role="user", content=[text_block(prompt)],
                parent_id=parent, created_at=asked, metadata=dict(msg_meta),
            ))
            parent = msg_id
            if first_prompt is None:
                first_prompt = prompt
        else:
            skipped["request/no prompt text"] += 1

        text = _response_text(req.get("response"), skipped)
        result = req.get("result") if isinstance(req.get("result"), dict) else {}
        if not text:
            skipped["response/no text"] += 1
            last_ts = asked or last_ts
            continue
        elapsed = (result.get("timings") or {}).get("totalElapsed") if isinstance(result.get("timings"), dict) else None
        answered = asked
        if asked is not None and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            answered = _ms(req.get("timestamp") + elapsed) or asked
        meta = dict(msg_meta)
        if req.get("isCanceled") is True:
            meta["canceled"] = True
        error = result.get("errorDetails")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            meta["error"] = error["message"]
        model = _model(req)
        first_model = first_model or model
        response_id = req.get("responseId") if isinstance(req.get("responseId"), str) else f"{request_id}:response"
        msg_id = f"{session_id}:{response_id}"
        messages.append(Message(
            id=msg_id, role="assistant", content=[text_block(text)],
            parent_id=parent, model=model, created_at=answered, metadata=meta,
        ))
        parent = msg_id
        last_ts = answered or last_ts

    if not messages:
        return None
    created = _ms(session.get("creationDate")) or messages[0].created_at
    stamps = [t for t in (_ms(session.get("lastMessageDate")), last_ts) if t is not None]
    updated = max(stamps) if stamps else created
    if created is None:
        created = updated
    if created is None:
        return None
    conv = Conversation(
        id=session_id,
        title=_title(session, first_prompt),
        source="copilot",
        model=first_model,
        created_at=created,
        updated_at=max(created, updated),
        tags=["copilot"],
    )
    for msg in messages:
        conv.add_message(msg)
    conv.metadata["importer_mode"] = "conversation_only"
    conv.metadata["client"] = "vscode"
    conv.metadata.update(_workspace(path))
    for key, field in (("session_version", "version"),
                       ("initial_location", "initialLocation"),
                       ("responder", "responderUsername")):
        value = session.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "":
            conv.metadata[key] = value
    conv.metadata["_provenance"] = {
        "source_type": "copilot_chat",
        "source_file": str(path),
        "source_id": session_id,
        "source_format": layout,
    }
    return conv


def _rank(session: Dict[str, Any], path: Path) -> Tuple[float, int]:
    """The current copy of a session: latest ``lastMessageDate``, then the
    most recently written file (a deletion or an edit may leave
    ``lastMessageDate`` alone, but VS Code rewrites the live file)."""
    last = session.get("lastMessageDate")
    ok = isinstance(last, (int, float)) and not isinstance(last, bool)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return (float(last) if ok else 0.0, mtime)


def _user_dir(p: Path) -> Optional[Path]:
    """VS Code's ``User`` directory above *p*, if *p* lies in its storage."""
    for d in [p, *p.parents]:
        if d.name in ("workspaceStorage", "globalStorage"):
            return d.parent
    return None


def _candidates(p: Path) -> List[Path]:
    """Files to read for an import of *p*.

    A session can have copies in several places (both layouts side by
    side, and storage migrations), so importing part of VS Code's storage
    also reads every other copy of the sessions found there. Otherwise a
    stale copy imported on its own would replace the current one.
    """
    found = list(_session_files(p))
    user = _user_dir(p.resolve())
    if user is None or p.resolve() == user:
        return found
    stems = {f.stem for f in found}
    everywhere = [f for f in _session_files(user) if f.stem in stems]
    return sorted(set(found) | set(everywhere))


def import_path(path: str) -> List[Conversation]:
    """Import one session file, or every session under a directory."""
    skipped: Counter = Counter()
    best: Dict[str, Tuple[Tuple[float, int], Dict[str, Any], Path, str]] = {}
    for f in _candidates(Path(path)):
        try:
            loaded = _load(f, skipped)
        except Exception as e:  # one unreadable session must not sink the batch
            skipped["<session failed>"] += 1
            logger.warning("Skipping Copilot Chat session %s: %s", f, e)
            continue
        if loaded is None:
            continue
        layout, session = loaded
        rank = _rank(session, f)
        sid = session["sessionId"]
        if sid not in best or rank > best[sid][0]:
            if sid in best:
                skipped["<duplicate session file>"] += 1
            best[sid] = (rank, session, f, layout)
        else:
            skipped["<duplicate session file>"] += 1
    convs: List[Conversation] = []
    for sid, (_, session, f, layout) in best.items():
        if not session["requests"]:  # the current copy was cleared
            skipped["<empty session>"] += 1
            continue
        try:
            conv = _build(session, f, layout, skipped)
        except Exception as e:
            skipped["<session failed>"] += 1
            logger.warning("Skipping Copilot Chat session %s: %s", f, e)
            continue
        if conv is not None:
            convs.append(conv)
    if skipped:
        logger.info(
            "Copilot Chat import: %d conversation(s); skipped by type: %s",
            len(convs), dict(skipped.most_common()),
        )
    return convs
