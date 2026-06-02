"""Export conversations as an arkiv archive.

Writes the bundle as a directory, a ``.zip``, or a ``.tar.gz`` depending on
the file extension of ``path``. All three layouts contain the same files:

- ``conversations.jsonl``  -- one record per message (arkiv universal format)
- ``README.md``            -- YAML frontmatter (ECHO self-description)
- ``schema.yaml``          -- metadata key statistics

Compression choice prioritizes longevity: ``.zip`` and ``.tar.gz`` are both
ubiquitous (every OS and scripting language has supported them for 30+ years).
"""
import gzip
import io
import json
import os
import tarfile
import zipfile
from datetime import date
from typing import Any, Dict, List

from llm_memex.models import Conversation


def _detect_compression(path: str) -> str:
    """Infer output format from *path*'s extension.

    Returns one of: ``"zip"``, ``"tar.gz"``, ``"dir"``.
    """
    lower = path.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "tar.gz"
    return "dir"


def export(conversations: List[Conversation], path: str, **kwargs) -> None:
    """Export conversations to an arkiv archive.

    Output format is inferred from *path*'s extension:

    - ``path.zip``           -> single zip file
    - ``path.tar.gz``/`.tgz` -> single gzip-compressed tarball
    - any other path         -> directory containing the three files

    Parameters
    ----------
    include_notes : bool
        Include message-level notes in record metadata (default True).
    db : Database | None
        Database instance for querying notes.
    """
    include_notes = kwargs.get("include_notes", True)
    db = kwargs.get("db")
    records = _build_records(conversations, include_notes=include_notes, db=db)

    jsonl_bytes = _records_to_jsonl_bytes(records)
    schema_bytes = _schema_yaml_bytes(
        _compute_schema(records), record_count=len(records)
    )
    readme_bytes = _readme_bytes(num_conversations=len(conversations))

    kind = _detect_compression(path)
    if kind == "zip":
        _write_zip(path, jsonl_bytes, schema_bytes, readme_bytes)
    elif kind == "tar.gz":
        _write_tar_gz(path, jsonl_bytes, schema_bytes, readme_bytes)
    else:
        os.makedirs(path, exist_ok=True)
        _write_file(os.path.join(path, "conversations.jsonl"), jsonl_bytes)
        _write_file(os.path.join(path, "schema.yaml"), schema_bytes)
        _write_file(os.path.join(path, "README.md"), readme_bytes)


def _records_to_jsonl_bytes(records: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    for rec in records:
        buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return buf.getvalue().encode("utf-8")


def _write_file(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _write_zip(
    path: str, jsonl: bytes, schema_yaml: bytes, readme: bytes
) -> None:
    """Write the three bundle files into a single .zip archive."""
    # DEFLATE is the universally-supported compressor; good text ratio.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("conversations.jsonl", jsonl)
        zf.writestr("schema.yaml", schema_yaml)
        zf.writestr("README.md", readme)


def _write_tar_gz(
    path: str, jsonl: bytes, schema_yaml: bytes, readme: bytes
) -> None:
    """Write the three bundle files into a single .tar.gz archive.

    The output is byte-reproducible: ``tarfile.open(path, "w:gz")`` would stamp
    the current wall-clock time into the gzip header, so two exports of
    identical data would differ. We instead drive a ``gzip.GzipFile`` with
    ``mtime=0`` (fixed gzip header) and set every ``TarInfo.mtime`` to 0 (fixed
    per-member timestamps), making the bundle deterministic. The .zip path is
    already deterministic (zipfile uses a fixed default date), so this brings
    .tar.gz to parity.
    """
    with open(path, "wb") as raw:
        # mtime=0 zeroes the timestamp field in the gzip header. filename=""
        # suppresses the FNAME field, which GzipFile would otherwise populate
        # from ``raw.name`` (the output path), leaking the file name into the
        # header and breaking byte-reproducibility across differently-named
        # outputs of identical data.
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tf:
                for name, data in (
                    ("conversations.jsonl", jsonl),
                    ("schema.yaml", schema_yaml),
                    ("README.md", readme),
                ):
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    info.mtime = 0  # fixed per-member timestamp
                    tf.addfile(info, io.BytesIO(data))


def _build_records(
    conversations: List[Conversation],
    *,
    include_notes: bool = True,
    db: Any = None,
) -> List[Dict[str, Any]]:
    """Convert conversations to arkiv records (one per message)."""
    records = []
    for conv in conversations:
        # Pre-fetch message-level notes for this conversation
        msg_notes: Dict[str, List[Dict[str, Any]]] = {}
        if include_notes and db:
            for note in db.get_notes(
                conversation_id=conv.id, target_kind="message"
            ):
                mid = note.get("message_id")
                if mid:
                    msg_notes.setdefault(mid, []).append(note)

        conv_tags = conv.tags
        for msg in conv.messages.values():
            text = msg.get_text()
            if not text:
                continue

            timestamp = (
                msg.created_at.isoformat()
                if msg.created_at
                else conv.created_at.isoformat()
            )

            metadata: Dict[str, Any] = {"conversation_id": conv.id, "role": msg.role}
            if conv.title:
                metadata["conversation_title"] = conv.title
            if conv.source:
                metadata["source"] = conv.source
            if msg.model:
                metadata["model"] = msg.model
            elif conv.model:
                metadata["model"] = conv.model
            metadata["message_id"] = msg.id
            if conv_tags:
                metadata["tags"] = conv_tags

            # Attach message-level notes
            notes_for_msg = msg_notes.get(msg.id, [])
            if notes_for_msg:
                metadata["notes"] = [
                    {"id": n["id"], "text": n["text"]} for n in notes_for_msg
                ]

            records.append(
                {
                    "mimetype": "text/plain",
                    "content": text,
                    "timestamp": timestamp,
                    "metadata": metadata,
                }
            )
    return records


def _compute_schema(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute metadata key statistics from emitted records.

    For each metadata key, track: type, count, and either enumerated values
    (if ≤20 unique) or a single example (if >20).
    """
    key_stats: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        meta = rec.get("metadata", {})
        for key, value in meta.items():
            if key not in key_stats:
                # ``first_example`` captures the first real (raw, JSON-
                # serializable) value seen for this key, in record order. When
                # a key crosses the >20-unique cap we emit this deterministic
                # value rather than pulling an arbitrary element out of a set
                # (set iteration order depends on PYTHONHASHSEED, which would
                # break byte-stability of schema.yaml, and ``str()`` of a
                # hashed list/dict yields tuple-repr / sorted-JSON garbage).
                key_stats[key] = {
                    "type": _json_type(value),
                    "count": 0,
                    "values": set(),
                    "first_example": value,
                }
            key_stats[key]["count"] += 1
            # Track unique values (cap collection at 21 to know if >20)
            vals = key_stats[key]["values"]
            if isinstance(vals, set) and len(vals) <= 20:
                vals.add(_hashable(value))

    # Convert sets to lists or pick an example
    result = {}
    for key, stats in key_stats.items():
        entry: Dict[str, Any] = {"type": stats["type"], "count": stats["count"]}
        vals = stats["values"]
        if isinstance(vals, set) and len(vals) <= 20:
            entry["values"] = sorted(str(v) for v in vals)
        else:
            # >20 unique values: emit the deterministic first real value
            # (faithful and JSON-serializable), not an arbitrary set element.
            entry["example"] = stats["first_example"]
        result[key] = entry
    return result


def _json_type(value: Any) -> str:
    """Return the JSON type name for a Python value."""
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _hashable(value: Any) -> Any:
    """Make a value hashable for set storage.

    Recurses into lists so a list of dicts becomes a tuple of JSON strings
    (each dict JSON-serialized) rather than a tuple of unhashable dicts.
    """
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _schema_yaml_bytes(
    schema: Dict[str, Dict[str, Any]], record_count: int
) -> bytes:
    """Render schema.yaml as bytes."""
    import yaml

    # ``_compute_schema`` already returns per-key entries shaped exactly as
    # ``{type, count, values|example}`` (with ``values`` as a list), in the
    # insertion order this YAML needs, so it can be embedded directly.
    doc = {
        "conversations": {
            "record_count": record_count,
            "metadata_keys": schema,
        }
    }
    buf = io.StringIO()
    buf.write("# Auto-generated by llm-memex. Edit freely.\n")
    yaml.safe_dump(
        doc,
        buf,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    return buf.getvalue().encode("utf-8")


def _readme_bytes(num_conversations: int) -> bytes:
    """Render README.md as bytes."""
    # Look up via installed package metadata rather than `from llm_memex
    # import __version__` — the latter is ambiguous under pytest where
    # `tests/llm_memex/` shadows the real package on sys.path.
    try:
        from importlib.metadata import version as _pkg_version
        __version__ = _pkg_version("llm-memex")
    except Exception:
        __version__ = "unknown"

    today = date.today().isoformat()
    lines = [
        "---",
        "name: llm-memex conversations archive",
        f"description: \"{num_conversations} conversations exported from llm-memex\"",
        f"datetime: {today}",
        f"generator: llm-memex {__version__}",
        "contents:",
        "  - path: conversations.jsonl",
        "    description: Conversation messages in arkiv universal record format",
        "---",
        "",
        "# llm-memex Conversations Archive",
        "",
        f"This archive contains {num_conversations} conversation(s) exported from llm-memex",
        "in [arkiv](https://github.com/alonzo-church/arkiv) universal record format.",
        "",
        "Each record in `conversations.jsonl` represents one message with metadata",
        "linking it back to its conversation, speaker role, and source platform.",
        "",
        "To import into arkiv:",
        "",
        "```bash",
        "arkiv import README.md --db archive.db",
        "```",
        "",
        "To re-import into llm-memex (round-trip):",
        "",
        "```bash",
        "llm-memex import <this bundle> --format arkiv",
        "```",
        "",
    ]
    return "\n".join(lines).encode("utf-8")
