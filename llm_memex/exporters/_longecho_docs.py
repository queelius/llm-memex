"""READMEs, checksums, and destination rules for the longecho export.

The root ``README.md`` is the interface of the archive: a reader who has none
of these tools, and does not know what llm-memex is, starts there. This module
renders it (and the viewer's ``site/README.md``), writes ``SHA256SUMS``, and
decides whether an existing folder may be replaced. It produces strings and
performs the file operations the exporter asks for; it knows nothing about how
the other parts of the folder are built.
"""
import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import yaml

GENERATOR_PREFIX = "llm-memex longecho"
SUMS_NAME = "SHA256SUMS"
README_NAME = "README.md"

_HASH_CHUNK = 1024 * 1024  # 1 MB reads, so a multi-GB database never loads at once
_SHA256_HEX_LEN = 64

# (description, sql) pairs shown in the README as `sqlite3 data/conversations.db "<sql>"`.
# The SQL uses single quotes only, so it survives being wrapped in double quotes
# in a shell. Keep them runnable against the projected snapshot (no FTS5).
README_QUERIES: List[Tuple[str, str]] = [
    (
        "The 20 newest conversations, with when they started and where they came from:",
        "SELECT created_at, title, source FROM conversations "
        "ORDER BY created_at DESC LIMIT 20;",
    ),
    (
        "Messages that mention a word (here: sqlite), with the title of their conversation:",
        "SELECT c.title, m.role, m.content FROM messages m "
        "JOIN conversations c ON c.id = m.conversation_id "
        "WHERE m.content LIKE '%sqlite%' LIMIT 10;",
    ),
]


@dataclass
class ArchiveSummary:
    """Everything the root README states about the archive."""

    conversations: int
    messages: int
    sources: List[str]
    first: Optional[str]  # ISO dates, YYYY-MM-DD; None for an empty database
    last: Optional[str]
    owner: Optional[str]
    include_notes: bool
    include_archived: bool
    site_has_data: bool  # False when the database was too large for the viewer
    exported_at: str  # ISO 8601


def _version() -> str:
    # Package metadata rather than `from llm_memex import __version__`, which
    # is ambiguous under pytest where tests/llm_memex/ shadows the package (C10).
    try:
        from importlib.metadata import version

        return version("llm-memex")
    except Exception:
        return "unknown"


def _generator() -> str:
    return f"{GENERATOR_PREFIX} {_version()}"


def summarize(db_path: Path) -> dict:
    """Counts, sorted distinct sources, and the date range of a snapshot.

    Opened read-only with plain sqlite3: the ``Database`` class would switch the
    published file to WAL mode.
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        sources = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT source FROM conversations "
                "WHERE source IS NOT NULL AND source != '' ORDER BY source"
            )
        ]
        first, last = conn.execute(
            "SELECT substr(MIN(created_at), 1, 10), substr(MAX(created_at), 1, 10) "
            "FROM conversations"
        ).fetchone()
    finally:
        conn.close()
    return {
        "conversations": conversations,
        "messages": messages,
        "sources": sources,
        "first": first,
        "last": last,
    }


# ---------------------------------------------------------------------------
# READMEs
# ---------------------------------------------------------------------------

def _frontmatter(meta: Dict[str, object]) -> str:
    dumped = yaml.safe_dump(
        meta, sort_keys=False, allow_unicode=True, width=1_000_000
    )
    return f"---\n{dumped}---\n"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _date_range(summary: ArchiveSummary) -> Optional[str]:
    if summary.first is None or summary.last is None:
        return None
    if summary.first == summary.last:
        return summary.first
    return f"{summary.first} to {summary.last}"


def _description(summary: ArchiveSummary) -> str:
    text = f"{_count(summary.conversations, 'conversation')} "
    text += f"({_count(summary.messages, 'message')}) with AI assistants"
    if summary.sources:
        text += f" from {', '.join(summary.sources)}"
    span = _date_range(summary)
    if span:
        text += f", {span}"
    return text + "."


def render_root_readme(summary: ArchiveSummary) -> str:
    """The archive's root README: frontmatter plus a body for a reader with no context."""
    name = "llm-memex archive"
    if summary.owner:
        name += f" of {summary.owner}"
    contents = [
        {"path": "text/index.md",
         "description": "Every conversation as a plain Markdown file; start here "
                        "with any text editor"},
        {"path": "site/index.html",
         "description": "A viewer you open by double-clicking it in a web browser"
                        if summary.site_has_data else
                        "A viewer page; this archive is too large for it to load "
                        "the conversations by itself"},
        {"path": "data/conversations.db",
         "description": "All conversations and messages as a SQLite database"},
        {"path": "data/arkiv/conversations.jsonl",
         "description": "The same messages as JSON Lines, one JSON object per line"},
        {"path": SUMS_NAME,
         "description": "A checksum for every file, to confirm nothing has changed"},
    ]
    meta = {
        "name": name,
        "description": _description(summary),
        "datetime": summary.exported_at,
        "generator": _generator(),
        "contents": contents,
    }
    return _frontmatter(meta) + "\n" + _root_body(summary)


def _root_body(summary: ArchiveSummary) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Archive of AI conversations")
    add("")
    add("## What this is")
    add("")
    if summary.owner:
        add(f"This folder is an archive of conversations between {summary.owner} "
            "and AI assistants (chatbots and coding assistants).")
    else:
        add("This folder is a personal archive of conversations with AI assistants "
            "(chatbots and coding assistants).")
    add("")
    if summary.conversations == 0:
        add("It holds 0 conversations: the database it was made from was empty "
            "(or everything in it was left out, see below).")
    else:
        sentence = (f"It holds {_count(summary.conversations, 'conversation')} "
                    f"and {_count(summary.messages, 'message')}")
        if summary.sources:
            sentence += f" from these sources: {', '.join(summary.sources)}"
        span = _date_range(summary)
        if span:
            sentence += f". The oldest conversation began on {summary.first}"
            if summary.first != summary.last:
                sentence += f" and the newest on {summary.last}"
        add(sentence + ".")
    add("")
    add(f"Exported on {summary.exported_at}.")
    add("")
    add("Everything here is an ordinary file. You do not need the program that "
        "made this folder to read it.")
    add("")

    add("## How to open it")
    add("")
    add("Start with the first option; each one needs less than the next.")
    add("")
    add("1. **Read it in a text editor.** The `text/` folder has one Markdown "
        "file (a plain text file) for every conversation, grouped by year. "
        "Open `text/index.md` for a list of all of them by date, each with a "
        "link to its file.")
    if summary.site_has_data:
        add("2. **Browse it in a web browser.** Double-click `site/index.html`. "
            "It opens straight from the folder, with no server and no internet "
            "connection, and lets you search and read the conversations.")
    else:
        add("2. **The browser viewer.** This archive is too large for the browser "
            "viewer, so double-clicking `site/index.html` will not show the "
            "conversations. Read `text/` or query `data/` instead.")
    add("3. **Query the database.** `data/conversations.db` is a SQLite database. "
        "With the `sqlite3` program (free, and on most computers) run, from this "
        "folder:")
    add("")
    for description, sql in README_QUERIES:
        add(f"   {description}")
        add("")
        add("   ```")
        add(f'   sqlite3 data/conversations.db "{sql}"')
        add("   ```")
        add("")
    add("   The tables are `conversations` (one row each) and `messages` "
        "(`conversation_id` points at the conversation). Message `content` is "
        "JSON text.")
    add("4. **Read the JSON Lines.** `data/arkiv/conversations.jsonl` has the same "
        "messages, one JSON object per line, in a format called arkiv. "
        "`data/arkiv/README.md` and `data/arkiv/schema.yaml` describe it. Any "
        "tool that reads JSON can open it.")
    add("")
    add("Pictures and other media that the conversations refer to, if there are "
        "any, are in `data/assets/`.")
    add("")

    add("## What was left out")
    add("")
    if summary.include_archived:
        add("- Archived conversations are included, because the export was asked "
            "to keep them.")
    else:
        add("- Archived conversations (ones the owner had set aside) are left out "
            "and appear nowhere in this folder, including the counts above.")
    if summary.include_notes:
        add("- Notes the owner attached to conversations are included.")
    else:
        add("- Notes the owner attached to conversations are left out.")
    add("- The database's full-text search index is not included. It is large, it "
        "can be rebuilt from the messages, and it could still hold text from "
        "conversations that were left out. The `LIKE` query above searches the "
        "messages directly instead.")
    add("")

    add("## How to verify")
    add("")
    add(f"`{SUMS_NAME}` lists a SHA-256 checksum for every other file in this "
        "folder. To check that nothing has been damaged or changed, run this "
        "from this folder:")
    add("")
    add("```")
    add(f"sha256sum -c {SUMS_NAME}")
    add("```")
    add("")
    add("Each line should end with `OK`.")
    add("")
    return "\n".join(lines)


def render_site_readme() -> str:
    """README for the ``site/`` viewer, stamped so longecho sees it as a view."""
    meta = {
        "name": "llm-memex viewer",
        "description": "A web page for browsing and searching the conversations "
                       "in this archive. It opens straight from the folder.",
        "generator": _generator(),
    }
    body = (
        "# Viewer\n"
        "\n"
        "Double-click `index.html` to open the viewer in a web browser. It works "
        "offline, from the folder, with no server.\n"
        "\n"
        "It reads a copy of `../data/conversations.db` (stored here as the "
        "script files in `db/`). The archive's own README, one level up, "
        "explains the rest.\n"
    )
    return _frontmatter(meta) + "\n" + body


# ---------------------------------------------------------------------------
# SHA256SUMS
# ---------------------------------------------------------------------------

_ESCAPES = (("\\", "\\\\"), ("\n", "\\n"), ("\r", "\\r"))


def _escape_name(name: str) -> Tuple[str, bool]:
    """GNU coreutils escaping: the line is prefixed with a backslash when needed."""
    escaped = name
    for raw, esc in _ESCAPES:
        escaped = escaped.replace(raw, esc)
    return escaped, escaped != name


def _unescape_name(escaped: str) -> Optional[str]:
    out: List[str] = []
    chars = iter(escaped)
    for ch in chars:
        if ch != "\\":
            out.append(ch)
            continue
        nxt = next(chars, None)
        if nxt == "\\":
            out.append("\\")
        elif nxt == "n":
            out.append("\n")
        elif nxt == "r":
            out.append("\r")
        else:
            return None  # a lone or unknown escape: malformed
    return "".join(out)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(out_dir: Path) -> None:
    """Write ``SHA256SUMS`` (sha256sum format) for every regular file under *out_dir*.

    Symlinks are neither followed nor listed, so nothing outside *out_dir* can
    be hashed or later deleted on its behalf. ``SHA256SUMS`` itself is excluded.
    """
    out_dir = Path(out_dir)
    entries: List[Tuple[str, Path]] = []
    for dirpath, _dirnames, filenames in os.walk(out_dir, followlinks=False):
        for filename in filenames:
            full = Path(dirpath) / filename
            if full.is_symlink() or not full.is_file():
                continue
            rel = full.relative_to(out_dir).as_posix()
            if rel != SUMS_NAME:
                entries.append((rel, full))
    entries.sort(key=lambda entry: entry[0])

    lines = []
    for rel, full in entries:
        name, escaped = _escape_name(rel)
        prefix = "\\" if escaped else ""
        lines.append(f"{prefix}{_hash_file(full)}  {name}\n")

    # Replace atomically, so an interrupted run never leaves a half-written file.
    tmp = out_dir / f".{SUMS_NAME}.tmp"
    tmp.write_bytes("".join(lines).encode("utf-8", "surrogateescape"))
    os.replace(tmp, out_dir / SUMS_NAME)


def _parse_sums(text: str) -> List[str]:
    """Paths listed in a SHA256SUMS text; malformed lines are skipped."""
    paths: List[str] = []
    for line in text.split("\n"):
        escaped = line.startswith("\\")
        body = line[1:] if escaped else line
        split = _SHA256_HEX_LEN
        digest, separator, name = body[:split], body[split:split + 2], body[split + 2:]
        if (
            len(digest) != _SHA256_HEX_LEN
            or any(c not in "0123456789abcdefABCDEF" for c in digest)
            or separator != "  "
            or not name
        ):
            continue
        if escaped:
            name = _unescape_name(name)
            if name is None:
                continue
        paths.append(name)
    return paths


# ---------------------------------------------------------------------------
# Destination
# ---------------------------------------------------------------------------

def _readme_generator(readme: Path) -> Optional[str]:
    """The ``generator`` frontmatter value of *readme*, or None if absent/unreadable."""
    try:
        lines = readme.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return None
    try:
        meta = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None
    generator = meta.get("generator")
    return str(generator) if generator is not None else None


def _safe_listed_file(out_dir: Path, root: Path, name: str) -> Optional[Path]:
    """The path of a listed file that is safe to delete, else None.

    The listing is data from disk and may be damaged or hostile: only relative
    paths with no ``..`` that resolve (through any symlinks) to an existing
    regular file inside *out_dir* qualify.
    """
    if not name or "\0" in name:
        return None
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    candidate = out_dir / rel
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved == root or not resolved.is_relative_to(root):
        return None
    if resolved.is_dir() or not resolved.is_file():
        return None
    return candidate


def prepare_destination(out_dir: Path) -> None:
    """Make *out_dir* ready for a fresh export, or refuse.

    A missing or empty folder is created/accepted. A non-empty one is replaced
    only if it is a complete previous longecho export by this tool: its root
    README is stamped ``generator: llm-memex longecho ...`` and ``SHA256SUMS``
    exists. Exactly the files that ``SHA256SUMS`` lists are deleted (then the
    directories that leaves empty); anything a person added stays.

    Order matters for recovery: the root README, which holds the ownership
    stamp, goes last (after ``SHA256SUMS``), so every interruption leaves a
    folder this function either resumes or explains.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"{out_dir} exists and is not a directory")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        return
    if not any(out_dir.iterdir()):
        return

    found = _readme_generator(out_dir / README_NAME)
    if found is None or not found.startswith(GENERATOR_PREFIX):
        stamp = f"generator {found!r}" if found is not None else "no generator stamp"
        raise ValueError(
            f"{out_dir} is not empty and was not written by `{GENERATOR_PREFIX}` "
            f"(its README.md has {stamp}); refusing to overwrite it. "
            "Choose an empty or new folder."
        )
    sums = out_dir / SUMS_NAME
    if not sums.is_file():
        raise ValueError(
            f"{out_dir} looks like an interrupted previous export: its README.md "
            f"is stamped but {SUMS_NAME} is missing, so what to replace is "
            "unknown. Remove the folder and export again."
        )

    root = out_dir.resolve()
    listed = _parse_sums(sums.read_bytes().decode("utf-8", "surrogateescape"))
    deleted_parents = set()
    readme_listed = False
    for name in listed:
        path = _safe_listed_file(out_dir, root, name)
        if path is None:
            continue
        if PurePosixPath(name) == PurePosixPath(README_NAME):
            readme_listed = True  # deleted last, below
            continue
        path.unlink()
        deleted_parents.update(PurePosixPath(name).parents)

    # Directories the deleted files leave empty, deepest first; never out_dir.
    for rel in sorted(deleted_parents, key=lambda p: len(p.parts), reverse=True):
        if not rel.parts:
            continue
        try:
            (out_dir / rel).rmdir()
        except OSError:
            pass  # not empty (a person's file lives there), or a symlink

    sums.unlink()
    if readme_listed:
        (out_dir / README_NAME).unlink()
