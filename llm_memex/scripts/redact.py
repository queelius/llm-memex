"""Redact sensitive content from conversations (word/message/conversation level)."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# -- Data Structures ---------------------------------------------------------


@dataclass
class Match:
    conversation_id: str
    message_id: str
    term: str
    start: int
    end: int
    block_index: int


@dataclass
class ScanResult:
    conversation_id: str
    message_id: str
    matches: list[Match] = field(default_factory=list)
    content: list[dict] = field(default_factory=list)


# -- Detection Engine --------------------------------------------------------


def compile_matchers(words=None, patterns=None, pattern_file=None):
    """Compile words and patterns into a list of (compiled_regex, label) tuples.

    Words get word-boundary matching and case-insensitive flags.
    Patterns are used as-is.
    """
    matchers = []

    if words:
        for word in words:
            regex = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            matchers.append((regex, word))

    # Combine explicit patterns and pattern-file patterns
    raw_patterns = list(patterns or [])
    if pattern_file:
        raw_patterns.extend(load_pattern_file(pattern_file))

    for pattern in raw_patterns:
        matchers.append((re.compile(pattern), pattern))

    if not matchers:
        raise ValueError("No words, patterns, or pattern file provided.")

    return matchers


def load_pattern_file(path):
    """Load patterns from a file, one per line. Skips comments (#) and blanks.

    Resolves bare filenames against built-in patterns/ dir first, then
    ~/.memex/scripts/patterns/, then treats as absolute/relative path.
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        # Try built-in patterns dir
        builtin = Path(__file__).parent / "patterns" / path
        if builtin.exists():
            p = builtin
        else:
            # Try user patterns dir
            user = Path.home() / ".memex" / "scripts" / "patterns" / path
            if user.exists():
                p = user

    lines = p.read_text().strip().splitlines()
    return [
        line.strip() for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


# Keys whose values are structural metadata (not user/model text), skipped
# when walking non-text blocks for secrets. A match in a tool name or id is
# almost certainly coincidental, and rewriting it would corrupt the block.
_STRUCTURAL_KEYS = {"type", "id", "name", "tool_use_id", "tool_name", "role", "model"}


def _walk_strings(obj, skip_keys=_STRUCTURAL_KEYS):
    """Yield every string leaf in a nested content block.

    Used to find secrets hiding in tool_use ``input`` dicts, tool_result
    ``content``, and thinking blocks — the places Claude Code sessions are
    most likely to capture credentials (cat of a key file, a Bash command,
    etc.). Skips structural keys so a coincidental match on a type/name
    isn't reported.
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            yield from _walk_strings(v, skip_keys)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v, skip_keys)


def scan_message(content, matchers, conversation_id, message_id):
    """Scan a message's content blocks for matcher hits.

    Text blocks yield precise offset matches (word-level redactable). Every
    other block type (tool_use, tool_result, thinking, ...) is scanned via
    its string leaves; a hit there is recorded as a *structural* match
    (start=end=-1), which the apply path escalates to full message-level
    redaction since precise in-place rewriting of structured JSON is unsafe.
    Returns a ScanResult with all matches found.
    """
    result = ScanResult(
        conversation_id=conversation_id,
        message_id=message_id,
        content=content,
    )

    for block_idx, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "")
            for regex, term in matchers:
                for m in regex.finditer(text):
                    result.matches.append(Match(
                        conversation_id=conversation_id,
                        message_id=message_id,
                        term=term,
                        start=m.start(),
                        end=m.end(),
                        block_index=block_idx,
                    ))
        else:
            # Non-text block: detect secrets in any string leaf. Record one
            # structural match per term so the dry run is honest; the apply
            # path redacts the whole message regardless of count.
            seen_terms = set()
            for s in _walk_strings(block):
                for regex, term in matchers:
                    if term in seen_terms:
                        continue
                    if regex.search(s):
                        seen_terms.add(term)
                        result.matches.append(Match(
                            conversation_id=conversation_id,
                            message_id=message_id,
                            term=term,
                            start=-1,
                            end=-1,
                            block_index=block_idx,
                        ))

    return result


def check_match_mode(matches, mode, matchers):
    """Check if matches satisfy the match mode.

    'any': at least one matcher produced a hit.
    'all': every matcher produced at least one hit.
    """
    if mode == "any":
        return bool(matches)
    if mode == "all":
        matched_terms = {m.term for m in matches}
        required_terms = {term for _, term in matchers}
        return required_terms.issubset(matched_terms)
    return False


def register_args(parser):
    """Add redact-specific CLI arguments."""
    parser.add_argument("--words", help="Comma-separated literal terms to match")
    parser.add_argument("--patterns", help="Comma-separated regex patterns")
    parser.add_argument("--pattern-file", help="File with one pattern per line")
    parser.add_argument("--level", choices=["word", "message", "conversation"],
                        default="word", help="Redaction level (default: word)")
    parser.add_argument("--match-mode", choices=["any", "all"], default="any",
                        help="'any' (default) or 'all' terms must match")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive review, apply all")


def run(db, args, apply=False):
    """Scan and optionally redact content."""
    words = [w.strip() for w in args.words.split(",")] if args.words else None
    patterns = [p.strip() for p in args.patterns.split(",")] if args.patterns else None
    matchers = compile_matchers(words=words, patterns=patterns,
                                pattern_file=args.pattern_file)

    # Scan all messages
    conv_rows = db.execute_sql("SELECT id FROM conversations")
    pending = []  # list of ScanResults
    conv_hits = {}  # conversation_id -> list of ScanResults (for conversation-level)

    for row in conv_rows:
        conv_id = row["id"]
        messages = db.execute_sql(
            "SELECT id, content FROM messages WHERE conversation_id=? ORDER BY created_at",
            (conv_id,),
        )
        for msg_row in messages:
            content = json.loads(msg_row["content"]) if isinstance(msg_row["content"], str) else msg_row["content"]
            result = scan_message(content, matchers, conv_id, msg_row["id"])
            if not result.matches:
                continue
            if args.level == "conversation":
                conv_hits.setdefault(conv_id, []).append(result)
            elif check_match_mode(result.matches, args.match_mode, matchers):
                pending.append(result)

    # For conversation-level: check match mode across all messages in conv
    if args.level == "conversation":
        for conv_id, results in conv_hits.items():
            all_matches = [m for r in results for m in r.matches]
            if check_match_mode(all_matches, args.match_mode, matchers):
                pending.append(ScanResult(
                    conversation_id=conv_id,
                    message_id="(all)",
                    matches=all_matches,
                    content=[],
                ))

    stats = _compute_stats(pending, args.level)
    note_hits = _scan_notes(db, matchers)
    stats["notes_matched"] = len(note_hits)

    if not apply:
        if pending:
            _print_dry_run(pending, args.level, stats)
        if note_hits:
            print(f"\n  [NOTE]  {len(note_hits)} marginalia note(s) contain matches "
                  f"(scrubbed on --apply).")
        if not pending and not note_hits:
            print("No matches found.")
        return stats

    if args.yes:
        failed = []
        for result in pending:
            try:
                _apply_single(db, result, args.level)
            except Exception as exc:  # noqa: BLE001 - surface, don't abort batch
                # Per-item failure must not abort the whole batch (REDACT-4).
                # The failed item's writes are atomic, so nothing is half-applied.
                failed.append(result.conversation_id)
                print(f"  [FAILED] conv {result.conversation_id}: {exc}")
        if failed:
            stats["failed"] = failed
            print(f"\n{len(failed)} conversation(s) failed: {', '.join(failed)}")
    else:
        interactive_stats = interactive_review(pending, db, args.level)
        stats.update(interactive_stats)

    # Scrub the denormalized title/summary fields too (they are exported by
    # every exporter), independent of which messages matched.
    stats["fields_redacted"] = _redact_conversation_fields(db, matchers)
    # Scrub marginalia note text (also exported), FTS-safely.
    stats["notes_redacted"] = _redact_notes(db, matchers)

    return stats


def _compute_stats(pending, level):
    stats = {"total_matches": len(pending), "word_redactions": 0,
             "message_redactions": 0, "conversation_deletions": 0}
    if level == "word":
        stats["word_redactions"] = sum(len(r.matches) for r in pending)
    elif level == "message":
        stats["message_redactions"] = len(pending)
    elif level == "conversation":
        stats["conversation_deletions"] = len(pending)
    return stats


def _print_dry_run(pending, level, stats):
    if not pending:
        print("No matches found.")
        return

    for result in pending:
        conv_short = result.conversation_id[:12]
        if level == "word":
            for match in result.matches:
                if match.start < 0:
                    print(f"  [BLOCK] conv {conv_short}... msg {result.message_id}: "
                          f"matched '{match.term}' in a non-text block "
                          f"(tool/thinking) — whole message will be redacted")
                else:
                    print(f"  [WORD]  conv {conv_short}... msg {result.message_id}: "
                          f"matched '{match.term}' at {match.start}:{match.end}")
        elif level == "message":
            terms = ", ".join(sorted({m.term for m in result.matches}))
            print(f"  [MSG]   conv {conv_short}... msg {result.message_id}: "
                  f"matches: {terms}")
        elif level == "conversation":
            terms = ", ".join(sorted({m.term for m in result.matches}))
            print(f"  [CONV]  conv {conv_short}...: matches across messages: {terms}")

    print("\nSummary:")
    if stats["word_redactions"]:
        print(f"  Word-level redactions:  {stats['word_redactions']}")
    if stats["message_redactions"]:
        print(f"  Message-level redactions: {stats['message_redactions']}")
    if stats["conversation_deletions"]:
        print(f"  Conversation deletions: {stats['conversation_deletions']}")
    print("\nRe-run with --apply to commit changes.")


# -- Mutation Engine ---------------------------------------------------------


def redact_word_level(content, matches):
    """Replace matched spans with [REDACTED] in text blocks.

    Processes matches right-to-left within each block to preserve offsets.
    """
    result = copy.deepcopy(content)
    # Group matches by block_index
    by_block = {}
    for m in matches:
        by_block.setdefault(m.block_index, []).append(m)

    for block_idx, block_matches in by_block.items():
        if block_idx >= len(result):
            continue
        block = result[block_idx]
        if block.get("type") != "text":
            continue
        text = block["text"]
        # Merge overlapping/identical spans before replacement
        sorted_matches = sorted(block_matches, key=lambda x: (x.start, x.end))
        merged_spans = []
        for m in sorted_matches:
            if merged_spans and m.start <= merged_spans[-1][1]:
                merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], m.end))
            else:
                merged_spans.append((m.start, m.end))
        # Replace right-to-left to preserve offsets
        for start, end in reversed(merged_spans):
            text = text[:start] + "[REDACTED]" + text[end:]
        block["text"] = text

    return result


def redact_message_level():
    """Return replacement content for a fully redacted message."""
    return [{"type": "text", "text": "[REDACTED]"}]


def redact_text(text, matchers):
    """Replace every matcher hit in a plain string with [REDACTED].

    Returns (new_text, changed). Overlapping spans are merged and replaced
    right-to-left so offsets stay valid.
    """
    if not text:
        return text, False
    spans = []
    for regex, _ in matchers:
        spans.extend((m.start(), m.end()) for m in regex.finditer(text))
    if not spans:
        return text, False
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in reversed(merged):
        text = text[:start] + "[REDACTED]" + text[end:]
    return text, True


def _redact_conversation_fields(db, matchers):
    """Scrub matcher hits from every conversation's title and summary.

    These denormalized fields are exported by every exporter (markdown, json,
    arkiv, and the HTML conversation list), so a secret left in a title or
    summary would leak even after message-level redaction. Returns the count
    of conversations updated.
    """
    rows = db.execute_sql("SELECT id, title, summary FROM conversations")
    updated = 0
    for row in rows:
        new_title, t_changed = redact_text(row["title"], matchers)
        new_summary, s_changed = redact_text(row["summary"], matchers)
        if t_changed or s_changed:
            db.execute_sql(
                "UPDATE conversations SET title=?, summary=? WHERE id=?",
                (new_title, new_summary, row["id"]),
            )
            updated += 1
    return updated


def _scan_notes(db, matchers):
    """Return the ids of marginalia notes whose text matches any matcher.

    Returns [] on a pre-v4 schema (no notes table).
    """
    try:
        rows = db.execute_sql("SELECT id, text FROM notes")
    except Exception:  # noqa: BLE001 - missing table on old schemas
        return []
    return [
        r["id"] for r in rows
        if any(regex.search(r["text"] or "") for regex, _ in matchers)
    ]


def _redact_notes(db, matchers):
    """Scrub matcher hits from marginalia note text. Returns count updated.

    Notes travel in arkiv and HTML exports, so a secret in a note leaks just
    like one in a message. Uses ``db.update_note`` so ``notes_fts`` stays in
    sync — a direct UPDATE would drift the FTS index.
    """
    updated = 0
    for note_id in _scan_notes(db, matchers):
        row = db.execute_sql("SELECT text FROM notes WHERE id=?", (note_id,))
        if not row:
            continue
        new_text, changed = redact_text(row[0]["text"], matchers)
        if changed:
            db.update_note(note_id, new_text)
            updated += 1
    return updated


def _stage_original_content_enrichment(db, conversation_id, message_id, content):
    """Stage (without committing) an `original_content` enrichment holding the
    pre-redaction content.

    Intentionally writes to ``db.conn`` WITHOUT a commit so it shares a single
    transaction with the subsequent ``update_message_content``. The commit (or
    rollback) inside ``update_message_content`` then flushes (or discards) this
    enrichment together with the message change, making the pair atomic
    (REDACT-4). A crash/raise between them can never leave a committed plaintext
    enrichment beside an un-redacted message.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.conn.execute(
        "INSERT OR REPLACE INTO enrichments "
        "(conversation_id,type,value,source,confidence,created_at) "
        "VALUES (?,?,?,?,?,?)",
        (
            conversation_id,
            "original_content",
            json.dumps({"message_id": message_id, "content": content}),
            "redact",
            None,
            now,
        ),
    )


def _apply_single(db, result, level):
    """Apply a single redaction action to the database.

    For word/message levels the original-content enrichment and the message
    update are committed as ONE transaction (see
    ``_stage_original_content_enrichment``): both apply or neither does.
    """
    if level in ("word", "message"):
        # A structural match (start < 0) means a secret sits inside a
        # tool_use/tool_result/thinking block. We cannot safely rewrite
        # arbitrary structured JSON in place, so escalate that message to
        # full message-level redaction — the secret is guaranteed gone, and
        # the pre-redaction content is preserved in the original_content
        # enrichment for undo.
        has_structural = any(m.start < 0 for m in result.matches)
        if level == "message" or has_structural:
            new_content = redact_message_level()
        else:
            new_content = redact_word_level(result.content, result.matches)
        # Single transaction: stage the original-content enrichment (no commit),
        # then update the message. update_message_content commits, flushing both
        # atomically. If anything raises, roll back so the staged enrichment is
        # discarded too: never a committed plaintext enrichment beside an
        # un-redacted message (REDACT-4). We own the rollback here rather than
        # relying on update_message_content's internal rollback, so atomicity
        # holds no matter where the failure originates.
        try:
            _stage_original_content_enrichment(
                db, result.conversation_id, result.message_id, result.content)
            db.update_message_content(result.conversation_id, result.message_id,
                                      new_content)
        except Exception:
            db.conn.rollback()
            raise
    elif level == "conversation":
        db.delete_conversation(result.conversation_id)


# -- Interactive Review ------------------------------------------------------


def interactive_review(pending, db, level, input_fn=None):
    """Interactively review each match before applying."""
    if input_fn is None:
        input_fn = input
    auto_terms = set()
    stats = {"redacted": 0, "skipped": 0}

    for i, result in enumerate(pending):
        # Auto-apply if all terms in this result are in auto_terms
        if level == "word" and all(m.term in auto_terms for m in result.matches):
            _apply_single(db, result, level)
            stats["redacted"] += 1
            continue

        conv_short = result.conversation_id[:12]
        if level == "word":
            approved_matches = [m for m in result.matches if m.term in auto_terms]
            unapproved = [m for m in result.matches if m.term not in auto_terms]
            reviewed_matches = list(approved_matches)
            quit_requested = False
            for m in unapproved:
                block = result.content[m.block_index] if m.block_index < len(result.content) else {}
                if m.start < 0 or not (isinstance(block, dict) and block.get("type") == "text"):
                    btype = block.get("type", "?") if isinstance(block, dict) else "?"
                    preview = f"(match in {btype} block — whole message will be redacted)"
                else:
                    text = block.get("text", "")
                    preview = "..." + text[max(0, m.start - 20):m.end + 20] + "..."
                print(f"\n[{i+1}/{len(pending)}] conv {conv_short}... msg {result.message_id}:")
                print(f"  {preview}")
                choice = input_fn("  [r]edact  [s]kip  [a]ll  [q]uit\n> ").strip().lower()
                if choice == "a":
                    auto_terms.add(m.term)
                    reviewed_matches.append(m)
                elif choice == "r":
                    reviewed_matches.append(m)
                elif choice == "q":
                    quit_requested = True
                    break
                # "s" skips this term (don't add to reviewed_matches)
            if quit_requested:
                if reviewed_matches:
                    partial = ScanResult(result.conversation_id, result.message_id,
                                         reviewed_matches, result.content)
                    _apply_single(db, partial, level)
                    stats["redacted"] += 1
                return stats
            if reviewed_matches:
                partial = ScanResult(result.conversation_id, result.message_id,
                                     reviewed_matches, result.content)
                _apply_single(db, partial, level)
                stats["redacted"] += 1
            else:
                stats["skipped"] += 1
        else:
            terms = ", ".join(sorted({m.term for m in result.matches}))
            print(f"\n[{i+1}/{len(pending)}] conv {conv_short}... msg {result.message_id}:")
            print(f"  matches: {terms}")
            choice = input_fn("  [r]edact  [s]kip  [q]uit\n> ").strip().lower()
            if choice == "r":
                _apply_single(db, result, level)
                stats["redacted"] += 1
            elif choice == "s":
                stats["skipped"] += 1
            elif choice == "q":
                return stats

    return stats
