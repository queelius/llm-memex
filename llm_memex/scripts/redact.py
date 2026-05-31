"""Redact sensitive content from conversations (word/message/conversation level)."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
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


def scan_message(content, matchers, conversation_id, message_id):
    """Scan a message's content blocks for matcher hits.

    Only scans text blocks; non-text blocks are ignored.
    Returns a ScanResult with all matches found.
    """
    result = ScanResult(
        conversation_id=conversation_id,
        message_id=message_id,
        content=content,
    )

    for block_idx, block in enumerate(content):
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
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

    if not apply:
        _print_dry_run(pending, args.level, stats)
        return stats

    if args.yes:
        for result in pending:
            _apply_single(db, result, args.level)
    else:
        interactive_stats = interactive_review(pending, db, args.level)
        stats.update(interactive_stats)

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


def _apply_single(db, result, level):
    """Apply a single redaction action to the database."""
    if level == "word":
        new_content = redact_word_level(result.content, result.matches)
        db.save_enrichment(
            result.conversation_id, "original_content",
            json.dumps({"message_id": result.message_id, "content": result.content}),
            "redact")
        db.update_message_content(result.conversation_id, result.message_id,
                                  new_content)
    elif level == "message":
        db.save_enrichment(
            result.conversation_id, "original_content",
            json.dumps({"message_id": result.message_id, "content": result.content}),
            "redact")
        db.update_message_content(result.conversation_id, result.message_id,
                                  redact_message_level())
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
                text = result.content[m.block_index]["text"] if m.block_index < len(result.content) else ""
                preview = text[max(0, m.start - 20):m.end + 20]
                print(f"\n[{i+1}/{len(pending)}] conv {conv_short}... msg {result.message_id}:")
                print(f"  ...{preview}...")
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
