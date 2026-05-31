"""Bulk-enrich trivial conversations using heuristics."""
from __future__ import annotations

import json
import re


GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|"
    r"what'?s\s+up|sup|yo)\b",
    re.IGNORECASE,
)

TRIVIAL_PATTERNS = re.compile(
    r"^(test|testing|ignore|asdf|aaa+|\.+|…+|\?+|!+)$",
    re.IGNORECASE,
)


def extract_user_text(content_json):
    """Extract plain text from a message's content JSON."""
    try:
        blocks = json.loads(content_json) if isinstance(content_json, str) else content_json
        if isinstance(blocks, list):
            return " ".join(
                b.get("text", "") for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def classify_conversation(conv, messages):
    """Classify a conversation and return enrichments to add."""
    enrichments = []
    msg_count = conv["message_count"]
    title = conv["title"] or ""
    is_untitled = title.lower() in ("", "untitled conversation", "new chat", "untitled")

    if msg_count == 0:
        enrichments.append({
            "type": "importance", "value": "trivial",
            "source": "heuristic", "confidence": 1.0,
        })
        return enrichments

    user_texts = [
        extract_user_text(m["content"])
        for m in messages if m["role"] == "user"
    ]
    total_user_text = " ".join(user_texts).strip()
    first_user_text = user_texts[0] if user_texts else ""

    if first_user_text and GREETING_PATTERNS.match(first_user_text):
        enrichments.append({
            "type": "topic", "value": "greeting",
            "source": "heuristic", "confidence": 0.9,
        })

    if first_user_text and TRIVIAL_PATTERNS.match(first_user_text):
        enrichments.append({
            "type": "importance", "value": "trivial",
            "source": "heuristic", "confidence": 0.95,
        })
        return enrichments

    if not total_user_text:
        enrichments.append({
            "type": "importance", "value": "trivial",
            "source": "heuristic", "confidence": 0.95,
        })
        return enrichments

    if is_untitled and len(first_user_text) < 20:
        enrichments.append({
            "type": "importance", "value": "trivial",
            "source": "heuristic", "confidence": 0.9,
        })
        return enrichments

    if msg_count <= 2 and len(total_user_text) < 20:
        enrichments.append({
            "type": "importance", "value": "trivial",
            "source": "heuristic", "confidence": 0.85,
        })
        return enrichments

    if msg_count <= 2 and len(total_user_text) < 100:
        enrichments.append({
            "type": "importance", "value": "brief",
            "source": "heuristic", "confidence": 0.7,
        })

    return enrichments


def register_args(parser):
    """Add script-specific arguments."""
    parser.add_argument("--max-messages", type=int, default=4,
                        help="Only scan conversations with at most N messages")


def run(db, args, apply=False):
    """Scan conversations and classify as trivial/brief/greeting."""
    candidates = db.execute_sql(
        "SELECT id, title, message_count, source "
        "FROM conversations WHERE message_count <= ? "
        "ORDER BY message_count, title",
        (args.max_messages,),
    )
    print(f"Scanning {len(candidates)} conversations with <={args.max_messages} messages...")

    stats = {"trivial": 0, "brief": 0, "greeting": 0, "skipped": 0}
    pending = []

    for conv in candidates:
        existing = db.execute_sql(
            "SELECT type FROM enrichments "
            "WHERE conversation_id=? AND type='importance'",
            (conv["id"],),
        )
        if existing:
            stats["skipped"] += 1
            continue

        messages = db.execute_sql(
            "SELECT role, content FROM messages "
            "WHERE conversation_id=? ORDER BY created_at",
            (conv["id"],),
        )

        enrichments = classify_conversation(conv, messages)
        if not enrichments:
            continue

        for e in enrichments:
            if e["value"] == "trivial":
                stats["trivial"] += 1
            elif e["value"] == "brief":
                stats["brief"] += 1
            if e.get("type") == "topic" and e["value"] == "greeting":
                stats["greeting"] += 1

        first_text = extract_user_text(messages[0]["content"]) if messages else "(empty)"
        preview = first_text[:60] + "..." if len(first_text) > 60 else first_text
        labels = ", ".join(f'{e["type"]}={e["value"]}' for e in enrichments)
        pending.append((conv, enrichments, preview, labels))

    for conv, enrichments, preview, labels in pending:
        flag = "WRITE" if apply else "DRY"
        print(f"  [{flag}] {conv['id'][:12]}... ({conv['message_count']}msg) "
              f"{conv['title']!r}: [{labels}] {preview!r}")
        if apply:
            db.save_enrichments(conv["id"], enrichments)

    print(f"\n{'Applied' if apply else 'Would apply'}:")
    print(f"  trivial:  {stats['trivial']}")
    print(f"  brief:    {stats['brief']}")
    print(f"  greeting: {stats['greeting']}")
    print(f"  skipped:  {stats['skipped']} (already enriched)")

    if not apply and (stats["trivial"] + stats["brief"] > 0):
        print("\nRe-run with --apply to write enrichments.")

    return stats
