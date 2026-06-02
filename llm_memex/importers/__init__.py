"""Convention-based importers. Each module provides detect() and import_path()."""
import math
from datetime import datetime, timezone
from typing import Any, List, Optional


def _epoch_to_utc(value: float) -> Optional[datetime]:
    """Convert a numeric epoch (seconds) to a timezone-aware UTC datetime.

    Returns None for non-finite values or epochs outside the representable range,
    instead of raising OverflowError / OSError / ValueError.
    """
    if not math.isfinite(value):
        return None
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse a timestamp from various formats (int/float epoch, ISO string).

    All returned datetimes are timezone-aware UTC: numeric epochs are interpreted
    as UTC, ISO strings are parsed and (if naive) treated as UTC, then converted
    to UTC. This keeps mixed-source timestamps mutually comparable.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is an int subclass; a boolean is never a meaningful timestamp.
        return None
    if isinstance(value, (int, float)):
        return _epoch_to_utc(float(value))
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        try:
            epoch = float(value)
        except (ValueError, OverflowError):
            return None
        return _epoch_to_utc(epoch)
    return None


def detect_model(data: dict, message_keys: List[str], default: str) -> Optional[str]:
    """Detect model from conversation data by scanning top-level then messages.

    Always returns a string or None: a non-string top-level "model" (e.g. a dict
    or list from a malformed export) is rejected rather than passed through.

    Args:
        data: Conversation dict.
        message_keys: Keys to try for the message list (e.g. ["chat_messages", "messages"]).
        default: Fallback model name.
    """
    model = data.get("model")
    if isinstance(model, str):
        return model
    for key in message_keys:
        if key in data:
            seq = data[key]
            if isinstance(seq, list):
                for msg in seq:
                    if isinstance(msg, dict):
                        candidate = msg.get("model")
                        if isinstance(candidate, str) and candidate:
                            return candidate
            break
    return default
