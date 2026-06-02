"""Asset management for media blocks — resolution, copying, and rendering."""
from __future__ import annotations

import base64
import binascii
import hashlib
import glob
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_memex.models import Conversation

logger = logging.getLogger(__name__)

# Characters permitted in a derived file extension (after the leading dot).
# Same alphanumeric-plus-limited-punctuation whitelist used to sanitize names.
_EXT_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


# ── Media type helpers ──────────────────────────────────────────

_EXT_MAP = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "application/pdf": ".pdf",
}

# Reverse map: file extension -> canonical media type. Used to correct
# mislabeled media_type values after we locate the actual file on disk.
# OpenAI exports occasionally list image/png for what's actually a .wav.
_EXT_TO_MEDIA_TYPE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".pdf": "application/pdf",
}


def _media_type_from_path(path: Path) -> str | None:
    """Return canonical media_type from a file extension, or None if unknown."""
    return _EXT_TO_MEDIA_TYPE.get(path.suffix.lower())


def _media_type_to_ext(media_type: str) -> str:
    """Map a MIME type to a file extension.

    ``media_type`` is attacker-controlled (it comes straight from an export),
    so any extension derived from its subtype is validated against an
    alphanumeric-plus-limited-punctuation whitelist. A subtype containing
    shell metacharacters, spaces, slashes, path-traversal dots, or anything
    else outside the whitelist falls back to ".bin" rather than leaking junk
    into the on-disk filename (MCA-6).
    """
    if media_type in _EXT_MAP:
        return _EXT_MAP[media_type]
    # Fallback: use subtype (e.g. "image/tiff" -> ".tiff"), but only if it is
    # composed entirely of whitelisted characters and is not a bare-dots
    # traversal token (e.g. "x/.." must not become "..").
    parts = media_type.split("/")
    if len(parts) == 2:
        subtype = parts[1]
        if subtype and all(c in _EXT_ALLOWED for c in subtype) and subtype.strip(".") != "":
            return f".{subtype}"
    return ".bin"


def _safe_filename(name: str | None, msg_id: str, index: int, media_type: str) -> str:
    """Generate a safe, unique filename for an asset."""
    ext = _media_type_to_ext(media_type)
    if name:
        # Sanitize: keep alphanums, hyphens, underscores, dots
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        # Ensure correct extension
        if not safe.lower().endswith(ext):
            safe = safe + ext
        return safe
    # No name: generate from message ID and block index
    short_id = msg_id[:8]
    return f"{short_id}_{index}{ext}"


def _collision_rename(filepath: Path) -> Path:
    """Append short hash to filename if it already exists."""
    if not filepath.exists():
        return filepath
    # Hash the stem + a counter to create unique name
    stem = filepath.stem
    ext = filepath.suffix
    for i in range(1, 1000):
        h = hashlib.md5(f"{stem}_{i}".encode()).hexdigest()[:6]
        candidate = filepath.parent / f"{stem}_{h}{ext}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique name for {filepath}")


# ── OpenAI asset resolution ────────────────────────────────────

def _correct_media_type(block: dict, resolved_path: Path) -> None:
    """Correct media_type on a media block based on the actual file extension.

    OpenAI exports have been observed mislabeling media (e.g. image/png for
    what's actually a .wav). When we locate the real file, trust the extension.
    """
    inferred = _media_type_from_path(resolved_path)
    if inferred and block.get("media_type") != inferred:
        block["media_type"] = inferred


def resolve_openai_assets(conv: Conversation, source_dir: Path) -> int:
    """Resolve OpenAI asset URLs to local file paths.

    Supports three URL schemes observed across OpenAI export generations:
    - ``file-service://file-{ID}`` (older format): a loose file ``{ID}-*`` at
      the top of the source dir, or under ``dalle-generations/``.
    - ``sediment://file_{SHA}`` (newer format, post-2024): a file
      ``file_{SHA}*`` under ``{source_dir}/{conv.id}/{kind}/`` where kind is
      ``image``, ``audio``, etc.
    - ``file-service://file-{ID}`` where the file now lives under the conv-id
      subdirectory (mixed exports).

    For each located file, the block's URL is rewritten to the absolute path,
    and ``media_type`` is corrected from the actual file extension when the
    originally-declared type doesn't match.
    """
    count = 0
    conv_dir = source_dir / conv.id  # newer per-conversation layout

    for msg in conv.messages.values():
        for block in msg.content:
            if block.get("type") != "media":
                continue
            url = block.get("url", "")

            if url.startswith("sediment://"):
                file_id = url.replace("sediment://", "")
                # Search under conv_dir/{kind}/ first, then anywhere under conv_dir
                patterns = [
                    str(conv_dir / "*" / f"{file_id}*"),
                    str(conv_dir / "**" / f"{file_id}*"),
                ]
            elif url.startswith("file-service://file-"):
                file_id = url.replace("file-service://", "")
                patterns = [
                    str(source_dir / f"{file_id}-*"),
                    str(source_dir / "dalle-generations" / f"{file_id}-*"),
                    # Mixed exports: newer exports sometimes relocate older refs
                    str(conv_dir / "*" / f"{file_id}*"),
                    str(conv_dir / "**" / f"{file_id}*"),
                ]
            else:
                continue

            resolved: Path | None = None
            for pattern in patterns:
                recursive = "**" in pattern
                matches = glob.glob(pattern, recursive=recursive)
                if matches:
                    resolved = Path(matches[0]).resolve()
                    break

            if resolved is not None:
                block["url"] = str(resolved)
                _correct_media_type(block, resolved)
                count += 1

    return count


# ── Asset copying ──────────────────────────────────────────────

def copy_assets(conv: Conversation, asset_dir: Path) -> int:
    """Copy referenced assets into asset_dir, rewriting URLs to relative paths.

    Handles three cases:
    1. Absolute file path → copy file, set url to assets/{filename}
    2. Base64 data (no usable url) → decode, write, set url, delete data key
    3. Already relative assets/ URL → skip (idempotent)
    """
    asset_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for msg in conv.messages.values():
        for i, block in enumerate(msg.content):
            if block.get("type") != "media":
                continue
            url = block.get("url", "")
            media_type = block.get("media_type", "application/octet-stream")

            # Already a relative assets/ path — skip
            if url.startswith("assets/"):
                continue

            # Case 1: absolute file path
            if url and Path(url).is_absolute() and Path(url).is_file():
                src = Path(url)
                filename = _safe_filename(
                    block.get("filename") or src.name, msg.id, i, media_type
                )
                dest = _collision_rename(asset_dir / filename)
                shutil.copy2(str(src), str(dest))
                block["url"] = f"assets/{dest.name}"
                # Drop any co-present stale base64 so it does not re-serialize,
                # for consistency with the base64 branch below.
                block.pop("data", None)
                count += 1
                continue

            # Case 2: base64 data with no usable file URL
            data = block.get("data")
            if data:
                # media_type is attacker-controlled and the base64 payload may
                # be malformed (e.g. "Incorrect padding"). Decode with
                # validate=False (consistent, default behaviour) but never let
                # a single bad block abort the whole import loop (MCA-1): log
                # and skip, leaving the block's data intact for later recovery.
                try:
                    decoded = base64.b64decode(data)
                except (binascii.Error, ValueError) as exc:
                    logger.warning(
                        "Skipping media block with malformed base64 "
                        "(msg %s, block %d): %s",
                        msg.id, i, exc,
                    )
                    continue
                filename = _safe_filename(
                    block.get("filename"), msg.id, i, media_type
                )
                dest = _collision_rename(asset_dir / filename)
                dest.write_bytes(decoded)
                block["url"] = f"assets/{dest.name}"
                del block["data"]
                count += 1
                continue
    return count


# ── Source-type dispatcher ─────────────────────────────────────

def resolve_source_assets(conv: Conversation, source_dir: Path, source_type: str) -> int:
    """Resolve source-specific asset references before copying.

    Only OpenAI needs resolution (file-service:// URLs).
    Anthropic/Gemini use base64 inline — handled directly by copy_assets.
    """
    if source_type == "openai":
        return resolve_openai_assets(conv, source_dir)
    return 0
