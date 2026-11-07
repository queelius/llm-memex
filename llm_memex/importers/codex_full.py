"""Import OpenAI Codex CLI sessions with full fidelity.

The same rollout files, layouts, thread assembly, branches and subagent
links as ``codex`` (see that module), plus what the conversation-only import
leaves out: tool calls (``tool_use``) and their output (``tool_result``),
readable reasoning summaries (``thinking``) and inter-agent messages. Each
record is one message, so the tree keeps the order and fork points the
rollout records. Encrypted reasoning, developer instructions, tool
declarations, token counts and UI events are still skipped and counted.

Like ``claude_code`` and ``claude_code_full``, both importers use
``source="codex"`` and the same conversation ids; ``importer_mode`` and the
provenance ``source_type`` tell them apart. Keep them in separate databases.
"""
from typing import List

from llm_memex.importers.codex import detect  # noqa: F401  (re-exported)
from llm_memex.importers.codex import import_path as _import_codex
from llm_memex.models import Conversation

# Bump when a change makes this importer read the same source differently;
# conversations stamped with an older revision are re-imported (cli.py).
IMPORTER_REVISION = 1


def import_path(path: str) -> List[Conversation]:
    """Import one rollout file, or every rollout file under a directory."""
    return _import_codex(path, full=True)
