"""Tests for memex importers: OpenAI, Anthropic, Gemini, Claude Code."""
import pytest
import json
import math
from datetime import datetime, timezone

from llm_memex.importers import parse_timestamp, detect_model
from llm_memex.importers._claude_code_common import parse_iso as cc_parse_iso
from llm_memex.importers.openai import detect as openai_detect
from llm_memex.importers.openai import import_path as openai_import
from llm_memex.importers.anthropic import detect as anthropic_detect
from llm_memex.importers.anthropic import import_path as anthropic_import
from llm_memex.importers.gemini import detect as gemini_detect
from llm_memex.importers.gemini import import_path as gemini_import
from llm_memex.importers.claude_code import detect as claude_code_detect
from llm_memex.importers.claude_code import import_path as claude_code_import
from llm_memex.importers.claude_code_full import detect as claude_code_full_detect
from llm_memex.importers.claude_code_full import import_path as claude_code_full_import


# ---------- OpenAI ----------

class TestOpenAIDetect:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "export.json"
        f.write_text(json.dumps([{
            "id": "conv1",
            "mapping": {"node1": {"message": {"content": {"parts": ["hi"]}}}},
        }]))
        assert openai_detect(str(f)) is True

    def test_invalid_format(self, tmp_path):
        f = tmp_path / "other.json"
        f.write_text(json.dumps({"not": "openai"}))
        assert openai_detect(str(f)) is False

    def test_nonexistent_file(self, tmp_path):
        assert openai_detect(str(tmp_path / "nope.json")) is False

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json at all")
        assert openai_detect(str(f)) is False

    def test_empty_list(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("[]")
        assert openai_detect(str(f)) is False

    def test_binary_file(self, tmp_path):
        f = tmp_path / "image.jpg"
        f.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)
        assert openai_detect(str(f)) is False


class TestOpenAIImport:
    def test_simple_conversation(self, tmp_path):
        data = [{
            "id": "conv1", "title": "Test Chat",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "root": {"id": "root", "children": ["m1"], "message": None},
                "m1": {
                    "id": "m1", "parent": "root", "children": ["m2"],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
                "m2": {
                    "id": "m2", "parent": "m1", "children": [],
                    "message": {
                        "id": "m2", "author": {"role": "assistant"},
                        "content": {"parts": ["hi there"]},
                        "create_time": 1700000001,
                        "metadata": {"model_slug": "gpt-4"},
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert len(convs) == 1
        conv = convs[0]
        assert conv.title == "Test Chat"
        assert conv.source == "openai"
        assert conv.model == "gpt-4"
        assert len(conv.messages) == 2

    def test_multimodal_content(self, tmp_path):
        data = [{
            "id": "conv2", "title": "Image Chat",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {
                            "parts": [
                                "look at this",
                                {"asset_pointer": "file://img.png"},
                            ],
                        },
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "media"
        assert msg.content[1]["url"] == "file://img.png"

    def test_skips_empty_system_messages(self, tmp_path):
        data = [{
            "id": "conv3", "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "sys": {
                    "id": "sys", "parent": None, "children": ["m1"],
                    "message": {
                        "id": "sys", "author": {"role": "system"},
                        "content": {"parts": []},
                        "create_time": 1700000000,
                    },
                },
                "m1": {
                    "id": "m1", "parent": "sys", "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        # System message skipped, only user message
        assert len(convs[0].messages) == 1

    def test_empty_mapping(self, tmp_path):
        data = [{"id": "conv4", "mapping": {}}]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert len(convs) == 0

    def test_provenance_metadata(self, tmp_path):
        data = [{
            "id": "conv1", "title": "Test",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        prov = convs[0].metadata.get("_provenance")
        assert prov is not None
        assert prov["source_type"] == "openai"
        assert prov["source_id"] == "conv1"
        assert prov["source_file"] == str(f)

    def test_tool_use_content(self, tmp_path):
        data = [{
            "id": "conv5", "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "assistant"},
                        "content": {
                            "parts": [
                                {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "test"}},
                            ],
                        },
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert msg.content[0]["type"] == "tool_use"
        assert msg.content[0]["name"] == "search"


# ---------- Anthropic ----------

class TestAnthropicDetect:
    def test_chat_messages_format(self, tmp_path):
        f = tmp_path / "claude.json"
        f.write_text(json.dumps([{
            "uuid": "abc", "name": "My Chat",
            "chat_messages": [{"sender": "human", "text": "hi"}],
        }]))
        assert anthropic_detect(str(f)) is True

    def test_uuid_name_format(self, tmp_path):
        f = tmp_path / "claude.json"
        f.write_text(json.dumps([{"uuid": "abc", "name": "Chat"}]))
        assert anthropic_detect(str(f)) is True

    def test_invalid_format(self, tmp_path):
        f = tmp_path / "other.json"
        f.write_text(json.dumps({"some": "data"}))
        assert anthropic_detect(str(f)) is False

    def test_nonexistent_file(self, tmp_path):
        assert anthropic_detect(str(tmp_path / "nope.json")) is False


class TestAnthropicImport:
    def test_chat_messages_format(self, tmp_path):
        data = [{
            "uuid": "conv1", "name": "Test Chat",
            "created_at": "2024-01-01T00:00:00Z",
            "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "hello"},
                {"uuid": "m2", "sender": "assistant", "text": "hi back"},
            ],
        }]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        assert len(convs) == 1
        conv = convs[0]
        assert conv.title == "Test Chat"
        assert conv.source == "anthropic"
        assert len(conv.messages) == 2
        msgs = list(conv.messages.values())
        assert msgs[0].role == "user"
        assert msgs[0].get_text() == "hello"
        assert msgs[1].role == "assistant"

    def test_multipart_content(self, tmp_path):
        data = [{
            "uuid": "conv2", "name": "Multi",
            "chat_messages": [{
                "uuid": "m1", "sender": "assistant",
                "content": [
                    {"type": "text", "text": "here's the code"},
                    {"type": "tool_use", "id": "tu1", "name": "edit", "input": {"file": "a.py"}},
                ],
            }],
        }]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "tool_use"

    def test_attachments(self, tmp_path):
        data = [{
            "uuid": "conv3", "name": "Attach",
            "chat_messages": [{
                "uuid": "m1", "sender": "human",
                "text": "see attached",
                "attachments": [{"file_name": "report.pdf", "file_type": "application/pdf"}],
            }],
        }]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "media"
        assert msg.content[1]["filename"] == "report.pdf"

    def test_tags_default(self, tmp_path):
        data = [{"uuid": "conv4", "name": "Tags", "chat_messages": []}]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        assert "anthropic" in convs[0].tags
        assert "claude" in convs[0].tags

    def test_provenance_metadata(self, tmp_path):
        data = [{
            "uuid": "conv1", "name": "Test",
            "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "hello"},
            ],
        }]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        prov = convs[0].metadata.get("_provenance")
        assert prov is not None
        assert prov["source_type"] == "anthropic"
        assert prov["source_id"] == "conv1"


# ---------- Gemini ----------

class TestGeminiDetect:
    def test_conversations_key(self, tmp_path):
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps({"conversations": [{"id": "c1"}]}))
        assert gemini_detect(str(f)) is True

    def test_turns_key(self, tmp_path):
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps({"turns": [{"role": "user"}]}))
        assert gemini_detect(str(f)) is True

    def test_conversation_id_key(self, tmp_path):
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps({"conversation_id": "abc"}))
        assert gemini_detect(str(f)) is True

    def test_list_with_gemini_marker(self, tmp_path):
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps([{"model": "gemini-1.5-pro", "turns": []}]))
        assert gemini_detect(str(f)) is True

    def test_invalid_format(self, tmp_path):
        f = tmp_path / "other.json"
        f.write_text(json.dumps({"some": "data"}))
        assert gemini_detect(str(f)) is False

    def test_nonexistent_file(self, tmp_path):
        assert gemini_detect(str(tmp_path / "nope.json")) is False

    def test_openai_export_mentioning_gemini_not_detected(self, tmp_path):
        """An OpenAI export (has 'mapping') that merely mentions gemini must not be claimed (IMP-1)."""
        f = tmp_path / "openai.json"
        f.write_text(json.dumps([
            {"title": "gemini vs gpt-4", "mapping": {"root": {"id": "root", "message": None}}}
        ]))
        assert gemini_detect(str(f)) is False

    def test_anthropic_export_mentioning_gemini_not_detected(self, tmp_path):
        """An Anthropic export (has 'chat_messages') that mentions gemini must not be claimed (IMP-1)."""
        f = tmp_path / "anthropic.json"
        f.write_text(json.dumps([
            {"name": "compare bard", "chat_messages": [{"text": "x"}]}
        ]))
        assert gemini_detect(str(f)) is False


class TestGeminiImport:
    def test_turns_format(self, tmp_path):
        data = {
            "conversations": [{
                "id": "conv1", "title": "Gemini Chat",
                "created_at": 1700000000,
                "turns": [
                    {"id": "m1", "role": "user", "parts": [{"text": "hello"}]},
                    {"id": "m2", "role": "model", "parts": [{"text": "hi!"}]},
                ],
            }],
        }
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        assert len(convs) == 1
        conv = convs[0]
        assert conv.title == "Gemini Chat"
        assert conv.source == "gemini"
        assert len(conv.messages) == 2
        msgs = list(conv.messages.values())
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"  # "model" mapped to "assistant"

    def test_inline_data(self, tmp_path):
        data = {
            "conversations": [{
                "id": "conv2", "title": "Image",
                "turns": [{
                    "id": "m1", "role": "user",
                    "parts": [
                        {"text": "what's this?"},
                        {"inline_data": {"mime_type": "image/jpeg", "data": "base64data"}},
                    ],
                }],
            }],
        }
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "media"
        assert msg.content[1]["data"] == "base64data"

    def test_simple_text_content(self, tmp_path):
        data = {
            "conversations": [{
                "id": "conv3", "title": "Simple",
                "turns": [{"id": "m1", "role": "user", "content": "plain text"}],
            }],
        }
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert msg.get_text() == "plain text"

    def test_string_parts(self, tmp_path):
        data = {
            "conversations": [{
                "id": "conv4", "title": "Strings",
                "turns": [{"id": "m1", "role": "user", "parts": ["hello", "world"]}],
            }],
        }
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert len(msg.content) == 2
        assert msg.content[0]["text"] == "hello"
        assert msg.content[1]["text"] == "world"

    def test_tags_default(self, tmp_path):
        data = {"conversations": [{"id": "c1", "title": "T", "turns": []}]}
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        assert "google" in convs[0].tags
        assert "gemini" in convs[0].tags

    def test_model_detection(self, tmp_path):
        data = {
            "conversations": [{
                "id": "c1", "title": "T", "model": "gemini-1.5-pro",
                "turns": [],
            }],
        }
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        assert convs[0].model == "gemini-1.5-pro"

    def test_provenance_metadata(self, tmp_path):
        data = {
            "conversations": [{
                "id": "conv1", "title": "Gemini Chat",
                "turns": [
                    {"id": "m1", "role": "user", "parts": [{"text": "hello"}]},
                ],
            }],
        }
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        prov = convs[0].metadata.get("_provenance")
        assert prov is not None
        assert prov["source_type"] == "gemini"
        assert prov["source_id"] == "conv1"


# ---------- Claude Code ----------

def _cc_event(event_type, uuid="u1", parent_uuid=None, session_id="sess-123",
              slug="cool-testing-session", timestamp="2026-02-18T10:00:00Z",
              user_type="external", is_sidechain=False, message=None, **extra):
    """Helper to build a Claude Code JSONL event."""
    rec = {
        "type": event_type,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "sessionId": session_id,
        "slug": slug,
        "timestamp": timestamp,
        "userType": user_type,
        "isSidechain": is_sidechain,
    }
    if message is not None:
        rec["message"] = message
    rec.update(extra)
    return rec


def _write_jsonl(path, events):
    """Write events as JSONL file."""
    path.write_text("\n".join(json.dumps(e) for e in events))


class TestClaudeCodeDetect:
    def test_detect_jsonl(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, [_cc_event("user")])
        assert claude_code_detect(str(f)) is True

    def test_detect_rejects_json(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"type": "user", "sessionId": "abc"}))
        assert claude_code_detect(str(f)) is False

    def test_detect_rejects_non_claude_jsonl(self, tmp_path):
        f = tmp_path / "other.jsonl"
        f.write_text(json.dumps({"foo": "bar", "type": "something_else"}))
        assert claude_code_detect(str(f)) is False

    def test_detect_nonexistent_file(self, tmp_path):
        assert claude_code_detect(str(tmp_path / "nope.jsonl")) is False

    def test_detect_invalid_json(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text("not json at all")
        assert claude_code_detect(str(f)) is False


class TestClaudeCodeImport:
    def test_import_basic(self, tmp_path):
        """Import a transcript with 2 user + 2 assistant messages."""
        events = [
            _cc_event("user", uuid="u1", timestamp="2026-02-18T10:00:00Z",
                      message={"role": "user", "content": "Hello, help me with Python"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      timestamp="2026-02-18T10:00:01Z",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Sure, I can help!"}]}),
            _cc_event("user", uuid="u2", parent_uuid="a1",
                      timestamp="2026-02-18T10:00:02Z",
                      message={"role": "user", "content": "How do I sort a list?"}),
            _cc_event("assistant", uuid="a2", parent_uuid="u2",
                      timestamp="2026-02-18T10:00:03Z",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Use sorted() or list.sort()"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        assert len(convs) == 1
        conv = convs[0]
        assert conv.id == "sess-123"
        assert conv.title == "Cool Testing Session"
        assert conv.source == "claude_code"
        assert conv.model == "claude-opus-4-6"
        assert "claude-code" in conv.tags
        assert len(conv.messages) == 4
        assert conv.metadata["importer_mode"] == "conversation_only"

        # Verify linear chain
        msgs = list(conv.messages.values())
        assert msgs[0].parent_id is None
        assert msgs[1].parent_id == msgs[0].id
        assert msgs[2].parent_id == msgs[1].id
        assert msgs[3].parent_id == msgs[2].id

    def test_rewind_branch_uses_parentuuid_not_linear(self, tmp_path):
        """LLM-5: a rewind/edit re-parents onto an earlier ancestor. The
        importer must honor parentUuid, not chain to the linear predecessor."""
        events = [
            _cc_event("user", uuid="u1", parent_uuid=None,
                      message={"role": "user", "content": "first question"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "first answer"}]}),
            _cc_event("user", uuid="u2", parent_uuid="a1",
                      message={"role": "user", "content": "follow-up v1"}),
            _cc_event("assistant", uuid="a2", parent_uuid="u2",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "answer v1"}]}),
            # Rewind: the follow-up is edited, branching from a1 again (NOT a2).
            _cc_event("user", uuid="u3", parent_uuid="a1",
                      message={"role": "user", "content": "follow-up v2"}),
            _cc_event("assistant", uuid="a3", parent_uuid="u3",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "answer v2"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        conv = claude_code_import(str(f))[0]
        assert conv.messages["u1"].parent_id is None       # root
        assert conv.messages["u2"].parent_id == "a1"
        # The rewound turn branches from a1, not from the linear predecessor a2.
        assert conv.messages["u3"].parent_id == "a1"
        assert conv.messages["a3"].parent_id == "u3"

    def test_skipped_parent_walks_to_nearest_imported_ancestor(self, tmp_path):
        """LLM-5: when a record's parentUuid points at a skipped record, the
        importer walks up to the nearest imported ancestor (diverging from
        the linear predecessor)."""
        events = [
            _cc_event("user", uuid="u1", parent_uuid=None,
                      message={"role": "user", "content": "q1"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "answer 1"}]}),
            _cc_event("user", uuid="u2", parent_uuid="a1",
                      message={"role": "user", "content": "q2"}),
            _cc_event("assistant", uuid="a2", parent_uuid="u2",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "answer 2"}]}),
            # tool_use-only assistant: SKIPPED by conversation_only; branches
            # from a1.
            _cc_event("assistant", uuid="askip", parent_uuid="a1",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "tool_use", "id": "t1",
                                            "name": "Read", "input": {}}]}),
            # Its child's parentUuid is the skipped record; nearest imported
            # ancestor is a1 (NOT the linear predecessor a2).
            _cc_event("user", uuid="u3", parent_uuid="askip",
                      message={"role": "user", "content": "q3"}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        conv = claude_code_import(str(f))[0]
        assert "askip" not in conv.messages  # skipped
        assert conv.messages["u3"].parent_id == "a1"

    def test_import_skips_tool_use(self, tmp_path):
        """Assistant messages with only tool_use (no text) are skipped."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Read my file"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "tool_use", "id": "tu1",
                                            "name": "Read", "input": {"path": "a.py"}}]}),
            _cc_event("assistant", uuid="a2", parent_uuid="a1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Here is your file."}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        assert len(convs[0].messages) == 2  # user + text assistant only

    def test_import_skips_tool_results(self, tmp_path):
        """User messages that are tool results (not external) are skipped."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Check something"}),
            # Tool result: userType=internal
            _cc_event("user", uuid="u2", parent_uuid="u1",
                      user_type="internal",
                      message={"role": "user", "content": [
                          {"type": "tool_result", "tool_use_id": "tu1",
                           "content": "file contents here"}
                      ]}),
            _cc_event("assistant", uuid="a1", parent_uuid="u2",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Done!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        assert len(convs[0].messages) == 2  # external user + text assistant only

    def test_import_skips_sidechain(self, tmp_path):
        """Messages with isSidechain=true are skipped."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      is_sidechain=True,
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Sidechain response"}]}),
            _cc_event("assistant", uuid="a2", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Main response"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        msgs = list(convs[0].messages.values())
        assert len(msgs) == 2
        assert msgs[1].get_text() == "Main response"

    def test_import_skips_progress_and_system(self, tmp_path):
        """Non-message event types (progress, system, file-history-snapshot) are skipped."""
        events = [
            _cc_event("progress", uuid="p1", data={"type": "hook_progress"}),
            _cc_event("system", uuid="s1", message={"role": "system", "content": "init"}),
            _cc_event("file-history-snapshot", uuid="f1",
                      snapshot={"trackedFileBackups": {}}),
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        assert len(convs[0].messages) == 2

    def test_import_extracts_model(self, tmp_path):
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hi"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-sonnet-4-6",
                               "content": [{"type": "text", "text": "Hello!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        assert convs[0].model == "claude-sonnet-4-6"

    def test_import_provenance_metadata(self, tmp_path):
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hi"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hello!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        prov = convs[0].metadata.get("_provenance")
        assert prov is not None
        assert prov["source_type"] == "claude_code"
        assert prov["source_id"] == "sess-123"
        assert prov["source_file"] == str(f)

    def test_import_empty_session(self, tmp_path):
        """Session with only system/progress events returns empty list."""
        events = [
            _cc_event("progress", uuid="p1", data={"type": "hook_progress"}),
            _cc_event("system", uuid="s1", message={"role": "system", "content": "init"}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        assert convs == []

    def test_import_strips_thinking_blocks(self, tmp_path):
        """Thinking blocks in assistant content are stripped."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Think about this"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "thinking", "text": "Let me think..."},
                                   {"type": "text", "text": "Here is my answer."},
                               ]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        msg = list(convs[0].messages.values())[1]
        assert msg.get_text() == "Here is my answer."
        assert len(msg.content) == 1  # Only text block, no thinking

    def test_import_joins_multiple_text_blocks(self, tmp_path):
        """Multiple text blocks in one assistant turn are joined."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Explain"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "text", "text": "First part."},
                                   {"type": "tool_use", "id": "tu1", "name": "Read",
                                    "input": {"path": "a.py"}},
                                   {"type": "text", "text": "Second part."},
                               ]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        msg = list(convs[0].messages.values())[1]
        assert "First part." in msg.get_text()
        assert "Second part." in msg.get_text()
        assert len(msg.content) == 1  # Joined into single text block

    def test_import_skips_corrupted_lines(self, tmp_path):
        """Corrupted/NUL-padded lines in JSONL are skipped gracefully."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        # Write valid lines + corrupted NUL bytes (simulating truncated write)
        content = "\n".join(json.dumps(e) for e in events) + "\n" + "\x00" * 100
        f.write_text(content)
        convs = claude_code_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 2


# ---------- Directory detection + import ----------

@pytest.mark.parametrize(
    "import_fn", [claude_code_import, claude_code_full_import],
    ids=["conversation_only", "full"],
)
class TestClaudeCodeSessionContext:
    """Where and with which client a session ran, so the catalog can be
    filtered by project ("my llm-memex sessions") and re-read by layout."""

    def _session(self, tmp_path, *, second_cwd="/work/proj", second_version="2.1.0"):
        events = [
            _cc_event("user", uuid="u1", cwd="/work/proj", gitBranch="main",
                      version="2.0.0",
                      message={"role": "user", "content": "hi"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1", cwd=second_cwd,
                      gitBranch="feature", version=second_version,
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "hello"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        return f

    def test_records_cwd_branch_and_client_version(self, tmp_path, import_fn):
        conv = import_fn(str(self._session(tmp_path)))[0]
        # cwd/branch: where the session started; version: the newest client
        # that wrote to it (a resumed session can span upgrades).
        assert conv.metadata["cwd"] == "/work/proj"
        assert conv.metadata["git_branch"] == "main"
        assert conv.metadata["client_version"] == "2.1.0"

    def test_first_cwd_wins_when_session_moves(self, tmp_path, import_fn):
        conv = import_fn(str(self._session(tmp_path, second_cwd="/elsewhere")))[0]
        assert conv.metadata["cwd"] == "/work/proj"

    def test_provenance_names_the_layout(self, tmp_path, import_fn):
        conv = import_fn(str(self._session(tmp_path)))[0]
        assert conv.metadata["_provenance"]["source_format"] == "claude-code-jsonl/1"

    def test_absent_context_is_omitted_not_null(self, tmp_path, import_fn):
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [
            _cc_event("user", uuid="u1", message={"role": "user", "content": "hi"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "yo"}]}),
        ])
        meta = import_fn(str(f))[0].metadata
        assert not {"cwd", "git_branch", "client_version"} & meta.keys()


class TestClaudeCodeDirectoryDetect:
    def test_directory_with_jsonl(self, tmp_path):
        """Directory containing Claude Code .jsonl files is detected."""
        sub = tmp_path / "projects" / "myproject"
        sub.mkdir(parents=True)
        f = sub / "session.jsonl"
        _write_jsonl(f, [_cc_event("user")])
        assert claude_code_detect(str(tmp_path / "projects")) is True

    def test_directory_without_jsonl(self, tmp_path):
        """Empty directory is not detected."""
        d = tmp_path / "empty"
        d.mkdir()
        assert claude_code_detect(str(d)) is False

    def test_directory_with_non_claude_jsonl(self, tmp_path):
        """Directory with non-Claude Code .jsonl is not detected."""
        d = tmp_path / "other"
        d.mkdir()
        f = d / "data.jsonl"
        f.write_text(json.dumps({"foo": "bar", "type": "something_else"}))
        assert claude_code_detect(str(d)) is False


class TestClaudeCodeDirectoryImport:
    def test_import_directory(self, tmp_path):
        """Importing a directory finds and imports all .jsonl sessions."""
        d = tmp_path / "sessions"
        d.mkdir()
        # Session A
        events_a = [
            _cc_event("user", uuid="u1", session_id="sess-a", slug="session-a",
                      message={"role": "user", "content": "Hello A"}),
            _cc_event("assistant", uuid="a1", session_id="sess-a", slug="session-a",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Reply A"}]}),
        ]
        _write_jsonl(d / "a.jsonl", events_a)
        # Session B in a subdirectory
        sub = d / "sub"
        sub.mkdir()
        events_b = [
            _cc_event("user", uuid="u1", session_id="sess-b", slug="session-b",
                      message={"role": "user", "content": "Hello B"}),
            _cc_event("assistant", uuid="a1", session_id="sess-b", slug="session-b",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Reply B"}]}),
        ]
        _write_jsonl(sub / "b.jsonl", events_b)

        convs = claude_code_import(str(d))
        assert len(convs) == 2
        ids = {c.id for c in convs}
        assert ids == {"sess-a", "sess-b"}

    def test_import_directory_skips_non_claude(self, tmp_path):
        """Non-Claude .jsonl files in directory are skipped."""
        d = tmp_path / "mixed"
        d.mkdir()
        # Valid Claude Code session
        events = [
            _cc_event("user", uuid="u1", session_id="sess-1", slug="valid",
                      message={"role": "user", "content": "Hi"}),
            _cc_event("assistant", uuid="a1", session_id="sess-1", slug="valid",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hello"}]}),
        ]
        _write_jsonl(d / "valid.jsonl", events)
        # Non-Claude JSONL
        (d / "other.jsonl").write_text(json.dumps({"foo": "bar"}))

        convs = claude_code_import(str(d))
        assert len(convs) == 1
        assert convs[0].id == "sess-1"

    def test_import_empty_directory(self, tmp_path):
        """Importing an empty directory returns no conversations."""
        d = tmp_path / "empty"
        d.mkdir()
        convs = claude_code_import(str(d))
        assert convs == []


class TestOpenAIDirectoryDetect:
    def test_directory_with_conversations_json(self, tmp_path):
        """Directory containing conversations.json is detected."""
        d = tmp_path / "openai_export"
        d.mkdir()
        (d / "conversations.json").write_text(json.dumps([{
            "id": "c1",
            "mapping": {"m1": {"message": {"content": {"parts": ["hi"]}}}},
        }]))
        assert openai_detect(str(d)) is True

    def test_directory_without_conversations_json(self, tmp_path):
        """Directory without conversations.json is not detected."""
        d = tmp_path / "random"
        d.mkdir()
        assert openai_detect(str(d)) is False

    def test_directory_with_wrong_conversations_json(self, tmp_path):
        """Directory with non-OpenAI conversations.json is not detected."""
        d = tmp_path / "wrong"
        d.mkdir()
        (d / "conversations.json").write_text(json.dumps({"not": "openai"}))
        assert openai_detect(str(d)) is False


class TestOpenAIDirectoryImport:
    def test_import_directory(self, tmp_path):
        """Importing a directory reads conversations.json inside it."""
        d = tmp_path / "openai_export"
        d.mkdir()
        data = [{
            "id": "conv1", "title": "Test Chat",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        (d / "conversations.json").write_text(json.dumps(data))
        convs = openai_import(str(d))
        assert len(convs) == 1
        assert convs[0].title == "Test Chat"


class TestAnthropicDirectoryDetect:
    def test_rejects_directory(self, tmp_path):
        """Anthropic importer rejects directories."""
        d = tmp_path / "some_dir"
        d.mkdir()
        assert anthropic_detect(str(d)) is False


class TestGeminiDirectoryDetect:
    def test_rejects_directory(self, tmp_path):
        """Gemini importer rejects directories."""
        d = tmp_path / "some_dir"
        d.mkdir()
        assert gemini_detect(str(d)) is False


# ---------- Claude Code Full ----------

class TestClaudeCodeFullDetect:
    def test_detect_delegates_to_shared(self, tmp_path):
        """Full importer uses the same detection as conversation_only."""
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, [_cc_event("user")])
        assert claude_code_full_detect(str(f)) is True

    def test_detect_rejects_non_claude(self, tmp_path):
        f = tmp_path / "other.jsonl"
        f.write_text(json.dumps({"foo": "bar", "type": "something_else"}))
        assert claude_code_full_detect(str(f)) is False

    def test_detect_directory(self, tmp_path):
        sub = tmp_path / "projects" / "myproject"
        sub.mkdir(parents=True)
        _write_jsonl(sub / "session.jsonl", [_cc_event("user")])
        assert claude_code_full_detect(str(tmp_path / "projects")) is True


class TestClaudeCodeFullImport:
    def test_rewind_branch_uses_parentuuid(self, tmp_path):
        """LLM-5: the full-fidelity importer must also honor parentUuid so a
        rewind/edit branch is not linearized onto the wrong ancestor."""
        events = [
            _cc_event("user", uuid="u1", parent_uuid=None,
                      message={"role": "user", "content": "q1"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "answer 1"}]}),
            _cc_event("user", uuid="u2", parent_uuid="a1",
                      message={"role": "user", "content": "follow-up v1"}),
            _cc_event("assistant", uuid="a2", parent_uuid="u2",
                      message={"role": "assistant", "model": "m",
                               "content": [{"type": "text", "text": "answer v1"}]}),
            _cc_event("user", uuid="u3", parent_uuid="a1",  # rewind onto a1
                      message={"role": "user", "content": "follow-up v2"}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        conv = claude_code_full_import(str(f))[0]
        assert conv.messages["u3"].parent_id == "a1"  # not the predecessor a2
        assert conv.messages["u1"].parent_id is None

    def test_import_preserves_tool_use(self, tmp_path):
        """Tool use blocks are preserved in full import."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Read my file"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "text", "text": "Let me read that."},
                                   {"type": "tool_use", "id": "tu1",
                                    "name": "Read", "input": {"path": "a.py"}},
                               ]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        assert len(convs) == 1
        msg = list(convs[0].messages.values())[1]
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "tool_use"
        assert msg.content[1]["name"] == "Read"
        assert msg.content[1]["input"] == {"path": "a.py"}

    def test_import_preserves_thinking(self, tmp_path):
        """Thinking blocks are preserved and normalized to text field."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Think about this"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "thinking", "thinking": "Let me consider..."},
                                   {"type": "text", "text": "Here is my answer."},
                               ]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        msg = list(convs[0].messages.values())[1]
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "thinking"
        assert msg.content[0]["text"] == "Let me consider..."
        assert msg.content[1]["type"] == "text"

    def test_import_preserves_tool_results(self, tmp_path):
        """User tool_result messages are preserved."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Check something"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "tool_use", "id": "tu1",
                                    "name": "Read", "input": {"path": "a.py"}},
                               ]}),
            _cc_event("user", uuid="u2", parent_uuid="a1",
                      user_type="internal",
                      message={"role": "user", "content": [
                          {"type": "tool_result", "tool_use_id": "tu1",
                           "content": "file contents here"}
                      ]}),
            _cc_event("assistant", uuid="a2", parent_uuid="u2",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Done!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        assert len(convs[0].messages) == 4
        # Check the tool_result message
        msgs = list(convs[0].messages.values())
        assert msgs[2].role == "user"
        assert msgs[2].content[0]["type"] == "tool_result"
        assert msgs[2].content[0]["tool_use_id"] == "tu1"
        assert msgs[2].content[0]["content"] == "file contents here"

    def test_import_preserves_tool_result_error(self, tmp_path):
        """Tool result errors are preserved."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Do something"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "tool_use", "id": "tu1",
                                    "name": "Bash", "input": {"cmd": "fail"}},
                               ]}),
            _cc_event("user", uuid="u2", parent_uuid="a1",
                      user_type="internal",
                      message={"role": "user", "content": [
                          {"type": "tool_result", "tool_use_id": "tu1",
                           "content": "command not found", "is_error": True}
                      ]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        msgs = list(convs[0].messages.values())
        assert msgs[2].content[0]["is_error"] is True

    def test_import_text_only_unchanged(self, tmp_path):
        """Text-only messages work the same as conversation_only importer."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi there!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        assert len(convs) == 1
        msgs = list(convs[0].messages.values())
        assert msgs[0].get_text() == "Hello"
        assert msgs[1].get_text() == "Hi there!"

    def test_import_skips_sidechain(self, tmp_path):
        """Sidechain messages are still skipped in full import."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      is_sidechain=True,
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Sidechain"}]}),
            _cc_event("assistant", uuid="a2", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Main"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        msgs = list(convs[0].messages.values())
        assert len(msgs) == 2
        assert msgs[1].get_text() == "Main"

    def test_import_skips_progress_and_system(self, tmp_path):
        """Progress and system events are skipped."""
        events = [
            _cc_event("progress", uuid="p1", data={"type": "hook_progress"}),
            _cc_event("system", uuid="s1", message={"role": "system", "content": "init"}),
            _cc_event("file-history-snapshot", uuid="f1",
                      snapshot={"trackedFileBackups": {}}),
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        assert len(convs[0].messages) == 2

    def test_import_metadata(self, tmp_path):
        """Full import sets importer_mode='full' and provenance source_type='claude_code_full'."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hi"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hello!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        conv = convs[0]
        assert conv.metadata["importer_mode"] == "full"
        assert conv.source == "claude_code"
        assert "claude-code" in conv.tags
        prov = conv.metadata["_provenance"]
        assert prov["source_type"] == "claude_code_full"
        assert prov["source_id"] == "sess-123"
        assert prov["source_file"] == str(f)

    def test_import_empty_session(self, tmp_path):
        """Session with no importable messages returns empty list."""
        events = [
            _cc_event("progress", uuid="p1", data={"type": "hook_progress"}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        assert convs == []

    def test_import_directory(self, tmp_path):
        """Directory import works via shared directory walker."""
        d = tmp_path / "sessions"
        d.mkdir()
        events_a = [
            _cc_event("user", uuid="u1", session_id="sess-a", slug="session-a",
                      message={"role": "user", "content": "Hello A"}),
            _cc_event("assistant", uuid="a1", session_id="sess-a", slug="session-a",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Reply A"}]}),
        ]
        _write_jsonl(d / "a.jsonl", events_a)
        events_b = [
            _cc_event("user", uuid="u1", session_id="sess-b", slug="session-b",
                      message={"role": "user", "content": "Hello B"}),
            _cc_event("assistant", uuid="a1", session_id="sess-b", slug="session-b",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "tool_use", "id": "tu1",
                                    "name": "Bash", "input": {"cmd": "ls"}},
                               ]}),
        ]
        _write_jsonl(d / "b.jsonl", events_b)
        convs = claude_code_full_import(str(d))
        assert len(convs) == 2
        ids = {c.id for c in convs}
        assert ids == {"sess-a", "sess-b"}

    def test_import_assistant_only_tool_use(self, tmp_path):
        """Assistant turn with only tool_use (no text) is preserved in full import."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Read a.py"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "tool_use", "id": "tu1",
                                    "name": "Read", "input": {"path": "a.py"}},
                               ]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        # Both messages preserved (conversation_only would skip the tool_use-only assistant)
        assert len(convs[0].messages) == 2
        msg = list(convs[0].messages.values())[1]
        assert msg.content[0]["type"] == "tool_use"


# ---------- Claude Code Full: Subagent Import ----------

class TestClaudeCodeFullSubagentImport:
    """Tests for subagent import in the full-fidelity importer."""

    def _make_parent_session(self, d, session_id="sess-parent", slug="parent-session"):
        """Create a parent session JSONL file."""
        events = [
            _cc_event("user", uuid="u1", session_id=session_id, slug=slug,
                      message={"role": "user", "content": "Do something"}),
            _cc_event("assistant", uuid="a1", session_id=session_id, slug=slug,
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "On it!"}]}),
        ]
        f = d / f"{session_id}.jsonl"
        _write_jsonl(f, events)
        return f

    def _make_subagent(self, parent_dir, agent_id, session_id="sess-parent"):
        """Create a subagent JSONL file in <session_id>/subagents/ dir."""
        subdir = parent_dir / session_id / "subagents"
        subdir.mkdir(parents=True, exist_ok=True)
        events = [
            _cc_event("user", uuid="su1", session_id=session_id, slug=agent_id,
                      is_sidechain=True,
                      message={"role": "user", "content": "Subagent task"}),
            _cc_event("assistant", uuid="sa1", session_id=session_id, slug=agent_id,
                      is_sidechain=True,
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [
                                   {"type": "tool_use", "id": "tu1",
                                    "name": "Read", "input": {"path": "a.py"}},
                               ]}),
            _cc_event("user", uuid="su2", session_id=session_id, slug=agent_id,
                      user_type="internal", is_sidechain=True,
                      message={"role": "user", "content": [
                          {"type": "tool_result", "tool_use_id": "tu1",
                           "content": "file contents"}
                      ]}),
            _cc_event("assistant", uuid="sa2", session_id=session_id, slug=agent_id,
                      is_sidechain=True,
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Done with subagent work"}]}),
        ]
        f = subdir / f"{agent_id}.jsonl"
        _write_jsonl(f, events)
        return f

    def test_single_file_imports_subagents(self, tmp_path):
        """Importing a single parent file also imports its subagents."""
        d = tmp_path / "sessions"
        d.mkdir()
        parent_file = self._make_parent_session(d)
        self._make_subagent(d, "compact")

        convs = claude_code_full_import(str(parent_file))
        assert len(convs) == 2
        # Parent first, child second
        assert convs[0].id == "sess-parent"
        assert convs[0].parent_conversation_id is None
        assert convs[1].id == "sess-parent:compact"
        assert convs[1].parent_conversation_id == "sess-parent"

    def test_subagent_id_is_deterministic(self, tmp_path):
        """Subagent ID is {sessionId}:{agentId}."""
        d = tmp_path / "sessions"
        d.mkdir()
        self._make_parent_session(d)
        self._make_subagent(d, "my_agent")

        convs = claude_code_full_import(str(d / "sess-parent.jsonl"))
        child = [c for c in convs if c.parent_conversation_id][0]
        assert child.id == "sess-parent:my_agent"

    def test_subagent_metadata(self, tmp_path):
        """Subagent gets agent_id in metadata."""
        d = tmp_path / "sessions"
        d.mkdir()
        self._make_parent_session(d)
        self._make_subagent(d, "prompt_suggestion")

        convs = claude_code_full_import(str(d / "sess-parent.jsonl"))
        child = [c for c in convs if c.parent_conversation_id][0]
        assert child.metadata["agent_id"] == "prompt_suggestion"

    def test_subagent_tags(self, tmp_path):
        """Subagents get appropriate tags based on agent type."""
        d = tmp_path / "sessions"
        d.mkdir()
        self._make_parent_session(d)
        self._make_subagent(d, "compact")
        self._make_subagent(d, "prompt_suggestion")
        self._make_subagent(d, "custom_agent")

        convs = claude_code_full_import(str(d / "sess-parent.jsonl"))
        children = {c.metadata.get("agent_id"): c for c in convs if c.parent_conversation_id}

        # compact agent gets compact tag
        assert "claude-code-agent" in children["compact"].tags
        assert "claude-code-compact" in children["compact"].tags

        # prompt_suggestion agent gets prompt tag
        assert "claude-code-agent" in children["prompt_suggestion"].tags
        assert "claude-code-prompt-suggestion" in children["prompt_suggestion"].tags

        # custom agent only gets base agent tag
        assert "claude-code-agent" in children["custom_agent"].tags
        assert "claude-code-compact" not in children["custom_agent"].tags

    def test_subagent_sidechain_records_imported(self, tmp_path):
        """Subagent's isSidechain=true records ARE imported (not skipped)."""
        d = tmp_path / "sessions"
        d.mkdir()
        self._make_parent_session(d)
        self._make_subagent(d, "compact")

        convs = claude_code_full_import(str(d / "sess-parent.jsonl"))
        child = [c for c in convs if c.parent_conversation_id][0]
        # The subagent has 4 records (user, assistant+tool_use, user+tool_result, assistant)
        assert child.message_count == 4

    def test_directory_import_includes_subagents(self, tmp_path):
        """Directory import returns parents before children."""
        d = tmp_path / "sessions"
        d.mkdir()
        self._make_parent_session(d, "sess-a", "session-a")
        self._make_subagent(d, "compact", "sess-a")

        # Second session without subagents
        sub = d / "sub"
        sub.mkdir()
        self._make_parent_session(sub, "sess-b", "session-b")

        convs = claude_code_full_import(str(d))
        ids = [c.id for c in convs]
        # Both parents present, plus one child
        assert "sess-a" in ids
        assert "sess-a:compact" in ids
        assert "sess-b" in ids
        # Parent appears before its child
        assert ids.index("sess-a") < ids.index("sess-a:compact")

    def test_no_subagent_dir_is_fine(self, tmp_path):
        """Session without subagents/ directory works normally."""
        d = tmp_path / "sessions"
        d.mkdir()
        self._make_parent_session(d)

        convs = claude_code_full_import(str(d / "sess-parent.jsonl"))
        assert len(convs) == 1
        assert convs[0].id == "sess-parent"

    def test_empty_subagent_file_skipped(self, tmp_path):
        """Subagent JSONL with no importable messages is skipped."""
        d = tmp_path / "sessions"
        d.mkdir()
        self._make_parent_session(d)
        subdir = d / "sess-parent" / "subagents"
        subdir.mkdir(parents=True)
        # Empty subagent (only progress events, all sidechain)
        events = [
            _cc_event("progress", uuid="p1", is_sidechain=True,
                      data={"type": "hook_progress"}),
        ]
        _write_jsonl(subdir / "empty_agent.jsonl", events)

        convs = claude_code_full_import(str(d / "sess-parent.jsonl"))
        assert len(convs) == 1  # Only parent

    def test_conversation_only_skips_subagents(self, tmp_path):
        """The conversation_only importer still skips subagent directories."""
        d = tmp_path / "sessions"
        d.mkdir()
        # Parent
        events = [
            _cc_event("user", uuid="u1", session_id="sess-a", slug="session-a",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", session_id="sess-a", slug="session-a",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi!"}]}),
        ]
        _write_jsonl(d / "sess-a.jsonl", events)
        # Subagent
        subdir = d / "subagents"
        subdir.mkdir()
        sub_events = [
            _cc_event("user", uuid="su1", session_id="sess-a", slug="compact",
                      is_sidechain=True,
                      message={"role": "user", "content": "Agent work"}),
            _cc_event("assistant", uuid="sa1", session_id="sess-a", slug="compact",
                      is_sidechain=True,
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Done"}]}),
        ]
        _write_jsonl(subdir / "compact.jsonl", sub_events)

        convs = claude_code_import(str(d))
        assert len(convs) == 1
        assert convs[0].id == "sess-a"


# ---------- Malformed/adversarial input hardening ----------

# IMP-2 / IMP-7: parse_timestamp robustness and tz-awareness

class TestParseTimestampHardening:
    def test_out_of_range_numeric_epoch(self):
        """A wildly out-of-range numeric epoch returns None, not a crash (IMP-2)."""
        # 1e20 seconds is far beyond datetime's representable range.
        assert parse_timestamp(1e20) is None

    def test_negative_huge_numeric_epoch(self):
        """A hugely negative epoch returns None instead of OverflowError (IMP-2)."""
        assert parse_timestamp(-1e20) is None

    def test_nan_epoch(self):
        """NaN epoch returns None (IMP-2)."""
        assert parse_timestamp(float("nan")) is None

    def test_infinity_epoch(self):
        """Infinity epoch returns None (IMP-2)."""
        assert parse_timestamp(float("inf")) is None
        assert parse_timestamp(float("-inf")) is None

    def test_out_of_range_string_epoch(self):
        """An out-of-range string epoch returns None instead of crashing (IMP-2)."""
        assert parse_timestamp("1e20") is None

    def test_numeric_epoch_is_utc_aware(self):
        """Numeric epoch yields a timezone-aware UTC datetime (IMP-7)."""
        dt = parse_timestamp(1700000000)
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timezone.utc.utcoffset(None)
        # 1700000000 == 2023-11-14T22:13:20 UTC
        assert dt == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)

    def test_iso_naive_treated_as_utc(self):
        """A naive ISO string is treated as UTC and returned tz-aware (IMP-7)."""
        dt = parse_timestamp("2024-01-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_iso_with_z_is_utc_aware(self):
        """An ISO string with trailing Z is returned tz-aware UTC (IMP-7)."""
        dt = parse_timestamp("2024-01-01T00:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_iso_with_offset_converted_to_utc(self):
        """An ISO string with a non-UTC offset is converted to UTC (IMP-7)."""
        dt = parse_timestamp("2024-01-01T05:00:00+05:00")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_numeric_and_iso_are_comparable(self):
        """Mixed numeric and ISO timestamps are comparable (no naive/aware clash) (IMP-7)."""
        a = parse_timestamp(1700000000)
        b = parse_timestamp("2024-01-01T00:00:00Z")
        # Comparison must not raise TypeError.
        assert (a < b) or (a >= b)


# IMP minor: detect_model must return a str or None even for non-string model values

class TestDetectModelHardening:
    def test_dict_model_value_not_returned(self):
        """A dict model value is rejected: result is a str-or-None, never the dict (IMP minor)."""
        result = detect_model({"model": {"name": "x"}}, ["messages"], "default")
        assert result is None or isinstance(result, str)
        assert not isinstance(result, dict)
        # Malformed top-level model falls through to the default.
        assert result == "default"

    def test_list_model_value_not_returned(self):
        """A list model value is rejected: result is a str-or-None, never the list (IMP minor)."""
        result = detect_model({"model": ["a", "b"]}, ["messages"], "default")
        assert result is None or isinstance(result, str)
        assert not isinstance(result, list)
        assert result == "default"

    def test_dict_model_with_usable_message_model(self):
        """A malformed top-level model still lets a valid message-level model win (IMP minor)."""
        data = {"model": {"bad": "dict"}, "messages": [{"model": "gpt-4"}]}
        result = detect_model(data, ["messages"], "default")
        assert result == "gpt-4"

    def test_string_model_value_preserved(self):
        """A string model value is returned unchanged."""
        assert detect_model({"model": "gpt-4"}, ["messages"], "default") == "gpt-4"

    def test_missing_model_falls_to_default(self):
        assert detect_model({}, ["messages"], "default") == "default"

    def test_non_dict_message_in_scan_ignored(self):
        """Non-dict entries in the message list don't crash the scan (IMP minor)."""
        data = {"messages": ["not a dict", {"model": "gpt-4"}]}
        assert detect_model(data, ["messages"], "default") == "gpt-4"


# IMP-3: OpenAI importer with null/non-dict fields

class TestOpenAIMalformed:
    def test_null_content(self, tmp_path):
        """A JSON-null content field does not crash import (IMP-3)."""
        data = [{
            "id": "conv1", "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": None,
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 1

    def test_null_author(self, tmp_path):
        """A JSON-null author field does not crash import (IMP-3)."""
        data = [{
            "id": "conv1", "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": None,
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert len(convs) == 1
        msg = list(convs[0].messages.values())[0]
        assert msg.role == "unknown"

    def test_non_dict_mapping(self, tmp_path):
        """A non-dict mapping yields no conversation rather than crashing (IMP-3)."""
        data = [{"id": "conv1", "mapping": ["not", "a", "dict"]}]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert convs == []

    def test_non_dict_node(self, tmp_path):
        """A non-dict node in the mapping is skipped (IMP-3)."""
        data = [{
            "id": "conv1", "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "bad": "not a dict",
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 1

    def test_non_dict_message(self, tmp_path):
        """A non-dict message value is skipped (IMP-3)."""
        data = [{
            "id": "conv1", "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "bad": {"id": "bad", "parent": None, "children": [], "message": "oops"},
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 1

    def test_create_time_zero_not_now(self, tmp_path):
        """create_time == 0 routes through parse_timestamp (epoch 0), not now() (IMP-3)."""
        data = [{
            "id": "conv1", "create_time": 0, "update_time": 0,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 0,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        conv = convs[0]
        # Epoch 0 is 1970, definitely not "now" (this test runs well after 2020).
        assert conv.created_at.year == 1970
        msg = list(conv.messages.values())[0]
        assert msg.created_at is not None
        assert msg.created_at.year == 1970

    def test_title_default_consistent(self, tmp_path):
        """OpenAI title defaults to 'Untitled Conversation' like the others (IMP-3)."""
        data = [{
            "id": "conv1", "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert convs[0].title == "Untitled Conversation"

    def test_create_time_out_of_range_falls_back(self, tmp_path):
        """An out-of-range create_time does not crash; falls back gracefully (IMP-3)."""
        data = [{
            "id": "conv1", "create_time": 1e20, "update_time": 1e20,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1e20,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_text(json.dumps(data))
        convs = openai_import(str(f))
        assert len(convs) == 1
        # created_at falls back to now() (current year), message created_at is None.
        msg = list(convs[0].messages.values())[0]
        assert msg.created_at is None


# IMP-5/IMP-6: OpenAI file-level robustness (BOM)

class TestOpenAIBom:
    def test_detect_with_bom(self, tmp_path):
        """A UTF-8 BOM does not break detect() (IMP-6)."""
        f = tmp_path / "export.json"
        payload = json.dumps([{
            "id": "conv1",
            "mapping": {"m1": {"message": {"content": {"parts": ["hi"]}}}},
        }])
        f.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
        assert openai_detect(str(f)) is True

    def test_import_with_bom(self, tmp_path):
        """A UTF-8 BOM does not break import_path() (IMP-6)."""
        data = [{
            "id": "conv1", "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]
        f = tmp_path / "export.json"
        f.write_bytes(b"\xef\xbb\xbf" + json.dumps(data).encode("utf-8"))
        convs = openai_import(str(f))
        assert len(convs) == 1
        assert convs[0].id == "conv1"


# IMP-4: Anthropic importer with null/non-list container, non-dict elements, numeric ts

class TestAnthropicMalformed:
    def test_null_messages_container(self, tmp_path):
        """A null chat_messages container does not crash (IMP-4)."""
        data = [{"uuid": "conv1", "name": "Test", "chat_messages": None}]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 0

    def test_dict_messages_container(self, tmp_path):
        """A dict (non-list) messages container is coerced to empty (IMP-4)."""
        data = [{"uuid": "conv1", "name": "Test", "chat_messages": {"k": "v"}}]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 0

    def test_non_dict_message_element(self, tmp_path):
        """A non-dict element in chat_messages is skipped (IMP-4)."""
        data = [{
            "uuid": "conv1", "name": "Test",
            "chat_messages": ["not a dict", {"uuid": "m1", "sender": "human", "text": "hi"}],
        }]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 1

    def test_numeric_message_timestamp(self, tmp_path):
        """A numeric message timestamp is parsed, not crashed on (IMP-4)."""
        data = [{
            "uuid": "conv1", "name": "Test",
            "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "hi",
                 "created_at": 1700000000},
            ],
        }]
        f = tmp_path / "claude.json"
        f.write_text(json.dumps(data))
        convs = anthropic_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert msg.created_at is not None
        assert msg.created_at.year == 2023

    def test_detect_with_bom(self, tmp_path):
        """A UTF-8 BOM does not break Anthropic detect() (IMP-6)."""
        f = tmp_path / "claude.json"
        payload = json.dumps([{"uuid": "abc", "name": "Chat", "chat_messages": []}])
        f.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
        assert anthropic_detect(str(f)) is True

    def test_import_with_bom(self, tmp_path):
        """A UTF-8 BOM does not break Anthropic import_path() (IMP-6)."""
        data = [{"uuid": "conv1", "name": "Test",
                 "chat_messages": [{"uuid": "m1", "sender": "human", "text": "hi"}]}]
        f = tmp_path / "claude.json"
        f.write_bytes(b"\xef\xbb\xbf" + json.dumps(data).encode("utf-8"))
        convs = anthropic_import(str(f))
        assert len(convs) == 1
        assert convs[0].id == "conv1"


# IMP-4 / IMP-8: Gemini importer with null/non-list container, string parts, numeric ts

class TestGeminiMalformed:
    def test_null_messages_container(self, tmp_path):
        """A null turns container does not crash (IMP-4)."""
        data = {"conversations": [{"id": "c1", "title": "T", "turns": None}]}
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 0

    def test_non_dict_message_element(self, tmp_path):
        """A non-dict turn element is skipped (IMP-4)."""
        data = {"conversations": [{
            "id": "c1", "title": "T",
            "turns": ["not a dict", {"id": "m1", "role": "user", "parts": [{"text": "hi"}]}],
        }]}
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 1

    def test_string_parts_not_iterated_charwise(self, tmp_path):
        """A string 'parts' value is not iterated character-by-character (IMP-8)."""
        data = {"conversations": [{
            "id": "c1", "title": "T",
            "turns": [{"id": "m1", "role": "user", "parts": "hello"}],
        }]}
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        msg = list(convs[0].messages.values())[0]
        # Must NOT produce 5 single-character text blocks.
        text_blocks = [b for b in msg.content if b.get("type") == "text"]
        assert all(b["text"] != "h" for b in text_blocks)
        # Falls back to empty content (single empty text block), not char soup.
        assert len(msg.content) <= 1

    def test_numeric_message_timestamp(self, tmp_path):
        """A numeric message timestamp is parsed, not crashed on (IMP-4)."""
        data = {"conversations": [{
            "id": "c1", "title": "T",
            "turns": [{"id": "m1", "role": "user", "parts": [{"text": "hi"}],
                       "timestamp": 1700000000}],
        }]}
        f = tmp_path / "gemini.json"
        f.write_text(json.dumps(data))
        convs = gemini_import(str(f))
        msg = list(convs[0].messages.values())[0]
        assert msg.created_at is not None
        assert msg.created_at.year == 2023

    def test_detect_with_bom(self, tmp_path):
        """A UTF-8 BOM does not break Gemini detect() (IMP-6)."""
        f = tmp_path / "gemini.json"
        payload = json.dumps({"conversations": [{"id": "c1"}]})
        f.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
        assert gemini_detect(str(f)) is True

    def test_import_with_bom(self, tmp_path):
        """A UTF-8 BOM does not break Gemini import_path() (IMP-6)."""
        data = {"conversations": [{"id": "c1", "title": "T",
                                   "turns": [{"id": "m1", "role": "user",
                                              "parts": [{"text": "hi"}]}]}]}
        f = tmp_path / "gemini.json"
        f.write_bytes(b"\xef\xbb\xbf" + json.dumps(data).encode("utf-8"))
        convs = gemini_import(str(f))
        assert len(convs) == 1
        assert convs[0].id == "c1"


# IMP-3/IMP-4: Claude Code importers with non-dict message and numeric timestamp

class TestClaudeCodeMalformed:
    def test_parse_iso_non_string(self):
        """parse_iso tolerates a non-string timestamp, returning None (IMP-4)."""
        assert cc_parse_iso(1700000000) is None
        assert cc_parse_iso(None) is None
        assert cc_parse_iso({"k": "v"}) is None

    def test_parse_iso_valid_string(self):
        """parse_iso still parses a valid ISO string."""
        dt = cc_parse_iso("2026-02-18T10:00:00Z")
        assert dt is not None
        assert dt.year == 2026

    def test_non_dict_message_conversation_only(self, tmp_path):
        """A non-dict 'message' value does not crash conversation_only import (IMP-4)."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            # message is a string, not a dict
            _cc_event("assistant", uuid="a1", parent_uuid="u1", message="oops"),
            _cc_event("assistant", uuid="a2", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        assert len(convs) == 1
        # The bad assistant is skipped; user + good assistant remain.
        assert len(convs[0].messages) == 2

    def test_non_dict_message_full(self, tmp_path):
        """A non-dict 'message' value does not crash full import (IMP-4)."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1", message="oops"),
            _cc_event("assistant", uuid="a2", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_full_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 2

    def test_numeric_timestamp_conversation_only(self, tmp_path):
        """A numeric timestamp does not crash conversation_only import (IMP-4)."""
        events = [
            _cc_event("user", uuid="u1", timestamp=1700000000,
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1", timestamp=1700000001,
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        _write_jsonl(f, events)
        convs = claude_code_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 2
        # Numeric timestamp is not parseable as ISO, so created_at is None.
        for msg in convs[0].messages.values():
            assert msg.created_at is None


# IMP-5: Claude Code common file reading tolerates non-UTF8 bytes

class TestClaudeCodeNonUtf8:
    def test_parse_records_tolerates_non_utf8(self, tmp_path):
        """A non-UTF8 byte in the JSONL file is tolerated, valid lines still parse (IMP-5)."""
        events = [
            _cc_event("user", uuid="u1",
                      message={"role": "user", "content": "Hello"}),
            _cc_event("assistant", uuid="a1", parent_uuid="u1",
                      message={"role": "assistant", "model": "claude-opus-4-6",
                               "content": [{"type": "text", "text": "Hi!"}]}),
        ]
        f = tmp_path / "session.jsonl"
        good = "\n".join(json.dumps(e) for e in events).encode("utf-8")
        # Inject an invalid UTF-8 byte (0xff) on its own corrupt line.
        f.write_bytes(good + b"\n" + b"\xff\xfe garbage line\n")
        convs = claude_code_import(str(f))
        assert len(convs) == 1
        assert len(convs[0].messages) == 2

    def test_detect_file_tolerates_non_utf8(self, tmp_path):
        """detect_file tolerates a non-UTF8 byte and still detects a valid session (IMP-5)."""
        events = [_cc_event("user", uuid="u1",
                            message={"role": "user", "content": "Hello"})]
        f = tmp_path / "session.jsonl"
        good = "\n".join(json.dumps(e) for e in events).encode("utf-8")
        f.write_bytes(good + b"\n" + b"\xff\xfe garbage\n")
        assert claude_code_detect(str(f)) is True


# ── Codex ─────────────────────────────────────────────────────
# Synthetic rollouts in the two on-disk layouts measured from real Codex
# sessions (2025-08 .. 2026-10). See llm_memex/importers/codex.py.

from llm_memex.importers import codex as codex_mod  # noqa: E402
from llm_memex.importers.codex import detect as codex_detect  # noqa: E402
from llm_memex.importers.codex import import_path as codex_import  # noqa: E402


def _cx(rtype, payload, ordinal, ts="2026-07-01T10:00:00Z"):
    return {"timestamp": ts, "type": rtype, "payload": payload, "ordinal": ordinal}


def _cx_meta(thread="th-1", ordinal=0, source="cli", cli_version="0.150.0", **extra):
    payload = {
        "id": thread, "session_id": thread, "timestamp": "2026-07-01T10:00:00Z",
        "cwd": "/work/proj", "originator": "codex-tui", "cli_version": cli_version,
        "source": source, "git": {"branch": "main", "commit_hash": "abc",
                                  "repository_url": "x"},
    }
    payload.update(extra)
    return _cx("session_meta", payload, ordinal)


def _cx_msg(role, text, ordinal, ts="2026-07-01T10:00:01Z"):
    ptype = "output_text" if role == "assistant" else "input_text"
    return _cx("response_item", {"type": "message", "role": role,
                                 "content": [{"type": ptype, "text": text}]},
               ordinal, ts)


def _cx_write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _cx_v2_session(thread="th-1", source="cli"):
    return [
        _cx_meta(thread, source=source),
        _cx("turn_context", {"model": "gpt-5.5", "cwd": "/work/proj",
                             "approval_policy": "never"}, 1),
        _cx("response_item", {"type": "message", "role": "developer",
                              "content": [{"type": "input_text",
                                           "text": "<permissions>...</permissions>"}]}, 2),
        _cx("response_item", {"type": "message", "role": "user", "content": [
            {"type": "input_text",
             "text": "<environment_context>\n  <cwd>/work/proj</cwd>\n</environment_context>"},
        ]}, 3),
        _cx_msg("user", "Fix the failing test please", 4),
        _cx("event_msg", {"type": "item_completed", "item": {
            "type": "UserMessage", "id": "x",
            "content": [{"type": "text", "text": "Fix the failing test please"}]}}, 5),
        _cx("response_item", {"type": "reasoning", "summary": [],
                              "encrypted_content": "gAAA"}, 6),
        _cx("response_item", {"type": "function_call", "name": "shell",
                              "arguments": "{}", "call_id": "c1"}, 7),
        _cx("response_item", {"type": "function_call_output", "call_id": "c1",
                              "output": "ok"}, 8),
        _cx_msg("assistant", "Fixed: the fixture was stale.", 9,
                ts="2026-07-01T10:05:00Z"),
        _cx("event_msg", {"type": "item_completed", "item": {
            "type": "AgentMessage", "id": "y",
            "content": [{"type": "text", "text": "Fixed: the fixture was stale."}]}}, 10),
        _cx("brand_new_record_type", {"anything": 1}, 11),
    ]


class TestCodexDetect:
    def test_detects_both_layouts(self, tmp_path):
        v2 = _cx_write(tmp_path / "a.jsonl", _cx_v2_session())
        v1 = _cx_write(tmp_path / "b.jsonl", [
            {"id": "old-1", "timestamp": "2025-08-07T23:15:55Z", "instructions": None},
        ])
        assert codex_detect(str(v2)) and codex_detect(str(v1))
        assert codex_detect(str(tmp_path))

    def test_rejects_claude_code_and_other_jsonl(self, tmp_path):
        cc = tmp_path / "cc.jsonl"
        _write_jsonl(cc, [_cc_event("user", message={"role": "user", "content": "hi"})])
        other = _cx_write(tmp_path / "o.jsonl", [{"type": "x", "id": "1"}])
        assert not codex_detect(str(cc)) and not codex_detect(str(other))

    def test_claude_code_detectors_do_not_claim_codex(self, tmp_path):
        f = _cx_write(tmp_path / "a.jsonl", _cx_v2_session())
        assert not claude_code_detect(str(f))
        assert not claude_code_full_detect(str(f))


class TestCodexImportLayout2:
    def _conv(self, tmp_path, **kw):
        f = _cx_write(tmp_path / "rollout-2026-07-01T10-00-00-th-1.jsonl",
                      _cx_v2_session(**kw))
        convs = codex_import(str(f))
        assert len(convs) == 1
        return convs[0]

    def test_conversation_skeleton(self, tmp_path):
        conv = self._conv(tmp_path)
        msgs = list(conv.messages.values())
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].get_text() == "Fix the failing test please"
        assert msgs[1].get_text() == "Fixed: the fixture was stale."
        assert msgs[1].parent_id == msgs[0].id and msgs[0].parent_id is None
        assert msgs[1].model == "gpt-5.5"
        assert conv.id == "th-1" and conv.source == "codex" and conv.tags == ["codex"]
        assert conv.model == "gpt-5.5"
        assert conv.updated_at > conv.created_at

    def test_message_ids_are_thread_segment_ordinal(self, tmp_path):
        conv = self._conv(tmp_path)
        seg = "2026-07-01T10:00:00Z"  # the segment's own session_meta timestamp
        assert list(conv.messages) == [f"th-1:{seg}:o4", f"th-1:{seg}:o9"]

    def test_title_falls_back_to_first_human_message(self, tmp_path):
        assert self._conv(tmp_path).title == "Fix the failing test please"

    def test_session_context_and_provenance(self, tmp_path):
        conv = self._conv(tmp_path)
        assert conv.metadata["cwd"] == "/work/proj"
        assert conv.metadata["git_branch"] == "main"
        assert conv.metadata["client_version"] == "0.150.0"
        assert conv.metadata["originator"] == "codex-tui"
        assert conv.metadata["importer_mode"] == "conversation_only"
        prov = conv.metadata["_provenance"]
        assert prov["source_type"] == "codex"
        assert prov["source_format"] == "codex-rollout/2"
        assert prov["source_id"] == "th-1"

    def test_image_input_becomes_media_block(self, tmp_path):
        records = _cx_v2_session()
        records.insert(5, _cx("response_item", {"type": "message", "role": "user",
                                                "content": [
            {"type": "input_text", "text": "<image name=[Image #1]>"},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
            {"type": "input_text", "text": "</image>"},
            {"type": "input_text", "text": "what is this?"},
        ]}, 50))
        f = _cx_write(tmp_path / "r.jsonl", records)
        user = [m for m in codex_import(str(f))[0].messages.values()
                if m.role == "user"][1]
        media = [b for b in user.content if b["type"] == "media"]
        assert media == [{"type": "media", "media_type": "image/png",
                          "data": "iVBORw0KGgo="}]
        # The <image ...> / </image> wrapper parts are split across parts, so
        # neither is a single element; they stay as text, in order.
        texts = [b["text"] for b in user.content if b["type"] == "text"]
        assert texts == ["<image name=[Image #1]>", "</image>", "what is this?"]

    def test_lenient_json_and_truncated_last_line(self, tmp_path):
        f = tmp_path / "r.jsonl"
        lines = [json.dumps(r) for r in _cx_v2_session()]
        # A raw control character inside a string (Codex writes these).
        lines.insert(5, json.dumps(_cx_msg("user", "tabXHERE", 40)).replace("X", "\t"))
        lines.append('{"timestamp": "2026-07-01T10:06:00Z", "type": "respon')
        f.write_text("\n".join(lines) + "\n")
        texts = [m.get_text() for m in codex_import(str(f))[0].messages.values()]
        assert "tab\tHERE" in texts


class TestCodexClientContext:
    @pytest.mark.parametrize("text", [
        "<environment_context>\n<cwd>/x</cwd>\n</environment_context>",
        "<codex_internal_context source=\"x\">\nstuff\n</codex_internal_context>",
        "  <heartbeat>\ntick\n</heartbeat>\n",
        "<a-new-tag>\nwhatever\n</a-new-tag>",
        # AGENTS.md: a heading line, then the block (548 real titles were this).
        "# AGENTS.md instructions for /work/proj\n\n<INSTRUCTIONS>\n- be terse\n</INSTRUCTIONS>",
        "<outer>\n<outer>\nnested same name\n</outer>\n</outer>",
    ])
    def test_block_elements_are_client_context(self, text):
        assert codex_mod._is_client_context(text, set())

    @pytest.mark.parametrize("text", [
        "Fix this <b>now</b>",
        "<div>hello world, why does this render wrong?</div>",
        '<svg width="10"></svg>',
        "<a>x</a> please compare with <a>y</a>",
        "<a>\nx\n</a> please compare with <a>\ny\n</a>",
        "<note>\nfirst\n</note>\nand my question\n<note>\nsecond\n</note>",
        "<image name=[Image #1]>",
        "a < b and c > d",
        "<>",
        "# Just a heading",
    ])
    def test_human_text_is_not_client_context(self, text):
        assert not codex_mod._is_client_context(text, set())

    def test_ui_stream_record_overrides_shape(self):
        reply = "<send_user_message_question_reply>\nyes, ship it\n</send_user_message_question_reply>"
        assert codex_mod._is_client_context(reply, set())
        assert not codex_mod._is_client_context(reply, {reply})

    def test_import_keeps_submitted_block_and_drops_unsubmitted(self, tmp_path):
        reply = "<send_user_message_question_reply>\nyes, ship it\n</send_user_message_question_reply>"
        records = _cx_v2_session()
        records += [
            _cx("response_item", {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": reply}]}, 20),
            _cx("event_msg", {"type": "item_completed", "item": {
                "type": "UserMessage", "id": "z",
                "content": [{"type": "text", "text": reply}]}}, 21),
        ]
        f = _cx_write(tmp_path / "r.jsonl", records)
        texts = [m.get_text() for m in codex_import(str(f))[0].messages.values()]
        assert reply in texts
        assert not any("environment_context" in t for t in texts)

    def test_agents_md_never_becomes_the_title(self, tmp_path):
        records = _cx_v2_session()
        records.insert(3, _cx("response_item", {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "# AGENTS.md instructions for /w\n\n<INSTRUCTIONS>\nx\n</INSTRUCTIONS>"},
        ]}, 30))
        f = _cx_write(tmp_path / "r.jsonl", records)
        conv = codex_import(str(f))[0]
        assert conv.title == "Fix the failing test please"
        assert not any("AGENTS.md" in m.get_text() for m in conv.messages.values())


class TestCodexImportLayout1:
    def test_pre_envelope_layout(self, tmp_path):
        f = _cx_write(tmp_path / "rollout-2025-08-07T23-15-55-old.jsonl", [
            {"id": "old-1", "timestamp": "2025-08-07T23:15:55Z", "instructions": None,
             "git": {"branch": "dev", "commit_hash": "c", "repository_url": "u"}},
            {"record_type": "state"},
            {"type": "message", "id": None, "role": "user",
             "content": [{"type": "input_text", "text": "hello codex"}]},
            {"type": "reasoning", "id": "r", "summary": [], "encrypted_content": "x"},
            {"type": "message", "id": None, "role": "assistant",
             "content": [{"type": "output_text", "text": "hello human"}]},
        ])
        conv = codex_import(str(f))[0]
        assert conv.id == "old-1"
        assert [m.get_text() for m in conv.messages.values()] == ["hello codex", "hello human"]
        assert conv.metadata["_provenance"]["source_format"] == "codex-rollout/1"
        assert conv.metadata["git_branch"] == "dev"
        assert "client_version" not in conv.metadata


class TestCodexSplitThreads:
    """A thread split across rollout files: the later file resumes from a fork
    point, and the earlier file's records past it are a kept branch."""

    def _split(self, tmp_path):
        day = tmp_path / "sessions" / "2026" / "10" / "13"
        first = [
            _cx_meta("th-s", cli_version="0.159.0"),
            _cx_msg("user", "q1", 1), _cx_msg("assistant", "a1", 2),
            _cx_msg("user", "q2 (abandoned branch)", 3),
            _cx_msg("assistant", "a2 (abandoned branch)", 4),
        ]
        # Second segment resumes at ordinal 3: its records reuse ordinals 3-4
        # with different content (measured on real split threads).
        second = [
            _cx_meta("th-s", ordinal=3, cli_version="0.163.0",
                     timestamp="2026-10-15T00:00:00Z"),
            _cx_msg("user", "q2 (resumed)", 4), _cx_msg("assistant", "a2 (resumed)", 5),
        ]
        _cx_write(day / "rollout-2026-10-13T00-00-00-th-s.jsonl", first)
        _cx_write(day.parent / "15" / "rollout-2026-10-15T00-00-00-th-s.jsonl", second)
        return tmp_path / "sessions"

    def test_one_conversation_with_both_branches(self, tmp_path):
        convs = codex_import(str(self._split(tmp_path)))
        assert len(convs) == 1
        msgs = convs[0].messages
        assert len(msgs) == 6
        # Resumed branch forks from a1, the last message before ordinal 3.
        s0, s1 = "th-s:2026-07-01T10:00:00Z", "th-s:2026-10-15T00:00:00Z"
        assert msgs[f"{s1}:o4"].parent_id == f"{s0}:o2"
        assert msgs[f"{s0}:o3"].parent_id == f"{s0}:o2"
        assert msgs[f"{s1}:o5"].parent_id == f"{s1}:o4"

    def test_split_thread_metadata(self, tmp_path):
        conv = codex_import(str(self._split(tmp_path)))[0]
        assert conv.metadata["client_version"] == "0.163.0"  # newest segment
        assert len(conv.metadata["source_files"]) == 2
        assert conv.metadata["_provenance"]["source_file"].endswith(
            "rollout-2026-10-13T00-00-00-th-s.jsonl")

    def test_message_ids_stable_when_a_segment_is_added(self, tmp_path):
        sessions = self._split(tmp_path)
        before = set(codex_import(str(sessions))[0].messages)
        _cx_write(sessions / "2026" / "10" / "17" / "rollout-2026-10-17T00-00-00-th-s.jsonl",
                  [_cx_meta("th-s", ordinal=6, timestamp="2026-10-17T00:00:00Z"),
                   _cx_msg("user", "q3", 6)])
        after = set(codex_import(str(sessions))[0].messages)
        assert before < after


    def test_single_segment_import_assembles_the_whole_thread(self, tmp_path):
        """Importing one file (or the cron's per-file fallback) must not replace
        a split thread with part of itself or renumber its messages."""
        sessions = self._split(tmp_path)
        whole = codex_import(str(sessions))[0]
        one_file = next((sessions / "2026" / "10" / "15").glob("*.jsonl"))
        part = codex_import(str(one_file))[0]
        assert set(part.messages) == set(whole.messages)
        sub = codex_import(str(sessions / "2026" / "10" / "13"))[0]
        assert set(sub.messages) == set(whole.messages)

    def test_segment_in_another_layout_is_kept(self, tmp_path):
        day = tmp_path / "sessions" / "2025" / "10" / "01"
        _cx_write(day / "rollout-2025-10-01T00-00-00-mix.jsonl", [
            {"id": "mix", "timestamp": "2025-10-01T00:00:00Z", "instructions": None},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "old q"}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "old a"}]},
        ])
        _cx_write(day / "rollout-2025-10-02T00-00-00-mix.jsonl", [
            _cx_meta("mix", ordinal=5, timestamp="2025-10-02T00:00:00Z"),
            _cx_msg("user", "new q", 6),
        ])
        conv = codex_import(str(tmp_path / "sessions"))[0]
        assert sorted(m.get_text() for m in conv.messages.values()) == [
            "new q", "old a", "old q"]
        new_q = next(m for m in conv.messages.values() if m.get_text() == "new q")
        assert conv.messages[new_q.parent_id].get_text() == "old a"
        assert conv.metadata["_provenance"]["source_format"] == \
            "codex-rollout/1+codex-rollout/2"

    def test_mixed_and_repeated_ordinals_never_collide(self, tmp_path):
        records = [_cx_meta("dup"), _cx_msg("user", "u1", 1), _cx_msg("assistant", "a1", 2)]
        no_ord = _cx_msg("user", "u2", 0)
        del no_ord["ordinal"]
        records += [no_ord, _cx_msg("assistant", "a2", 2)]  # ordinal 2 repeated
        conv = codex_import(str(_cx_write(tmp_path / "r.jsonl", records)))[0]
        assert len(conv.messages) == 4
        for m in conv.messages.values():
            assert m.parent_id != m.id
        roots = [m for m in conv.messages.values() if m.parent_id is None]
        assert len(roots) == 1

    def test_one_failing_thread_does_not_sink_the_batch(self, tmp_path, monkeypatch):
        _cx_write(tmp_path / "a.jsonl", _cx_v2_session("good"))
        _cx_write(tmp_path / "b.jsonl", _cx_v2_session("bad"))
        real = codex_mod._import_thread

        def flaky(paths, skipped):
            if "b.jsonl" in paths[0]:
                raise TypeError("an unforeseen shape")
            return real(paths, skipped)
        monkeypatch.setattr(codex_mod, "_import_thread", flaky)
        assert [c.id for c in codex_import(str(tmp_path))] == ["good"]


class TestCodexTitlesAndSubagents:
    def _home(self, tmp_path):
        sessions = tmp_path / ".codex" / "sessions" / "2026" / "07" / "01"
        _cx_write(sessions / "rollout-2026-07-01T10-00-00-parent.jsonl",
                  _cx_v2_session("parent"))
        child = _cx_v2_session("child", source={"subagent": {"thread_spawn": {
            "parent_thread_id": "parent", "depth": 1, "agent_nickname": "Ada",
            "agent_role": "explorer", "agent_path": "x"}}})
        # Child file sorts before its parent by name; import must still order
        # parents first (parent_conversation_id is a foreign key).
        _cx_write(sessions / "rollout-2026-07-01T09-00-00-child.jsonl", child)
        _cx_write(sessions / "rollout-2026-07-01T11-00-00-guard.jsonl",
                  _cx_v2_session("guard", source={"subagent": {"other": "guardian"}}))
        _cx_write(sessions / "rollout-2026-07-01T12-00-00-orphan.jsonl",
                  _cx_v2_session("orphan", source={"subagent": {"thread_spawn": {
                      "parent_thread_id": "pruned", "depth": 1}}}))
        index = tmp_path / ".codex" / "session_index.jsonl"
        index.write_text("\n".join(json.dumps(r) for r in [
            {"id": "parent", "thread_name": "Old name", "updated_at": "x"},
            {"id": "parent", "thread_name": "Fix flaky test", "updated_at": "y"},
        ]) + "\n")
        return tmp_path / ".codex" / "sessions"

    def test_titles_from_session_index_latest_wins(self, tmp_path):
        by_id = {c.id: c for c in codex_import(str(self._home(tmp_path)))}
        assert by_id["parent"].title == "Fix flaky test"
        assert by_id["child"].title == "Fix the failing test please"

    def test_subagents_linked_and_ordered(self, tmp_path):
        convs = codex_import(str(self._home(tmp_path)))
        ids = [c.id for c in convs]
        by_id = {c.id: c for c in convs}
        assert ids.index("parent") < ids.index("child")
        child = by_id["child"]
        assert child.parent_conversation_id == "parent"
        assert "codex-subagent" in child.tags
        assert child.metadata["agent_nickname"] == "Ada"
        assert child.metadata["subagent_kind"] == "thread_spawn"
        assert by_id["guard"].metadata["subagent_kind"] == "guardian"
        assert by_id["guard"].parent_conversation_id is None

    def test_subagent_without_human_turn_titled_by_nickname(self, tmp_path):
        records = [
            _cx_meta("sub", source={"subagent": {"thread_spawn": {
                "parent_thread_id": "p", "depth": 1, "agent_nickname": "Ada",
                "agent_role": "explorer"}}}),
            _cx("response_item", {"type": "agent_message", "author": "/root",
                                  "recipient": "/root/ada", "content": []}, 1),
            _cx_msg("assistant", "done", 2),
        ]
        f = _cx_write(tmp_path / "r.jsonl", records)
        assert codex_import(str(f))[0].title == "Subagent Ada (explorer)"

    def test_missing_parent_keeps_metadata_without_link(self, tmp_path):
        orphan = {c.id: c for c in codex_import(str(self._home(tmp_path)))}["orphan"]
        assert orphan.parent_conversation_id is None
        assert orphan.metadata["parent_thread_id"] == "pruned"
