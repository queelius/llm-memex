"""Tests for scripts framework discovery and CLI integration."""
import argparse
import json
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_memex.db import Database
from llm_memex.models import Conversation, Message, text_block
from llm_memex.scripts import discover_scripts, load_script


class TestScriptDiscovery:
    def test_discover_finds_builtin_scripts(self, tmp_path):
        """Scripts in the built-in dir are discovered."""
        script = tmp_path / "hello.py"
        script.write_text(
            '"""Say hello."""\n'
            'def register_args(parser): pass\n'
            'def run(db, args, apply=False): return {}\n'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                scripts = discover_scripts()
        assert "hello" in scripts
        assert scripts["hello"]["description"] == "Say hello."

    def test_discover_skips_underscore_files(self, tmp_path):
        """Files starting with _ are skipped."""
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "_utils.py").write_text(
            '"""Private."""\ndef register_args(p): pass\ndef run(d,a,apply=False): return {}'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                scripts = discover_scripts()
        assert len(scripts) == 0

    def test_discover_user_scripts(self, tmp_path):
        """User scripts in ~/.memex/scripts/ are discovered."""
        builtin = tmp_path / "builtin"
        builtin.mkdir()
        user = tmp_path / "user"
        user.mkdir()
        (user / "custom.py").write_text(
            '"""Custom script."""\ndef register_args(p): pass\ndef run(d,a,apply=False): return {}'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=builtin):
            with patch("llm_memex.scripts._user_dir", return_value=user):
                scripts = discover_scripts()
        assert "custom" in scripts

    def test_load_script_returns_module(self, tmp_path):
        """load_script returns a module with the convention interface."""
        (tmp_path / "example.py").write_text(
            '"""Example."""\ndef register_args(p): pass\ndef run(d,a,apply=False): return {"ok": True}'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                mod = load_script("example")
        assert hasattr(mod, "register_args")
        assert hasattr(mod, "run")

    def test_load_script_not_found_raises(self, tmp_path):
        """load_script raises ValueError for unknown scripts."""
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                with pytest.raises(ValueError, match="not found"):
                    load_script("nonexistent")

    def test_user_shadows_builtin(self, tmp_path):
        """User script with same name shadows built-in."""
        builtin = tmp_path / "builtin"
        builtin.mkdir()
        user = tmp_path / "user"
        user.mkdir()
        (builtin / "dupe.py").write_text(
            '"""Built-in."""\ndef register_args(p): pass\ndef run(d,a,apply=False): return {}'
        )
        (user / "dupe.py").write_text(
            '"""User version."""\ndef register_args(p): pass\ndef run(d,a,apply=False): return {}'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=builtin):
            with patch("llm_memex.scripts._user_dir", return_value=user):
                scripts = discover_scripts()
        assert scripts["dupe"]["description"] == "User version."

    def test_discover_skips_invalid_modules(self, tmp_path):
        """Modules without required interface are skipped."""
        (tmp_path / "broken.py").write_text(
            '"""No interface."""\ndef something(): pass\n'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                scripts = discover_scripts()
        assert len(scripts) == 0


class TestCLIRun:
    """CLI tests use _cmd_run directly to avoid _get_version import issue
    (tests/llm_memex/__init__.py shadows the real llm_memex package)."""

    def test_run_list(self, tmp_path, capsys):
        """memex run --list shows available scripts."""
        from llm_memex.cli import _cmd_run
        (tmp_path / "hello.py").write_text(
            '"""Say hello."""\ndef register_args(p): pass\ndef run(d,a,apply=False): return {}'
        )
        args = argparse.Namespace(name=None, list=True, apply=False, verbose=False,
                                  db=str(tmp_path))
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                _cmd_run(args, [])
        out = capsys.readouterr().out
        assert "hello" in out
        assert "Say hello." in out

    def test_run_no_name_shows_list(self, tmp_path, capsys):
        """memex run without name shows available scripts or 'no scripts'."""
        from llm_memex.cli import _cmd_run
        (tmp_path / "hello.py").write_text(
            '"""Say hello."""\ndef register_args(p): pass\ndef run(d,a,apply=False): return {}'
        )
        args = argparse.Namespace(name=None, list=False, apply=False, verbose=False,
                                  db=str(tmp_path))
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                _cmd_run(args, [])
        out = capsys.readouterr().out
        assert "Available scripts" in out

    def test_run_unknown_errors(self, tmp_path, capsys):
        """memex run nonexistent prints error and exits."""
        from llm_memex.cli import _cmd_run
        args = argparse.Namespace(name="nonexistent", list=False, apply=False,
                                  verbose=False, db=str(tmp_path))
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                with pytest.raises(SystemExit):
                    _cmd_run(args, [])

    def test_run_executes_script(self, tmp_path, capsys):
        """memex run <name> executes the script's run function."""
        from llm_memex.cli import _cmd_run
        (tmp_path / "hello.py").write_text(
            '"""Say hello."""\n'
            'def register_args(p): pass\n'
            'def run(db, args, apply=False):\n'
            '    print("hello from script")\n'
            '    return {}\n'
        )
        db_dir = tmp_path / "testdb"
        db_dir.mkdir()
        args = argparse.Namespace(name="hello", list=False, apply=False,
                                  verbose=False, db=str(db_dir))
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                _cmd_run(args, [])
        out = capsys.readouterr().out
        assert "hello from script" in out


def _make_conv(id="c1", title="Test", msg_text="hello"):
    now = datetime.now()
    conv = Conversation(id=id, created_at=now, updated_at=now, title=title,
                        source="test", model="gpt-4")
    conv.add_message(Message(id="m1", role="user", content=[text_block(msg_text)]))
    return conv


class TestScriptDiscoveryEdgeCases:
    def test_discover_does_not_execute_modules_that_raise_on_import(self, tmp_path):
        """Discovery parses statically: a module whose IMPORT would raise is
        still listed (its top-level code is never executed during discovery).

        REDACT-3: discovery must not exec_module. A module that would blow up
        on import (or run arbitrary side effects) must not do so merely from
        being discovered/listed. The failure is deferred to load_script.
        """
        (tmp_path / "badmod.py").write_text(
            '"""Bad module."""\n'
            'import nonexistent_module_xyz\n'
            'def register_args(p): pass\n'
            'def run(d,a,apply=False): return {}\n'
        )
        (tmp_path / "goodmod.py").write_text(
            '"""Good module."""\ndef register_args(p): pass\ndef run(d,a,apply=False): return {}'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                scripts = discover_scripts()
        # badmod has the required interface (statically) and is listed without
        # ever importing nonexistent_module_xyz.
        assert "badmod" in scripts
        assert scripts["badmod"]["description"] == "Bad module."
        assert "goodmod" in scripts

    def test_user_script_not_executed_during_discovery(self, tmp_path):
        """REDACT-3: a module dropped in the user scripts dir is NOT executed
        by discovery or listing.

        Arbitrary code at module top level (here: writing a sentinel file and
        raising) must not run merely because `llm-memex run --list` enumerates
        scripts. Discovery reads the description statically via ast and defers
        any execution to load_script.
        """
        builtin = tmp_path / "builtin"
        builtin.mkdir()
        user = tmp_path / "user"
        user.mkdir()
        sentinel = tmp_path / "SENTINEL_EXECUTED"
        # Top-level code writes a sentinel AND raises. If discovery imports it,
        # the sentinel appears (and/or the exception is swallowed).
        (user / "evil.py").write_text(
            '"""Evil script."""\n'
            'from pathlib import Path\n'
            f'Path({str(sentinel)!r}).write_text("executed")\n'
            'raise RuntimeError("top-level code ran")\n'
            'def register_args(p): pass\n'
            'def run(d,a,apply=False): return {}\n'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=builtin):
            with patch("llm_memex.scripts._user_dir", return_value=user):
                scripts = discover_scripts()
        assert not sentinel.exists(), "discovery executed user module top-level code"
        # Still discoverable + described, just not executed.
        assert "evil" in scripts
        assert scripts["evil"]["description"] == "Evil script."

    def test_run_list_does_not_execute_user_scripts(self, tmp_path):
        """REDACT-3: `run --list` enumerates user scripts without executing them."""
        from llm_memex.cli import _cmd_run
        builtin = tmp_path / "builtin"
        builtin.mkdir()
        user = tmp_path / "user"
        user.mkdir()
        sentinel = tmp_path / "SENTINEL_LIST"
        (user / "evil.py").write_text(
            '"""Evil list script."""\n'
            'from pathlib import Path\n'
            f'Path({str(sentinel)!r}).write_text("executed")\n'
            'def register_args(p): pass\n'
            'def run(d,a,apply=False): return {}\n'
        )
        args = argparse.Namespace(name=None, list=True, apply=False, verbose=False,
                                  db=str(tmp_path))
        with patch("llm_memex.scripts._builtin_dir", return_value=builtin):
            with patch("llm_memex.scripts._user_dir", return_value=user):
                _cmd_run(args, [])
        assert not sentinel.exists(), "run --list executed a user script"

    def test_extract_description_no_docstring(self, tmp_path):
        """Module without docstring gets empty description."""
        (tmp_path / "nodoc.py").write_text(
            'def register_args(p): pass\ndef run(d,a,apply=False): return {}'
        )
        with patch("llm_memex.scripts._builtin_dir", return_value=tmp_path):
            with patch("llm_memex.scripts._user_dir", return_value=tmp_path / "nope"):
                scripts = discover_scripts()
        assert "nodoc" in scripts
        assert scripts["nodoc"]["description"] == ""


class TestEnrichTrivialScript:
    def test_has_convention_interface(self):
        mod = load_script("enrich_trivial")
        assert hasattr(mod, "register_args")
        assert hasattr(mod, "run")
        assert mod.__doc__

    def test_dry_run_returns_stats(self, tmp_db_path):
        """Dry run scans but doesn't write."""
        db = Database(tmp_db_path)
        conv = _make_conv(msg_text="hi")
        conv.message_count = 1
        db.save_conversation(conv)
        db.close()

        db = Database(tmp_db_path, readonly=True)
        mod = load_script("enrich_trivial")
        script_args = argparse.Namespace(max_messages=4)
        stats = mod.run(db, script_args, apply=False)
        assert stats["greeting"] >= 0  # returns stats dict
        # DB unchanged — no enrichments
        db.close()
        db = Database(tmp_db_path, readonly=True)
        enrichments = db.get_enrichments("c1")
        assert len(enrichments) == 0
        db.close()

    def test_apply_writes_enrichments(self, tmp_db_path):
        """Apply mode writes enrichments to DB."""
        db = Database(tmp_db_path)
        # Empty conversation (0 messages) — trivial with confidence 1.0
        now = datetime.now()
        conv = Conversation(id="empty1", created_at=now, updated_at=now,
                            title="Empty", message_count=0, source="test")
        db.save_conversation(conv)

        mod = load_script("enrich_trivial")
        script_args = argparse.Namespace(max_messages=4)
        stats = mod.run(db, script_args, apply=True)
        assert stats["trivial"] >= 1

        enrichments = db.get_enrichments("empty1")
        importance_vals = [e for e in enrichments if e["type"] == "importance"]
        assert len(importance_vals) == 1
        assert importance_vals[0]["value"] == "trivial"
        db.close()

    def test_skips_already_enriched_conversations(self, tmp_db_path):
        """Conversations with existing importance enrichments are skipped."""
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(id="already", created_at=now, updated_at=now,
                            title="Already Done", message_count=0, source="test")
        db.save_conversation(conv)
        db.save_enrichment("already", "importance", "high", "user")

        mod = load_script("enrich_trivial")
        script_args = argparse.Namespace(max_messages=4)
        stats = mod.run(db, script_args, apply=True)
        assert stats["skipped"] >= 1
        # Original enrichment still there, not overwritten
        enrichments = db.get_enrichments("already")
        assert len(enrichments) == 1
        assert enrichments[0]["value"] == "high"
        db.close()


class TestClassifyConversation:
    """Unit tests for classify_conversation covering all classification branches."""

    def _make_conv_dict(self, msg_count=1, title="Test"):
        return {"id": "c1", "message_count": msg_count, "title": title, "source": "test"}

    def _make_msg(self, text, role="user"):
        return {"role": role, "content": json.dumps([{"type": "text", "text": text}])}

    def test_trivial_pattern_match(self):
        """Messages matching TRIVIAL_PATTERNS (e.g. 'test', 'asdf') are trivial."""
        from llm_memex.scripts.enrich_trivial import classify_conversation
        conv = self._make_conv_dict(msg_count=1)
        messages = [self._make_msg("test")]
        enrichments = classify_conversation(conv, messages)
        importance = [e for e in enrichments if e["type"] == "importance"]
        assert len(importance) == 1
        assert importance[0]["value"] == "trivial"
        assert importance[0]["confidence"] == 0.95

    def test_trivial_pattern_aaa(self):
        """Repeated characters like 'aaa' are trivial."""
        from llm_memex.scripts.enrich_trivial import classify_conversation
        conv = self._make_conv_dict(msg_count=1)
        messages = [self._make_msg("aaaa")]
        enrichments = classify_conversation(conv, messages)
        importance = [e for e in enrichments if e["type"] == "importance"]
        assert len(importance) == 1
        assert importance[0]["value"] == "trivial"

    def test_no_user_text_is_trivial(self):
        """Conversation with messages but no user text content is trivial."""
        from llm_memex.scripts.enrich_trivial import classify_conversation
        conv = self._make_conv_dict(msg_count=1)
        # Only assistant messages, no user text
        messages = [{"role": "assistant", "content": json.dumps([{"type": "text", "text": "hi"}])}]
        enrichments = classify_conversation(conv, messages)
        importance = [e for e in enrichments if e["type"] == "importance"]
        assert len(importance) == 1
        assert importance[0]["value"] == "trivial"
        assert importance[0]["confidence"] == 0.95

    def test_untitled_with_short_text_is_trivial(self):
        """Untitled conversation with short first user text is trivial."""
        from llm_memex.scripts.enrich_trivial import classify_conversation
        conv = self._make_conv_dict(msg_count=1, title="Untitled")
        messages = [self._make_msg("ok")]
        enrichments = classify_conversation(conv, messages)
        importance = [e for e in enrichments if e["type"] == "importance"]
        assert len(importance) == 1
        assert importance[0]["value"] == "trivial"
        assert importance[0]["confidence"] == 0.9

    def test_short_two_message_conv_is_trivial(self):
        """2-message conversation with very short user text is trivial."""
        from llm_memex.scripts.enrich_trivial import classify_conversation
        conv = self._make_conv_dict(msg_count=2, title="Named Conversation")
        messages = [self._make_msg("hi"), self._make_msg("ok", role="assistant")]
        enrichments = classify_conversation(conv, messages)
        importance = [e for e in enrichments if e["type"] == "importance"]
        assert len(importance) == 1
        assert importance[0]["value"] == "trivial"
        assert importance[0]["confidence"] == 0.85

    def test_short_two_message_conv_is_brief(self):
        """2-message conversation with moderate text is classified as brief."""
        from llm_memex.scripts.enrich_trivial import classify_conversation
        conv = self._make_conv_dict(msg_count=2, title="Named Conversation")
        # Text longer than 20 chars but under 100
        messages = [
            self._make_msg("How do I install numpy in Python?"),
            self._make_msg("pip install numpy", role="assistant"),
        ]
        enrichments = classify_conversation(conv, messages)
        importance = [e for e in enrichments if e["type"] == "importance"]
        assert len(importance) == 1
        assert importance[0]["value"] == "brief"
        assert importance[0]["confidence"] == 0.7

    def test_greeting_detected(self):
        """Greeting patterns produce a topic=greeting enrichment."""
        from llm_memex.scripts.enrich_trivial import classify_conversation
        conv = self._make_conv_dict(msg_count=1)
        messages = [self._make_msg("hello there, how are you doing today?")]
        enrichments = classify_conversation(conv, messages)
        topic = [e for e in enrichments if e["type"] == "topic" and e["value"] == "greeting"]
        assert len(topic) == 1
        assert topic[0]["confidence"] == 0.9

    def test_non_trivial_conversation_no_enrichments(self):
        """Substantive conversation with enough content gets no enrichments."""
        from llm_memex.scripts.enrich_trivial import classify_conversation
        conv = self._make_conv_dict(msg_count=4, title="Named Conversation")
        messages = [
            self._make_msg("Can you explain how Python generators work? I want to understand "
                           "yield and how they differ from regular functions that return lists."),
            self._make_msg("Generators are a special type of function...", role="assistant"),
            self._make_msg("Can you show an example with send()?"),
            self._make_msg("Sure, here's an example...", role="assistant"),
        ]
        enrichments = classify_conversation(conv, messages)
        assert len(enrichments) == 0

    def test_extract_user_text_with_invalid_json(self):
        """extract_user_text handles invalid JSON gracefully."""
        from llm_memex.scripts.enrich_trivial import extract_user_text
        assert extract_user_text("not valid json{{{") == ""

    def test_extract_user_text_with_non_list_json(self):
        """extract_user_text handles non-list JSON gracefully."""
        from llm_memex.scripts.enrich_trivial import extract_user_text
        assert extract_user_text('{"type": "text"}') == ""
