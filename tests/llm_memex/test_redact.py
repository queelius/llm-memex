"""Tests for the redact script — detection, mutation, and interactive review."""
import argparse
import json
from datetime import datetime
from unittest.mock import patch

import pytest

from llm_memex.db import Database
from llm_memex.models import Conversation, Message, text_block


def _make_conv_with_text(text, id="c1", title="Test"):
    now = datetime.now()
    conv = Conversation(id=id, created_at=now, updated_at=now, title=title,
                        source="test", message_count=1)
    conv.add_message(Message(id="m1", role="user", content=[text_block(text)]))
    return conv


def _make_args(words=None, patterns=None, pattern_file=None, level="word",
               yes=False, all_mode=False):
    return argparse.Namespace(
        words=words, patterns=patterns, pattern_file=pattern_file,
        level=level, yes=yes, match_mode="all" if all_mode else "any",
    )


# ── Detection Engine ────────────────────────────────────────────


class TestCompileMatchers:
    def test_words_case_insensitive(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(words=["fuck"])
        assert matchers[0][0].search("What the Fuck")

    def test_words_word_boundary(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(words=["ass"])
        assert matchers[0][0].search("what an ass")
        assert not matchers[0][0].search("class method")

    def test_patterns_regex(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(patterns=[r"sk-[a-zA-Z0-9]{20,}"])
        assert matchers[0][0].search("key: sk-abc123def456ghi789jkl0")

    def test_combined(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(words=["bad"], patterns=[r"\d{3}-\d{4}"])
        assert len(matchers) == 2

    def test_empty_raises(self):
        from llm_memex.scripts.redact import compile_matchers
        with pytest.raises(ValueError):
            compile_matchers()


class TestLoadPatternFile:
    def test_loads_patterns(self, tmp_path):
        from llm_memex.scripts.redact import load_pattern_file
        f = tmp_path / "test.txt"
        f.write_text("pattern1\npattern2\n")
        patterns = load_pattern_file(str(f))
        assert len(patterns) == 2

    def test_skips_comments_and_blanks(self, tmp_path):
        from llm_memex.scripts.redact import load_pattern_file
        f = tmp_path / "test.txt"
        f.write_text("# comment\n\npattern1\n  # indented comment\n")
        patterns = load_pattern_file(str(f))
        assert len(patterns) == 1


class TestScanMessage:
    def test_finds_word_in_text_block(self):
        from llm_memex.scripts.redact import compile_matchers, scan_message
        content = [{"type": "text", "text": "what the fuck"}]
        matchers = compile_matchers(words=["fuck"])
        result = scan_message(content, matchers, "c1", "m1")
        assert len(result.matches) == 1
        assert result.matches[0].start == 9
        assert result.matches[0].end == 13

    def test_skips_non_text_blocks(self):
        from llm_memex.scripts.redact import compile_matchers, scan_message
        content = [{"type": "tool_use", "name": "fuck"}]
        matchers = compile_matchers(words=["fuck"])
        result = scan_message(content, matchers, "c1", "m1")
        assert len(result.matches) == 0

    def test_multiple_matches(self):
        from llm_memex.scripts.redact import compile_matchers, scan_message
        content = [{"type": "text", "text": "fuck this shit"}]
        matchers = compile_matchers(words=["fuck", "shit"])
        result = scan_message(content, matchers, "c1", "m1")
        assert len(result.matches) == 2

    def test_multiple_blocks(self):
        from llm_memex.scripts.redact import compile_matchers, scan_message
        content = [
            {"type": "text", "text": "clean text"},
            {"type": "text", "text": "has fuck in it"},
        ]
        matchers = compile_matchers(words=["fuck"])
        result = scan_message(content, matchers, "c1", "m1")
        assert len(result.matches) == 1
        assert result.matches[0].block_index == 1


class TestMatchMode:
    def test_any_single_match(self):
        from llm_memex.scripts.redact import compile_matchers, scan_message, check_match_mode
        content = [{"type": "text", "text": "just fuck"}]
        matchers = compile_matchers(words=["fuck", "shit"])
        result = scan_message(content, matchers, "c1", "m1")
        assert check_match_mode(result.matches, "any", matchers) is True

    def test_all_requires_every_matcher(self):
        from llm_memex.scripts.redact import compile_matchers, scan_message, check_match_mode
        content = [{"type": "text", "text": "just fuck"}]
        matchers = compile_matchers(words=["fuck", "shit"])
        result = scan_message(content, matchers, "c1", "m1")
        assert check_match_mode(result.matches, "all", matchers) is False

    def test_all_all_match(self):
        from llm_memex.scripts.redact import compile_matchers, scan_message, check_match_mode
        content = [{"type": "text", "text": "fuck this shit"}]
        matchers = compile_matchers(words=["fuck", "shit"])
        result = scan_message(content, matchers, "c1", "m1")
        assert check_match_mode(result.matches, "all", matchers) is True


# ── Mutation Engine ─────────────────────────────────────────────


class TestRedactWordLevel:
    def test_replaces_match(self):
        from llm_memex.scripts.redact import Match, redact_word_level
        content = [{"type": "text", "text": "what the fuck"}]
        matches = [Match("c1", "m1", "fuck", 9, 13, 0)]
        result = redact_word_level(content, matches)
        assert result[0]["text"] == "what the [REDACTED]"

    def test_preserves_non_text_blocks(self):
        from llm_memex.scripts.redact import Match, redact_word_level
        content = [{"type": "text", "text": "fuck"}, {"type": "tool_use", "id": "x"}]
        matches = [Match("c1", "m1", "fuck", 0, 4, 0)]
        result = redact_word_level(content, matches)
        assert result[1] == {"type": "tool_use", "id": "x"}

    def test_multiple_matches_in_block(self):
        from llm_memex.scripts.redact import Match, redact_word_level
        content = [{"type": "text", "text": "fuck this shit"}]
        matches = [Match("c1", "m1", "fuck", 0, 4, 0), Match("c1", "m1", "shit", 10, 14, 0)]
        result = redact_word_level(content, matches)
        assert result[0]["text"] == "[REDACTED] this [REDACTED]"

    def test_does_not_mutate_original(self):
        from llm_memex.scripts.redact import Match, redact_word_level
        content = [{"type": "text", "text": "fuck"}]
        matches = [Match("c1", "m1", "fuck", 0, 4, 0)]
        redact_word_level(content, matches)
        assert content[0]["text"] == "fuck"  # original unchanged


class TestRedactMessageLevel:
    def test_replaces_entire_content(self):
        from llm_memex.scripts.redact import redact_message_level
        result = redact_message_level()
        assert result == [{"type": "text", "text": "[REDACTED]"}]


class TestRunDryRun:
    def test_reports_but_no_changes(self, tmp_db_path):
        """Dry run shows matches but doesn't modify DB."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="word")
        stats = run(db, args, apply=False)
        assert stats["word_redactions"] > 0
        # DB unchanged
        reloaded = db.load_conversation("c1")
        assert "fuck" in reloaded.messages["m1"].get_text()
        db.close()


class TestRunApplyBatch:
    def test_word_level(self, tmp_db_path):
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="word", yes=True)
        stats = run(db, args, apply=True)
        reloaded = db.load_conversation("c1")
        assert "[REDACTED]" in reloaded.messages["m1"].get_text()
        # Original stored as enrichment
        enrichments = db.get_enrichments("c1")
        originals = [e for e in enrichments if e["type"] == "original_content"]
        assert len(originals) == 1
        db.close()

    def test_message_level(self, tmp_db_path):
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="message", yes=True)
        run(db, args, apply=True)
        reloaded = db.load_conversation("c1")
        assert reloaded.messages["m1"].get_text() == "[REDACTED]"
        db.close()

    def test_conversation_level(self, tmp_db_path):
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="conversation", yes=True)
        run(db, args, apply=True)
        assert db.load_conversation("c1") is None
        db.close()

    def test_all_mode(self, tmp_db_path):
        """--all requires every term to match."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("just drunk")
        db.save_conversation(conv)
        args = _make_args(words="drunk,kim", level="message", yes=True, all_mode=True)
        run(db, args, apply=True)
        # "kim" not present, so no redaction
        reloaded = db.load_conversation("c1")
        assert "drunk" in reloaded.messages["m1"].get_text()
        db.close()

    def test_no_matches_returns_zero_stats(self, tmp_db_path):
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("perfectly clean text")
        db.save_conversation(conv)
        args = _make_args(words="profanity", level="word", yes=True)
        stats = run(db, args, apply=True)
        assert stats["word_redactions"] == 0
        db.close()


# ── Interactive Review ──────────────────────────────────────────


class TestInteractiveMode:
    def _setup_conv(self, tmp_db_path, text="what the fuck"):
        from llm_memex.scripts.redact import compile_matchers, scan_message, ScanResult
        db = Database(tmp_db_path)
        conv = _make_conv_with_text(text)
        db.save_conversation(conv)
        return db

    def test_redact_choice(self, tmp_db_path):
        """'r' redacts the match."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message
        db = self._setup_conv(tmp_db_path)
        matchers = compile_matchers(words=["fuck"])
        content = [{"type": "text", "text": "what the fuck"}]
        result = scan_message(content, matchers, "c1", "m1")
        responses = iter(["r"])
        stats = interactive_review([result], db, "word", input_fn=lambda _: next(responses))
        assert stats["redacted"] == 1
        reloaded = db.load_conversation("c1")
        assert "[REDACTED]" in reloaded.messages["m1"].get_text()
        db.close()

    def test_skip_choice(self, tmp_db_path):
        """'s' skips without modifying."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message
        db = self._setup_conv(tmp_db_path)
        matchers = compile_matchers(words=["fuck"])
        content = [{"type": "text", "text": "what the fuck"}]
        result = scan_message(content, matchers, "c1", "m1")
        responses = iter(["s"])
        stats = interactive_review([result], db, "word", input_fn=lambda _: next(responses))
        assert stats["skipped"] == 1
        reloaded = db.load_conversation("c1")
        assert "fuck" in reloaded.messages["m1"].get_text()
        db.close()

    def test_quit_stops_early(self, tmp_db_path):
        """'q' stops and returns what's been done."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message
        db = Database(tmp_db_path)
        # Two conversations with profanity
        conv1 = _make_conv_with_text("what the fuck", id="c1")
        conv2 = _make_conv_with_text("oh shit", id="c2")
        db.save_conversation(conv1)
        db.save_conversation(conv2)
        matchers = compile_matchers(words=["fuck", "shit"])
        result1 = scan_message([{"type": "text", "text": "what the fuck"}], matchers, "c1", "m1")
        result2 = scan_message([{"type": "text", "text": "oh shit"}], matchers, "c2", "m1")
        responses = iter(["r", "q"])
        stats = interactive_review([result1, result2], db, "word",
                                   input_fn=lambda _: next(responses))
        assert stats["redacted"] == 1
        # First redacted, second untouched
        r1 = db.load_conversation("c1")
        assert "[REDACTED]" in r1.messages["m1"].get_text()
        r2 = db.load_conversation("c2")
        assert "shit" in r2.messages["m1"].get_text()
        db.close()

    def test_all_choice_auto_redacts(self, tmp_db_path):
        """'a' redacts this and all same-term matches."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message
        db = Database(tmp_db_path)
        conv1 = _make_conv_with_text("what the fuck", id="c1")
        conv2 = _make_conv_with_text("oh fuck again", id="c2")
        db.save_conversation(conv1)
        db.save_conversation(conv2)
        matchers = compile_matchers(words=["fuck"])
        result1 = scan_message([{"type": "text", "text": "what the fuck"}], matchers, "c1", "m1")
        result2 = scan_message([{"type": "text", "text": "oh fuck again"}], matchers, "c2", "m1")
        # Only prompted once — 'a' auto-applies to second
        responses = iter(["a"])
        stats = interactive_review([result1, result2], db, "word",
                                   input_fn=lambda _: next(responses))
        assert stats["redacted"] == 2
        r1 = db.load_conversation("c1")
        assert "[REDACTED]" in r1.messages["m1"].get_text()
        r2 = db.load_conversation("c2")
        assert "[REDACTED]" in r2.messages["m1"].get_text()
        db.close()

    def test_message_level_interactive(self, tmp_db_path):
        """Interactive mode works for message-level redaction."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message
        db = self._setup_conv(tmp_db_path)
        matchers = compile_matchers(words=["fuck"])
        content = [{"type": "text", "text": "what the fuck"}]
        result = scan_message(content, matchers, "c1", "m1")
        responses = iter(["r"])
        stats = interactive_review([result], db, "message", input_fn=lambda _: next(responses))
        assert stats["redacted"] == 1
        reloaded = db.load_conversation("c1")
        assert reloaded.messages["m1"].get_text() == "[REDACTED]"
        db.close()

    def test_run_interactive_via_apply_no_yes(self, tmp_db_path):
        """run() with apply=True, yes=False triggers interactive review."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="word", yes=False)
        # Patch input to auto-respond 'r'
        with patch("llm_memex.scripts.redact.interactive_review") as mock_review:
            mock_review.return_value = {"redacted": 1, "skipped": 0}
            stats = run(db, args, apply=True)
            assert mock_review.called
        db.close()


# ── Built-in Pattern Files ──────────────────────────────────────


class TestBuiltinPatterns:
    def test_api_keys_file_exists(self):
        from pathlib import Path
        p = Path(__file__).parent.parent.parent / "llm_memex" / "scripts" / "patterns" / "api_keys.txt"
        assert p.exists()

    def test_pii_file_exists(self):
        from pathlib import Path
        p = Path(__file__).parent.parent.parent / "llm_memex" / "scripts" / "patterns" / "pii.txt"
        assert p.exists()

    def test_api_keys_matches_openai(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(pattern_file="api_keys.txt")
        assert any(m[0].search("sk-proj-abc123def456ghi789jkl012mno") for m in matchers)

    def test_api_keys_matches_github(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(pattern_file="api_keys.txt")
        assert any(m[0].search("ghp_ABCDEFghijklmnopqrstuvwxyz0123456789") for m in matchers)

    def test_api_keys_matches_aws(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(pattern_file="api_keys.txt")
        assert any(m[0].search("AKIAIOSFODNN7EXAMPLE") for m in matchers)

    def test_pii_matches_email(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(pattern_file="pii.txt")
        assert any(m[0].search("user@example.com") for m in matchers)

    def test_pii_matches_phone(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(pattern_file="pii.txt")
        assert any(m[0].search("555-123-4567") for m in matchers)

    def test_pii_matches_ssn(self):
        from llm_memex.scripts.redact import compile_matchers
        matchers = compile_matchers(pattern_file="pii.txt")
        assert any(m[0].search("123-45-6789") for m in matchers)

    def test_load_via_bare_filename(self):
        """Bare filename resolves to built-in patterns/ dir."""
        from llm_memex.scripts.redact import load_pattern_file
        patterns = load_pattern_file("api_keys.txt")
        assert len(patterns) >= 3  # at least OpenAI, GitHub, AWS


# ── Additional Detection Edge Cases ────────────────────────────


class TestLoadPatternFileResolution:
    def test_user_patterns_dir_fallback(self, tmp_path):
        """When builtin dir lacks file, user patterns dir is tried."""
        from llm_memex.scripts.redact import load_pattern_file
        user_patterns = tmp_path / ".memex" / "scripts" / "patterns"
        user_patterns.mkdir(parents=True)
        (user_patterns / "custom.txt").write_text("user_pattern_1\nuser_pattern_2\n")
        with patch("pathlib.Path.home", return_value=tmp_path):
            patterns = load_pattern_file("custom.txt")
        assert patterns == ["user_pattern_1", "user_pattern_2"]

    def test_nonexistent_pattern_file_raises(self):
        """Loading a pattern file that doesn't exist anywhere raises."""
        from llm_memex.scripts.redact import load_pattern_file
        with pytest.raises(FileNotFoundError):
            load_pattern_file("totally_nonexistent_file_xyz.txt")


class TestCheckMatchModeEdgeCases:
    def test_unknown_mode_returns_false(self):
        """An unrecognized mode returns False."""
        from llm_memex.scripts.redact import check_match_mode, Match
        matches = [Match("c1", "m1", "word", 0, 4, 0)]
        assert check_match_mode(matches, "unknown_mode", []) is False

    def test_any_with_no_matches_returns_false(self):
        """'any' mode with empty matches returns False."""
        from llm_memex.scripts.redact import check_match_mode
        assert check_match_mode([], "any", []) is False


# ── Additional Mutation Edge Cases ─────────────────────────────


class TestRedactWordLevelEdgeCases:
    def test_out_of_bounds_block_index_skipped(self):
        """Match referencing a block_index beyond content length is skipped."""
        from llm_memex.scripts.redact import Match, redact_word_level
        content = [{"type": "text", "text": "hello"}]
        # block_index=5 is out of bounds
        matches = [Match("c1", "m1", "hello", 0, 5, 5)]
        result = redact_word_level(content, matches)
        assert result[0]["text"] == "hello"  # unchanged

    def test_match_on_non_text_block_skipped(self):
        """Match referencing a non-text block is skipped."""
        from llm_memex.scripts.redact import Match, redact_word_level
        content = [{"type": "tool_use", "text": "some tool data"}]
        matches = [Match("c1", "m1", "some", 0, 4, 0)]
        result = redact_word_level(content, matches)
        # Non-text block unchanged
        assert result[0]["text"] == "some tool data"

    def test_adjacent_matches_both_redacted(self):
        """Adjacent matches in same block are both redacted correctly."""
        from llm_memex.scripts.redact import Match, redact_word_level
        content = [{"type": "text", "text": "bad bad"}]
        matches = [
            Match("c1", "m1", "bad", 0, 3, 0),
            Match("c1", "m1", "bad", 4, 7, 0),
        ]
        result = redact_word_level(content, matches)
        assert result[0]["text"] == "[REDACTED] [REDACTED]"

    def test_overlapping_matches_merged(self):
        """Overlapping matches from different matchers don't corrupt output."""
        from llm_memex.scripts.redact import Match, redact_word_level
        # Same span matched by both a word and a pattern
        content = [{"type": "text", "text": "what the fuck"}]
        matches = [
            Match("c1", "m1", "fuck", 9, 13, 0),
            Match("c1", "m1", "f..k", 9, 13, 0),
        ]
        result = redact_word_level(content, matches)
        assert result[0]["text"] == "what the [REDACTED]"

    def test_partially_overlapping_matches_merged(self):
        """Partially overlapping matches produce a single [REDACTED]."""
        from llm_memex.scripts.redact import Match, redact_word_level
        content = [{"type": "text", "text": "abcdefghij"}]
        # Match 1: positions 2-6, Match 2: positions 4-8 (overlap at 4-6)
        matches = [
            Match("c1", "m1", "pat1", 2, 6, 0),
            Match("c1", "m1", "pat2", 4, 8, 0),
        ]
        result = redact_word_level(content, matches)
        assert result[0]["text"] == "ab[REDACTED]ij"


# ── Additional Dry Run Output Tests ────────────────────────────


class TestDryRunOutput:
    def test_message_level_dry_run(self, tmp_db_path, capsys):
        """Dry run at message level prints MSG indicator."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="message")
        stats = run(db, args, apply=False)
        assert stats["message_redactions"] == 1
        output = capsys.readouterr().out
        assert "MSG" in output
        db.close()

    def test_conversation_level_dry_run(self, tmp_db_path, capsys):
        """Dry run at conversation level prints CONV indicator."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="conversation")
        stats = run(db, args, apply=False)
        assert stats["conversation_deletions"] == 1
        output = capsys.readouterr().out
        assert "CONV" in output
        db.close()

    def test_dry_run_no_matches_prints_no_matches(self, tmp_db_path, capsys):
        """Dry run with no matches prints 'No matches found.'."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("perfectly clean")
        db.save_conversation(conv)
        args = _make_args(words="nonexistent", level="word")
        run(db, args, apply=False)
        output = capsys.readouterr().out
        assert "No matches found" in output
        db.close()


# ── Conversation-Level Match Mode ──────────────────────────────


class TestConversationLevelMatchMode:
    def test_conversation_all_mode_across_messages(self, tmp_db_path):
        """Conversation-level with all mode checks across all messages in conv."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(id="c1", created_at=now, updated_at=now, title="Test",
                            source="test", message_count=2)
        conv.add_message(Message(id="m1", role="user",
                                 content=[text_block("the word fuck appears here")]))
        conv.add_message(Message(id="m2", role="assistant",
                                 content=[text_block("the word shit appears here")],
                                 parent_id="m1"))
        db.save_conversation(conv)
        # Both terms present across messages - should match in conversation-level all mode
        args = _make_args(words="fuck,shit", level="conversation", yes=True, all_mode=True)
        stats = run(db, args, apply=True)
        assert stats["conversation_deletions"] == 1
        assert db.load_conversation("c1") is None
        db.close()

    def test_conversation_all_mode_incomplete_no_delete(self, tmp_db_path):
        """Conversation-level all mode: if not all terms found, no deletion."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("only fuck here")
        db.save_conversation(conv)
        args = _make_args(words="fuck,shit", level="conversation", yes=True, all_mode=True)
        stats = run(db, args, apply=True)
        assert stats["conversation_deletions"] == 0
        assert db.load_conversation("c1") is not None
        db.close()


# ── Interactive Review Edge Cases ──────────────────────────────


class TestInteractiveReviewEdgeCases:
    def test_message_level_skip(self, tmp_db_path):
        """Interactive message-level 's' skips without modification."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        matchers = compile_matchers(words=["fuck"])
        content = [{"type": "text", "text": "what the fuck"}]
        result = scan_message(content, matchers, "c1", "m1")
        responses = iter(["s"])
        stats = interactive_review([result], db, "message",
                                   input_fn=lambda _: next(responses))
        assert stats["skipped"] == 1
        reloaded = db.load_conversation("c1")
        assert "fuck" in reloaded.messages["m1"].get_text()
        db.close()

    def test_message_level_quit(self, tmp_db_path):
        """Interactive message-level 'q' stops early."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message
        db = Database(tmp_db_path)
        conv1 = _make_conv_with_text("what the fuck", id="c1")
        conv2 = _make_conv_with_text("what the shit", id="c2")
        db.save_conversation(conv1)
        db.save_conversation(conv2)
        matchers = compile_matchers(words=["fuck", "shit"])
        result1 = scan_message([text_block("what the fuck")], matchers, "c1", "m1")
        result2 = scan_message([text_block("what the shit")], matchers, "c2", "m1")
        responses = iter(["q"])
        stats = interactive_review([result1, result2], db, "message",
                                   input_fn=lambda _: next(responses))
        assert stats["redacted"] == 0
        # Neither should be modified
        assert db.load_conversation("c1") is not None
        assert db.load_conversation("c2") is not None
        db.close()

    def test_all_choice_honors_auto_on_skip(self, tmp_db_path):
        """When term A is auto-approved and term B is skipped, A is still redacted."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message
        db = Database(tmp_db_path)
        # Message 1: has "fuck" — user says 'a' (auto all "fuck")
        conv1 = _make_conv_with_text("what the fuck", id="c1")
        # Message 2: has both "fuck" and "shit" — "fuck" is auto, user skips "shit"
        now = datetime.now()
        conv2 = Conversation(id="c2", created_at=now, updated_at=now, title="Test",
                             source="test", message_count=1)
        conv2.add_message(Message(id="m1", role="user",
                                  content=[text_block("fuck this shit")]))
        db.save_conversation(conv1)
        db.save_conversation(conv2)
        matchers = compile_matchers(words=["fuck", "shit"])
        result1 = scan_message([text_block("what the fuck")], matchers, "c1", "m1")
        result2 = scan_message([text_block("fuck this shit")], matchers, "c2", "m1")
        # First result: 'a' to auto-approve "fuck". Second result: 's' to skip "shit"
        responses = iter(["a", "s"])
        stats = interactive_review([result1, result2], db, "word",
                                   input_fn=lambda _: next(responses))
        # c1 fully redacted
        r1 = db.load_conversation("c1")
        assert "[REDACTED]" in r1.messages["m1"].get_text()
        # c2: "fuck" should be redacted (auto), "shit" should remain
        r2 = db.load_conversation("c2")
        text = r2.messages["m1"].get_text()
        assert "fuck" not in text
        assert "[REDACTED]" in text
        assert "shit" in text
        db.close()

    def test_conversation_level_interactive_redact(self, tmp_db_path):
        """Interactive conversation-level 'r' deletes the conversation."""
        from llm_memex.scripts.redact import interactive_review, compile_matchers, scan_message, ScanResult, Match
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("what the fuck")
        db.save_conversation(conv)
        # Build a conversation-level ScanResult (message_id="(all)", content=[])
        combined = ScanResult(
            conversation_id="c1",
            message_id="(all)",
            matches=[Match("c1", "m1", "fuck", 9, 13, 0)],
            content=[],
        )
        responses = iter(["r"])
        stats = interactive_review([combined], db, "conversation",
                                   input_fn=lambda _: next(responses))
        assert stats["redacted"] == 1
        assert db.load_conversation("c1") is None
        db.close()


# ── Pattern-Based End-to-End ───────────────────────────────────


class TestPatternBasedRedaction:
    def test_regex_pattern_end_to_end(self, tmp_db_path):
        """run() with --patterns performs regex-based redaction."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("my key is sk-proj-abc123def456ghi789jkl012mno please")
        db.save_conversation(conv)
        # Use a comma-free regex (commas in quantifiers like {20,} get split
        # by run()'s CSV parser for --patterns)
        args = _make_args(patterns=r"sk-\w+", level="word", yes=True)
        stats = run(db, args, apply=True)
        assert stats["word_redactions"] >= 1
        reloaded = db.load_conversation("c1")
        assert "[REDACTED]" in reloaded.messages["m1"].get_text()
        assert "sk-proj" not in reloaded.messages["m1"].get_text()
        db.close()

    def test_pattern_file_end_to_end(self, tmp_db_path, tmp_path):
        """run() with --pattern-file loads and applies patterns."""
        from llm_memex.scripts.redact import run
        pf = tmp_path / "patterns.txt"
        pf.write_text(r"secret_\w+")
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("the password is secret_abc123 ok")
        db.save_conversation(conv)
        args = _make_args(pattern_file=str(pf), level="word", yes=True)
        stats = run(db, args, apply=True)
        assert stats["word_redactions"] >= 1
        reloaded = db.load_conversation("c1")
        assert "secret_abc123" not in reloaded.messages["m1"].get_text()
        db.close()


# ── Redact + DB Integration ────────────────────────────────────


class TestRedactDBIntegration:
    def test_word_redact_updates_fts_index(self, tmp_db_path):
        """After word-level redaction, the original term is no longer FTS-searchable."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("the secret password is hunter2")
        db.save_conversation(conv)
        args = _make_args(words="hunter2", level="word", yes=True)
        run(db, args, apply=True)
        # "hunter2" should not be findable via FTS
        results = db.search_messages("hunter2")
        assert len(results) == 0
        # But surrounding text should still be findable
        results = db.search_messages("password")
        assert len(results) >= 1
        db.close()

    def test_message_redact_removes_all_text_from_fts(self, tmp_db_path):
        """After message-level redaction, original text is not FTS-searchable."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("the unique_search_term_xyz is here")
        db.save_conversation(conv)
        args = _make_args(words="unique_search_term_xyz", level="message", yes=True)
        run(db, args, apply=True)
        results = db.search_messages("unique_search_term_xyz")
        assert len(results) == 0
        db.close()

    def test_conversation_delete_cleans_fts_and_enrichments(self, tmp_db_path):
        """Conversation-level redaction cleans up FTS, enrichments, and provenance."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("fuck this particular conversation")
        db.save_conversation(conv)
        db.save_enrichment("c1", "topic", "profanity", "heuristic")
        db.save_provenance("c1", source_type="test", source_file="test.json")
        args = _make_args(words="fuck", level="conversation", yes=True)
        run(db, args, apply=True)
        # Everything gone
        assert db.load_conversation("c1") is None
        assert db.get_enrichments("c1") == []
        assert db.get_provenance("c1") == []
        results = db.search_messages("particular")
        assert len(results) == 0
        db.close()

    def test_redact_multiblock_message_preserves_clean_blocks(self, tmp_db_path):
        """Word redaction in a multi-block message only affects matching blocks."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(id="c1", created_at=now, updated_at=now, title="Test",
                            source="test", message_count=1)
        conv.add_message(Message(id="m1", role="user", content=[
            {"type": "text", "text": "clean text here"},
            {"type": "text", "text": "some fuck in this block"},
        ]))
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="word", yes=True)
        run(db, args, apply=True)
        reloaded = db.load_conversation("c1")
        content = reloaded.messages["m1"].content
        assert content[0]["text"] == "clean text here"
        assert "[REDACTED]" in content[1]["text"]
        assert "fuck" not in content[1]["text"]
        db.close()

    def test_redact_only_matching_messages_in_conversation(self, tmp_db_path):
        """Word-level redaction only modifies messages that contain matches."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        now = datetime.now()
        conv = Conversation(id="c1", created_at=now, updated_at=now, title="Test",
                            source="test", message_count=2)
        conv.add_message(Message(id="m1", role="user",
                                 content=[text_block("clean message here")]))
        conv.add_message(Message(id="m2", role="assistant",
                                 content=[text_block("this has fuck in it")],
                                 parent_id="m1"))
        db.save_conversation(conv)
        args = _make_args(words="fuck", level="word", yes=True)
        run(db, args, apply=True)
        reloaded = db.load_conversation("c1")
        assert reloaded.messages["m1"].get_text() == "clean message here"
        assert "[REDACTED]" in reloaded.messages["m2"].get_text()
        db.close()

    def test_redact_multiple_conversations_selectively(self, tmp_db_path):
        """Redaction across multiple conversations only affects matching ones."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv1 = _make_conv_with_text("has the bad word fuck", id="c1")
        conv2 = _make_conv_with_text("perfectly clean text", id="c2")
        conv3 = _make_conv_with_text("also has fuck here", id="c3")
        db.save_conversation(conv1)
        db.save_conversation(conv2)
        db.save_conversation(conv3)
        args = _make_args(words="fuck", level="word", yes=True)
        stats = run(db, args, apply=True)
        assert stats["word_redactions"] == 2
        # c1 and c3 redacted
        r1 = db.load_conversation("c1")
        assert "[REDACTED]" in r1.messages["m1"].get_text()
        r3 = db.load_conversation("c3")
        assert "[REDACTED]" in r3.messages["m1"].get_text()
        # c2 untouched
        r2 = db.load_conversation("c2")
        assert r2.messages["m1"].get_text() == "perfectly clean text"
        db.close()

    def test_word_redact_stores_original_content_enrichment(self, tmp_db_path):
        """Word-level redaction saves an original_content enrichment with recoverable content."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("contains secret here")
        db.save_conversation(conv)
        args = _make_args(words="secret", level="word", yes=True)
        run(db, args, apply=True)
        enrichments = db.get_enrichments("c1")
        originals = [e for e in enrichments if e["type"] == "original_content"]
        assert len(originals) == 1
        assert originals[0]["source"] == "redact"
        # The enrichment value contains the original content as JSON
        stored = json.loads(originals[0]["value"])
        assert stored["message_id"] == "m1"
        assert stored["content"] == [{"type": "text", "text": "contains secret here"}]
        db.close()

    def test_redact_scrubs_conversation_title(self, tmp_db_path):
        """Redaction must scrub the conversation title, not just messages (REDACT-2).

        The title is a denormalized field exported by every exporter, so a
        secret left in it would leak even after message redaction."""
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("a clean message", title="my secret key")
        db.save_conversation(conv)
        args = _make_args(words="secret", level="word", yes=True)
        run(db, args, apply=True)
        reloaded = db.load_conversation("c1")
        assert "secret" not in (reloaded.title or "").lower()
        assert "[REDACTED]" in reloaded.title
        db.close()


# ── REDACT-4: Atomicity of the two writes per redaction ────────


class TestApplySingleAtomicity:
    """_apply_single performs an enrichment write and a message-content write.

    REDACT-4: those two writes must be atomic. A crash/raise between them must
    not leave a committed plaintext `original_content` enrichment while the
    message is still un-redacted (or vice versa).
    """

    def test_word_level_consistent_on_success(self, tmp_db_path):
        """On success: message is redacted AND the original_content enrichment
        holds the pre-redaction content (both writes applied, consistently)."""
        from llm_memex.scripts.redact import compile_matchers, scan_message, _apply_single
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("contains secret here")
        db.save_conversation(conv)
        matchers = compile_matchers(words=["secret"])
        content = [{"type": "text", "text": "contains secret here"}]
        result = scan_message(content, matchers, "c1", "m1")
        _apply_single(db, result, "word")
        # Message redacted
        reloaded = db.load_conversation("c1")
        text = reloaded.messages["m1"].get_text()
        assert "[REDACTED]" in text
        assert "secret" not in text
        # Enrichment holds the ORIGINAL (pre-redaction) content
        originals = [e for e in db.get_enrichments("c1")
                     if e["type"] == "original_content"]
        assert len(originals) == 1
        stored = json.loads(originals[0]["value"])
        assert stored["message_id"] == "m1"
        assert stored["content"] == [{"type": "text", "text": "contains secret here"}]
        db.close()

    def test_word_level_rolls_back_if_message_update_fails(self, tmp_db_path):
        """If the message-content write raises, NO original_content enrichment
        is left committed (the enrichment write rolls back too)."""
        from llm_memex.scripts.redact import compile_matchers, scan_message, _apply_single
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("contains secret here")
        db.save_conversation(conv)
        matchers = compile_matchers(words=["secret"])
        content = [{"type": "text", "text": "contains secret here"}]
        result = scan_message(content, matchers, "c1", "m1")

        with patch.object(db, "update_message_content",
                          side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                _apply_single(db, result, "word")

        # No plaintext enrichment leaked
        originals = [e for e in db.get_enrichments("c1")
                     if e["type"] == "original_content"]
        assert originals == []
        # Message left intact (still contains the secret, not half-redacted)
        reloaded = db.load_conversation("c1")
        assert "secret" in reloaded.messages["m1"].get_text()
        assert "[REDACTED]" not in reloaded.messages["m1"].get_text()
        db.close()

    def test_message_level_rolls_back_if_message_update_fails(self, tmp_db_path):
        """Same atomicity guarantee for level='message'."""
        from llm_memex.scripts.redact import compile_matchers, scan_message, _apply_single
        db = Database(tmp_db_path)
        conv = _make_conv_with_text("contains secret here")
        db.save_conversation(conv)
        matchers = compile_matchers(words=["secret"])
        content = [{"type": "text", "text": "contains secret here"}]
        result = scan_message(content, matchers, "c1", "m1")

        with patch.object(db, "update_message_content",
                          side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                _apply_single(db, result, "message")

        originals = [e for e in db.get_enrichments("c1")
                     if e["type"] == "original_content"]
        assert originals == []
        reloaded = db.load_conversation("c1")
        assert "secret" in reloaded.messages["m1"].get_text()
        db.close()


class TestBatchPerItemFailure:
    """REDACT-4: the --yes batch loop must catch per-item failures, continue,
    and surface which conversations failed."""

    def test_batch_continues_and_reports_failed_conv(self, tmp_db_path):
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv1 = _make_conv_with_text("has secret one", id="c1")
        conv2 = _make_conv_with_text("has secret two", id="c2")
        db.save_conversation(conv1)
        db.save_conversation(conv2)

        real_update = db.update_message_content

        def flaky_update(conversation_id, message_id, content):
            if conversation_id == "c1":
                raise RuntimeError("boom on c1")
            return real_update(conversation_id, message_id, content)

        args = _make_args(words="secret", level="word", yes=True)
        with patch.object(db, "update_message_content", side_effect=flaky_update):
            stats = run(db, args, apply=True)

        # The batch did not abort: c2 was still redacted.
        r2 = db.load_conversation("c2")
        assert "[REDACTED]" in r2.messages["m1"].get_text()
        # c1 failed cleanly: not half-applied, no leaked enrichment.
        r1 = db.load_conversation("c1")
        assert "secret" in r1.messages["m1"].get_text()
        assert [e for e in db.get_enrichments("c1")
                if e["type"] == "original_content"] == []
        # The failure is surfaced (which conversation failed).
        assert "c1" in stats.get("failed", [])
        assert "c2" not in stats.get("failed", [])
        db.close()


def _make_conv_with_blocks(blocks, id="c1", title="Test"):
    """Build a one-message conversation whose content is arbitrary blocks."""
    now = datetime.now()
    conv = Conversation(id=id, created_at=now, updated_at=now, title=title,
                        source="test", message_count=1)
    conv.add_message(Message(id="m1", role="assistant", content=blocks))
    return conv


class TestNonTextBlockRedaction:
    """B6: secrets in tool_use/tool_result/thinking blocks must be detected
    and removed, not silently passed over with a false all-clear."""

    def test_scan_detects_secret_in_tool_result_block(self):
        from llm_memex.scripts.redact import compile_matchers, scan_message
        matchers = compile_matchers(words=["sk-SECRET123"])
        content = [{"type": "tool_result",
                    "content": "the api key is sk-SECRET123 ok"}]
        result = scan_message(content, matchers, "c1", "m1")
        assert result.matches, "secret in tool_result block was not detected"
        # Recorded as a structural match (no precise text offset).
        assert all(m.start < 0 for m in result.matches)

    def test_scan_skips_structural_keys(self):
        """A coincidental match on a block's type/name must not fire."""
        from llm_memex.scripts.redact import compile_matchers, scan_message
        matchers = compile_matchers(words=["tool_use"])
        content = [{"type": "tool_use", "name": "tool_use", "input": {"x": "safe"}}]
        result = scan_message(content, matchers, "c1", "m1")
        assert not result.matches

    def test_apply_escalates_nontext_match_to_message_level(self, tmp_db_path):
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        conv = _make_conv_with_blocks([
            {"type": "tool_result", "content": "leaked sk-SECRET123 here"},
        ])
        db.save_conversation(conv)
        args = _make_args(words="sk-SECRET123", level="word", yes=True)
        run(db, args, apply=True)
        reloaded = db.load_conversation("c1")
        # Whole message redacted; the secret is gone from stored content.
        dumped = json.dumps(reloaded.messages["m1"].content)
        assert "sk-SECRET123" not in dumped, "secret survived in a non-text block"
        # Pre-redaction content preserved for undo.
        originals = [e for e in db.get_enrichments("c1")
                     if e["type"] == "original_content"]
        assert len(originals) == 1
        db.close()

    def test_notes_are_redacted_on_apply(self, tmp_db_path):
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv_with_text("clean message"))
        db.add_note(conversation_id="c1", text="reminder: token is sk-NOTESECRET")
        args = _make_args(words="sk-NOTESECRET", level="word", yes=True)
        stats = run(db, args, apply=True)
        assert stats["notes_redacted"] == 1
        rows = db.execute_sql("SELECT text FROM notes")
        assert "sk-NOTESECRET" not in rows[0]["text"]
        assert "[REDACTED]" in rows[0]["text"]
        db.close()

    def test_dry_run_reports_note_matches(self, tmp_db_path, capsys):
        from llm_memex.scripts.redact import run
        db = Database(tmp_db_path)
        db.save_conversation(_make_conv_with_text("clean message"))
        db.add_note(conversation_id="c1", text="secret sk-NOTESECRET")
        args = _make_args(words="sk-NOTESECRET", level="word")
        stats = run(db, args, apply=False)
        out = capsys.readouterr().out
        assert stats["notes_matched"] == 1
        assert "No matches found." not in out  # honest: notes DID match
        assert "marginalia note" in out
        # And the note is untouched by a dry run.
        assert "sk-NOTESECRET" in db.execute_sql("SELECT text FROM notes")[0]["text"]
        db.close()
