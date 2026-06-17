"""Tests for memex CLI: import, export, mcp, version, show."""
import json
import os
import subprocess
import sys


class TestCLIVersion:
    def test_version_flag(self):
        # Read version from the installed package metadata to avoid
        # pytest's tests/llm_memex/__init__.py shadowing the real llm_memex package.
        from importlib.metadata import version
        pkg_version = version("llm-memex")
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "--version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert pkg_version in result.stdout


class TestCLIImport:
    def test_import_openai(self, tmp_path):
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "Test",
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
        }]))
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 1 conversation" in result.stdout

    def test_import_forced_format(self, tmp_path):
        db_dir = tmp_path / "db"
        export_file = tmp_path / "data.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "Test",
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
        }]))
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--format", "openai", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 1 conversation" in result.stdout

    def test_import_no_importer(self, tmp_path):
        db_dir = tmp_path / "db"
        export_file = tmp_path / "unknown.json"
        export_file.write_text(json.dumps({"random": "data"}))
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "no importer found" in result.stderr


class TestCLIImportClaudeCode:
    def test_import_claude_code(self, tmp_path):
        from llm_memex.db import Database
        db_dir = tmp_path / "db"
        export_file = tmp_path / "session.jsonl"
        events = [
            {
                "type": "user", "uuid": "u1", "parentUuid": None,
                "sessionId": "sess-cli-test", "slug": "cli-test-session",
                "timestamp": "2026-02-18T10:00:00Z",
                "userType": "external", "isSidechain": False,
                "message": {"role": "user", "content": "Hello from CLI test"},
            },
            {
                "type": "assistant", "uuid": "a1", "parentUuid": "u1",
                "sessionId": "sess-cli-test", "slug": "cli-test-session",
                "timestamp": "2026-02-18T10:00:01Z",
                "userType": "external", "isSidechain": False,
                "message": {"role": "assistant", "model": "claude-opus-4-6",
                            "content": [{"type": "text", "text": "Hello!"}]},
            },
        ]
        export_file.write_text("\n".join(json.dumps(e) for e in events))
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 1 conversation" in result.stdout
        # Verify provenance was saved. Auto-import prefers claude_code_full
        # when both detect the same file, so accept either source_type.
        db = Database(str(db_dir))
        prov = db.get_provenance("sess-cli-test")
        assert len(prov) == 1
        assert prov[0]["source_type"] in ("claude_code", "claude_code_full")
        db.close()


class TestCLIImportProvenance:
    def test_import_persists_provenance(self, tmp_path):
        from llm_memex.db import Database
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "Test",
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
        }]))
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Verify provenance was saved to DB
        db = Database(str(db_dir))
        prov = db.get_provenance("c1")
        assert len(prov) == 1
        assert prov[0]["source_type"] == "openai"
        assert prov[0]["source_id"] == "c1"
        assert str(export_file) in prov[0]["source_file"]
        db.close()


class TestCLIImportRecursive:
    """Tests for memex import --recursive directory import."""

    def _make_openai_file(self, path, conv_id="c1", title="Test"):
        """Write a minimal OpenAI export JSON file."""
        path.write_text(json.dumps([{
            "id": conv_id, "title": title,
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
        }]))

    def _make_claude_code_file(self, path, session_id="sess-1"):
        """Write a minimal Claude Code JSONL file."""
        events = [
            {
                "type": "user", "uuid": "u1", "parentUuid": None,
                "sessionId": session_id, "slug": "test-session",
                "timestamp": "2026-02-18T10:00:00Z",
                "userType": "external", "isSidechain": False,
                "message": {"role": "user", "content": "hello"},
            },
            {
                "type": "assistant", "uuid": "a1", "parentUuid": "u1",
                "sessionId": session_id, "slug": "test-session",
                "timestamp": "2026-02-18T10:00:01Z",
                "userType": "external", "isSidechain": False,
                "message": {"role": "assistant", "model": "claude-opus-4-6",
                            "content": [{"type": "text", "text": "hi"}]},
            },
        ]
        path.write_text("\n".join(json.dumps(e) for e in events))

    def test_recursive_single_file(self, tmp_path):
        """Directory with one importable file imports successfully via recursive walk."""
        db_dir = tmp_path / "db"
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        # Use a named file (not conversations.json) so OpenAI dir detection doesn't claim it
        self._make_openai_file(src_dir / "export.json")

        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--recursive", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "1 conversation(s) from 1 file(s)" in result.stdout

    def test_recursive_mixed_files(self, tmp_path):
        """Directory with importable and non-importable files imports only matching."""
        db_dir = tmp_path / "db"
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        # Use a named file (not conversations.json) so OpenAI dir detection doesn't claim it
        self._make_openai_file(src_dir / "export.json", conv_id="c1")
        (src_dir / "README.md").write_text("# Not importable")
        (src_dir / "config.yaml").write_text("key: value")

        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--recursive", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "1 conversation(s) from 1 file(s)" in result.stdout
        assert "2 skipped" in result.stdout

    def test_recursive_required_for_unrecognized_directory(self, tmp_path):
        """Passing an unrecognized directory without --recursive errors."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        # Files that no importer claims as a directory structure
        (src_dir / "random.txt").write_text("hello")

        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--db", str(tmp_path / "db")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "is a directory" in result.stderr
        assert "--recursive" in result.stderr

    def test_recursive_nested_subdirectories(self, tmp_path):
        """Files in nested subdirectories are all found."""
        db_dir = tmp_path / "db"
        src_dir = tmp_path / "sources"
        sub1 = src_dir / "project-a"
        sub2 = src_dir / "project-b" / "deep"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)
        self._make_openai_file(sub1 / "conv1.json", conv_id="c1")
        self._make_openai_file(sub2 / "conv2.json", conv_id="c2")
        # Non-importable at root level
        (src_dir / "notes.txt").write_text("just notes")

        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--recursive", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "2 conversation(s) from 2 file(s)" in result.stdout
        assert "1 skipped" in result.stdout

        # Verify both conversations are in the database
        from llm_memex.db import Database
        with Database(str(db_dir), readonly=True) as db:
            c1 = db.load_conversation("c1")
            c2 = db.load_conversation("c2")
            assert c1 is not None
            assert c2 is not None

    def test_recursive_with_format_flag(self, tmp_path):
        """--format works with --recursive to force a specific importer."""
        db_dir = tmp_path / "db"
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        # Write OpenAI data with a non-standard extension
        self._make_openai_file(src_dir / "data.txt", conv_id="forced")

        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--recursive", "--format", "openai", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "1 conversation(s) from 1 file(s)" in result.stdout


class TestCLIImportDirectory:
    """Tests for directory import without --recursive (importer claims directory)."""

    def test_openai_directory_import(self, tmp_path):
        """OpenAI export directory with conversations.json imports without --recursive."""
        db_dir = tmp_path / "db"
        src_dir = tmp_path / "openai_export"
        src_dir.mkdir()
        (src_dir / "conversations.json").write_text(json.dumps([{
            "id": "c1", "title": "OpenAI Chat",
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
        }]))

        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 1 conversation(s)" in result.stdout

    def test_claude_code_directory_import(self, tmp_path):
        """Claude Code project directory imports without --recursive."""
        db_dir = tmp_path / "db"
        src_dir = tmp_path / "claude_sessions"
        src_dir.mkdir()
        events = [
            {
                "type": "user", "uuid": "u1", "parentUuid": None,
                "sessionId": "sess-dir", "slug": "dir-test",
                "timestamp": "2026-02-18T10:00:00Z",
                "userType": "external", "isSidechain": False,
                "message": {"role": "user", "content": "hello from dir"},
            },
            {
                "type": "assistant", "uuid": "a1", "parentUuid": "u1",
                "sessionId": "sess-dir", "slug": "dir-test",
                "timestamp": "2026-02-18T10:00:01Z",
                "userType": "external", "isSidechain": False,
                "message": {"role": "assistant", "model": "claude-opus-4-6",
                            "content": [{"type": "text", "text": "hi"}]},
            },
        ]
        (src_dir / "session.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events)
        )

        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 1 conversation(s)" in result.stdout


class TestCLIImportSkipUnchanged:
    """Tests for skip-if-unchanged optimization and --force flag."""

    def _make_openai_file(self, path, conv_id="c1", title="Test",
                          update_time=1700000001):
        """Write a minimal OpenAI export JSON file."""
        path.write_text(json.dumps([{
            "id": conv_id, "title": title,
            "create_time": 1700000000, "update_time": update_time,
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
        }]))

    def test_reimport_unchanged_skips(self, tmp_path):
        """Re-importing identical data reports conversations as unchanged."""
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        self._make_openai_file(export_file)

        # First import
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--db", str(db_dir)],
            capture_output=True, text=True, check=True,
        )
        # Second import — should skip
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 0 conversation(s)" in result.stdout
        assert "1 unchanged" in result.stdout

    def test_force_reimports_unchanged(self, tmp_path):
        """--force re-imports even if unchanged."""
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        self._make_openai_file(export_file)

        # First import
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--db", str(db_dir)],
            capture_output=True, text=True, check=True,
        )
        # Second import with --force
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--force", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 1 conversation(s)" in result.stdout
        # With --force, no "N unchanged" count should appear
        assert " unchanged)" not in result.stdout

    def test_modified_conversation_reimported(self, tmp_path):
        """Conversation with changed update_time gets re-imported."""
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"

        # First import
        self._make_openai_file(export_file, update_time=1700000001)
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--db", str(db_dir)],
            capture_output=True, text=True, check=True,
        )
        # Update the file with a new update_time
        self._make_openai_file(export_file, update_time=1700099999)
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 1 conversation(s)" in result.stdout
        assert " unchanged)" not in result.stdout

    def test_recursive_with_unchanged(self, tmp_path):
        """Recursive import shows unchanged count in summary."""
        db_dir = tmp_path / "db"
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        self._make_openai_file(src_dir / "conv1.json", conv_id="c1")
        self._make_openai_file(src_dir / "conv2.json", conv_id="c2")

        # First import
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--recursive", "--db", str(db_dir)],
            capture_output=True, text=True, check=True,
        )
        # Re-import — both should be unchanged
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(src_dir),
             "--recursive", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 0 conversation(s)" in result.stdout
        assert "2 unchanged" in result.stdout


class TestCLIImportSkipUnchangedUnit:
    """Unit tests for Database.conversation_unchanged()."""

    def test_conversation_unchanged_true(self, tmp_db_path):
        from llm_memex.db import Database
        from llm_memex.models import Conversation
        from datetime import datetime

        with Database(tmp_db_path) as db:
            conv = Conversation(
                id="c1", title="Test",
                created_at=datetime(2023, 1, 1),
                updated_at=datetime(2023, 1, 2),
                message_count=5,
            )
            db.save_conversation(conv)
            assert db.conversation_unchanged("c1", datetime(2023, 1, 2), 5) is True

    def test_conversation_unchanged_false_different_time(self, tmp_db_path):
        from llm_memex.db import Database
        from llm_memex.models import Conversation
        from datetime import datetime

        with Database(tmp_db_path) as db:
            conv = Conversation(
                id="c1", title="Test",
                created_at=datetime(2023, 1, 1),
                updated_at=datetime(2023, 1, 2),
                message_count=5,
            )
            db.save_conversation(conv)
            assert db.conversation_unchanged("c1", datetime(2023, 6, 1), 5) is False

    def test_conversation_unchanged_false_different_count(self, tmp_db_path):
        from llm_memex.db import Database
        from llm_memex.models import Conversation
        from datetime import datetime

        with Database(tmp_db_path) as db:
            conv = Conversation(
                id="c1", title="Test",
                created_at=datetime(2023, 1, 1),
                updated_at=datetime(2023, 1, 2),
                message_count=5,
            )
            db.save_conversation(conv)
            assert db.conversation_unchanged("c1", datetime(2023, 1, 2), 10) is False

    def test_conversation_unchanged_nonexistent(self, tmp_db_path):
        from llm_memex.db import Database
        from datetime import datetime

        with Database(tmp_db_path) as db:
            assert db.conversation_unchanged("nope", datetime(2023, 1, 1), 0) is False

    def test_reimport_preserves_enrichments_and_summary(self, tmp_path):
        """CF1: re-importing a CHANGED conversation must not wipe user
        enrichments or a user-set summary. The importer carries neither, so
        the default re-import path was silently destroying both."""
        import types
        from datetime import datetime

        from llm_memex.cli import _save_convs
        from llm_memex.db import Database
        from llm_memex.models import Conversation

        db_dir = tmp_path / "db"
        args = types.SimpleNamespace(
            db=str(db_dir), merge=False, force=False, no_copy_assets=True
        )
        with Database(str(db_dir)) as db:
            v1 = Conversation(
                id="c1", title="Topic", source="openai",
                created_at=datetime(2023, 1, 1), updated_at=datetime(2023, 1, 2),
            )
            v1.message_count = 3  # __post_init__ resets to len(messages)
            _save_convs([v1], tmp_path, args, db)

            # User enriches + sets a summary post-import.
            db.save_enrichment("c1", "summary", "hand-written summary", "user")
            db.save_enrichment("c1", "importance", "high", "user")
            db.conn.execute(
                "UPDATE conversations SET summary=? WHERE id=?",
                ("user summary", "c1"),
            )
            db.conn.commit()

            # Re-import a CHANGED version (more messages); importer carries
            # no summary and no enrichments.
            v2 = Conversation(
                id="c1", title="Topic", source="openai",
                created_at=datetime(2023, 1, 1), updated_at=datetime(2023, 1, 3),
                summary=None,
            )
            v2.message_count = 5
            imported, unchanged = _save_convs([v2], tmp_path, args, db)
            assert (imported, unchanged) == (1, 0)  # changed -> re-saved

            enr = {(e["type"], e["value"]) for e in db.get_enrichments("c1")}
            assert ("summary", "hand-written summary") in enr
            assert ("importance", "high") in enr

            row = db.execute_sql(
                "SELECT summary FROM conversations WHERE id=?", ("c1",)
            )
            assert row[0]["summary"] == "user summary"

    def test_merge_writes_provenance_and_strips_it_from_metadata(self, tmp_path):
        """LLM-6: --merge with a platform importer must write the provenance
        table for a newly created conversation and must not serialize the
        internal _provenance dict into the conversation metadata JSON."""
        import json as _json
        import types
        from datetime import datetime

        from llm_memex.cli import _save_convs
        from llm_memex.db import Database
        from llm_memex.models import Conversation

        db_dir = tmp_path / "db"
        args = types.SimpleNamespace(
            db=str(db_dir), merge=True, force=False, no_copy_assets=True
        )
        with Database(str(db_dir)) as db:
            conv = Conversation(
                id="c1", title="T", source="openai",
                created_at=datetime(2023, 1, 1), updated_at=datetime(2023, 1, 2),
            )
            conv.metadata["_provenance"] = {
                "source_type": "openai",
                "source_file": "/x/conv.json",
                "source_id": "src-1",
                "source_hash": "abc",
            }
            _save_convs([conv], tmp_path, args, db)

            prov_rows = db.get_provenance("c1")
            assert prov_rows, "merge did not write the provenance table"
            assert prov_rows[0]["source_type"] == "openai"

            stored = db.execute_sql(
                "SELECT metadata FROM conversations WHERE id=?", ("c1",)
            )
            meta = _json.loads(stored[0]["metadata"] or "{}")
            assert "_provenance" not in meta


class TestCLIExport:
    def test_export_markdown(self, tmp_path):
        # First import, then export
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "Exported Chat",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello world"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]))
        # Import
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        # Export
        out_file = tmp_path / "out.md"
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(out_file),
             "--format", "markdown", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Exported 1 conversation" in result.stdout
        content = out_file.read_text()
        assert "hello world" in content

    def test_export_json(self, tmp_path):
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "Test",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hi"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]))
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        out_file = tmp_path / "out.json"
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(out_file),
             "--format", "json", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(out_file.read_text())
        assert len(data) == 1
        assert data[0]["id"] == "c1"


class TestCLIExportArkiv:
    """Tests for the arkiv exporter (JSONL + README.md + schema.yaml)."""

    def _import_conversation(self, tmp_path, *, conv_id="c1", title="My Chat",
                              source="chatgpt", model="gpt-4", text="hello world"):
        """Import a single conversation and return db_dir."""
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": conv_id, "title": title,
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": ["m2"],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": [text]},
                        "create_time": 1700000000,
                    },
                },
                "m2": {
                    "id": "m2", "parent": "m1", "children": [],
                    "message": {
                        "id": "m2", "author": {"role": "assistant"},
                        "content": {"parts": ["hi there"]},
                        "create_time": 1700000001,
                        "metadata": {"model_slug": model},
                    },
                },
            },
        }]))
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True, check=True,
        )
        return db_dir

    def test_export_creates_directory_with_files(self, tmp_path):
        db_dir = self._import_conversation(tmp_path)
        out_dir = tmp_path / "arkiv_out"
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(out_dir),
             "--format", "arkiv", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Exported 1 conversation" in result.stdout
        assert (out_dir / "conversations.jsonl").exists()
        assert (out_dir / "README.md").exists()
        assert (out_dir / "schema.yaml").exists()

    def test_jsonl_records_have_correct_fields(self, tmp_path):
        db_dir = self._import_conversation(tmp_path)
        out_dir = tmp_path / "arkiv_out"
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(out_dir),
             "--format", "arkiv", "--db", str(db_dir)],
            capture_output=True, text=True, check=True,
        )
        records = []
        with open(out_dir / "conversations.jsonl") as f:
            for line in f:
                records.append(json.loads(line))

        assert len(records) == 2  # user + assistant messages

        # Check first record (user message)
        rec = records[0]
        assert rec["mimetype"] == "text/plain"
        assert rec["content"] == "hello world"
        assert "timestamp" in rec
        meta = rec["metadata"]
        assert meta["conversation_id"] == "c1"
        assert meta["conversation_title"] == "My Chat"
        assert meta["role"] == "user"
        assert meta["source"] == "openai"
        assert meta["message_id"] == "m1"

        # Check second record (assistant message)
        rec2 = records[1]
        assert rec2["content"] == "hi there"
        assert rec2["metadata"]["role"] == "assistant"
        assert rec2["metadata"]["model"] == "gpt-4"

    def test_readme_has_yaml_frontmatter(self, tmp_path):
        db_dir = self._import_conversation(tmp_path)
        out_dir = tmp_path / "arkiv_out"
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(out_dir),
             "--format", "arkiv", "--db", str(db_dir)],
            capture_output=True, text=True, check=True,
        )
        content = (out_dir / "README.md").read_text()
        assert content.startswith("---\n")
        assert "name: llm-memex conversations archive" in content
        assert "1 conversations exported from llm-memex" in content
        assert "generator: llm-memex" in content
        assert "conversations.jsonl" in content

    def test_schema_yaml_has_metadata_keys(self, tmp_path):
        db_dir = self._import_conversation(tmp_path)
        out_dir = tmp_path / "arkiv_out"
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(out_dir),
             "--format", "arkiv", "--db", str(db_dir)],
            capture_output=True, text=True, check=True,
        )
        content = (out_dir / "schema.yaml").read_text()
        assert "conversations:" in content
        assert "record_count: 2" in content
        assert "metadata_keys:" in content
        assert "role:" in content
        assert "type: string" in content
        assert "values:" in content

    def test_empty_export_produces_valid_output(self, tmp_path):
        """Exporting with no conversations should still produce valid files."""
        db_dir = tmp_path / "db"
        # Create an empty database by importing nothing — just init via export
        os.makedirs(db_dir, exist_ok=True)
        # Initialize DB with an import that we'll ignore — use direct DB
        from llm_memex.db import Database
        with Database(str(db_dir)) as db:
            pass  # just creates schema

        out_dir = tmp_path / "arkiv_out"
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(out_dir),
             "--format", "arkiv", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert (out_dir / "conversations.jsonl").exists()
        # JSONL should be empty
        content = (out_dir / "conversations.jsonl").read_text()
        assert content == ""


class TestCLIShow:
    def _import_one(self, tmp_path):
        """Helper: import one conversation, return db_dir."""
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "Show Test",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": ["m2"],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["hello world"]},
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
        }]))
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        return db_dir

    def test_show_list(self, tmp_path):
        db_dir = self._import_one(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "c1" in result.stdout
        assert "Show Test" in result.stdout

    def test_show_conversation(self, tmp_path):
        db_dir = self._import_one(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show", "c1", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "# Show Test" in result.stdout
        assert "hello world" in result.stdout
        assert "hi there" in result.stdout

    def test_show_not_found(self, tmp_path):
        db_dir = self._import_one(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show", "nope", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "not found" in result.stderr


class TestCLIShowSearch:
    """Tests for memex show --search and empty-state messaging."""

    def _import_one(self, tmp_path):
        """Helper: import one conversation with searchable content, return db_dir."""
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "Quantum Computing Chat",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": [],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["Tell me about quantum entanglement"]},
                        "create_time": 1700000000,
                    },
                },
            },
        }]))
        subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        return db_dir

    def test_empty_db_message(self, tmp_path):
        """Empty database shows 'No conversations found.' message."""
        db_dir = tmp_path / "empty_db"
        # Create an empty database first (show requires an existing DB)
        from llm_memex.db import Database
        with Database(str(db_dir)) as _db:
            pass
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "No conversations found." in result.stdout

    def test_search_with_match(self, tmp_path):
        """--search with matching term shows the conversation."""
        db_dir = self._import_one(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show", "--search", "quantum", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "c1" in result.stdout
        assert "Quantum Computing Chat" in result.stdout

    def test_search_no_match(self, tmp_path):
        """--search with no matching term shows appropriate message."""
        db_dir = self._import_one(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show", "--search", "xyznonexistent", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "No conversations matching 'xyznonexistent'." in result.stdout

    def test_search_ignored_when_id_given(self, tmp_path):
        """--search is ignored when a conversation ID is provided."""
        db_dir = self._import_one(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show", "c1", "--search", "quantum", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Shows the specific conversation, not a search listing
        assert "# Quantum Computing Chat" in result.stdout


class TestCLIImportAssets:
    def test_import_with_assets(self, tmp_path):
        """Import OpenAI file with an image asset, verify it's copied and rendered."""
        from llm_memex.db import Database

        db_dir = tmp_path / "db"
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create a fake image file matching the asset_pointer
        img_file = source_dir / "file-img001-photo.png"
        img_file.write_bytes(b"\x89PNG fake image data")

        # Create OpenAI export with an image asset_pointer
        export_file = source_dir / "conversations.json"
        export_file.write_text(json.dumps([{
            "id": "c-media", "title": "Media Test",
            "create_time": 1700000000, "update_time": 1700000001,
            "mapping": {
                "m1": {
                    "id": "m1", "parent": None, "children": ["m2"],
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"parts": ["check out this image"]},
                        "create_time": 1700000000,
                    },
                },
                "m2": {
                    "id": "m2", "parent": "m1", "children": [],
                    "message": {
                        "id": "m2", "author": {"role": "assistant"},
                        "content": {"parts": [
                            {"asset_pointer": "file-service://file-img001",
                             "content_type": "image_asset_pointer"},
                        ]},
                        "create_time": 1700000001,
                    },
                },
            },
        }]))

        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import",
             str(export_file), "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Imported 1 conversation" in result.stdout

        # Verify assets directory was created with the image
        assets_dir = db_dir / "assets"
        assert assets_dir.exists()
        asset_files = list(assets_dir.iterdir())
        assert len(asset_files) == 1
        assert asset_files[0].read_bytes() == b"\x89PNG fake image data"

        # Verify `memex show` renders the image as markdown
        show_result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show", "c-media", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert show_result.returncode == 0
        assert "![" in show_result.stdout

    def test_import_no_copy_assets_flag(self, tmp_path):
        """--no-copy-assets skips asset copying."""
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "Test",
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
        }]))
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--no-copy-assets", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Assets directory should not exist (no media to copy, but flag works)
        assert not (db_dir / "assets").exists()


class TestCLIListFormats:
    def test_import_list_formats(self):
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", "--list-formats"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Available import formats:" in result.stdout
        # Check all built-in importers are listed
        for name in ("openai", "anthropic", "gemini", "claude_code"):
            assert name in result.stdout

    def test_export_list_formats(self):
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", "--list-formats"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Available export formats:" in result.stdout
        # Check all built-in exporters are listed (json_export.py → json)
        for name in ("json", "markdown", "html"):
            assert name in result.stdout

    def test_import_unknown_format_shows_available(self):
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", "/dev/null",
             "--format", "nonexistent"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "unknown format" in result.stderr
        assert "openai" in result.stderr  # lists available formats

    def test_import_unknown_format_without_file(self):
        """Bad --format should error about the format, not about missing file."""
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", "--format", "bogus"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "unknown format" in result.stderr
        assert "file" not in result.stderr.lower()  # not a "missing file" error

    def test_export_unknown_format_shows_available(self, tmp_path):
        db_dir = tmp_path / "db"
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(tmp_path / "out.txt"),
             "--format", "nonexistent", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "unknown export format" in result.stderr
        assert "json" in result.stderr  # lists available formats

    def test_import_list_formats_without_file(self):
        """--list-formats should work without providing a file argument."""
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "import", "--list-formats"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Available import formats:" in result.stdout

    def test_export_list_formats_without_output(self):
        """--list-formats should work without providing an output argument."""
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", "--list-formats"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Available export formats:" in result.stdout


class TestDiscoverFormats:
    """Unit tests for _discover_formats, _discover_importers, _discover_exporters."""

    def test_discover_importers_returns_all_builtins(self):
        from llm_memex.cli import _discover_importers
        importers = _discover_importers()
        assert "openai" in importers
        assert "anthropic" in importers
        assert "gemini" in importers
        assert "claude_code" in importers

    def test_discover_importers_have_descriptions(self):
        from llm_memex.cli import _discover_importers
        importers = _discover_importers()
        for name, info in importers.items():
            assert info["description"], f"Importer {name} has no description"

    def test_discover_exporters_returns_all_builtins(self):
        from llm_memex.cli import _discover_exporters
        exporters = _discover_exporters()
        assert "json" in exporters
        assert "markdown" in exporters
        assert "html" in exporters

    def test_discover_exporters_strips_suffix(self):
        """json_export.py should be discovered as 'json', not 'json_export'."""
        from llm_memex.cli import _discover_exporters
        exporters = _discover_exporters()
        assert "json" in exporters
        assert "json_export" not in exporters

    def test_discover_exporters_have_descriptions(self):
        from llm_memex.cli import _discover_exporters
        exporters = _discover_exporters()
        for name, info in exporters.items():
            assert info["description"], f"Exporter {name} has no description"

    def test_discover_formats_skips_underscored_files(self):
        from llm_memex.cli import _discover_importers
        importers = _discover_importers()
        # __init__.py should not appear
        assert "__init__" not in importers

    def test_user_plugin_directory(self, tmp_path, monkeypatch):
        """User plugins from ~/.memex/importers/ are discovered."""
        from llm_memex.cli import _discover_formats
        user_dir = tmp_path / "user_importers"
        user_dir.mkdir()
        # Write a minimal importer plugin
        plugin = user_dir / "custom.py"
        plugin.write_text(
            '"""Custom test importer."""\n'
            'def detect(f): return False\n'
            'def import_path(f): return []\n'
        )
        builtin_dir = tmp_path / "empty_builtins"
        builtin_dir.mkdir()
        formats = _discover_formats(builtin_dir, user_dir, ("detect", "import_path"))
        assert "custom" in formats
        assert "Custom test importer." in formats["custom"]["description"]


class TestCLIMissingDb:
    """Test that mistyped --db paths produce clear errors, not silent no-ops."""

    def test_show_missing_db_errors(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "show",
             "--db", str(tmp_path / "nonexistent")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "database not found" in result.stderr.lower()

    def test_export_missing_db_errors(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(tmp_path / "out.md"),
             "--format", "markdown", "--db", str(tmp_path / "nonexistent")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "database not found" in result.stderr.lower()

    def test_run_missing_db_errors(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "run", "enrich_trivial",
             "--db", str(tmp_path / "nonexistent")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "database not found" in result.stderr.lower()


class TestCLIHelp:
    def test_no_command_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "llm-memex" in result.stdout.lower()


class TestCliConfigCache:
    def test_load_cli_config_is_env_aware(self, tmp_path, monkeypatch):
        """_load_cli_config must not serve a stale config after MEMEX_CONFIG
        changes within the process (ARCH-7)."""
        from llm_memex import cli
        if hasattr(cli._load_cli_config, "_cache"):
            del cli._load_cli_config._cache
        try:
            a = tmp_path / "a.yaml"
            a.write_text("primary: aaa\ndatabases: {aaa: {path: /tmp/a}}\n")
            b = tmp_path / "b.yaml"
            b.write_text("primary: bbb\ndatabases: {bbb: {path: /tmp/b}}\n")
            monkeypatch.setenv("MEMEX_CONFIG", str(a))
            cfg_a = cli._load_cli_config()
            monkeypatch.setenv("MEMEX_CONFIG", str(b))
            cfg_b = cli._load_cli_config()
            assert cfg_a["primary"] == "aaa"
            assert cfg_b["primary"] == "bbb"
        finally:
            if hasattr(cli._load_cli_config, "_cache"):
                del cli._load_cli_config._cache
