"""Thin CLI for memex. Delegates to core for all logic."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _load_cli_config():
    """Load config for CLI commands. Cached per resolved config path so a
    MEMEX_CONFIG change within the process is not served a stale config."""
    config_path = os.environ.get(
        "MEMEX_CONFIG", os.path.expanduser("~/.memex/config.yaml")
    )
    cache = getattr(_load_cli_config, "_cache", None)
    if cache is None or cache[0] != config_path:
        from llm_memex.config import load_config
        _load_cli_config._cache = (config_path, load_config(config_path))
    return _load_cli_config._cache[1]


def _resolve_db_path(name_or_path: str) -> str:
    """Resolve a --db value: config database name or literal path."""
    config = _load_cli_config()
    databases = config.get("databases", {})
    if name_or_path in databases:
        return os.path.expanduser(databases[name_or_path]["path"])
    return os.path.expanduser(name_or_path)


def _default_db() -> str:
    """Default --db value: primary from config, env var, or ~/.memex/default."""
    config = _load_cli_config()
    primary = config.get("primary")
    databases = config.get("databases", {})
    if primary and primary in databases:
        return primary
    return os.environ.get("MEMEX_DATABASE_PATH", "~/.memex/default")


def _open_db(args, readonly=True):
    """Open a database, exiting with an error if not found."""
    from llm_memex.db import Database

    db_name = args.db or _default_db()
    db_path = _resolve_db_path(db_name)
    try:
        return Database(db_path, readonly=readonly)
    except FileNotFoundError:
        print(f"Error: database not found: {db_name}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="llm-memex", description="Personal conversation knowledge base"
    )
    parser.add_argument(
        "--version", action="version", version=f"llm-memex {_get_version()}"
    )
    sub = parser.add_subparsers(dest="command")

    # import
    imp = sub.add_parser("import", help="Import conversations from a file or directory")
    imp.add_argument("file", nargs="?", help="File or directory to import")
    imp.add_argument("--format", help="Force importer format (use --list-formats to see available)")
    imp.add_argument("--list-formats", action="store_true", help="List available import formats")
    imp.add_argument("--recursive", "-r", action="store_true",
                     help="Recursively import all detected files from a directory")
    imp.add_argument("--force", action="store_true",
                     help="Re-import all conversations even if unchanged "
                          "(REPLACE mode: cascades delete messages/tags/enrichments)")
    imp.add_argument("--merge", action="store_true",
                     help="Merge into existing conversations without "
                          "clobbering enrichments or conversation metadata. "
                          "Adds only new messages and tags via INSERT OR IGNORE. "
                          "Use this when round-tripping annotations from an "
                          "exported HTML bundle back to the primary DB.")
    imp.add_argument("--no-copy-assets", action="store_true",
                     help="Skip copying media assets into the database directory")
    imp.add_argument(
        "--db",
        help="Database directory",
        default=None,
    )

    # export
    exp = sub.add_parser("export", help="Export conversations")
    exp.add_argument("output", nargs="?", help="Output file path")
    exp.add_argument("--format", default="markdown", help="Export format (use --list-formats to see available)")
    exp.add_argument("--list-formats", action="store_true", help="List available export formats")
    exp.add_argument("--no-notes", action="store_true",
                     help="Exclude notes from exported output")
    exp.add_argument("--include-archived", action="store_true",
                     help="Include soft-deleted (archived) conversations in the "
                          "export (HTML export excludes them by default)")
    exp.add_argument(
        "--db",
        help="Database directory",
        default=None,
    )

    # show
    show = sub.add_parser("show", help="Display a conversation")
    show.add_argument("id", nargs="?", help="Conversation ID (omit to list all)")
    show.add_argument("--search", help="Search conversations by text (FTS)")
    show.add_argument("--raw", action="store_true", help="(deprecated, now always prints markdown)")
    show.add_argument(
        "--db",
        help="Database directory",
        default=None,
    )

    # mcp
    sub.add_parser("mcp", help="Start MCP server")

    # run
    run_p = sub.add_parser("run", help="Run a memex script")
    run_p.add_argument("name", nargs="?", help="Script name (omit or use --list to list)")
    run_p.add_argument("--list", action="store_true", help="List available scripts")
    run_p.add_argument("--apply", action="store_true", help="Commit changes (default: dry-run)")
    run_p.add_argument("--verbose", action="store_true", help="Verbose output")
    run_p.add_argument(
        "--db",
        help="Database directory",
        default=None,
    )

    args, remaining = parser.parse_known_args()
    if args.command == "import":
        _cmd_import(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "mcp":
        _cmd_mcp(args)
    elif args.command == "run":
        _cmd_run(args, remaining)
    else:
        parser.print_help()


def _get_version():
    from llm_memex import __version__

    return __version__


def _list_formats(formats, label):
    """Print available formats with descriptions."""
    print(f"Available {label} formats:\n")
    for name, info in sorted(formats.items()):
        print(f"  {name:20s}  {info['description']}")
    print()


def _print_import_result(imported, unchanged, db_path):
    """Print the 'Imported N conversation(s) [(M unchanged)] into <db>' line."""
    parts = [f"Imported {imported} conversation(s)"]
    if unchanged:
        parts.append(f"({unchanged} unchanged)")
    parts.append(f"into {db_path}")
    print(" ".join(parts))


def _cmd_import(args):
    import logging
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stderr,
    )
    if args.list_formats:
        _list_formats(_discover_importers(), "import")
        return
    if args.format:
        importers = _discover_importers()
        if args.format not in importers:
            print(f"Error: unknown format '{args.format}'. "
                  f"Available: {', '.join(sorted(importers))}",
                  file=sys.stderr)
            sys.exit(1)
    if not args.file:
        print("Error: the following arguments are required: file", file=sys.stderr)
        sys.exit(1)

    from llm_memex.db import Database

    db_path = _resolve_db_path(args.db or _default_db())
    file_path = Path(args.file).resolve()

    if file_path.is_dir():
        # Try importers directly on the directory first
        convs = _auto_import(str(file_path), args.format, exit_on_fail=False)
        if convs is not None:
            # An importer claimed the directory
            with Database(db_path) as db:
                n_imp, n_unch = _save_convs(convs, file_path, args, db)
                _print_import_result(n_imp, n_unch, db_path)
        elif args.recursive:
            # Fallback: walk directory, try each file
            with Database(db_path) as db:
                stats = {"imported": 0, "unchanged": 0, "files_imported": 0, "files_skipped": 0}
                for child in sorted(file_path.rglob("*")):
                    if not child.is_file():
                        continue
                    n_imp, n_unch = _import_one(child, args, db)
                    if n_imp > 0 or n_unch > 0:
                        stats["imported"] += n_imp
                        stats["unchanged"] += n_unch
                        stats["files_imported"] += 1
                    else:
                        stats["files_skipped"] += 1
                parts = [f"Imported {stats['imported']} conversation(s) from "
                         f"{stats['files_imported']} file(s) "
                         f"({stats['files_skipped']} skipped)"]
                if stats["unchanged"]:
                    parts.append(f"({stats['unchanged']} unchanged)")
                parts.append(f"into {db_path}")
                print(" ".join(parts))
        else:
            print(f"Error: '{args.file}' is a directory. "
                  f"Use --recursive to import all files, "
                  f"or point at a recognized export directory.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        with Database(db_path) as db:
            n_imp, n_unch = _import_one(file_path, args, db, exit_on_miss=True)
            _print_import_result(n_imp, n_unch, db_path)


def _materialize_arkiv_notes(conv, db):
    """Pull any _arkiv_notes off each message's metadata and INSERT OR IGNORE
    into the notes table. The arkiv importer stashes message-level notes
    there during import so the conversation-save step doesn't need to know
    about notes. INSERT OR IGNORE keeps this idempotent on re-imports.
    """
    notes = []
    for msg in conv.messages.values():
        arkiv_notes = msg.metadata.get("_arkiv_notes") if msg.metadata else None
        if not arkiv_notes:
            continue
        for n in arkiv_notes:
            if not n.get("id") or not n.get("text"):
                continue
            notes.append({
                "id": n["id"],
                "conversation_id": conv.id,
                "message_id": msg.id,
                "target_kind": "message",
                "text": n["text"],
            })
        # Clean up so save_conversation doesn't re-serialize internal metadata.
        msg.metadata.pop("_arkiv_notes", None)
    if notes:
        db.merge_notes(notes)


def _save_convs(convs, source_path, args, db):
    """Save imported conversations to the database.

    Returns (imported_count, unchanged_count).
    source_path is used as the base for asset resolution.

    Behavior modes:
      - default:  INSERT OR REPLACE per conversation (cascade delete
                  messages/tags/enrichments/provenance first). Used for
                  fresh imports from platform exports.
      - --force:  same as default but also bypass the unchanged-check.
      - --merge:  for each conversation, keep existing DB state and only
                  INSERT OR IGNORE new messages + tags. Preserves
                  enrichments and provenance. Materialize any message-level
                  notes present in arkiv metadata as notes rows. This is
                  the round-trip mode for merging SPA-added annotations
                  back into the primary DB.
    """
    from llm_memex.assets import resolve_source_assets, copy_assets

    db_path = _resolve_db_path(args.db or _default_db())
    source_dir = source_path if source_path.is_dir() else source_path.parent
    asset_dir = Path(db_path) / "assets"
    imported = 0
    unchanged = 0
    for conv in convs:
        if getattr(args, "merge", False):
            # Merge mode: preserve existing state, add only new messages and
            # tags. Also materialize any arkiv message-level notes.
            stats = db.merge_conversation(conv)
            if stats["created_conversation"] or stats["added_messages"] or stats["added_tags"]:
                imported += 1
            else:
                unchanged += 1
            _materialize_arkiv_notes(conv, db)
            continue

        # Skip-if-unchanged check (bypass with --force)
        if not args.force and db.conversation_unchanged(
            conv.id, conv.updated_at, conv.message_count
        ):
            unchanged += 1
            continue

        prov = conv.metadata.pop("_provenance", None)
        if not args.no_copy_assets:
            source_type = prov.get("source_type", "") if prov else ""
            resolve_source_assets(conv, source_dir, source_type)
            copy_assets(conv, asset_dir)
        # save_conversation does a full replace of the conversation's child
        # rows (it DELETEs enrichments). Enrichments are derived OR
        # user-authored (summary/topic/importance/excerpt set via
        # update_conversations), and the importer never carries them, so a
        # routine re-import of an active conversation would silently wipe
        # them. Capture them before the replace and re-insert after, exactly
        # as provenance is re-written below.
        existing_enrichments = db.get_enrichments(conv.id)
        db.save_conversation(conv)
        if existing_enrichments:
            db.save_enrichments(conv.id, existing_enrichments)
        if prov:
            db.save_provenance(
                conv.id,
                source_type=prov.get("source_type", "unknown"),
                source_file=prov.get("source_file"),
                source_id=prov.get("source_id"),
                source_hash=prov.get("source_hash"),
            )
        imported += 1

        # Progress to stderr (every 50 conversations)
        if len(convs) > 10 and (imported + unchanged) % 50 == 0:
            print(f"\rImporting: {imported + unchanged}/{len(convs)} "
                  f"conversations ({imported} imported, {unchanged} unchanged)...",
                  end="", file=sys.stderr, flush=True)

    # Clear progress line
    if len(convs) > 10 and (imported + unchanged) > 0:
        print("\r" + " " * 79 + "\r", end="", file=sys.stderr, flush=True)

    return (imported, unchanged)


def _import_one(file_path, args, db, exit_on_miss=False):
    """Import a single file into the database.

    Returns (imported_count, unchanged_count).
    If exit_on_miss=True, calls sys.exit(1) when no importer matches.
    """
    convs = _auto_import(str(file_path), args.format, exit_on_fail=exit_on_miss)
    if convs is None:
        return (0, 0)
    return _save_convs(convs, file_path, args, db)


def _discover_formats(directory, user_directory, required_attrs, strip_suffix=""):
    """Discover plugin modules from built-in and user directories.

    Returns dict mapping name to {"path": Path, "description": str, "module": mod}.
    User plugins shadow built-in plugins of the same name (with warning).
    strip_suffix removes a naming suffix (e.g. "_export") so json_export.py → "json".
    """
    formats = {}
    builtin_names = set()

    for d in [directory, user_directory]:
        if not d.exists():
            continue
        for py_file in sorted(d.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            name = py_file.stem
            if strip_suffix and name.endswith(strip_suffix):
                name = name[: -len(strip_suffix)]
            if d == directory:
                builtin_names.add(name)
            elif name in builtin_names:
                print(
                    f"Warning: user plugin '{name}' shadows built-in. "
                    f"Rename to avoid conflicts, or use --format to select explicitly.",
                    file=sys.stderr,
                )
            try:
                spec = importlib.util.spec_from_file_location(name, py_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception:
                continue
            if all(hasattr(mod, a) for a in required_attrs):
                doc = getattr(mod, "__doc__", None) or ""
                formats[name] = {
                    "path": py_file,
                    "description": doc.strip().split("\n")[0].strip(),
                    "module": mod,
                }
    return formats


def _discover_importers():
    return _discover_formats(
        Path(__file__).parent / "importers",
        Path.home() / ".memex" / "importers",
        ("detect", "import_path"),
    )


def _discover_exporters():
    return _discover_formats(
        Path(__file__).parent / "exporters",
        Path.home() / ".memex" / "exporters",
        ("export",),
        strip_suffix="_export",
    )


def _run_import_path(module, file_path, exit_on_fail):
    """Call a module's import_path, converting failures to exit (or None).

    On success returns the imported conversations. On exception, prints an
    error and exits when exit_on_fail, otherwise prints a warning and returns
    None.
    """
    try:
        return module.import_path(file_path)
    except Exception as e:
        if exit_on_fail:
            print(f"Error: failed to import {file_path}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Warning: failed to import {file_path}: {e}", file=sys.stderr)
        return None


def _auto_import(file_path, format_name=None, exit_on_fail=True):
    """Auto-detect importer and import path (file or directory).

    Returns list of conversations, or None if no importer matched and exit_on_fail=False.
    """
    importers = _discover_importers()
    if format_name:
        if format_name not in importers:
            print(f"Error: unknown format '{format_name}'. "
                  f"Available: {', '.join(sorted(importers))}",
                  file=sys.stderr)
            sys.exit(1)
        return _run_import_path(importers[format_name]["module"], file_path, exit_on_fail)
    # Preferred order: claude_code_full wins over claude_code when both
    # detect (both share the same detect function, so otherwise the
    # alphabetically-earlier skeleton importer would always win).
    _PREFERRED = ("claude_code_full",)
    ordered = list(_PREFERRED) + [
        n for n in importers.keys() if n not in _PREFERRED
    ]
    for name in ordered:
        info = importers.get(name)
        if info is None:
            continue
        if info["module"].detect(file_path):
            return _run_import_path(info["module"], file_path, exit_on_fail)
    if exit_on_fail:
        print(f"Error: no importer found for {file_path}", file=sys.stderr)
        sys.exit(1)
    return None


def _cmd_show(args):
    with _open_db(args) as db:
        if args.id is None:
            # List mode
            cursor = None
            found = False
            while True:
                result = db.query_conversations(
                    query=args.search, limit=50, cursor=cursor,
                )
                for item in result["items"]:
                    found = True
                    tags = f"  [{item['tags_csv']}]" if item.get("tags_csv") else ""
                    print(f"{item['id']}  {item['message_count']:3d} msgs  {item['title'] or '(untitled)'}{tags}")
                if not result["has_more"]:
                    break
                cursor = result["next_cursor"]
            if not found:
                if args.search:
                    print(f"No conversations matching '{args.search}'.")
                else:
                    print("No conversations found.")
            return

        conv = db.load_conversation(args.id)
        if conv is None:
            print(f"Error: conversation '{args.id}' not found", file=sys.stderr)
            sys.exit(1)

        print(_render_conversation_md(conv))


def _render_conversation_md(conv):
    """Render a conversation as markdown text."""
    lines = []
    lines.append(f"# {conv.title or conv.id}")
    lines.append("")
    meta = []
    if conv.source:
        meta.append(f"**Source:** {conv.source}")
    if conv.model:
        meta.append(f"**Model:** {conv.model}")
    if conv.tags:
        meta.append(f"**Tags:** {', '.join(conv.tags)}")
    if conv.message_count:
        meta.append(f"**Messages:** {conv.message_count}")
    if meta:
        lines.append(" | ".join(meta))
        lines.append("")
    lines.append("---")
    lines.append("")

    for i, path_msgs in enumerate(conv.get_all_paths()):
        if i > 0:
            lines.append("---")
            lines.append("")
        for msg in path_msgs:
            text = msg.get_content_md()
            lines.append(f"### {msg.role}")
            lines.append("")
            lines.append(text)
            lines.append("")

    return "\n".join(lines)


def _cmd_export(args):
    if args.list_formats:
        _list_formats(_discover_exporters(), "export")
        return
    exporters = _discover_exporters()
    if args.format not in exporters:
        print(f"Error: unknown export format '{args.format}'. "
              f"Available: {', '.join(sorted(exporters))}",
              file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: the following arguments are required: output", file=sys.stderr)
        sys.exit(1)

    exporter_mod = exporters[args.format]["module"]

    with _open_db(args) as db:
        # Load conversations in chunks to avoid memory exhaustion
        convs = []
        cursor = None
        while True:
            result = db.query_conversations(limit=100, cursor=cursor)
            for item in result["items"]:
                conv = db.load_conversation(item["id"])
                if conv is not None:
                    convs.append(conv)
            if not result["has_more"]:
                break
            cursor = result["next_cursor"]
        include_notes = not getattr(args, "no_notes", False)
        include_archived = getattr(args, "include_archived", False)
        exporter_mod.export(
            convs, args.output,
            db_path=db.db_path,
            db=db,
            include_notes=include_notes,
            include_archived=include_archived,
        )
        print(f"Exported {len(convs)} conversation(s) to {args.output}")


def _cmd_run(args, remaining):
    from llm_memex.db import Database
    from llm_memex.scripts import discover_scripts, load_script

    if args.list or not args.name:
        scripts = discover_scripts()
        if not scripts:
            print("No scripts available.")
            return
        print("Available scripts:\n")
        for name, info in sorted(scripts.items()):
            print(f"  {name:20s}  {info['description']}")
        print(f"\nUsage: memex run <name> [--apply] [script args...]")
        return

    try:
        mod = load_script(args.name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build a sub-parser for the script's own arguments
    script_parser = argparse.ArgumentParser(prog=f"llm-memex run {args.name}")
    mod.register_args(script_parser)
    script_args = script_parser.parse_args(remaining)

    with _open_db(args, readonly=not args.apply) as db:
        result = mod.run(db, script_args, apply=args.apply)
        if args.verbose and result:
            print(f"\nResult: {result}")


def _cmd_mcp(args):
    from llm_memex.mcp import main as mcp_main

    mcp_main()


if __name__ == "__main__":
    main()
