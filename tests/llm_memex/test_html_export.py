"""Tests for HTML SPA template."""

from llm_memex.exporters.html_template import get_template


class TestHtmlTemplate:
    def test_template_is_valid_html(self):
        html = get_template()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "sql-wasm.js" in html
        assert "initApp" in html

    def test_template_contains_key_elements(self):
        html = get_template()
        assert 'id="search-box"' in html
        assert 'id="conv-list"' in html
        assert 'id="settings-overlay"' in html
        assert "anthropic-dangerous-direct-browser-access" in html

    def test_template_contains_css_custom_properties(self):
        html = get_template()
        assert "--bg:" in html
        assert "--text:" in html
        assert "--border:" in html
        assert "--text-accent:" in html
        assert "--font-mono:" in html

    def test_template_contains_layout_structure(self):
        """Single-column layout: top-bar + main, no sidebar, no timeline."""
        html = get_template()
        assert 'id="app"' in html
        assert 'id="top-bar"' in html
        assert 'id="main"' in html
        assert 'id="top-brand"' in html
        # Sidebar and timeline were removed in the single-column refresh
        assert 'id="sidebar"' not in html
        assert 'id="timeline"' not in html

    def test_template_contains_db_loading_cascade(self):
        html = get_template()
        # URL param check
        assert "URLSearchParams" in html
        assert 'get("db")' in html
        # Default fetch tries .db.gz first (smaller), falls back to .db
        assert "./conversations.db.gz" in html
        assert "./conversations.db" in html
        # Gzip decompression uses native DecompressionStream (no lib)
        assert 'new DecompressionStream("gzip")' in html
        # File picker fallback
        assert 'id="drop-zone"' in html
        assert 'id="file-input"' in html

    def test_template_contains_query_helpers(self):
        html = get_template()
        assert "function query(sql" in html
        # The write helper wraps db.run()
        write_helper = "function " + "exec" + "(sql"
        assert write_helper in html
        assert "stmt.getAsObject()" in html
        assert "db.run(sql" in html

    def test_template_contains_placeholder_functions(self):
        html = get_template()
        assert "function onDbLoaded()" in html
        assert "function loadConversations()" in html
        assert "function openConversation(convId)" in html
        assert "function renderHome()" in html
        assert "function sendMessage()" in html
        assert "function downloadDb()" in html
        # Timeline is gone
        assert "function initTimeline" not in html

    def test_template_references_local_wasm(self):
        """sql-wasm is referenced as a local sibling file, not a CDN URL.

        This keeps exports self-contained and durable — no reliance on
        jsdelivr availability or version pinning drift.
        """
        html = get_template()
        assert "sql-wasm.wasm" in html
        assert "sql-wasm.js" in html
        assert "cdn.jsdelivr.net" not in html
        assert "jsdelivr" not in html

    def test_template_is_self_contained(self):
        """Template should have inline CSS and JS, no external stylesheets."""
        html = get_template()
        assert "<style>" in html
        assert "</style>" in html
        assert "<script>" in html
        assert "</script>" in html
        # Only external resource should be sql.js CDN
        assert 'rel="stylesheet"' not in html

    def test_template_contains_esc_helper(self):
        """Template should have the XSS-safe esc() helper."""
        html = get_template()
        assert "function esc(s)" in html
        assert "d.textContent" in html
        assert "d.innerHTML" in html

    def test_template_contains_active_filters_state(self):
        """Template should have the activeFilters state object."""
        html = get_template()
        assert "var activeFilters" in html
        assert "source: null" in html
        assert "tag: null" in html
        assert "starred: false" in html
        assert "dateFrom: null" in html
        assert "dateTo: null" in html

    def test_template_contains_render_filters(self):
        """Template should have the renderFilters function."""
        html = get_template()
        assert "function renderFilters()" in html
        assert "filter-chip" in html
        assert "data-filter" in html

    def test_template_contains_search_debounce(self):
        """onDbLoaded should wire up debounced search (200ms)."""
        html = get_template()
        assert "searchTimer" in html
        assert "clearTimeout(searchTimer)" in html
        assert "setTimeout" in html
        assert "200" in html

    def test_template_contains_filters_element(self):
        """Template should have the #filters container."""
        html = get_template()
        assert 'id="filters"' in html

    def test_template_contains_filter_chip_css(self):
        """Template should have CSS for filter chips."""
        html = get_template()
        assert ".filter-chip" in html
        assert ".filter-chip.active" in html

    def test_template_contains_fmt_date_helper(self):
        """Template should have the fmtDate helper."""
        html = get_template()
        assert "function fmtDate(iso)" in html

    def test_template_load_conversations_builds_sql(self):
        """loadConversations should build SQL with LIKE search and filter support."""
        html = get_template()
        # LIKE search (no FTS5)
        assert "LIKE '%' || ? || '%'" in html
        # Tag filter with EXISTS subquery
        assert "EXISTS (SELECT 1 FROM tags t WHERE t.conversation_id = c.id AND t.tag = ?)" in html
        # Source filter
        assert "c.source = ?" in html
        # Starred filter
        assert "c.starred_at IS NOT NULL" in html
        # Order and limit
        assert "ORDER BY c.updated_at DESC LIMIT 500" in html

    def test_template_on_db_loaded_calls_subroutines(self):
        """onDbLoaded wires the search box and routes to the initial view."""
        html = get_template()
        start = html.index("function onDbLoaded()")
        chunk = html[start:start + 1500]
        # onDbLoaded delegates rendering to the hash router, which resolves
        # to renderHome() when no hash is present.
        assert "renderRoute()" in chunk
        assert "searchWired" in chunk
        # Home-render itself delegates to these
        home_start = html.index("function renderHome()")
        home_chunk = html[home_start:home_start + 1000]
        assert "renderWelcomeStats()" in home_chunk
        assert "renderFilters()" in home_chunk
        assert "loadConversations()" in home_chunk

    def test_template_conversation_count_in_status(self):
        """onDbLoaded should query conversation count for status bar."""
        html = get_template()
        assert 'SELECT count(*) AS n FROM conversations' in html

    def test_template_load_conversations_uses_parameterized_queries(self):
        """loadConversations should use parameterized queries (not string concat)."""
        html = get_template()
        # Params are pushed to array and passed to query()
        assert "params.push(" in html
        assert "query(sql, params" in html


class TestHtmlRendering:
    """Tests for conversation viewer rendering functions (Task 3)."""

    def test_template_has_render_functions(self):
        html = get_template()
        assert 'function renderContent(' in html
        assert 'function renderMedia(' in html
        assert 'function renderMarkdown(' in html
        assert 'function openConversation(' in html

    def test_open_conversation_queries_db(self):
        """openConversation should query conversations, messages, and tags."""
        html = get_template()
        # Find openConversation body (up to next function)
        start = html.index("function openConversation(convId)")
        end = html.index("function renderConversation(")
        chunk = html[start:end]
        assert "FROM conversations WHERE id" in chunk
        assert "FROM messages WHERE conversation_id" in chunk
        assert "FROM tags WHERE conversation_id" in chunk

    def test_open_conversation_sets_active_id(self):
        """openConversation should set activeConvId."""
        html = get_template()
        assert "var activeConvId" in html
        start = html.index("function openConversation(convId)")
        chunk = html[start:start + 300]
        assert "activeConvId = convId" in chunk

    def test_render_conversation_shows_header(self):
        """renderConversation should render title, meta, and tags."""
        html = get_template()
        start = html.index("function renderConversation(")
        chunk = html[start:start + 1500]
        assert "conv-header" in chunk
        assert "conv-header-meta" in chunk
        assert "conv-header-tags" in chunk

    def test_render_conversation_shows_input_area(self):
        """renderConversation should unhide the input area."""
        html = get_template()
        start = html.index("function renderConversation(")
        end = html.index("function renderContent(")
        chunk = html[start:end]
        assert 'remove("hidden")' in chunk

    def test_render_content_handles_text_blocks(self):
        """renderContent should pass text blocks through renderMarkdown."""
        html = get_template()
        start = html.index("function renderContent(")
        chunk = html[start:start + 500]
        assert "renderMarkdown" in chunk

    def test_render_content_handles_media_blocks(self):
        """renderContent should pass media blocks through renderMedia."""
        html = get_template()
        start = html.index("function renderContent(")
        chunk = html[start:start + 500]
        assert "renderMedia" in chunk

    def test_render_content_skips_tool_and_thinking(self):
        """renderContent should only handle text and media, skipping other types."""
        html = get_template()
        start = html.index("function renderContent(")
        chunk = html[start:start + 500]
        # Should only process "text" and "media" types, skip everything else
        assert '"text"' in chunk
        assert '"media"' in chunk

    def test_render_media_image(self):
        """renderMedia should render images with img tag and lazy loading."""
        html = get_template()
        start = html.index("function renderMedia(")
        chunk = html[start:start + 1500]
        assert "<img" in chunk
        assert "loading" in chunk
        assert "lazy" in chunk

    def test_render_media_audio(self):
        """renderMedia should render audio with audio tag and controls."""
        html = get_template()
        start = html.index("function renderMedia(")
        chunk = html[start:start + 1500]
        assert "<audio" in chunk
        assert "controls" in chunk

    def test_render_media_video(self):
        """renderMedia should render video with video tag and controls."""
        html = get_template()
        start = html.index("function renderMedia(")
        chunk = html[start:start + 1500]
        assert "<video" in chunk
        assert "controls" in chunk

    def test_render_media_data_uri(self):
        """renderMedia should build data URIs from base64 data."""
        html = get_template()
        start = html.index("function renderMedia(")
        chunk = html[start:start + 1500]
        assert "data:" in chunk
        assert ";base64," in chunk

    def test_render_media_no_src_placeholder(self):
        """renderMedia should show filename placeholder when no URL or data."""
        html = get_template()
        start = html.index("function renderMedia(")
        chunk = html[start:start + 1500]
        assert "fname" in chunk or "filename" in chunk

    def test_render_media_link_fallback(self):
        """renderMedia should fall back to link for unknown media types."""
        html = get_template()
        start = html.index("function renderMedia(")
        # renderMedia ends where the next top-level function begins.
        end = html.index("function renderMarkdown(", start)
        chunk = html[start:end]
        assert "<a href" in chunk
        assert "target" in chunk

    def test_render_media_uses_attribute_escaping(self):
        """renderMedia must use escAttr (not esc) for attribute values to
        prevent quote-based attribute escape XSS."""
        html = get_template()
        start = html.index("function renderMedia(")
        end = html.index("function renderMarkdown(", start)
        chunk = html[start:end]
        # All src= and href= values should be wrapped in escAttr(), never plain esc()
        assert 'src="\' + escAttr(' in chunk
        assert 'href="\' + escAttr(' in chunk

    def test_render_media_has_url_scheme_allowlist(self):
        """renderMedia must reject javascript:/vbscript: URLs via safeMediaUrl."""
        html = get_template()
        # safeMediaUrl helper should exist and be called from renderMedia
        assert "function safeMediaUrl(" in html
        start = html.index("function renderMedia(")
        end = html.index("function renderMarkdown(", start)
        chunk = html[start:end]
        assert "safeMediaUrl(" in chunk

    def test_execute_tool_rejects_non_select(self):
        """executeTool must reject anything that isn't SELECT or EXPLAIN,
        including WITH (which can mutate via data-modifying CTE)."""
        html = get_template()
        start = html.index("function executeTool(")
        # Extract until the next function definition after executeTool.
        end = html.index("function ", start + 1)
        chunk = html[start:end]
        # First-word check should only allow SELECT and EXPLAIN
        assert '"SELECT"' in chunk
        assert '"EXPLAIN"' in chunk
        # WITH must NOT be on the allowlist (data-modifying CTEs bypass it)
        assert '"WITH"' not in chunk
        # And we check getRowsModified as defense in depth
        assert "getRowsModified" in chunk

    def test_render_markdown_escapes_html_first(self):
        """renderMarkdown should escape HTML entities before markdown transforms."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        # The first replacements should be &, <, >
        chunk = html[start:start + 400]
        amp_pos = chunk.index("&amp;")
        lt_pos = chunk.index("&lt;")
        gt_pos = chunk.index("&gt;")
        # Escaping should happen before markdown transforms
        assert amp_pos < lt_pos < gt_pos

    def test_render_markdown_handles_code_blocks(self):
        """renderMarkdown should handle fenced code blocks with language class."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        chunk = html[start:start + 1500]
        assert "```" in chunk
        assert "language-" in chunk
        assert "<pre>" in chunk
        assert "<code" in chunk

    def test_render_markdown_handles_inline_code(self):
        """renderMarkdown should handle inline code with backticks."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        chunk = html[start:start + 1500]
        # Inline code regex pattern
        assert "`" in chunk

    def test_render_markdown_handles_headings(self):
        """renderMarkdown should handle h1-h4 headings."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        chunk = html[start:start + 2000]
        assert "<h1>" in chunk
        assert "<h2>" in chunk
        assert "<h3>" in chunk
        assert "<h4>" in chunk

    def test_render_markdown_handles_emphasis(self):
        """renderMarkdown should handle bold, italic, and bold+italic."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        chunk = html[start:start + 2500]
        assert "<strong>" in chunk
        assert "<em>" in chunk

    def test_render_markdown_handles_links_and_images(self):
        """renderMarkdown should handle markdown links and images."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        chunk = html[start:start + 2500]
        assert "<a href" in chunk
        assert "<img src" in chunk

    def test_render_markdown_handles_lists(self):
        """renderMarkdown should handle unordered lists."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        chunk = html[start:start + 2500]
        assert "<li>" in chunk
        assert "<ul>" in chunk

    def test_render_markdown_handles_horizontal_rules(self):
        """renderMarkdown should handle horizontal rules."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        chunk = html[start:start + 2500]
        assert "<hr>" in chunk

    def test_render_markdown_code_block_protection(self):
        """renderMarkdown should protect code blocks from other transforms."""
        html = get_template()
        start = html.index("function renderMarkdown(")
        chunk = html[start:start + 2500]
        # Uses placeholder pattern to protect code blocks
        assert "codeBlocks" in chunk
        assert "inlineCodes" in chunk

    def test_conversation_content_json_parse(self):
        """renderConversation should parse content as JSON."""
        html = get_template()
        start = html.index("function renderConversation(")
        chunk = html[start:start + 2500]
        assert "JSON.parse" in chunk

    def test_conversation_content_string_fallback(self):
        """Should handle content that is plain string (not JSON array)."""
        html = get_template()
        start = html.index("function renderConversation(")
        chunk = html[start:start + 2500]
        # Should check if parsed is a string and wrap it
        assert "typeof parsed" in chunk or "string" in chunk

    def test_css_for_message_content_elements(self):
        """Template should have CSS for message content (code, images, etc)."""
        html = get_template()
        assert ".message-content pre" in html
        assert ".message-content code" in html
        assert ".message-content img" in html

    def test_css_for_conv_header(self):
        """Template should have CSS for conversation header."""
        html = get_template()
        assert ".conv-header" in html
        assert ".conv-tag" in html

    def test_messages_ordered_by_created_at(self):
        """openConversation should order messages by created_at."""
        html = get_template()
        start = html.index("function openConversation(convId)")
        end = html.index("function renderConversation(")
        chunk = html[start:end]
        assert "ORDER BY created_at" in chunk


class TestHashRouter:
    """Hash-based routing for bookmarks and browser back/forward."""

    def test_template_has_router_functions(self):
        html = get_template()
        assert "function parseRoute()" in html
        assert "function buildRoute(r)" in html
        assert "function setRoute(r, opts)" in html
        assert "function renderRoute()" in html

    def test_hashchange_listener_wired(self):
        html = get_template()
        assert 'addEventListener("hashchange", renderRoute)' in html
        assert "hashWired" in html

    def test_router_recognizes_all_routes(self):
        """parseRoute must recognize conv/search/marginalia routes."""
        html = get_template()
        start = html.index("function parseRoute()")
        chunk = html[start:start + 600]
        assert '"conv"' in chunk
        assert '"search"' in chunk
        assert '"marginalia"' in chunk

    def test_mode_functions_update_hash(self):
        """Mode entry-points call setRoute so URL reflects state."""
        html = get_template()
        # openConversation sets conv route
        cs = html.index("function openConversation(convId)")
        cchunk = html[cs:cs + 400]
        assert "setRoute({ mode: \"conv\", id: convId })" in cchunk
        # openMarginalia sets marginalia route
        ms = html.index("function openMarginalia()")
        mchunk = html[ms:ms + 400]
        assert 'setRoute({ mode: "marginalia" })' in mchunk
        # openSearchResults handles push-vs-replace for typing refinement
        ss = html.index("function openSearchResults(initialTerm)")
        schunk = html[ss:ss + 600]
        assert "chatMode === \"search\"" in schunk  # same-mode check
        assert "replace: true" in schunk
        # goHome resets to empty route
        gs = html.index("function goHome()")
        gchunk = html[gs:gs + 300]
        assert 'setRoute({ mode: "home" })' in gchunk

    def test_routing_guard_prevents_loops(self):
        """renderRoute sets routingFromHash so mode functions don't re-push."""
        html = get_template()
        assert "var routingFromHash = false;" in html
        assert "routingFromHash = true;" in html
        # setRoute respects the guard (returns false when suppressed)
        ss = html.index("function setRoute(r, opts)")
        schunk = html[ss:ss + 500]
        assert "if (routingFromHash) return false;" in schunk

    def test_parse_route_tolerates_bad_percent_encoding(self):
        """parseRoute uses safeDecode so malformed hashes don't throw."""
        html = get_template()
        assert "function safeDecode(s)" in html
        ps = html.index("function parseRoute()")
        pchunk = html[ps:ps + 700]
        assert "safeDecode(tail)" in pchunk

    def test_go_home_avoids_double_render(self):
        """goHome only renders manually when the hash didn't change."""
        html = get_template()
        gs = html.index("function goHome()")
        gchunk = html[gs:gs + 800]
        assert "var changed = setRoute" in gchunk
        assert "if (!changed) renderHome();" in gchunk


class TestXssSafety:
    """Defense-in-depth regression tests for the XSS surfaces found in review."""

    def test_add_note_btn_uses_data_attributes(self):
        """addNoteBtn serializes IDs as data-* attrs and event-delegates the click,
        rather than string-interpolating IDs into an onclick handler."""
        html = get_template()
        start = html.index("function addNoteBtn(targetKind, convId, msgId)")
        chunk = html[start:start + 800]
        assert 'data-action="add-note"' in chunk
        assert "escAttr(convId" in chunk
        # old dangerous pattern is gone
        assert 'onclick="openNoteComposer' not in html

    def test_add_note_click_is_event_delegated(self):
        html = get_template()
        assert "function wireNoteDelegation()" in html
        assert 'data-action=\\"add-note\\"' in html

    def test_safe_media_url_rejects_unsafe_chars(self):
        """safeMediaUrl refuses URLs containing quotes/angle brackets/whitespace."""
        html = get_template()
        start = html.index("function safeMediaUrl(url)")
        chunk = html[start:start + 600]
        # The defensive reject clause
        assert '["\\\'<>`\\s]' in chunk

    def test_edit_note_single_select(self):
        """editNote fetches target_kind + conversation_id + message_id in one SELECT."""
        html = get_template()
        start = html.index("function editNote(noteId, btn)")
        chunk = html[start:start + 1200]
        assert "SELECT target_kind, conversation_id, message_id FROM notes WHERE id = ?" in chunk

    def test_message_div_id_uses_escattr(self):
        """data-msg-id must use the quote-safe escAttr, not esc (HTML-2)."""
        html = get_template()
        assert "escAttr(msg.id)" in html
        assert "esc(msg.id)" not in html

    def test_note_edit_delete_are_event_delegated(self):
        """Note edit/delete buttons must not interpolate the note id into an inline
        onclick handler; they use data-action + delegation like add-note (HTML-1)."""
        html = get_template()
        assert 'onclick="editNote(' not in html
        assert 'onclick="deleteNoteUI(' not in html
        assert 'data-action="edit-note"' in html
        assert 'data-action="delete-note"' in html

    def test_db_param_rejects_cross_origin(self):
        """?db= must only accept same-origin relative paths, not arbitrary URLs (HTML-4)."""
        html = get_template()
        assert 'dbParam.slice(0, 2) !== "//"' in html

    def test_external_links_have_noopener(self):
        """target=_blank links must carry rel=noopener noreferrer (HTML-5)."""
        html = get_template()
        assert 'rel="noopener noreferrer"' in html
        assert 'target="_blank">' not in html

    def test_schema_ddl_cannot_break_out_of_script(self):
        """schema_ddl is embedded inside an inline <script>. The HTML tokenizer
        terminates a <script> at the first literal </script> regardless of JS
        quoting, so a schema_ddl carrying </script> (a view/trigger/index name
        or body) would break out into live markup (EXP-1). The closing-tag
        sequence must be neutralized so no literal </script> originates from
        the schema_ddl.
        """
        # A malicious/unlucky schema_ddl that closes the script element and
        # injects markup. The raw closing-tag sequence must NOT survive.
        evil = "x </script><img src=x onerror=alert(1)> y"
        html = get_template(schema_ddl=evil)
        # The exact injected markup must not appear verbatim.
        assert "</script><img src=x onerror=alert(1)>" not in html
        # No literal </script> may originate from the schema_ddl. The document
        # has exactly one legitimate </script> for the inline chat module plus
        # one for the sql-wasm.js loader script; the schema_ddl must contribute
        # none. Locate the SCHEMA_DDL assignment and assert no </script> appears
        # between it and the end of that JS string literal.
        marker = "var SCHEMA_DDL = '"
        idx = html.index(marker)
        # The schema string literal ends at the next "';\n" terminator.
        end = html.index("';\n", idx)
        ddl_region = html[idx:end]
        assert "</script>" not in ddl_region
        # The neutralized form (escaped slash) is what should be present.
        assert "<\\/script>" in ddl_region

    def test_schema_ddl_normal_still_embeds(self):
        """Neutralizing the closing-tag sequence must not corrupt normal schema
        DDL: the chat feature must still receive the schema text (EXP-1).
        """
        ddl = "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT);"
        html = get_template(schema_ddl=ddl)
        # A normal DDL has no "</", so it embeds byte-for-byte.
        assert ddl in html
        # And it lands inside the SCHEMA_DDL JS string literal.
        marker = "var SCHEMA_DDL = '"
        idx = html.index(marker)
        end = html.index("';\n", idx)
        assert ddl in html[idx:end]

    def test_csp_meta_present(self):
        """A Content-Security-Policy meta hardens object/base/frame/form vectors
        (HTML-3). script-src is intentionally left unrestricted so the offline
        file:// bundle keeps loading its sibling sql-wasm.js."""
        html = get_template()
        assert 'http-equiv="Content-Security-Policy"' in html
        assert "object-src 'none'" in html
        assert "base-uri 'none'" in html
        assert "frame-ancestors 'none'" in html


class TestVendoredAssets:
    def test_vendored_sqljs_files_present(self):
        """The vendored sql.js files must ship in the package (no CDN fallback);
        a packaging regression that drops them would silently break the SPA (ARCH-8)."""
        from llm_memex.exporters import html as html_mod
        for fname in html_mod._SQL_JS_FILES:
            assert (html_mod._VENDORED_DIR / fname).exists(), fname


class TestSpaArkivExport:
    """The SPA can export its current state as arkiv .jsonl.gz for round-tripping
    annotations back to the primary DB via `llm-memex import`."""

    def test_template_has_download_arkiv_button(self):
        html = get_template()
        assert 'id="download-arkiv-btn"' in html
        assert 'onclick="downloadArkiv()"' in html
        # The button label nudges users toward the round-trip destination
        assert "llm-memex import" in html

    def test_build_arkiv_records_shape(self):
        """The SPA's record builder produces the same shape as the Python
        exporter (see llm_memex/exporters/arkiv_export._build_records)."""
        html = get_template()
        assert "function _buildArkivRecords()" in html
        # Same metadata keys as Python exporter
        assert "conversation_id: msg.conversation_id" in html
        assert "role: msg.role" in html
        assert "message_id" in html
        # notes attachment path
        assert "notesByMsg" in html
        assert "meta.notes = notes" in html

    def test_arkiv_records_are_text_centric(self):
        """Skip non-text messages — matches the Python exporter's get_text()."""
        html = get_template()
        assert "function _extractMsgText(" in html
        assert 'type === "text"' in html

    def test_gzip_via_compression_stream(self):
        """No JS zip library dependency; uses the native CompressionStream."""
        html = get_template()
        assert "function _gzipBytes(bytes)" in html
        assert 'new CompressionStream("gzip")' in html

    def test_downloaded_filename_format(self):
        """Filename is date-stamped so multiple round-trips don't collide."""
        html = get_template()
        start = html.index("async function downloadArkiv()")
        chunk = html[start:start + 1500]
        assert 'a.download = "llm-memex-" + today + ".jsonl.gz"' in chunk


class TestLibrarianChatMode:
    """Librarian chat: memex-aware system prompt + a top-bar entry point."""

    def test_top_bar_has_chat_button(self):
        html = get_template()
        assert 'id="chat-toggle"' in html
        assert 'onclick="openChat()"' in html

    def test_has_open_chat_function(self):
        html = get_template()
        assert "function openChat()" in html

    def test_hash_route_includes_chat(self):
        html = get_template()
        start = html.index("function buildRoute(r)")
        chunk = html[start:start + 500]
        assert 'r.mode === "chat"' in chunk
        assert '"#/chat"' in chunk
        pstart = html.index("function parseRoute()")
        pchunk = html[pstart:pstart + 700]
        assert '"chat"' in pchunk

    def test_librarian_system_prompt_is_memex_aware(self):
        html = get_template()
        # Persona
        assert "librarian of this person" in html
        # Schema injection
        assert "DATABASE SCHEMA:" in html
        # Best practices list
        assert "ALWAYS LIMIT" in html
        # Example patterns
        assert "EXAMPLE PATTERNS:" in html
        # sql.js FTS limitation surfaced explicitly
        assert "FTS5 virtual tables are NOT available" in html
        assert "LIKE" in html

    def test_conversation_system_prompt_exists(self):
        """resumeConversation uses a memex-aware default, appended with user override."""
        html = get_template()
        assert "CONVERSATION_SYSTEM_PROMPT" in html
        assert "continuing a conversation" in html
        # User prompt composition: default + "\n\n" + user-set, not replace
        start = html.index("async function resumeConversation")
        chunk = html[start:start + 3000]
        assert "CONVERSATION_SYSTEM_PROMPT + \"\\n\\n\" + userPrompt" in chunk

    def test_input_visible_in_librarian_mode(self):
        html = get_template()
        assert 'chatMode === "librarian"' in html
        # refreshChatUiState reveals input-area in librarian mode
        assert "ask about your archive" in html

    def test_send_message_dispatches_to_librarian(self):
        html = get_template()
        start = html.index("function sendMessage()")
        chunk = html[start:start + 1200]
        assert 'chatMode === "librarian"' in chunk
        assert "askLibrarian(text)" in chunk


class TestStreamEpoch:
    """Regression: rapid conv-A → conv-B switch must not inject conv-A's stream
    into conv-B. Each mode transition bumps streamEpoch; resumeConversation
    captures it and bails when it changes."""

    def test_stream_epoch_global_exists(self):
        html = get_template()
        assert "var streamEpoch = 0;" in html

    def test_open_conversation_bumps_epoch(self):
        html = get_template()
        start = html.index("function openConversation(convId)")
        chunk = html[start:start + 400]
        assert "streamEpoch++" in chunk

    def test_resume_conversation_captures_and_checks_epoch(self):
        html = get_template()
        start = html.index("async function resumeConversation(userText)")
        chunk = html[start:start + 6000]
        assert "var myEpoch = streamEpoch;" in chunk
        assert "var streamConvId = activeConvId;" in chunk
        assert "myEpoch !== streamEpoch" in chunk
        # Post-stream DB writes use captured convId, not live activeConvId
        assert "[streamConvId, assistMsgId, \"assistant\"" in chunk

    def test_resume_conversation_maintains_messages_fts(self):
        """When messages_fts exists (live DB), user/assistant messages are
        mirrored to it via tryExec. When stripped (exported SPA), tryExec
        silently no-ops."""
        html = get_template()
        assert 'tryExec(\n        "INSERT INTO messages_fts' in html or \
               'tryExec(\\n        "INSERT INTO messages_fts' in html or \
               "INSERT INTO messages_fts (conversation_id, message_id, text)" in html
        # Assert at least one tryExec into messages_fts
        start = html.index("async function resumeConversation")
        chunk = html[start:start + 5000]
        assert "messages_fts" in chunk
        assert "tryExec" in chunk


class TestHtmlTimelineRemoved:
    """Single-column refresh deleted the canvas timeline bar; these regressions
    guard that nothing creeps back in by accident."""

    def test_template_has_no_timeline_dom(self):
        html = get_template()
        assert 'id="timeline-canvas"' not in html
        assert 'id="timeline-start"' not in html
        assert 'id="timeline-end"' not in html

    def test_template_has_no_timeline_functions(self):
        html = get_template()
        assert "function initTimeline" not in html
        assert "function drawTimeline" not in html

    def test_template_has_no_timeline_state(self):
        html = get_template()
        assert "var timelineData" not in html
        assert "var timelineSelection" not in html


class TestHtmlAnthropicIntegration:
    """Tests for Anthropic API integration, settings, and DB download (Task 5)."""

    def test_template_has_anthropic_api_code(self):
        """Anthropic API wiring is present, but no hardcoded endpoint default.

        The endpoint defaults to empty — chat is disabled until the user
        configures it in Settings. This prevents accidental exposure of
        the export author's proxy to anyone who downloads the bundle.
        """
        from llm_memex.exporters.html_template import get_template
        html = get_template()
        assert 'anthropic-dangerous-direct-browser-access' in html
        assert 'function sendMessage(' in html
        assert 'function downloadDb(' in html
        assert 'localStorage' in html
        # No hardcoded proxy URL default — chat must be opt-in per browser.
        assert 'metafunctor-edge.queelius.workers.dev' not in html

    def test_template_has_settings_form_fields(self):
        """Settings panel should have API key and system prompt fields."""
        html = get_template()
        assert 'id="setting-api-key"' in html
        assert 'id="setting-endpoint"' in html
        assert 'id="setting-system-prompt"' in html
        assert 'type="password"' in html

    def test_template_has_load_save_settings(self):
        """Template should have loadSettings and saveSettings functions."""
        html = get_template()
        assert 'function loadSettings()' in html
        assert 'function saveSettings()' in html

    def test_settings_has_provider_and_model_fields(self):
        """Users can choose between Anthropic and OpenAI-compatible providers."""
        html = get_template()
        assert 'id="setting-provider"' in html
        assert 'id="setting-model"' in html
        # Both provider options surfaced
        assert 'value="anthropic"' in html
        assert 'value="openai"' in html

    def test_template_has_chat_request_branching(self):
        """buildChatRequest/extractDeltaText branch by provider string."""
        html = get_template()
        assert "function buildChatRequest(provider, apiKey, model, systemPrompt, messages)" in html
        assert "function extractDeltaText(provider, data)" in html
        # Anthropic wire format
        assert '"x-api-key"' in html
        assert '"anthropic-version"' in html
        # OpenAI-compat wire format
        assert '"Authorization"' in html
        assert '"Bearer "' in html
        # OpenAI delta accessor
        assert "evt.choices" in html
        # Anthropic delta accessor
        assert 'evt.type === "content_block_delta"' in html

    def test_default_model_per_provider(self):
        """defaultModelFor returns a reasonable default for each provider."""
        html = get_template()
        start = html.index("function defaultModelFor(provider)")
        chunk = html[start:start + 300]
        assert "claude-sonnet-4-6" in chunk
        assert "gpt-4o-mini" in chunk

    def test_save_settings_persists_provider_and_model(self):
        """Saved keys include provider and model."""
        html = get_template()
        start = html.index("function saveSettings()")
        chunk = html[start:start + 800]
        assert '"llm_memex_provider"' in chunk
        assert '"llm_memex_model"' in chunk

    def test_load_settings_reads_localstorage(self):
        """loadSettings should read memex_api_key, memex_endpoint, memex_system_prompt."""
        html = get_template()
        start = html.index("function loadSettings()")
        chunk = html[start:start + 500]
        assert 'memex_api_key' in chunk
        assert 'memex_endpoint' in chunk
        assert 'memex_system_prompt' in chunk
        assert 'localStorage.getItem' in chunk

    def test_save_settings_writes_localstorage(self):
        """saveSettings should write to localStorage."""
        html = get_template()
        start = html.index("function saveSettings()")
        chunk = html[start:start + 500]
        assert 'localStorage.setItem' in chunk

    def test_on_db_loaded_calls_load_settings(self):
        """onDbLoaded should call loadSettings()."""
        html = get_template()
        start = html.index("function onDbLoaded()")
        end = html.index("function renderFilters()")
        chunk = html[start:end]
        assert "loadSettings()" in chunk

    def test_send_message_builds_history(self):
        """resumeConversation should query message history from DB."""
        html = get_template()
        start = html.index("function resumeConversation(")
        chunk = html[start:start + 3000]
        assert 'FROM messages WHERE conversation_id' in chunk
        assert 'ORDER BY created_at ASC' in chunk

    def test_send_message_streams_response(self):
        """sendMessage should use streaming (getReader + TextDecoder)."""
        html = get_template()
        start = html.index("function sendMessage()")
        end = html.index("function downloadDb()")
        chunk = html[start:end]
        assert 'getReader()' in chunk
        assert 'TextDecoder' in chunk
        assert 'content_block_delta' in chunk
        assert 'text_delta' in chunk

    def test_send_message_inserts_messages(self):
        """sendMessage should INSERT both user and assistant messages."""
        html = get_template()
        start = html.index("function sendMessage()")
        end = html.index("function downloadDb()")
        chunk = html[start:end]
        assert 'INSERT INTO messages' in chunk
        assert 'crypto.randomUUID()' in chunk

    def test_send_message_updates_conversation(self):
        """sendMessage should UPDATE conversation message_count and updated_at."""
        html = get_template()
        start = html.index("function sendMessage()")
        end = html.index("function downloadDb()")
        chunk = html[start:end]
        assert 'UPDATE conversations SET message_count' in chunk

    def test_send_message_disables_button_during_stream(self):
        """Send button should be disabled during streaming."""
        html = get_template()
        start = html.index("function sendMessage()")
        end = html.index("function downloadDb()")
        chunk = html[start:end]
        assert 'sendBtn.disabled = true' in chunk
        assert 'sendBtn.disabled = false' in chunk

    def test_send_message_uses_correct_headers(self):
        """sendMessage should send correct Anthropic API headers."""
        html = get_template()
        start = html.index("function sendMessage()")
        end = html.index("function downloadDb()")
        chunk = html[start:end]
        assert 'x-api-key' in chunk
        assert 'anthropic-version' in chunk
        assert '2023-06-01' in chunk
        assert 'anthropic-dangerous-direct-browser-access' in chunk

    def test_send_message_default_model(self):
        """Default model is claude-sonnet-4-6 (read near the agentic loop)."""
        html = get_template()
        start = html.index("runAgenticLoop")
        chunk = html[start:start + 3000]
        assert 'claude-sonnet-4-6' in chunk

    def test_download_db_exists(self):
        """downloadDb should export the database as a Blob download."""
        html = get_template()
        assert 'function downloadDb()' in html
        start = html.index("function downloadDb()")
        chunk = html[start:start + 500]
        assert 'db.export()' in chunk
        assert 'Blob' in chunk
        assert 'conversations.db' in chunk

    def test_settings_css_exists(self):
        """Template should have CSS for settings form fields."""
        html = get_template()
        assert '.settings-field' in html
        assert '.settings-actions' in html


# ---------------------------------------------------------------------------
# HTML exporter tests (Task 6)
# ---------------------------------------------------------------------------
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from llm_memex.models import Conversation, Message


def _make_conv(conv_id="c1", title="Test Conversation"):
    """Create a minimal Conversation for testing."""
    conv = Conversation(
        id=conv_id,
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 2),
        title=title,
    )
    msg = Message(id="m1", role="user", content=[{"type": "text", "text": "hello"}])
    conv.add_message(msg)
    return conv


class TestHtmlExporter:
    def test_export_creates_directory(self, tmp_path):
        from llm_memex.exporters.html import export

        out_dir = tmp_path / "site"
        export([_make_conv()], str(out_dir))
        assert out_dir.is_dir()
        assert (out_dir / "index.html").exists()

    def test_export_index_is_valid_html(self, tmp_path):
        from llm_memex.exporters.html import export

        out_dir = tmp_path / "site"
        export([_make_conv()], str(out_dir))
        html = (out_dir / "index.html").read_text()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "sql-wasm.js" in html

    def test_export_creates_existing_directory(self, tmp_path):
        """Export into an already-existing directory should not fail."""
        from llm_memex.exporters.html import export

        out_dir = tmp_path / "site"
        out_dir.mkdir()
        export([_make_conv()], str(out_dir))
        assert (out_dir / "index.html").exists()

    def test_export_no_db_path(self, tmp_path):
        """Without db_path, only index.html is written."""
        from llm_memex.exporters.html import export

        out_dir = tmp_path / "site"
        export([_make_conv()], str(out_dir))
        assert (out_dir / "index.html").exists()
        assert not (out_dir / "conversations.db").exists()
        assert not (out_dir / "assets").exists()

    def test_export_copies_db(self, tmp_path):
        """Default export gzips the DB for transfer; raw .db is removed."""
        import gzip as _gz
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            conv = _make_conv()
            db.save_conversation(conv)

        out_dir = tmp_path / "site"
        db_path = str(db_dir / "conversations.db")
        export([conv], str(out_dir), db_path=db_path)

        # The compressed form is shipped; raw .db is gone
        assert (out_dir / "conversations.db.gz").exists()
        assert not (out_dir / "conversations.db").exists()
        # Decompress and confirm it's a valid SQLite copy
        raw = (out_dir / "conversations.db")
        with _gz.open(out_dir / "conversations.db.gz", "rb") as fin, \
                open(raw, "wb") as fout:
            fout.write(fin.read())
        try:
            with Database(str(out_dir), readonly=True) as db2:
                result = db2.query_conversations()
                assert len(result["items"]) == 1
        finally:
            raw.unlink()

    def test_export_compress_db_false_keeps_raw(self, tmp_path):
        """compress_db=False opt-out emits the plain .db."""
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_make_conv())

        out_dir = tmp_path / "site"
        db_path = str(db_dir / "conversations.db")
        export([_make_conv()], str(out_dir), db_path=db_path, compress_db=False)
        assert (out_dir / "conversations.db").exists()
        assert not (out_dir / "conversations.db.gz").exists()

    def test_export_strips_original_content_enrichments(self, tmp_path):
        """Reversible-redaction plaintext must not ship in the published bundle (REDACT-1b)."""
        import json as _json
        import sqlite3 as _sqlite3
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_make_conv())
            db.save_enrichment(
                "c1", "original_content",
                _json.dumps({"message_id": "m1",
                             "content": [{"type": "text", "text": "sk-SUPER-SECRET-KEY"}]}),
                "redact",
            )

        out_dir = tmp_path / "site"
        db_path = str(db_dir / "conversations.db")
        export([_make_conv()], str(out_dir), db_path=db_path, compress_db=False)

        conn = _sqlite3.connect(str(out_dir / "conversations.db"))
        n = conn.execute(
            "SELECT COUNT(*) FROM enrichments WHERE type='original_content'"
        ).fetchone()[0]
        leaked = conn.execute(
            "SELECT COUNT(*) FROM enrichments WHERE value LIKE '%SUPER-SECRET%'"
        ).fetchone()[0]
        conn.close()
        assert n == 0 and leaked == 0, "redaction-undo plaintext leaked into the export"

    def test_export_copies_assets(self, tmp_path):
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_make_conv())

        # Create an assets directory with a test file
        assets_dir = db_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "test_image.png").write_bytes(b"\x89PNG fake image data")

        out_dir = tmp_path / "site"
        db_path = str(db_dir / "conversations.db")
        export([_make_conv()], str(out_dir), db_path=db_path)

        assert (out_dir / "assets").is_dir()
        assert (out_dir / "assets" / "test_image.png").exists()
        assert (out_dir / "assets" / "test_image.png").read_bytes() == b"\x89PNG fake image data"

    def test_export_no_assets_dir(self, tmp_path):
        """If there is no assets/ directory, export should still succeed."""
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_make_conv())

        out_dir = tmp_path / "site"
        db_path = str(db_dir / "conversations.db")
        export([_make_conv()], str(out_dir), db_path=db_path)

        assert (out_dir / "index.html").exists()
        assert (out_dir / "conversations.db.gz").exists()
        assert not (out_dir / "assets").exists()

    def test_export_memory_db_skips_copy(self, tmp_path):
        """db_path=':memory:' should not attempt to copy."""
        from llm_memex.exporters.html import export

        out_dir = tmp_path / "site"
        export([_make_conv()], str(out_dir), db_path=":memory:")
        assert (out_dir / "index.html").exists()
        assert not (out_dir / "conversations.db").exists()

    def test_export_replaces_existing_assets(self, tmp_path):
        """If dest assets/ already exists, it should be replaced."""
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_make_conv())

        # Source assets
        assets_dir = db_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "new.png").write_bytes(b"new")

        out_dir = tmp_path / "site"
        out_dir.mkdir()
        # Pre-existing dest assets with a stale file
        dest_assets = out_dir / "assets"
        dest_assets.mkdir()
        (dest_assets / "old.png").write_bytes(b"old")

        db_path = str(db_dir / "conversations.db")
        export([_make_conv()], str(out_dir), db_path=db_path)

        assert (out_dir / "assets" / "new.png").exists()
        assert not (out_dir / "assets" / "old.png").exists()

    def test_export_vendors_sql_js(self, tmp_path):
        """sql-wasm.js and sql-wasm.wasm are shipped alongside index.html."""
        from llm_memex.exporters.html import export

        out_dir = tmp_path / "site"
        export([_make_conv()], str(out_dir))
        assert (out_dir / "sql-wasm.js").exists()
        assert (out_dir / "sql-wasm.wasm").exists()
        # Sanity-check: the files have real content
        assert (out_dir / "sql-wasm.js").stat().st_size > 1000
        assert (out_dir / "sql-wasm.wasm").stat().st_size > 100_000

    def test_export_strips_fts5_tables(self, tmp_path):
        """FTS5 virtual tables are dropped before DB is copied to export.

        sql.js cannot use FTS5 (not compiled in), and the shadow tables
        typically account for ~50% of DB size. Stripping them halves
        the exported bundle size without functional loss.
        """
        import sqlite3
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_make_conv())

        db_path = str(db_dir / "conversations.db")
        # Confirm source DB has FTS5
        with sqlite3.connect(db_path) as conn:
            src_tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert "messages_fts" in src_tables
        assert "notes_fts" in src_tables

        out_dir = tmp_path / "site"
        # compress_db=False to inspect the raw .db in the test
        export([_make_conv()], str(out_dir), db_path=db_path, compress_db=False)

        # Confirm exported DB has no FTS5 tables
        with sqlite3.connect(str(out_dir / "conversations.db")) as conn:
            dst_tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert "messages_fts" not in dst_tables
        assert "notes_fts" not in dst_tables
        # Core tables still present
        assert "conversations" in dst_tables
        assert "messages" in dst_tables
        assert "notes" in dst_tables

    def test_export_db_uses_delete_journal_mode(self, tmp_path):
        """Export sets journal_mode=DELETE so no .db-wal / .db-shm sidecars
        travel with the bundle."""
        import sqlite3
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_make_conv())

        out_dir = tmp_path / "site"
        db_path = str(db_dir / "conversations.db")
        export([_make_conv()], str(out_dir), db_path=db_path, compress_db=False)

        dst = out_dir / "conversations.db"
        with sqlite3.connect(str(dst)) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "delete"
        assert not (out_dir / "conversations.db-wal").exists()
        assert not (out_dir / "conversations.db-shm").exists()

    def test_export_db_gz_is_valid_sqlite(self, tmp_path):
        """conversations.db.gz decompresses back to a valid SQLite file."""
        import gzip as _gz
        import sqlite3
        from llm_memex.db import Database
        from llm_memex.exporters.html import export

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        with Database(str(db_dir)) as db:
            db.save_conversation(_make_conv())

        out_dir = tmp_path / "site"
        db_path = str(db_dir / "conversations.db")
        export([_make_conv()], str(out_dir), db_path=db_path)

        # Decompress and open
        raw = tmp_path / "decompressed.db"
        with _gz.open(out_dir / "conversations.db.gz", "rb") as fin, \
                open(raw, "wb") as fout:
            fout.write(fin.read())
        with sqlite3.connect(str(raw)) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
        assert count == 1


class TestCLIExportHtml:
    def test_cli_export_html(self, tmp_path):
        """Full CLI round-trip: import OpenAI data, export as html."""
        db_dir = tmp_path / "db"
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps([{
            "id": "c1", "title": "HTML Export Test",
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
            [sys.executable, "-m", "llm_memex", "import", str(export_file),
             "--db", str(db_dir)],
            capture_output=True, text=True,
        )

        # Export as html
        out_dir = tmp_path / "site"
        result = subprocess.run(
            [sys.executable, "-m", "llm_memex", "export", str(out_dir),
             "--format", "html", "--db", str(db_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Exported 1 conversation" in result.stdout

        # Verify directory structure (DB is gzipped by default)
        assert (out_dir / "index.html").exists()
        assert (out_dir / "conversations.db.gz").exists()

        # Verify index.html content
        html = (out_dir / "index.html").read_text()
        assert "<!DOCTYPE html>" in html

        # Decompress to inspect the DB
        import gzip as _gz
        with _gz.open(out_dir / "conversations.db.gz", "rb") as fin, \
                open(out_dir / "conversations.db", "wb") as fout:
            fout.write(fin.read())

        # Verify the copied DB is usable
        from llm_memex.db import Database
        with Database(str(out_dir), readonly=True) as db2:
            result = db2.query_conversations()
            assert len(result["items"]) == 1
            assert result["items"][0]["title"] == "HTML Export Test"


class TestHtmlNotes:
    """Tests for notes (marginalia) support in the HTML SPA template."""

    def test_template_has_notes_css(self):
        html = get_template()
        assert ".note {" in html
        assert ".note-composer" in html
        assert ".add-note-btn" in html

    def test_template_has_marginalia_browse_button(self):
        html = get_template()
        assert 'id="marginalia-toggle"' in html
        assert 'openMarginalia()' in html

    def test_template_has_marginalia_function(self):
        html = get_template()
        assert "function openMarginalia()" in html
        # Queries notes joined with conversations for context
        assert "FROM notes n LEFT JOIN conversations c" in html
        # Filter input for searching notes
        assert 'marginalia-filter' in html
        # Renders cards with conversation title links
        assert "marginalia-card" in html

    def test_template_has_marginalia_css(self):
        html = get_template()
        assert ".marginalia-list" in html
        assert ".marginalia-card" in html
        assert ".marginalia-card-title" in html

    def test_input_area_only_visible_for_conversation_mode(self):
        """The note/continue input is hidden in home/marginalia/search modes."""
        html = get_template()
        # refreshChatUiState only reveals the input when reading a conversation
        assert 'chatMode === "conversation"' in html
        # Otherwise input-area is hidden
        assert 'inputArea.classList.add("hidden")' in html

    def test_template_has_search_results_mode(self):
        """Search is driven by the top search box, invoked with a query term."""
        html = get_template()
        assert 'id="search-box"' in html
        # openSearchResults now takes a term from the top search box
        assert "function openSearchResults(initialTerm)" in html
        # Query ranks conversations by match count
        assert "COUNT(*) AS match_count" in html
        assert "ORDER BY match_count DESC" in html

    def test_template_has_snippet_helpers(self):
        """Snippet extraction and safe highlight appending."""
        html = get_template()
        assert "function extractSnippet(" in html
        assert "function appendHighlightedText(" in html
        # The highlight helper uses DOM nodes (no innerHTML) for XSS safety
        assert "document.createElement(\"mark\")" in html
        assert "document.createTextNode(" in html

    def test_template_has_search_results_css(self):
        html = get_template()
        assert ".search-results-list" in html
        assert ".search-result-card" in html
        assert ".search-result-snippet mark" in html

    def test_template_has_welcome_hero_stats(self):
        """Welcome screen has stat tiles + source breakdown populated on DB load."""
        html = get_template()
        assert 'id="welcome-stats"' in html
        assert 'id="welcome-sources"' in html
        assert "function renderWelcomeStats()" in html
        assert ".welcome-stat-value" in html
        assert ".welcome-stat-label" in html
        # Queries for each stat
        assert "FROM conversations" in html
        assert "MIN(created_at)" in html

    def test_template_has_source_color_palette(self):
        """Each source has a distinct color dot; theme defines the palette."""
        html = get_template()
        assert "--src-openai" in html
        assert "--src-anthropic" in html
        assert "--src-claude-code" in html
        assert "--src-gemini" in html
        assert ".source-dot" in html

    def test_template_uses_warm_palette(self):
        """Warm off-white + bronze, not GitHub blue."""
        html = get_template()
        # GitHub-flavored colors should be gone
        assert "#0d1117" not in html
        assert "#58a6ff" not in html
        assert "#0969da" not in html
        # Warm tokens present (cream bg + bronze accent)
        assert "#faf7f0" in html or "#1a1714" in html  # warm bg tokens

    def test_message_role_labels_are_subtle(self):
        """Role labels use small-caps convention: small font + letter-spacing.

        The v1 design shouted with large ALL-CAPS labels. The refreshed
        design uses the modern small-caps convention: tiny font (~10-11px)
        with tracking so the uppercase reads as a label, not a headline.
        """
        html = get_template()
        start = html.index(".message-role {")
        block = html[start:start + 250]
        # Label is small (not headline-sized)
        assert "font-size: 10" in block or "font-size: 11" in block
        # Has letter-spacing so uppercase reads as a small-caps label
        assert "letter-spacing:" in block

    def test_template_has_notes_js(self):
        html = get_template()
        assert "function loadNotesForConversation" in html
        assert "function saveNote" in html
        assert "function deleteNoteUI" in html

    def test_template_librarian_mentions_notes(self):
        """Librarian system prompt tells the model how to query marginalia."""
        html = get_template("CREATE TABLE notes (id TEXT);")
        assert "notes table" in html
        assert "marginalia" in html

    def test_template_has_notes_render_functions(self):
        html = get_template()
        assert "function renderNotesForMessage" in html
        assert "function renderConversationNotes" in html
        assert "function addNoteBtn" in html

    def test_template_has_notes_edit_functions(self):
        html = get_template()
        assert "function editNote" in html
        assert "function openNoteComposer" in html

    def test_render_conversation_loads_notes(self):
        """renderConversation should call loadNotesForConversation."""
        html = get_template()
        start = html.index("function renderConversation(")
        end = html.index("function renderContent(")
        chunk = html[start:end]
        assert "loadNotesForConversation(conv.id)" in chunk

    def test_render_conversation_renders_conversation_notes(self):
        """renderConversation should render conversation-level notes."""
        html = get_template()
        start = html.index("function renderConversation(")
        end = html.index("function renderContent(")
        chunk = html[start:end]
        assert "renderConversationNotes(" in chunk

    def test_render_conversation_renders_message_notes(self):
        """renderConversation should render message-level notes."""
        html = get_template()
        start = html.index("function renderConversation(")
        end = html.index("function renderContent(")
        chunk = html[start:end]
        assert "renderNotesForMessage(" in chunk

    def test_render_conversation_has_note_buttons(self):
        """renderConversation should add note buttons for header and messages."""
        html = get_template()
        start = html.index("function renderConversation(")
        end = html.index("function renderContent(")
        chunk = html[start:end]
        assert 'addNoteBtn("conversation"' in chunk
        assert 'addNoteBtn("message"' in chunk

    def test_load_notes_graceful_degradation(self):
        """loadNotesForConversation should catch errors for missing table."""
        html = get_template()
        start = html.index("function loadNotesForConversation")
        chunk = html[start:start + 500]
        assert "try" in chunk
        assert "catch" in chunk

    def test_messages_have_data_msg_id(self):
        """Message divs should get data-msg-id attributes for note targeting."""
        html = get_template()
        start = html.index("function renderConversation(")
        end = html.index("function renderContent(")
        chunk = html[start:end]
        assert "data-msg-id" in chunk

    def test_notes_cache_variable(self):
        """Template should have the notesCache state variable."""
        html = get_template()
        assert "var notesCache = {}" in html
