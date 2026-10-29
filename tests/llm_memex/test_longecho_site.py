"""Tests for the file:// viewer options and the longecho site/ writers."""
import base64
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from llm_memex.exporters._longecho_site import (
    copy_sql_js,
    write_db_chunks,
    write_wasm_script,
)
from llm_memex.exporters.html_template import get_template

DATA = Path(__file__).parent / "data"
VENDORED = Path(__file__).parents[2] / "llm_memex" / "exporters" / "vendored"

STRICT_CSP = "img-src 'self' file: data: blob:; media-src 'self' file: data: blob:"


def _js_string_arg(text: str, prefix: str, suffix: str) -> bytes:
    """Strip a known call wrapper and decode its JSON string literal as base64."""
    assert text.startswith(prefix) and text.endswith(suffix), text[:80]
    literal = text[len(prefix):len(text) - len(suffix)]
    return base64.b64decode(json.loads(literal), validate=True)


def _site_template(**kw):
    opts = dict(media_base="../data/assets/", strict_network=True, db_chunks=3)
    opts.update(kw)
    return get_template("SCHEMA", **opts)


# -- template -----------------------------------------------------------------


def test_default_template_unchanged():
    golden = (DATA / "template_default.html").read_text(encoding="utf-8")
    assert get_template("SCHEMA") == golden


def test_explicit_defaults_match_golden():
    golden = (DATA / "template_default.html").read_text(encoding="utf-8")
    assert get_template("SCHEMA", media_base="assets/", strict_network=False,
                        db_chunks=None) == golden


def test_strict_template_has_csp_and_flags():
    page = _site_template()
    assert STRICT_CSP in page
    assert "var STRICT_NETWORK = true" in page
    assert 'var MEDIA_BASE = "../data/assets/"' in page
    assert "var DB_CHUNKS = 3" in page


def test_strict_csp_keeps_existing_directives_and_adds_no_others():
    page = _site_template()
    start = page.index('content="', page.index("Content-Security-Policy")) + len('content="')
    csp = page[start:page.index('"', start)]
    directives = [d.strip().split()[0] for d in csp.split(";")]
    assert directives == ["object-src", "base-uri", "frame-ancestors", "form-action",
                          "img-src", "media-src"]


def test_non_strict_site_template_keeps_default_csp():
    golden = (DATA / "template_default.html").read_text(encoding="utf-8")
    page = _site_template(strict_network=False)
    assert "img-src" not in page
    assert "var STRICT_NETWORK = false" in page
    csp_line = next(l for l in golden.splitlines() if "Content-Security-Policy" in l)
    assert csp_line in page


def test_chunked_page_loads_wasm_binary_before_inline_script():
    page = _site_template()
    wasm_js = page.index('<script src="sql-wasm.js"></script>')
    binary = page.index('<script src="sql-wasm-binary.js"></script>')
    inline = page.index("<script>\n")
    assert wasm_js < binary < inline


def test_media_options_without_chunks_use_fetch_cascade():
    page = _site_template(db_chunks=None)
    assert "var DB_CHUNKS = null" in page
    assert '<script src="sql-wasm-binary.js">' not in page


@pytest.mark.parametrize("bad", [-1, True, 1.5, "3"])
def test_rejects_invalid_db_chunks(bad):
    with pytest.raises((TypeError, ValueError)):
        get_template("SCHEMA", db_chunks=bad)


@pytest.mark.parametrize("kw", [{"media_base": "../data/assets"}, {"media_base": ""},
                                {"strict_network": 1}])
def test_rejects_invalid_media_options(kw):
    with pytest.raises((TypeError, ValueError)):
        get_template("SCHEMA", **kw)


def test_media_base_is_embedded_as_safe_js_string():
    hostile = get_template("SCHEMA", media_base='x"</script><b>/')
    plain = get_template("SCHEMA", media_base="x/")
    assert 'var MEDIA_BASE = "x\\"\\u003c/script>\\u003cb>/";' in hostile
    assert hostile.count("</script>") == plain.count("</script>")


# -- site writers -------------------------------------------------------------


def test_chunks_reassemble(tmp_path):
    payload = gzip.compress(os.urandom(40 * 1024))
    db_gz = tmp_path / "conversations.db.gz"
    db_gz.write_bytes(payload)
    site = tmp_path / "site"

    written = write_db_chunks(db_gz, site, chunk_bytes=16384)

    assert written == [site / "db" / f"{i:03d}.js" for i in range(3)]
    names = sorted(p.name for p in (site / "db").iterdir())
    assert names == ["000.js", "001.js", "002.js"]
    joined = b""
    for i, name in enumerate(names):
        text = (site / "db" / name).read_text(encoding="ascii")
        joined += _js_string_arg(text, f"llmMemexDbChunk({i},", ");\n")
    assert joined == payload


def test_chunk_sizes_slice_gzip_bytes(tmp_path):
    db_gz = tmp_path / "x.gz"
    db_gz.write_bytes(bytes(range(256)) * 10)  # 2560 bytes
    assert len(write_db_chunks(db_gz, tmp_path / "site", chunk_bytes=1000)) == 3
    sizes = [
        len(_js_string_arg((tmp_path / "site" / "db" / f"{i:03d}.js").read_text(),
                           f"llmMemexDbChunk({i},", ");\n"))
        for i in range(3)
    ]
    assert sizes == [1000, 1000, 560]


@pytest.mark.parametrize("bad", [0, -5])
def test_chunk_bytes_must_be_positive(tmp_path, bad):
    db_gz = tmp_path / "x.gz"
    db_gz.write_bytes(b"abc")
    with pytest.raises(ValueError):
        write_db_chunks(db_gz, tmp_path / "site", chunk_bytes=bad)


def test_empty_gzip_file_is_rejected(tmp_path):
    # 0 chunks means "too large" in the viewer, so an empty input must not map to it.
    db_gz = tmp_path / "x.gz"
    db_gz.write_bytes(b"")
    with pytest.raises(ValueError):
        write_db_chunks(db_gz, tmp_path / "site", chunk_bytes=10)


def test_wasm_script_decodes_to_vendored_wasm(tmp_path):
    assert write_wasm_script(tmp_path) == [tmp_path / "sql-wasm-binary.js"]
    text = (tmp_path / "sql-wasm-binary.js").read_text(encoding="ascii")
    decoded = _js_string_arg(text, "window.LLM_MEMEX_WASM_B64 = ", ";\n")
    assert decoded == (VENDORED / "sql-wasm.wasm").read_bytes()


def test_copy_sql_js(tmp_path):
    assert copy_sql_js(tmp_path / "site") == [tmp_path / "site" / "sql-wasm.js"]
    assert (tmp_path / "site" / "sql-wasm.js").read_bytes() == \
        (VENDORED / "sql-wasm.js").read_bytes()


# -- viewer JS, run in Node (vm + DOM stub in data/viewer_harness.js) ---------

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

DEFAULT_KW = {}
STRICT_KW = dict(media_base="../data/assets/", strict_network=True, db_chunks=3)


def _run_viewer(tmp_path: Path, program: str, *, site_dir: Path | None = None, **template_kw):
    """Run ``program`` (an async JS function body; ``ctx`` is the page's global
    scope) against the template's inline script and return its JSON result."""
    page = get_template("SCHEMA", **template_kw)
    inline = page.split("<script>\n", 1)[1].rsplit("</script>", 1)[0]
    script = tmp_path / "inline.js"
    script.write_text(inline, encoding="utf-8")
    args = [NODE, str(DATA / "viewer_harness.js"), str(script)]
    if site_dir is not None:
        args.append(str(site_dir))
    proc = subprocess.run(args, input=program, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


class _Elements(HTMLParser):
    """Collect (tag, attrs) for every start tag."""

    def __init__(self):
        super().__init__()
        self.found = []

    def handle_starttag(self, tag, attrs):
        self.found.append((tag, dict(attrs)))


def _elements(html: str):
    parser = _Elements()
    parser.feed(html)
    parser.close()
    return parser.found


ALLOWED_ATTRS = {"src", "alt", "loading", "href", "target", "rel", "class"}

XSS_PAYLOADS = [
    # Reviewer's repro: the link pass re-matched text inside the emitted <img src>.
    "![a](http://x/[b](https://e onerror=location=this.title "
    "title=javascript:alert%281%29 z) q)",
    "[a](https://x/[b](https://e onerror=alert(1) z))",
    # Inline code is a placeholder inside the URL / alt when tags are built.
    '[a](https://x/`" onerror=alert(1) x`)',
    '![`" onerror=alert(1) x`](https://x/y.png)',
    # A forged placeholder in the input.
    "\x00TG0\x00 [l](https://ok.example/) \x00IC0\x00",
    "![q](https://x/y.png\" onerror=alert(1) x=\")",
]


@needs_node
@pytest.mark.parametrize("kw", [DEFAULT_KW, STRICT_KW], ids=["default", "strict"])
def test_js_markdown_payloads_cannot_inject_attributes(tmp_path, kw):
    rendered = _run_viewer(
        tmp_path, f"return {json.dumps(XSS_PAYLOADS)}.map(ctx.renderMarkdown);", **kw)
    for payload, html in zip(XSS_PAYLOADS, rendered):
        for tag, attrs in _elements(html):
            assert set(attrs) <= ALLOWED_ATTRS, (payload, html)
            assert all("<" not in (v or "") and '"' not in (v or "") for v in attrs.values()), \
                (payload, html)
            assert tag in {"p", "img", "a", "code", "br", "em", "strong"}, (payload, html)


@needs_node
@pytest.mark.parametrize("kw", [DEFAULT_KW, STRICT_KW], ids=["default", "strict"])
def test_js_markdown_still_renders_plain_links_and_images(tmp_path, kw):
    html = _run_viewer(
        tmp_path,
        'return ctx.renderMarkdown("[ok **bold**](https://e.com/a?x=1) ![pic](assets/p.png)");',
        **kw)
    els = _elements(html)
    links = [a for t, a in els if t == "a"]
    imgs = [a for t, a in els if t == "img"]
    assert links and links[0]["href"] == "https://e.com/a?x=1"
    assert links[0]["rel"] == "noopener noreferrer"
    assert "<strong>bold</strong>" in html
    expected_src = "../data/assets/p.png" if kw else "assets/p.png"
    assert imgs and imgs[0]["src"] == expected_src


@needs_node
def test_js_local_media_url_rejects_traversal(tmp_path):
    result = _run_viewer(tmp_path, r"""
        return {
          based: ctx.safeMediaUrl("../data/assets/../../etc/passwd"),
          backslash: ctx.safeMediaUrl("assets/..\\x"),
          encoded: ctx.safeMediaUrl("assets/%2E%2E/x"),
          absolute: ctx.safeMediaUrl("assets//etc/x"),
          rest: ctx.localMediaUrl("..\\x"),
          ok: ctx.safeMediaUrl("assets/a/b.png"),
          okBased: ctx.safeMediaUrl("../data/assets/a.png"),
        };""", **STRICT_KW)
    assert result == {"based": "", "backslash": "", "encoded": "", "absolute": "",
                      "rest": "", "ok": "../data/assets/a/b.png",
                      "okBased": "../data/assets/a.png"}


@needs_node
def test_js_strict_remote_image_becomes_link(tmp_path):
    result = _run_viewer(tmp_path, """
        return [
          ctx.renderMedia({media_type: "image/png", url: "https://e.com/i.png", filename: "i.png"}),
          ctx.renderMarkdown("![cat](https://e.com/c.png)"),
        ];""", **STRICT_KW)
    for html in result:
        els = _elements(html)
        assert not [t for t, _ in els if t in ("img", "video", "audio")], html
        link = next(a for t, a in els if t == "a")
        assert link["href"].startswith("https://e.com/")
        assert link["rel"] == "noopener noreferrer"


def _build_site(tmp_path: Path, chunk_bytes: int = 30000):
    """A site dir holding a real gzipped 2000-row DB split into several chunks."""
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("create table t(x)")
    con.executemany("insert into t values (?)", [(os.urandom(50).hex(),) for _ in range(2000)])
    con.commit()
    con.close()
    db_gz = tmp_path / "t.db.gz"
    db_gz.write_bytes(gzip.compress(db.read_bytes()))
    site = tmp_path / "site"
    count = len(write_db_chunks(db_gz, site, chunk_bytes=chunk_bytes))
    assert count >= 3
    write_wasm_script(site)
    copy_sql_js(site)
    return site, count


OPEN_DB = """
    let rows = null;
    ctx.onDbLoaded = () => { rows = ctx.query("select count(*) as n from t")[0].n; };
    await ctx.initApp();
    return {
      rows,
      notice: ctx.document.getElementById("drop-zone-notice").textContent,
      loading: ctx.document.getElementById("loading").textContent,
      wasmFreed: ctx.LLM_MEMEX_WASM_B64 === null,
    };"""


@needs_node
def test_js_chunks_open_db(tmp_path):
    site, count = _build_site(tmp_path)
    result = _run_viewer(tmp_path, OPEN_DB, site_dir=site, db_chunks=count)
    assert result["rows"] == 2000 and result["wasmFreed"] is True
    assert result["notice"] == ""


@needs_node
def test_js_missing_chunk_is_named(tmp_path):
    site, count = _build_site(tmp_path)
    result = _run_viewer(tmp_path, OPEN_DB, site_dir=site, db_chunks=count + 1)
    assert result["rows"] is None
    assert f"db/{count:03d}.js" in result["notice"]
    assert f"db/{count:03d}.js" not in result["loading"]


@needs_node
def test_js_too_large_archive_shows_notice(tmp_path):
    site, _ = _build_site(tmp_path)
    result = _run_viewer(tmp_path, OPEN_DB, site_dir=site, db_chunks=0)
    assert result["rows"] is None
    assert "too large" in result["notice"]
    assert "text/" in result["notice"] and "data/" in result["notice"]
    assert "conversations.db" in result["notice"] and "SQLite" in result["notice"]
    assert "arkiv JSONL" in result["notice"]
    # shown once: in the picker, not repeated in the loading line behind it
    assert "too large" not in result["loading"]
