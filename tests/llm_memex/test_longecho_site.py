"""Tests for the file:// viewer options and the longecho site/ writers."""
import base64
import gzip
import json
import os
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


def test_too_large_archive_notice_points_at_text_and_data():
    page = _site_template(db_chunks=0)
    assert "var DB_CHUNKS = 0" in page
    assert "text/" in page and "data/" in page


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
    page = get_template("SCHEMA", media_base='x"</script><b>/')
    assert "</script><b>" not in page.split("<script>\n", 1)[1].split("</script>")[0]


# -- site writers -------------------------------------------------------------


def test_chunks_reassemble(tmp_path):
    payload = gzip.compress(os.urandom(40 * 1024))
    db_gz = tmp_path / "conversations.db.gz"
    db_gz.write_bytes(payload)
    site = tmp_path / "site"

    count = write_db_chunks(db_gz, site, chunk_bytes=16384)

    assert count == 3
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
    assert write_db_chunks(db_gz, tmp_path / "site", chunk_bytes=1000) == 3
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
    write_wasm_script(tmp_path)
    text = (tmp_path / "sql-wasm-binary.js").read_text(encoding="ascii")
    decoded = _js_string_arg(text, "window.LLM_MEMEX_WASM_B64 = ", ";\n")
    assert decoded == (VENDORED / "sql-wasm.wasm").read_bytes()


def test_copy_sql_js(tmp_path):
    copy_sql_js(tmp_path / "site")
    assert (tmp_path / "site" / "sql-wasm.js").read_bytes() == \
        (VENDORED / "sql-wasm.js").read_bytes()
