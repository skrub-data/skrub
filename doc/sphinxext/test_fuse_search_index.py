"""Unit tests for the fuse_search_index Sphinx extension.

Run with::

    pytest doc/sphinxext/test_fuse_search_index.py -v

No Sphinx installation is required for most tests; ``test_build_finished``
needs only the stdlib + the extension itself.
"""

import json
import textwrap
from types import SimpleNamespace

# The module under test lives next to this file.
import fuse_search_index as fsi

# ---------------------------------------------------------------------------
# _plain_text
# ---------------------------------------------------------------------------


class TestPlainText:
    def test_basic(self):
        assert fsi._plain_text("<p>Hello world</p>") == "Hello world"

    def test_skips_script(self):
        assert fsi._plain_text("<script>alert(1)</script>hello") == "hello"

    def test_skips_style(self):
        assert fsi._plain_text("<style>.a{color:red}</style>text") == "text"

    def test_skips_nav(self):
        assert fsi._plain_text("<nav>nav content</nav>body") == "body"

    def test_collapses_whitespace(self):
        assert fsi._plain_text("<p>a   b\n\tc</p>") == "a b c"

    def test_max_chars(self):
        result = fsi._plain_text("<p>" + "x" * 100 + "</p>", max_chars=10)
        assert len(result) == 10

    def test_nested_skip(self):
        html = "<nav><div><p>skip me</p></div></nav>keep me"
        assert fsi._plain_text(html) == "keep me"


# ---------------------------------------------------------------------------
# _extract_article_html
# ---------------------------------------------------------------------------


class TestExtractArticleHtml:
    def test_article_tag(self):
        html = "<html><article>content</article></html>"
        assert fsi._extract_article_html(html) == "content"

    def test_role_main_fallback(self):
        html = '<html><div role="main">main content</div></html>'
        assert fsi._extract_article_html(html) == "main content"

    def test_full_page_fallback(self):
        html = "<html><body>no special tag</body></html>"
        assert fsi._extract_article_html(html) == html

    def test_article_preferred_over_role_main(self):
        html = '<div role="main">main</div><article>article</article>'
        assert fsi._extract_article_html(html) == "article"


# ---------------------------------------------------------------------------
# _strip_h1
# ---------------------------------------------------------------------------


class TestStripH1:
    def test_removes_h1(self):
        html = "<h1>Page title</h1><p>body</p>"
        result = fsi._strip_h1(html)
        assert "Page title" not in result
        assert "<p>body</p>" in result

    def test_removes_only_first_h1(self):
        html = "<h1>First</h1><h1>Second</h1>"
        result = fsi._strip_h1(html)
        assert "First" not in result
        assert "Second" in result

    def test_no_h1(self):
        html = "<p>no heading</p>"
        assert fsi._strip_h1(html) == html

    def test_case_insensitive(self):
        html = "<H1>Title</H1><p>body</p>"
        result = fsi._strip_h1(html)
        assert "Title" not in result


# ---------------------------------------------------------------------------
# _extract_title
# ---------------------------------------------------------------------------


class TestExtractTitle:
    def test_basic(self):
        html = "<html><head><title>My Page</title></head></html>"
        assert fsi._extract_title(html) == "My Page"

    def test_strips_documentation_suffix_em_dash(self):
        html = "<title>My Page \u2014 skrub 0.5.0 documentation</title>"
        assert fsi._extract_title(html) == "My Page"

    def test_strips_suffix_html_entity(self):
        html = "<title>My Page &#8212; skrub 0.5 documentation</title>"
        assert fsi._extract_title(html) == "My Page"

    def test_strips_suffix_en_dash(self):
        html = "<title>My Page \u2013 skrub docs</title>"
        assert fsi._extract_title(html) == "My Page"

    def test_no_title_tag(self):
        assert fsi._extract_title("<html></html>") == ""

    def test_decodes_entities_in_title(self):
        html = "<title>filter() &amp; friends &#8212; skrub</title>"
        assert fsi._extract_title(html) == "filter() & friends"


# ---------------------------------------------------------------------------
# _split_sections
# ---------------------------------------------------------------------------


class TestSplitSections:
    def _make(self, *sections):
        """Build minimal article HTML with the given (id, heading, body) items."""
        parts = []
        for sec_id, heading, body in sections:
            parts.append(f'<section id="{sec_id}"><h2>{heading}</h2>{body}</section>')
        return "\n".join(parts)

    def test_single_section(self):
        html = self._make(("intro", "Introduction", "<p>body text</p>"))
        results = fsi._split_sections(html)
        assert len(results) == 1
        sec_id, heading, body = results[0]
        assert sec_id == "intro"
        assert heading == "Introduction"
        assert "body text" in body

    def test_multiple_sections(self):
        html = self._make(
            ("s1", "First", "<p>one</p>"),
            ("s2", "Second", "<p>two</p>"),
        )
        results = fsi._split_sections(html)
        assert [r[0] for r in results] == ["s1", "s2"]

    def test_skips_section_without_heading(self):
        html = '<section id="no-heading"><p>only a paragraph</p></section>'
        assert fsi._split_sections(html) == []

    def test_h1_not_included(self):
        # h1 is the page title; only h2–h4 should create section entries
        html = '<section id="top"><h1>Page Title</h1><p>body</p></section>'
        assert fsi._split_sections(html) == []

    def test_headerlink_stripped_from_heading(self):
        html = (
            '<section id="s1">'
            '<h2>Heading<a class="headerlink" href="#s1">\xb6</a></h2>'
            "<p>body</p>"
            "</section>"
        )
        results = fsi._split_sections(html)
        assert len(results) == 1
        assert results[0][1] == "Heading"

    def test_html_entities_decoded_in_heading(self):
        html = '<section id="s1"><h2>a &amp; b</h2><p>body</p></section>'
        results = fsi._split_sections(html)
        assert results[0][1] == "a & b"

    def test_nested_sections_both_indexed(self):
        # Both parent and child sections should appear as separate entries.
        html = textwrap.dedent("""\
            <section id="parent">
              <h2>Parent</h2>
              <p>parent body</p>
              <section id="child">
                <h3>Child</h3>
                <p>child body</p>
              </section>
            </section>
        """)
        results = fsi._split_sections(html)
        ids = [r[0] for r in results]
        assert "parent" in ids
        assert "child" in ids


# ---------------------------------------------------------------------------
# _build_finished — integration test
# ---------------------------------------------------------------------------


class TestBuildFinished:
    """Test the full pipeline: HTML files → fuse-search-index.js."""

    def _make_page(self, title, h2s=None):
        """Return minimal Sphinx-like HTML for a page."""
        h2_html = ""
        if h2s:
            for sec_id, heading, body in h2s:
                h2_html += (
                    f'<section id="{sec_id}"><h2>{heading}</h2><p>{body}</p></section>'
                )
        return textwrap.dedent(f"""\
            <html>
            <head><title>{title} &#8212; skrub 0.1 documentation</title></head>
            <body>
            <article>
              <h1>{title}</h1>
              <p>Intro paragraph.</p>
              {h2_html}
            </article>
            </body></html>
        """)

    def test_generates_index_file(self, tmp_path):
        (tmp_path / "_static").mkdir()
        page = tmp_path / "index.html"
        page.write_text(self._make_page("Home"), encoding="utf-8")

        app = SimpleNamespace(
            builder=SimpleNamespace(name="html"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=None)

        index_js = tmp_path / "_static" / "fuse-search-index.js"
        assert index_js.exists()
        content = index_js.read_text(encoding="utf-8")
        assert content.startswith("window.__FUSE_SEARCH_INDEX__ = ")
        assert content.endswith(";\n")

    def test_index_contains_page_entry(self, tmp_path):
        (tmp_path / "_static").mkdir()
        (tmp_path / "guide.html").write_text(
            self._make_page("My Guide", h2s=[("s1", "Section One", "detail")]),
            encoding="utf-8",
        )
        app = SimpleNamespace(
            builder=SimpleNamespace(name="html"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=None)

        js = (tmp_path / "_static" / "fuse-search-index.js").read_text()
        data = json.loads(js.split(" = ", 1)[1].rstrip(";\n"))

        urls = [e["url"] for e in data]
        assert "guide.html" in urls

        page_entry = next(e for e in data if e["url"] == "guide.html")
        assert page_entry["title"] == "My Guide"
        assert page_entry["type"] == "page"
        # h1 should be stripped from content
        assert "My Guide" not in page_entry["content"]

    def test_index_contains_section_entry(self, tmp_path):
        (tmp_path / "_static").mkdir()
        (tmp_path / "guide.html").write_text(
            self._make_page("My Guide", h2s=[("s1", "Section One", "detail text")]),
            encoding="utf-8",
        )
        app = SimpleNamespace(
            builder=SimpleNamespace(name="html"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=None)

        js = (tmp_path / "_static" / "fuse-search-index.js").read_text()
        data = json.loads(js.split(" = ", 1)[1].rstrip(";\n"))

        section_entry = next((e for e in data if e.get("url") == "guide.html#s1"), None)
        assert section_entry is not None
        assert section_entry["title"] == "Section One"
        assert section_entry["page"] == "My Guide"
        assert section_entry["type"] == "section"
        assert "detail text" in section_entry["content"]

    def test_skipped_pages_not_indexed(self, tmp_path):
        (tmp_path / "_static").mkdir()
        for name in ("genindex.html", "search.html", "py-modindex.html"):
            (tmp_path / name).write_text(self._make_page(name), encoding="utf-8")
        # Also write a legitimate page so the index isn't empty
        (tmp_path / "real.html").write_text(
            self._make_page("Real Page"), encoding="utf-8"
        )
        app = SimpleNamespace(
            builder=SimpleNamespace(name="html"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=None)

        js = (tmp_path / "_static" / "fuse-search-index.js").read_text()
        data = json.loads(js.split(" = ", 1)[1].rstrip(";\n"))
        urls = [e["url"] for e in data]

        assert "genindex.html" not in urls
        assert "search.html" not in urls
        assert "py-modindex.html" not in urls
        assert "real.html" in urls

    def test_api_pages_have_api_type(self, tmp_path):
        (tmp_path / "_static").mkdir()
        (tmp_path / "reference" / "generated").mkdir(parents=True)
        (tmp_path / "reference" / "generated" / "skrub.Foo.html").write_text(
            self._make_page("skrub.Foo"), encoding="utf-8"
        )
        app = SimpleNamespace(
            builder=SimpleNamespace(name="html"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=None)

        js = (tmp_path / "_static" / "fuse-search-index.js").read_text()
        data = json.loads(js.split(" = ", 1)[1].rstrip(";\n"))

        api_entry = next(
            e for e in data if e["url"] == "reference/generated/skrub.Foo.html"
        )
        assert api_entry["type"] == "api"
        # API pages should not produce section entries
        section_entries = [e for e in data if e.get("type") == "section"]
        assert section_entries == []

    def test_reference_toc_pages_skipped(self, tmp_path):
        (tmp_path / "_static").mkdir()
        (tmp_path / "reference").mkdir()
        (tmp_path / "reference" / "selectors.html").write_text(
            self._make_page("Selectors"), encoding="utf-8"
        )
        app = SimpleNamespace(
            builder=SimpleNamespace(name="html"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=None)

        js = (tmp_path / "_static" / "fuse-search-index.js").read_text()
        data = json.loads(js.split(" = ", 1)[1].rstrip(";\n"))
        assert not any(e["url"].startswith("reference/selectors") for e in data)

    def test_example_pages_have_example_type(self, tmp_path):
        (tmp_path / "_static").mkdir()
        (tmp_path / "auto_examples").mkdir()
        (tmp_path / "auto_examples" / "0000_getting_started.html").write_text(
            self._make_page("Getting Started"), encoding="utf-8"
        )
        app = SimpleNamespace(
            builder=SimpleNamespace(name="html"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=None)

        js = (tmp_path / "_static" / "fuse-search-index.js").read_text()
        data = json.loads(js.split(" = ", 1)[1].rstrip(";\n"))

        ex = next(
            e for e in data if e["url"] == "auto_examples/0000_getting_started.html"
        )
        assert ex["type"] == "example"

    def test_no_exception_propagated(self, tmp_path):
        """build-finished with exception=<something> must do nothing."""
        (tmp_path / "_static").mkdir()
        app = SimpleNamespace(
            builder=SimpleNamespace(name="html"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=RuntimeError("build failed"))
        assert not (tmp_path / "_static" / "fuse-search-index.js").exists()

    def test_non_html_builder_skipped(self, tmp_path):
        (tmp_path / "_static").mkdir()
        app = SimpleNamespace(
            builder=SimpleNamespace(name="latex"),
            outdir=str(tmp_path),
        )
        fsi._build_finished(app, exception=None)
        assert not (tmp_path / "_static" / "fuse-search-index.js").exists()
