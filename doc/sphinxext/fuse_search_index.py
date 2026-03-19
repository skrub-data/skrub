"""Sphinx extension that generates a Fuse.js-compatible JSON search index.

After the HTML build completes, this extension walks all generated HTML pages,
extracts their titles and body text, and writes
``_static/fuse-search-index.js``.  The JS file assigns a flat list of entry
objects to ``window.__FUSE_SEARCH_INDEX__``::

    [
      {
        "title": "...",
        "content": "...",
        "url": "path/to/page.html",        // or "page.html#section-id"
        "type": "page"                     // or "api" or "section"
      },
      ...
    ]

For non-API pages in the user guide / examples, each ``<section>`` element
with an ``id`` and a heading (h2–h4) also becomes its own entry so that users
can land directly on the relevant part of a page.
"""

from __future__ import annotations

import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML → plain text helpers
# ---------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Extract visible text from an HTML fragment, skipping code, nav etc."""

    _SKIP_TAGS = frozenset(
        {"script", "style", "head", "noscript", "nav", "footer", "aside", "button"}
    )

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self, max_chars: int = 3000) -> str:
        text = " ".join(self._parts)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]


def _plain_text(html_fragment: str, max_chars: int = 3000) -> str:
    ext = _TextExtractor()
    ext.feed(html_fragment)
    return ext.get_text(max_chars)


def _extract_article_html(full_html: str) -> str:
    """Return the HTML of the main article element, or the full page."""
    m = re.search(r"(<article\b[^>]*>)(.*?)</article>", full_html, re.DOTALL)
    if m:
        return m.group(2)
    m = re.search(
        r'<div\b[^>]*role=["\']main["\'][^>]*>(.*?)</div>', full_html, re.DOTALL
    )
    if m:
        return m.group(1)
    return full_html


def _strip_h1(article_html: str) -> str:
    """Remove the first h1 element so it isn't repeated in content snippets."""
    return re.sub(
        r"<h1\b[^>]*>.*?</h1>",
        "",
        article_html,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _extract_title(full_html: str) -> str:
    from html import unescape

    m = re.search(r"<title>(.*?)</title>", full_html, re.DOTALL)
    if not m:
        return ""
    title = unescape(m.group(1).strip())
    # Strip " — skrub X.Y.Z documentation" suffix
    title = re.sub(r"\s*[—\u2014\u2013-]\s*skrub.*$", "", title, flags=re.IGNORECASE)
    return title.strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

# Matches an opening <section id="..."> tag (Sphinx wraps every heading in one)
_SECTION_OPEN_RE = re.compile(
    r'<section\b[^>]*\bid=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE
)
# Matches any heading h2–h4 at the start of a section's content
_HEADING_RE = re.compile(r"<h[2-4]\b[^>]*>(.*?)</h[2-4]>", re.IGNORECASE | re.DOTALL)
# Closing tag
_SECTION_CLOSE_RE = re.compile(r"</section\s*>", re.IGNORECASE)


def _split_sections(article_html: str) -> list[tuple[str, str, str]]:
    """Return list of (section_id, heading_text, body_html) for each h2–h4
    section found in *article_html*.

    Both top-level and nested sections are indexed independently.  The
    depth-tracking is only used to find the correct matching ``</section>``
    boundary for each opening tag.
    """
    from html import unescape

    results = []
    html = article_html

    for open_m in _SECTION_OPEN_RE.finditer(html):
        sec_id = open_m.group(1)
        content_start = open_m.end()

        # Find the matching </section> by tracking open/close depth
        depth = 1
        search_pos = content_start
        close_pos = content_start
        while depth > 0:
            next_open = _SECTION_OPEN_RE.search(html, search_pos)
            next_close = _SECTION_CLOSE_RE.search(html, search_pos)
            if not next_close:
                break
            if next_open and next_open.start() < next_close.start():
                depth += 1
                search_pos = next_open.end()
            else:
                depth -= 1
                close_pos = next_close.start()
                search_pos = next_close.end()

        section_html = html[content_start:close_pos]

        # Only include sections whose first element is an h2–h4 heading
        heading_m = _HEADING_RE.search(section_html)
        if not heading_m:
            continue

        heading_html = heading_m.group(1)
        # Strip anchor permalink (<a class="headerlink"…>) inside headings
        heading_clean = re.sub(
            r'<a\b[^>]*class=["\'][^"\']*headerlink[^"\']*["\'][^>]*>.*?</a>',
            "",
            heading_html,
            flags=re.DOTALL,
        )
        heading_text = unescape(_plain_text(heading_clean, 200))

        # Body text = everything after the heading tag
        body_html = section_html[heading_m.end() :]

        results.append((sec_id, heading_text, body_html))

    return results


# ---------------------------------------------------------------------------
# Sphinx event handler
# ---------------------------------------------------------------------------

#: Exact relative paths that are always skipped (TOC / index pages).
_SKIP_EXACT = frozenset(
    {
        "documentation.html",
        "auto_examples/index.html",
        "auto_examples/sg_execution_times.html",
        "auto_examples/data_ops/index.html",
        "auto_examples/data_ops/sg_execution_times.html",
        "reference/index.html",
    }
)

#: Relative path prefixes that are skipped (search indexes, generated indexes…)
_SKIP_PREFIXES = (
    "_",
    "genindex",
    "search",
    "py-modindex",
    "sg_execution_times",
)

#: Relative path prefixes treated as API reference pages (no section splitting)
_API_PREFIXES = ("reference/generated/",)

#: Relative path prefixes treated as example-gallery pages
_EXAMPLE_PREFIXES = ("auto_examples/",)


def _build_finished(app, exception) -> None:
    if exception:
        return
    if app.builder.name not in ("html", "dirhtml"):
        return

    outdir = Path(app.outdir)
    static_dir = outdir / "_static"
    static_dir.mkdir(exist_ok=True)

    entries: list[dict] = []

    for html_path in sorted(outdir.rglob("*.html")):
        rel = html_path.relative_to(outdir)
        rel_str = rel.as_posix()

        # Skip utility / system pages
        if any(rel_str.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if rel_str in _SKIP_EXACT:
            continue
        # Skip reference index/TOC pages (e.g. reference/selectors.html) but
        # keep the actual generated API entries under reference/generated/.
        if rel_str.startswith("reference/") and not rel_str.startswith(
            "reference/generated/"
        ):
            continue

        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        title = _extract_title(html)
        if not title:
            title = rel_str

        article_html = _extract_article_html(html)

        is_api = any(rel_str.startswith(p) for p in _API_PREFIXES)
        is_example = any(rel_str.startswith(p) for p in _EXAMPLE_PREFIXES)

        if is_api:
            page_type = "api"
        elif is_example:
            page_type = "example"
        else:
            page_type = "page"

        # Page-level entry (strip h1 from content — it's already in title)
        entries.append(
            {
                "title": title,
                "content": _plain_text(_strip_h1(article_html)),
                "url": rel_str,
                "type": page_type,
            }
        )

        # Section-level entries (everything except raw API reference pages)
        if not is_api:
            for sec_id, heading_text, body_html in _split_sections(article_html):
                entries.append(
                    {
                        "title": heading_text,
                        "page": title,  # breadcrumb shown below the title
                        "content": _plain_text(body_html, 1500),
                        "url": f"{rel_str}#{sec_id}",
                        "type": "section",
                    }
                )

    # Write as a JS file (assigned to a global) so the search page can load
    # it via a plain <script> tag.  This works on file:// URLs too, unlike
    # fetch() which is blocked by the browser's CORS policy.
    index_path = static_dir / "fuse-search-index.js"
    index_path.write_text(
        "window.__FUSE_SEARCH_INDEX__ = "
        + json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    logger.info("[fuse_search_index] wrote %d entries → %s", len(entries), index_path)


# ---------------------------------------------------------------------------
# Sphinx setup
# ---------------------------------------------------------------------------


def setup(app):
    app.connect("build-finished", _build_finished)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
