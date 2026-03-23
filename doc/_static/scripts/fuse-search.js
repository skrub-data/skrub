/**
 * fuse-search.js — client-side Fuse.js search for skrub documentation.
 *
 * This script:
 *  1. Reads the pre-built index from window.__FUSE_SEARCH_INDEX__ (assigned
 *     by _static/fuse-search-index.js, loaded via a <script> tag).
 *  2. Pre-fills the search input from the ?q= URL parameter.
 *  3. Runs fuzzy search using Fuse.js and renders results.
 */

"use strict";

(function () {
  // -------------------------------------------------------------------------
  // Utilities
  // -------------------------------------------------------------------------

  /** Build a URL from a path relative to the doc root. */
  function docUrl(relPath) {
    // DOCUMENTATION_OPTIONS.URL_ROOT is injected by Sphinx into every page.
    // For search.html (always at the docs root) it is "" or "./".
    // We must NOT use "/" as a prefix: on file:// that resolves to the
    // filesystem root, producing broken links like file:///reference/….
    let root =
      (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.URL_ROOT) ||
      "";
    if (root === "/") root = "";
    return root + relPath;
  }

  /** Escape HTML special characters. */
  function escHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /**
   * Highlight every occurrence of ``term`` (exact string, case-insensitive)
   * inside ``text`` using <mark>.
   */
  function highlight(text, term) {
    if (!term) return escHtml(text);
    // Escape special regex characters in the term.
    const safe = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // Replace matches with <mark>…</mark>, keeping the original case.
    return escHtml(text).replace(
      new RegExp(safe, "gi"),
      (m) => `<mark>${m}</mark>`
    );
  }

  /**
   * Return a ≤200-character snippet of ``content`` around the first
   * occurrence of ``term``.
   */
  function snippet(content, term, maxLen) {
    maxLen = maxLen || 200;
    if (!content) return "";
    const lower = content.toLowerCase();
    // Find the first occurrence of the term (case-insensitive).
    // If not found, just return the start of the content.
    const idx = term ? lower.indexOf(term.toLowerCase()) : -1;
    let start = 0;
    // If the term is found, try to center the snippet around it.
    if (idx > 50) {
      start = idx - 50;
    }
    // Extract the snippet, adding ellipses if we had to cut off the start or end.
    let chunk = content.slice(start, start + maxLen);
    if (start > 0) chunk = "…" + chunk;
    if (start + maxLen < content.length) chunk += "…";
    return chunk;
  }

  // -------------------------------------------------------------------------
  // Render helpers
  // -------------------------------------------------------------------------

  function renderResults(results, query) {
    const container = document.getElementById("fuse-search-results");
    if (!results.length) {
      container.innerHTML =
        '<p class="fuse-no-results">No results found for <strong>' +
        escHtml(query) +
        "</strong>.</p>";
      return;
    }

    const parts = ['<ul class="fuse-results-list">'];
    results.forEach(function (r) {
      // item is the matched document from the search index, with all its fields
      const item = r.item;
      const url = docUrl(item.url);
      // typeBadge is a visual indicator of the item's type (API, section, example, guide)
      // item.type is expected to be one of "api", "section", "example", or "guide"
      // The type is defined in sphinxext/fuse_search_index.py when building the index
      const typeBadge =
        item.type === "api"
          ? '<span class="fuse-badge fuse-badge--api">API</span>'
          : item.type === "section"
          ? '<span class="fuse-badge fuse-badge--section">Section</span>'
          : item.type === "example"
          ? '<span class="fuse-badge fuse-badge--example">Example</span>'
          : '<span class="fuse-badge fuse-badge--guide">Guide</span>';

      // Highlight the query terms in the title and content snippet.
      const titleHtml = highlight(item.title, query);
      const breadcrumb = item.page
        ? '<span class="fuse-result-breadcrumb">' + escHtml(item.page) + '</span>'
        : '';
      const snip = snippet(item.content, query, 220);
      const snipHtml = snip ? "<p>" + highlight(snip, query) + "</p>" : "";

      // Build the HTML for the search result item
      parts.push(
        '<li class="fuse-result-item">' +
          '<a class="fuse-result-link" href="' + escHtml(url) + '">' +
            '<span class="fuse-result-title">' + titleHtml + "</span>" +
            typeBadge +
          "</a>" +
          breadcrumb +
          snipHtml +
        "</li>"
      );
    });
    parts.push("</ul>");
    container.innerHTML = parts.join("");
  }

  // Update the display of how many results were found. If count=0, hide it.
  function updateCount(count, total) {
    const el = document.getElementById("fuse-search-count");
    if (!el) return;
    if (count === 0) {
      el.textContent = "";
    } else {
      // Show "1 result" or "5 results (of 12)" if we had to cap the displayed results.
      el.textContent =
        count + " result" + (count !== 1 ? "s" : "") +
        (total !== count ? " (of " + total + ")" : "");
    }
  }

  // -------------------------------------------------------------------------
  // Search state
  // -------------------------------------------------------------------------

  let fuseInstance = null;
  let currentType = "all"; // "all" | "page" | "api" | "example" | "section"
  let currentQuery = "";

  function runSearch() {
    const q = currentQuery.trim();
    const container = document.getElementById("fuse-search-results");

    // If the query is empty, don't show any results (not even "no results found").
    if (!q) {
      container.innerHTML = "";
      updateCount(0, 0);
      return;
    }

    // If the Fuse instance isn't ready yet, show a loading message.
    if (!fuseInstance) {
      container.innerHTML = '<p class="fuse-loading">Loading index…</p>';
      return;
    }

    // Run the search query through Fuse.js to get matching results.
    let results = fuseInstance.search(q);

    // Filter by type tab
    if (currentType !== "all") {
      results = results.filter((r) => r.item.type === currentType);
    }

    // Cap at 40 results to stay readable
    const total = results.length;
    results = results.slice(0, 40);

    updateCount(results.length, total);
    renderResults(results, q);
  }

  // -------------------------------------------------------------------------
  // Initialisation
  // -------------------------------------------------------------------------

  function init() {
    const input = document.getElementById("fuse-search-input");
    const typeFilters = document.getElementById("fuse-type-filters");
    if (!input) return;

    // Pre-fill from URL query parameter.
    const params = new URLSearchParams(window.location.search);
    const urlQuery = params.get("q") || params.get("query") || "";
    if (urlQuery) {
      input.value = urlQuery;
      currentQuery = urlQuery;
    }

    // The search index is loaded via a plain <script> tag in search.html
    // (window.__FUSE_SEARCH_INDEX__), which works on file:// URLs unlike
    // fetch().  If it is not present yet, something went wrong at build time.
    const data = window.__FUSE_SEARCH_INDEX__;
    if (!Array.isArray(data)) {
      console.error("[fuse-search] window.__FUSE_SEARCH_INDEX__ not found.");
      const container = document.getElementById("fuse-search-results");
      if (container) {
        container.innerHTML =
          '<p class="admonition warning">Search index could not be loaded.</p>';
      }
      return;
    }

    fuseInstance = new Fuse(data, {
      // Search in title (higher weight), page breadcrumb, and content.
      keys: [
        { name: "title", weight: 2 },
        { name: "page", weight: 1.5 },
        { name: "content", weight: 1 },
      ],
      includeScore: true,
      threshold: 0.35,       // 0 = exact, 1 = match anything
      distance: 200,         // characters ahead where to check
      minMatchCharLength: 2,
      shouldSort: true,
      ignoreLocation: true,  // search anywhere in the string
    });

    // Show the type filters and run the initial search if we had a query from the URL.
    // If we're using search.html, the filters are always present
    // XXX we might want to remove this check and just assume the filters are always
    // there, since search.html is the only place that includes this script
    if (typeFilters) typeFilters.hidden = false;
    if (currentQuery) runSearch();

    // Live search as the user types (debounced).
    let debounceTimer;
    input.addEventListener("input", function () {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function () {
        currentQuery = input.value;
        // Keep URL in sync so the page is shareable.
        try {
          const sp = new URLSearchParams(window.location.search);
          if (currentQuery) {
            sp.set("q", currentQuery);
          } else {
            sp.delete("q");
          }
          history.replaceState(
            null,
            "",
            window.location.pathname + (sp.toString() ? "?" + sp.toString() : "")
          );
        } catch (_) {
          // history.replaceState may be restricted on file:// origins.
        }
        runSearch();
      }, 250);
    });

    // Submit — prevent page reload, the live search already handled it.
    const form = document.getElementById("fuse-search-form");
    if (form) {
      form.addEventListener("submit", function (e) {
        // Do nothing on submit, we already handle live search on input.
        e.preventDefault();
      });
    }

    // Type filter buttons.
    if (typeFilters) {
      typeFilters.addEventListener("change", function (e) {
        if (e.target.name === "fuse-type") {
          currentType = e.target.value;
          // Update active styling.
          typeFilters
            .querySelectorAll(".fuse-filter-btn")
            .forEach(function (btn) {
              btn.classList.toggle(
                "active",
                btn.dataset.type === currentType
              );
            });
          runSearch();
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
