// Included in node pages. When a node page is displayed in the drawer of
// index.html (see data_ops-report.js), clicks on links to other nodes or to
// source files are not followed here: we ask the index page to display them,
// so that it stays in charge of what is shown (and of the browser's history).
// Nothing happens when the page is opened on its own.
(function () {
    if (window.parent === window) {
        return;
    }
    document.addEventListener("click", (event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey ||
            event.shiftKey || event.altKey) {
            return;
        }
        const link = event.target.closest('a[href^="node_"], a[href^="python/"]');
        if (!link) {
            return;
        }
        event.preventDefault();
        // The target origin must be "*": for file:// URLs the origin is opaque,
        // so it cannot be named.
        window.parent.postMessage(
            {type: "skrub-report-link", href: link.getAttribute("href")}, "*");
    });
})();
