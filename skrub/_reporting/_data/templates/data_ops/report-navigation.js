// Included in node pages, which are displayed in the drawer iframe of
// index.html (see data_ops-report.js).
//
// A plain link click inside the iframe would add an entry to the browser's
// history, so that the back button would step through the nodes visited. We
// follow links with location.replace() instead. Links to the python files that
// created a node are not followed here: we ask the index page to show them in
// a separate drawer view, which can be dismissed to return to the node details.
(function () {
    const here = new URL(window.location.href);
    const reportDir = here.pathname.slice(
        0, here.pathname.lastIndexOf("/") + 1);
    const sourceDir = new URL("python/", here).pathname;
    // "node" or "source" for pages of this report, null for anything else
    // (external links, which are left to the browser).
    function classify(url) {
        const target = new URL(url, here);
        if (target.protocol !== here.protocol ||
            target.host !== here.host ||
            !target.pathname.startsWith(reportDir)) {
            return null;
        }
        return target.pathname.startsWith(sourceDir) ? "source" : "node";
    }
    // Returns true if the link was handled here.
    function follow(url) {
        const kind = classify(url);
        if (kind === null) {
            return false;
        }
        url = new URL(url, here).href;
        // When the page is not in the drawer (opened on its own) there is
        // nobody to ask, and we simply navigate.
        if (kind === "source" && window.parent !== window) {
            // The target origin must be "*": for file:// URLs the origin is
            // opaque, so it cannot be named.
            window.parent.postMessage({
                type: "skrub-report-source",
                url: url,
            }, "*");
        } else {
            window.location.replace(url);
        }
        return true;
    }
    document.addEventListener("click", (event) => {
        if (event.defaultPrevented || event.button !== 0 ||
            event.metaKey || event.ctrlKey || event.shiftKey ||
            event.altKey || !(event.target instanceof Element)) {
            return;
        }
        const link = event.target.closest("a");
        if (!link) {
            return;
        }
        const target = link.getAttribute("target");
        const href = link.getAttribute("href") ||
            link.getAttributeNS("http://www.w3.org/1999/xlink", "href");
        if ((target && target !== "_self") || !href ||
            link.hasAttribute("download")) {
            return;
        }
        if (follow(href)) {
            event.preventDefault();
        }
    });
})();
