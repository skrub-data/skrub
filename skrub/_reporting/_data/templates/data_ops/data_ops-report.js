// Index page of the report: the graph, plus a "drawer" showing a node page (or
// a source file) in an iframe next to it.
//
// This script controls what is displayed in the iframe. The browser's back
// button must leave the report rather than step through nodes, so the iframe is
// always navigated with location.replace(), which adds no history entry.
// When a link to another node ("previous", "next", ...) is clicked in the
// iframe, the event is captured and a message is posted so the navigation can
// be handled here (see report-navigation.js).
//
// The URL hash is kept up to date with replaceState so the URL for a specific
// node can be shared.
//
// There is a single persistent iframe, for nodes and source files.
// The browser keeps displaying the previous page until the next one
// is ready.
//
(function() {
    const frame = document.getElementById('drawer-frame');
    const closeButton = document.getElementById('close-drawer');
    const backButton = document.getElementById('back-to-node');
    const graphPanel = document.getElementById('graph-panel');
    const toggleGraphButton = document.getElementById('toggle-nav');
    // The current node is highlighted with a CSS rule rather than by editing
    // the graph, so that it works whether the graph was drawn by python or is
    // drawn later by the browser (in which case it may still be rendering).
    const highlightStyle = document.getElementById('current-node-style');
    const totalNNodes = Number(document.body.dataset.totalNodes);
    // The selected node, also while one of its source files is displayed.
    let currentNodeId = null;

    function navigate(url) {
        frame.contentWindow.location.replace(new URL(url, window.location.href));
    }

    // Button that shows or hides the graph panel.
    function setGraphOpen(isOpen) {
        graphPanel.toggleAttribute('data-is-open', isOpen);
        toggleGraphButton.setAttribute('aria-expanded', String(isOpen));
        toggleGraphButton.textContent = isOpen ? 'Hide graph' : 'Show graph';
    }

    // The toolbar has "Close" when a node is displayed, and only "Back to node
    // details" when a source file is.
    function setSourceMode(isSource) {
        closeButton.hidden = isSource;
        backButton.hidden = !isSource;
    }

    function nodeIdFromHash() {
        const match = /^#node_(\d+)$/.exec(window.location.hash);
        if (!match || Number(match[1]) >= totalNNodes) {
            return null;
        }
        return Number(match[1]);
    }

    // Keep the URL shareable (index.html#node_3) without adding history entries.
    function updateState(nodeId) {
        currentNodeId = nodeId;
        document.body.classList.toggle('drawer-open', nodeId !== null);
        // Never leave the page empty: without a node there is nothing to show
        // but the graph.
        toggleGraphButton.disabled = nodeId === null;
        if (nodeId === null) {
            setGraphOpen(true);
        }
        highlightStyle.textContent = nodeId === null ? '' : (
            `#node_${nodeId} polygon {` +
            ' fill: var(--current-node-color); stroke-width: 3; }');
        const hash = nodeId === null ? '' : `#node_${nodeId}`;
        if (window.location.hash !== hash) {
            const url = new URL(window.location.href);
            url.hash = hash;
            window.history.replaceState(null, '', url.href);
        }
    }

    function openNode(nodeId) {
        setSourceMode(false);
        navigate(`node_${nodeId}.html`);
        updateState(nodeId);
    }

    function openSource(url) {
        setSourceMode(true);
        navigate(url);
    }

    function closeDrawer() {
        setSourceMode(false);
        // do not show this page again if the drawer is reopened
        navigate('about:blank');
        updateState(null);
    }

    closeButton.addEventListener('click', closeDrawer);
    backButton.addEventListener('click', () => openNode(currentNodeId));

    // Node pages ask us to display the node or source file their links point to.
    window.addEventListener('message', (event) => {
        if (event.source !== frame.contentWindow ||
            !event.data || event.data.type !== 'skrub-report-link') {
            return;
        }
        const href = String(event.data.href);
        const node = /^node_(\d+)\.html$/.exec(href);
        if (node) {
            openNode(Number(node[1]));
        } else if (/^python\/[\w-]+\.html(#L\d+)?$/.test(href)) {
            openSource(href);
        }
    });

    // Graph nodes link to a hash such as "#node_3". We handle the click
    // ourselves because the default action would add a history entry. The
    // links are in an svg, where the attribute may be xlink:href.
    document.getElementById('graph-nav').addEventListener('click', (event) => {
        const link = event.target.closest('a');
        const href = link && (link.getAttribute('href') ||
            link.getAttributeNS('http://www.w3.org/1999/xlink', 'href'));
        const match = /^#node_(\d+)$/.exec(href);
        if (match && !event.metaKey && !event.ctrlKey && !event.shiftKey) {
            event.preventDefault();
            openNode(Number(match[1]));
        }
    });

    // The hash is edited by hand in the address bar of an open report. Our own
    // updates (replaceState) do not trigger this event.
    window.addEventListener('hashchange', () => {
        const nodeId = nodeIdFromHash();
        if (nodeId !== null) {
            openNode(nodeId);
        } else if (window.location.hash === '') {
            closeDrawer();
        }
    });
    if (nodeIdFromHash() !== null) {
        openNode(nodeIdFromHash());
    }

    toggleGraphButton.addEventListener('click', () => {
        setGraphOpen(!graphPanel.hasAttribute('data-is-open'));
    });
})();
