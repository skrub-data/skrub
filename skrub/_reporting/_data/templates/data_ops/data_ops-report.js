// Index page of the report: the graph, plus a "drawer" showing a node page (or
// a source file) in an iframe next to it.
//
// - Clicking a graph node, or loading index.html#node_3, opens that node. The
//   hash is kept up to date with replaceState so the URL for a specific node
//   can be shared.
// - The browser's back button must leave the report rather than step through
//   nodes. So the iframe is navigated with location.replace(), which adds no
//   history entry, both from here and from the pages inside the iframe (see
//   report-navigation.js). The same iframe is reused, so that the previous node
//   stays visible until the next one is ready rather than flashing an empty
//   drawer.
// - The report must work from file:// URLs, where the iframe is cross-origin
//   and we cannot read its location. Node pages therefore tell us what they
//   show via postMessage (see node.html and report-navigation.js).
(function() {
    const drawer = document.getElementById('drawer');
    const nodeView = document.getElementById('node-view');
    const sourceView = document.getElementById('source-view');
    const nodeFrameContainer = document.getElementById('node-frame-container');
    const sourceFrameContainer = document.getElementById('source-frame-container');
    // The current node is highlighted with a CSS rule rather than by editing
    // the graph, so that it works whether the graph was drawn by python or is
    // drawn later by the browser.
    const highlightStyle = document.getElementById('current-node-style');

    // The iframe showing the selected node; created when the first node is
    // opened and reused until the drawer is closed.
    let nodeFrame = null;
    const totalNNodes = Number(document.body.dataset.totalNodes);

    function makeFrame(description) {
        const frame = document.createElement('iframe');
        frame.title = description;
        return frame;
    }

    function nodeUrl(nodeId) {
        return new URL(`node_${nodeId}.html`, window.location.href).href;
    }

    // Keep the URL shareable (index.html#node_3) without adding history entries.
    function setNodeHash(nodeId) {
        const url = new URL(window.location.href);
        const hash = nodeId === null ? '' : `node_${nodeId}`;
        if (url.hash === (hash ? `#${hash}` : '')) {
            return;
        }
        url.hash = hash;
        window.history.replaceState(null, '', url.href);
    }

    function nodeIdFromHash() {
        const match = /^#node_(\d+)$/.exec(window.location.hash);
        if (!match) {
            return null;
        }
        const nodeId = Number(match[1]);
        return nodeId < totalNNodes ? nodeId : null;
    }

    function openNode(nodeId) {
        closeSource();

        if (nodeFrame) {
            // Navigate the existing iframe: the previous page stays visible
            // until the new one is ready, and replace() adds no history entry.
            nodeFrame.contentWindow.location.replace(nodeUrl(nodeId));
        } else {
            // The initial load of an iframe adds no history entry either.
            nodeFrame = makeFrame('Details of the selected node');
            nodeFrame.src = nodeUrl(nodeId);
            nodeFrameContainer.append(nodeFrame);
        }

        nodeView.hidden = false;
        drawer.hidden = false;
        document.body.classList.add('drawer-open');
        setNodeHash(nodeId);
    }

    function closeDrawer() {
        nodeFrameContainer.replaceChildren();
        nodeFrame = null;
        closeSource();
        drawer.hidden = true;
        document.body.classList.remove('drawer-open');
        highlightStyle.textContent = '';
        setNodeHash(null);
    }

    function openSource(url) {
        sourceFrameContainer.replaceChildren();
        const frame = makeFrame('Source code');
        // The node view stays displayed until the source page has loaded, to
        // avoid flashing an empty view. Meanwhile the source view is laid out
        // but invisible (not display: none), as the page scrolls to the
        // highlighted line when it loads.
        sourceView.dataset.loading = '';
        sourceView.hidden = false;
        frame.addEventListener('load', () => {
            nodeView.hidden = true;
            delete sourceView.dataset.loading;
        }, {once: true});
        frame.src = url;
        sourceFrameContainer.append(frame);

        drawer.hidden = false;
        document.body.classList.add('drawer-open');
    }

    function closeSource() {
        sourceFrameContainer.replaceChildren();
        delete sourceView.dataset.loading;
        sourceView.hidden = true;
        if (!drawer.hidden) {
            nodeView.hidden = false;
        }
    }

    document.getElementById('close-drawer').addEventListener('click', closeDrawer);
    document.getElementById('close-source').addEventListener('click', closeSource);

    // Messages from the node page: which node it shows (we update which node is
    // highlighted in the graph and the URL hash; it may differ from the one we
    // opened, as node pages have previous / next / parent / child links), and
    // requests to display a source file in the second drawer view.
    window.addEventListener('message', (event) => {
        if (!nodeFrame || event.source !== nodeFrame.contentWindow) {
            return;
        }
        if (!event.data || typeof event.data !== 'object') {
            return;
        }
        if (event.data.type === 'skrub-report-node-shown') {
            const nodeId = Number(event.data.nodeId);
            highlightStyle.textContent = (
                `#node_${nodeId} polygon {` +
                ' fill: var(--current-node-color); stroke-width: 3; }');
            setNodeHash(nodeId);
        } else if (event.data.type === 'skrub-report-source') {
            openSource(event.data.url);
        }
    });

    // Graph nodes link to a hash such as "#node_3". We handle the click
    // ourselves because the default action would add a history entry. The
    // links are in an svg, where the attribute may be xlink:href.
    document.addEventListener('click', (event) => {
        if (event.defaultPrevented || event.button !== 0 ||
            event.metaKey || event.ctrlKey || event.shiftKey ||
            event.altKey || !(event.target instanceof Element)) {
            return;
        }
        const link = event.target.closest('a');
        if (!link) {
            return;
        }
        const href = link.getAttribute('href') ||
            link.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
        const match = href && /^#node_(\d+)$/.exec(href);
        if (!match) {
            return;
        }
        event.preventDefault();
        openNode(Number(match[1]));
    });

    const initialNodeId = nodeIdFromHash();
    if (initialNodeId !== null) {
        openNode(initialNodeId);
    }

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

    // Button that shows or hides the graph panel.
    const graphPanel = document.getElementById('graph-panel');
    const toggleNav = document.getElementById('toggle-nav');
    // Label inside the graph toggle button.
    const toggleNavLabel = document.getElementById('toggle-nav-label');
    toggleNav.addEventListener('click', () => {
        const isOpen = graphPanel.toggleAttribute('data-is-open');
        toggleNav.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        toggleNav.title = isOpen ? 'Hide graph' : 'Show graph';
        toggleNavLabel.textContent = isOpen ? 'Hide graph' : 'Show graph';
    });
})();
