// Index page of the report: the graph, plus a "drawer" showing a node page (or
// a source file) in an iframe next to it.
//
// This script controls what is displayed in the iframe. The browser's back
// button must leave the report rather than step through nodes, so the iframe is
// always navigated with location.replace(), which adds no history entry.
// When a link to another node ("previous", "next", ...) is clicked in the
// node iframe, the event is captured and a message is posted so the navigation
// can be handled here (see report-navigation.js).
//
// The URL hash is kept up to date with replaceState so the URL for a specific
// node can be shared.
//
// The browser keeps displaying the previous node until the next one is ready.
//
// A source file is displayed in a second iframe, on top of the node iframe, so
// that the node page is still there, as we left it, when we come back. It is
// displayed once it has loaded, so that nothing flashes either.
//
(function() {
    const drawer = document.getElementById('drawer');
    const nodeFrame = document.getElementById('node-frame');
    let sourceFrame = document.getElementById('source-frame');
    const graphPanel = document.getElementById('graph-panel');
    const toggleGraphButton = document.getElementById('toggle-nav');
    // The current node is highlighted with a CSS rule rather than by editing
    // the graph, so that it works whether the graph was drawn by python or is
    // drawn later by the browser (in which case it may still be rendering).
    const highlightStyle = document.getElementById('current-node-style');
    const totalNNodes = Number(document.body.dataset.totalNodes);

    function navigate(frame, url) {
        frame.contentWindow.location.replace(new URL(url, window.location.href));
    }

    // Button that shows or hides the graph panel.
    function setGraphOpen(isOpen) {
        graphPanel.toggleAttribute('data-is-open', isOpen);
        toggleGraphButton.setAttribute('aria-expanded', String(isOpen));
        toggleGraphButton.textContent = isOpen ? 'Hide graph' : 'Show graph';
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
        closeSource();
        navigate(nodeFrame, `node_${nodeId}.html`);
        updateState(nodeId);
    }

    // The source page is loaded in a new iframe, which replaces the previous one
    // and is displayed (instead of the node page) once loaded. A new iframe is
    // needed because navigating one that shows the same page (or another line of
    // it) fires no load event. Dropping the previous iframe also drops its
    // events: there is nothing to cancel if the user goes elsewhere while the
    // page loads.
    function replaceSourceFrame(url) {
        const frame = document.createElement('iframe');
        frame.id = 'source-frame';
        frame.title = 'Source code';
        if (url !== null) {
            frame.addEventListener('load', () => {
                drawer.dataset.mode = 'source';
                // so that keyboard navigation continues in the page displayed
                // (the frame can only be focused once it is visible)
                frame.focus();
            }, {once: true});
            frame.src = new URL(url, window.location.href).href;
        }
        sourceFrame.replaceWith(frame);
        sourceFrame = frame;
    }

    function openSource(url) {
        replaceSourceFrame(url);
    }

    function closeSource() {
        replaceSourceFrame(null);
        drawer.dataset.mode = 'node';
    }

    function closeDrawer() {
        closeSource();
        // do not show this page again if the drawer is reopened
        navigate(nodeFrame, 'about:blank');
        updateState(null);
    }

    document.getElementById('close-drawer').addEventListener('click', closeDrawer);
    document.getElementById('back-to-node').addEventListener('click', () => {
        closeSource();
        nodeFrame.focus();
    });

    // Node pages ask us to display the node or source file their links point to.
    window.addEventListener('message', (event) => {
        if (event.source !== nodeFrame.contentWindow ||
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

    // Pan and zoom of the graph, with the Panzoom library (included in the page).
    // - drag: pan; buttons: zoom; Ctrl (or Cmd) + wheel, or pinch: zoom at the
    //   pointer; wheel: pan, as the graph used to scroll before.
    // - a drag must not click the node it ends on.
    const graphNav = document.getElementById('graph-nav');
    const graphCanvas = document.getElementById('graph-canvas');
        const MIN_SCALE = 0.05;
    const MAX_SCALE = 4;
    const WHEEL_FACTOR = 1.1;
    const panzoom = Panzoom(graphCanvas, {
        origin: '0 0', canvas: true, cursor: 'inherit',
        minScale: MIN_SCALE, maxScale: MAX_SCALE,
    });

    function setView(scale, x, y) {
        panzoom.zoom(scale, {animate: false});
        panzoom.pan(x, y, {animate: false});
    }

    // Initial view: the width of the graph fits the panel (never enlarged), top
    // of the graph visible.
    function resetView() {
        const scale = Math.min(1, graphNav.clientWidth / graphCanvas.offsetWidth);
        setView(Math.max(scale, MIN_SCALE), 0, 0);
    }

    // The whole graph, centered.
    function fitView() {
        const w = graphCanvas.offsetWidth, h = graphCanvas.offsetHeight;
        const vw = graphNav.clientWidth, vh = graphNav.clientHeight;
        if (!w || !h) {
            return;
        }
        const scale = Math.min(vw / w, vh / h, 1);
        setView(Math.max(scale, MIN_SCALE),
                (vw / scale - w) / 2, (vh / scale - h) / 2);
    }

    // Zoom keeping the point of the graph under (clientX, clientY) in place. We do
    // it ourselves because the transform origin is the top left corner: the
    // point under the pointer is at `p / scale - pan` in the graph's coordinates
    // (p being the position in the viewport), and that must not change.
    function zoomAround(factor, clientX, clientY) {
        const rect = graphNav.getBoundingClientRect();
        const px = clientX - rect.left - graphNav.clientLeft;
        const py = clientY - rect.top - graphNav.clientTop;
        const oldScale = panzoom.getScale();
        const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, oldScale * factor));
        const pan = panzoom.getPan();
        setView(newScale,
                pan.x + px / newScale - px / oldScale,
                pan.y + py / newScale - py / oldScale);
    }

    function zoomAtCenter(factor) {
        const rect = graphNav.getBoundingClientRect();
        zoomAround(factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
    }

    graphNav.addEventListener('wheel', (event) => {
        event.preventDefault();
        if (event.ctrlKey || event.metaKey) {
            // also what browsers send for a trackpad pinch
            zoomAround(event.deltaY < 0 ? WHEEL_FACTOR : 1 / WHEEL_FACTOR,
                       event.clientX, event.clientY);
        } else {
            const scale = panzoom.getScale();
            const pan = panzoom.getPan();
            panzoom.pan(pan.x - event.deltaX / scale, pan.y - event.deltaY / scale,
                        {animate: false});
        }
    }, {passive: false});

    // Do not treat the end of a drag as a click on a node.
    let pointerDownAt = null;
    let dragged = false;
    graphNav.addEventListener('pointerdown', (event) => {
        pointerDownAt = {x: event.clientX, y: event.clientY};
        dragged = false;
        graphNav.classList.add('panning');
    });
    graphNav.addEventListener('pointermove', (event) => {
        if (pointerDownAt && Math.hypot(
            event.clientX - pointerDownAt.x, event.clientY - pointerDownAt.y) > 4) {
            dragged = true;
        }
    });
    for (const name of ['pointerup', 'pointercancel']) {
        document.addEventListener(name, () => {
            pointerDownAt = null;
            graphNav.classList.remove('panning');
            // The click (if any) is dispatched right after pointerup. If the
            // pointer was released elsewhere there is none: do not let the flag
            // swallow a later click.
            setTimeout(() => { dragged = false; }, 0);
        });
    }
    graphNav.addEventListener('click', (event) => {
        if (dragged) {
            event.preventDefault();
            event.stopPropagation();
            dragged = false;
        }
    }, true);

    document.getElementById('zoom-in').addEventListener('click', () => zoomAtCenter(1.5));
    document.getElementById('zoom-out').addEventListener('click', () => zoomAtCenter(1 / 1.5));
    document.getElementById('zoom-fit').addEventListener('click', fitView);
    document.getElementById('zoom-reset').addEventListener('click', resetView);

    // The size of the graph is known once it is drawn (immediately, unless the
    // browser draws it).
    resetView();
    graphNav.addEventListener('graph-rendered', resetView);
})();
