(function() {
    // Node pages are displayed in the iframe and tell us which node they show
    // (see node.html). We cannot read the iframe's location ourselves: for
    // file:// URLs browsers treat it as cross-origin.
    //
    // The current node is highlighted with a CSS rule rather than by editing
    // the graph, so that it does not matter whether the graph is already
    // there: it may be rendered later by the browser (when graphviz is not
    // installed).
    const frame = document.getElementById('node-frame');
    const highlightStyle = document.getElementById('current-node-style');

    window.addEventListener('message', (event) => {
        if (event.source !== frame.contentWindow) {
            return;
        }
        if (!event.data || event.data.type !== 'skrub-report-node-shown') {
            return;
        }
        const nodeId = event.data.nodeId;
        // nodeId is null for pages that do not show a particular node
        highlightStyle.textContent = (nodeId === null) ? '' : (
            `#node_${Number(nodeId)} polygon {` +
            ' fill: var(--current-node-color); stroke-width: 3; }');
    });

    document.getElementById('toggle-nav').addEventListener('click', () => {
        document.querySelector('nav').toggleAttribute('data-is-open');
    });
})();
