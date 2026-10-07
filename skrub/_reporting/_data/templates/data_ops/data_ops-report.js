(function() {
    const graphDiv = document.getElementById('graph-nav');
    const frame = document.getElementById('node-frame');

    // The graph may be rendered asynchronously by the browser (when graphviz
    // is not installed). In that case the "graph-rendered" event is fired on
    // the graph container once the SVG is in the page.
    function whenGraphReady(callback) {
        if (graphDiv.querySelector('svg') !== null) {
            callback();
        } else {
            graphDiv.addEventListener('graph-rendered', callback, {once: true});
        }
    }

    function highlightNode(nodeId) {
        whenGraphReady(() => {
            for (const elem of graphDiv.querySelectorAll('.current-node')) {
                elem.classList.remove('current-node');
            }
            const nodeElem = document.getElementById(`node_${nodeId}`);
            if (nodeElem !== null) {
                nodeElem.classList.add('current-node');
            }
        });
    }

    function showNodeStatus() {
        const nodeStatus = JSON.parse(graphDiv.dataset.nodeStatus);
        for (const nodeId in nodeStatus) {
            const nodeElem = document.getElementById(`node_${nodeId}`);
            if (nodeElem === null) {
                continue;
            }
            switch (nodeStatus[nodeId]) {
            case 'success':
                nodeElem.classList.add('success-node');
                break;
            case 'error':
                nodeElem.classList.add('error-node');
                break;
            case 'skipped':
                nodeElem.classList.add('skipped-node');
                break;
            default:
                // eval=False was passed to .skb.report(), no particular styling
                // needed on any nodes.
                break;
            }
        }
    }
    whenGraphReady(showNodeStatus);

    // Node pages notify us when they are displayed in the iframe, whether it
    // was by clicking on the graph, on a link in a node page, or by using the
    // browser's back button. We keep the highlighted node and the URL's
    // fragment in sync so that reloading or sharing the URL shows the same
    // node. We cannot read the iframe's location directly: for file:// URLs
    // browsers treat it as cross-origin.
    window.addEventListener('message', (event) => {
        if (event.source !== frame.contentWindow) {
            return;
        }
        const data = event.data;
        if (data === null || typeof data !== 'object' ||
            data.type !== 'skrub-report-node-shown') {
            return;
        }
        highlightNode(data.nodeId);
        try {
            history.replaceState(null, '', `#node_${data.nodeId}`);
        } catch (e) {
            // some browsers restrict the history API for file:// URLs
        }
    });

    function showNodeFromHash() {
        const match = window.location.hash.match(/^#node_(\d+)$/);
        if (match === null) {
            return;
        }
        // the placeholder is a srcdoc, which takes precedence over src
        frame.removeAttribute('srcdoc');
        frame.src = `node_${match[1]}.html`;
    }
    showNodeFromHash();
    window.addEventListener('hashchange', showNodeFromHash);

    function toggleNav() {
        const nav = document.querySelector('nav');
        if (nav.hasAttribute('data-is-open')) {
            nav.removeAttribute('data-is-open');
        } else {
            nav.setAttribute('data-is-open', '');
        }
    }
    document.getElementById('toggle-nav').addEventListener('click', toggleNav);
})();
