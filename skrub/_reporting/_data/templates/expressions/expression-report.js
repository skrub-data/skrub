(function() {

    function highlightCurrentNode() {
        const graphDiv = document.getElementById('graph-nav');
        const currentNode = graphDiv.dataset.currentNodeId;
        const nodeElem = document.getElementById(currentNode);
        if (nodeElem === null){
            return;
        }
        nodeElem.classList.add('current-node');
    }
    highlightCurrentNode();

    function toggleNav() {
        const nav = document.querySelector("nav");
        if (nav === null){
            return;
        }
        if (nav.hasAttribute("data-is-open")) {
            nav.removeAttribute("data-is-open");
        } else {
            nav.setAttribute("data-is-open", "");
        }
    }
    const toggleNavButton = document.getElementById('toggle-nav');
    if (toggleNavButton !== null){
        toggleNavButton.addEventListener("click", toggleNav);
    }

    function showNodeStatus() {
        const graphDiv = document.getElementById('graph-nav');
        console.log(graphDiv.dataset.nodeStatus);
        const nodeStatus = JSON.parse(graphDiv.dataset.nodeStatus);
        console.log(nodeStatus);
        for (const nodeId in nodeStatus) {
            const nodeElem = document.getElementById(`node_${nodeId}`);
            console.log(nodeElem);
            switch (nodeStatus[nodeId]) {
            case 'success':
                nodeElem.classList.add('success-node');
                break;
            case 'error':
                nodeElem.classList.add('error-node');
                break;
                default:
                    nodeElem.classList.add('skipped-node');
                    break;
            }
        }
    }
    showNodeStatus();
})();
