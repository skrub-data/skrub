if (customElements.get('skrub-table-report') === undefined) {

    class Exchange {
        constructor() {
            this.subscribers = [];
        }

        add(sub) {
            this.subscribers.push(sub);
        }

        send(msg) {
            setTimeout(() => this.distribute(msg));
        }

        distribute(msg) {
            for (let sub of this.subscribers) {
                sub.send(msg);
            }
        }
    }

    class Manager {
        constructor(elem, exchange) {
            this.elem = elem;
            this.exchange = exchange;
        }

        send(msg) {
            if (msg.kind in this) {
                this[msg.kind](msg);
            }
        }

        hide() {
            this.elem.dataset.hidden = "";
        }

        show() {
            delete this.elem.dataset.hidden;
        }

        matchesColumnIdx(idx) {
            if (!this.elem.hasAttribute("data-column-idx")) {
                return false;
            }
            return Number(this.elem.dataset.columnIdx) === idx;
        }

        matchesColumnFilter({
            acceptedColumns
        }) {
            return acceptedColumns.has(Number(this.elem.dataset.columnIdx));
        }
    }

    class SkrubTableReport extends HTMLElement {
        static managerClasses = new Map();

        constructor() {
            super();
            this.exchange = new Exchange();
        }

        connectedCallback() {
            const template = document.getElementById(`${this.id}-template`);
            this.attachShadow({
                mode: "open"
            });
            this.shadowRoot.appendChild(template.content.cloneNode(true));
        }

        static register(managerClass) {
            SkrubTableReport.managerClasses.set(managerClass.name, managerClass);
        }

        init() {
            this.shadowRoot.querySelectorAll("[data-manager]").forEach((elem) => {
                for (let className of elem.dataset.manager.split(/\s+/)) {
                    const cls = SkrubTableReport.managerClasses.get(
                        className);
                    if (cls !== undefined) {
                        this.exchange.add(new cls(elem, this.exchange));
                    }
                }
            });
            this.shadowRoot.querySelectorAll("[data-show-on]").forEach((elem) => {
                this.exchange.add(new ShowOn(elem, this.exchange));
            });
            this.shadowRoot.querySelectorAll("[data-hide-on]").forEach((elem) => {
                this.exchange.add(new HideOn(elem, this.exchange));
            });
        }
    }
    customElements.define("skrub-table-report", SkrubTableReport);

    class ShowOn extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.showMsgKinds = this.elem.dataset.showOn.split(/\s+/);
        }

        send(msg) {
            if (this.showMsgKinds.includes(msg.kind)) {
                this.show();
            }
        }
    }

    class HideOn extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.hideMsgKinds = this.elem.dataset.hideOn.split(/\s+/);
        }

        send(msg) {
            if (this.hideMsgKinds.includes(msg.kind)) {
                this.hide();
            }
        }
    }

    class InvisibleInAssociationsTabPanel extends Manager {
        TAB_SELECTED(msg) {
            if (msg.targetPanelId === "column-associations-panel") {
                this.elem.dataset.notVisible = "";
            } else {
                delete this.elem.dataset.notVisible;
            }
        }
    }
    SkrubTableReport.register(InvisibleInAssociationsTabPanel);

    class ColumnFilter extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.filters = JSON.parse(atob(this.elem.dataset.allFiltersBase64));
            this.elem.addEventListener("change", () => this.onSelectChange());
        }

        onSelectChange() {
            const filterName = this.elem.value;
            const filterDisplayName = this.filters[filterName]["display_name"];
            const acceptedColumns = new Set(this.filters[filterName]["columns"]);
            const msg = {
                kind: "COLUMN_FILTER_CHANGED",
                filterName,
                filterDisplayName,
                acceptedColumns,
            };
            this.exchange.send(msg);
            const filterKind = acceptedColumns.size === 0 ? "EMPTY" : "NON_EMPTY";
            this.exchange.send({
                kind: `${filterKind}_COLUMN_FILTER_SELECTED`
            });
        }

        RESET_COLUMN_FILTER() {
            this.elem.value = "all()";
            this.onSelectChange();
        }

    }
    SkrubTableReport.register(ColumnFilter);

    class ResetColumnFilter extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.elem.addEventListener("click", () => this.exchange.send({
                kind: "RESET_COLUMN_FILTER"
            }));
        }
    }
    SkrubTableReport.register(ResetColumnFilter);

    class ColumnFilterName extends Manager {
        COLUMN_FILTER_CHANGED(msg) {
            this.elem.textContent = msg.filterDisplayName;
        }
    }
    SkrubTableReport.register(ColumnFilterName);

    class ColumnFilterMatchCount extends Manager {
        COLUMN_FILTER_CHANGED(msg) {
            this.elem.textContent = msg.acceptedColumns.size.toString();
        }
    }
    SkrubTableReport.register(ColumnFilterMatchCount);

    class SampleColumnSummary extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.elem.querySelector("[data-role='close-button']").addEventListener(
                "click",
                () => this.close());
        }

        close() {
            this.exchange.send({
                kind: "SAMPLE_COLUMN_SUMMARY_CLOSED"
            });
        }

        SAMPLE_TABLE_CELL_ACTIVATED(msg) {
            if (this.matchesColumnIdx(msg.columnIdx)) {
                this.show();
            } else {
                this.hide();
            }
        }

        SAMPLE_TABLE_CELL_DEACTIVATED() {
            this.hide();
        }
    }
    SkrubTableReport.register(SampleColumnSummary);

    class FilterableColumn extends Manager {
        COLUMN_FILTER_CHANGED(msg) {
            if (this.matchesColumnFilter(msg)) {
                delete this.elem.dataset.excludedByColumnFilter;
            } else {
                this.elem.dataset.excludedByColumnFilter = "";
            }
        }
    }
    SkrubTableReport.register(FilterableColumn);

    class SampleTableCell extends Manager {

        constructor(elem, exchange) {
            super(elem, exchange);
            this.elem.addEventListener("click", (event) => {
                this.activate();
                event.preventDefault();
            });
            // See forwardKeyboardEvent for details about captureKeys
            this.elem.dataset.captureKeys =
                "ArrowRight ArrowLeft ArrowUp ArrowDown Escape";
            this.elem.setAttribute("tabindex", -1);
            this.elem.oncopy = (event) => this.copyCell(event);
        }

        copyCell(event) {
            const selection = document.getSelection().toString();
            if (selection !== "") {
                return;
            }
            event.clipboardData.setData("text/plain", this.elem.dataset
                .valueRepr);
            event.preventDefault();
            this.elem.dataset.justCopied = "";
            setTimeout(() => this.elem.removeAttribute("data-just-copied"), 1000);
        }

        activate() {
            const msg = {
                kind: "SAMPLE_TABLE_CELL_ACTIVATED",
                cellId: this.elem.id,
                columnIdx: Number(this.elem.dataset.columnIdx),
                valueStr: this.elem.dataset.valueStr,
                valueRepr: this.elem.dataset.valueRepr,
            };
            this.exchange.send(msg);
        }

        deactivate() {
            if ("isActive" in this.elem.dataset) {
                this.exchange.send({
                    kind: "SAMPLE_TABLE_CELL_DEACTIVATED"
                });
            }
        }

        ACTIVATE_SAMPLE_TABLE_CELL(msg) {
            if (msg.cellId == this.elem.id) {
                this.activate();
            }
        }

        SAMPLE_TABLE_CELL_DEACTIVATED() {
            this.elem.setAttribute("tabindex", -1);
            this.elem.blur();
            delete this.elem.dataset.isActive;
            delete this.elem.dataset.isInActiveColumn;
        }

        SAMPLE_TABLE_CELL_ACTIVATED(msg) {
            if (msg.cellId === this.elem.id) {
                this.elem.dataset.isActive = "";
                this.elem.setAttribute("tabindex", 0);
                this.elem.contentEditable = "true";
                this.elem.focus({});
                this.elem.contentEditable = "false";
            } else {
                delete this.elem.dataset.isActive;
                this.elem.setAttribute("tabindex", -1);
            }
            if (this.matchesColumnIdx(msg.columnIdx)) {
                this.elem.dataset.isInActiveColumn = "";
            } else {
                delete this.elem.dataset.isInActiveColumn;
            }
        }

        SAMPLE_COLUMN_SUMMARY_CLOSED() {
            this.deactivate();
        }

        COLUMN_FILTER_CHANGED(msg) {
            if (this.elem.hasAttribute("data-column-idx") && !this
                .matchesColumnFilter(msg)) {
                this.deactivate();
            }
        }
    }
    SkrubTableReport.register(SampleTableCell);

    class SampleTable extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.root = this.elem.getRootNode();
            this.startI = Number(this.elem.dataset.startI) || 0;
            this.stopI = Number(this.elem.dataset.stopI) || 0;
            this.startJ = Number(this.elem.dataset.startJ) || 0;
            this.stopJ = Number(this.elem.dataset.stopJ) || 0;
            this.elem.addEventListener('keydown', (e) => this.onKeyDown(e));
            this.elem.addEventListener('skrub-keydown', (e) => this.onKeyDown(
                unwrapSkrubKeyDown(e)));
        }

        onKeyDown(event) {
            if (hasModifier(event)) {
                return;
            }
            const cell = event.target;
            let {
                i,
                j
            } = cell.dataset;
            [i, j] = [Number(i), Number(j)];
            if (isNaN(i) || isNaN(j)) {
                return;
            }
            let newCellId = null;
            switch (event.key) {
                case "ArrowLeft":
                    newCellId = this.findCellLeft(cell.id, i, j);
                    break;
                case "ArrowRight":
                    newCellId = this.findCellRight(cell.id, i, j);
                    break;
                case "ArrowUp":
                    newCellId = this.findCellUp(cell.id, i, j);
                    break;
                case "ArrowDown":
                    newCellId = this.findCellDown(cell.id, i, j);
                    break;
                case "Escape":
                    this.exchange.send({
                        kind: "SAMPLE_TABLE_CELL_DEACTIVATED"
                    });
                    event.preventDefault();
                    return;
                default:
                    return;
            }
            if (newCellId !== null) {
                this.exchange.send({
                    kind: "ACTIVATE_SAMPLE_TABLE_CELL",
                    cellId: newCellId
                });
                event.preventDefault();
                return;
            }
        }

        findNextCell(startCellId, i, j, next, stop) {
            [i, j] = next(i, j);
            while (!stop(i, j)) {
                const cell = this.elem.querySelector(`[data-spans__${i}__${j}]`);
                if (cell !== null && cell.id !== startCellId && !cell.hasAttribute(
                        "data-excluded-by-column-filter") && cell.dataset.role !==
                    "padding" && cell.dataset.role !== "ellipsis") {
                    return cell.id;
                }
                [i, j] = next(i, j);
            }
            return null;
        }

        findCellLeft(startCellId, i, j) {
            return this.findNextCell(startCellId, i, j, (i, j) => [i, j - 1], (i,
                j) => (j < this
                .startJ));
        }
        findCellRight(startCellId, i, j) {
            return this.findNextCell(startCellId, i, j, (i, j) => [i, j + 1], (i,
                j) => (this
                .stopJ <= j));
        }

        findCellUp(startCellId, i, j) {
            return this.findNextCell(startCellId, i, j, (i, j) => [i - 1, j], (i,
                j) => (i < this
                .startI));
        }

        findCellDown(startCellId, i, j) {
            return this.findNextCell(startCellId, i, j, (i, j) => [i + 1, j], (i,
                j) => (this
                .stopI <= i));
        }
    }
    SkrubTableReport.register(SampleTable);

    class SampleTableBar extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.display = elem.querySelector("[data-role='content-display']");
            this.displayValues = new Map();
        }

        updateDisplay() {
            this.display.textContent = this.displayValues.get(this.select.value) ||
                "";
        }

        SAMPLE_TABLE_CELL_ACTIVATED(msg) {
            this.display.textContent = msg.valueStr || "";
            this.display.dataset.copyText = msg.valueRepr || "";
        }

        SAMPLE_TABLE_CELL_DEACTIVATED() {
            this.display.textContent = "";
            this.display.removeAttribute("data-copy-text");
        }
    }
    SkrubTableReport.register(SampleTableBar);

    class TabList extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.tabs = new Map();
            this.elem.querySelectorAll("button[data-role='tab']").forEach(
                tab => {
                    const panel = tab.getRootNode().getElementById(tab.dataset
                        .targetPanelId);
                    this.tabs.set(tab, panel);
                    if (!this.firstTab) {
                        this.firstTab = tab;
                    }
                    this.lastTab = tab;
                    tab.addEventListener("click", () => this.selectTab(tab));
                    // See forwardKeyboardEvent for details about captureKeys
                    tab.dataset.captureKeys = "ArrowRight ArrowLeft";
                    tab.addEventListener("keydown", (event) => this.onKeyDown(
                        event));
                    tab.addEventListener("skrub-keydown", (event) => this
                        .onKeyDown(
                            unwrapSkrubKeyDown(event)));
                });
            this.selectTab(this.firstTab, false);
        }

        selectTab(tabToSelect, focus = true) {
            this.tabs.forEach((tabPanel, tab) => {
                delete tab.dataset.isSelected;
                tab.setAttribute("tabindex", -1);
                tabPanel.dataset.hidden = "";
            });
            tabToSelect.dataset.isSelected = "";
            tabToSelect.removeAttribute("tabindex");
            if (focus) {
                tabToSelect.focus();
            }
            delete this.tabs.get(tabToSelect).dataset.hidden;
            this.currentTab = tabToSelect;
            this.exchange.send({
                kind: "TAB_SELECTED",
                targetPanelId: this.currentTab.dataset.targetPanelId
            });
        }

        selectPreviousTab() {
            if (this.currentTab === this.firstTab) {
                this.selectTab(this.lastTab);
                return;
            }
            const keys = [...this.tabs.keys()];
            const idx = keys.indexOf(this.currentTab);
            this.selectTab(keys[idx - 1]);
        }

        selectNextTab() {
            if (this.currentTab === this.lastTab) {
                this.selectTab(this.firstTab);
                return;
            }
            const keys = [...this.tabs.keys()];
            const idx = keys.indexOf(this.currentTab);
            this.selectTab(keys[idx + 1]);
        }

        onKeyDown(event) {
            if (hasModifier(event)) {
                return;
            }
            switch (event.key) {
                case "ArrowLeft":
                    this.selectPreviousTab();
                    break;
                case "ArrowRight":
                    this.selectNextTab();
                    break;
                default:
                    return;
            }
            event.stopPropagation();
            event.preventDefault();
        }
    }
    SkrubTableReport.register(TabList);

    class sortableTable extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.elem.querySelectorAll("button[data-role='sort-button']").forEach(
                b => b.addEventListener("click", e => this.sort(e)));
        }

        getVal(row, tableColIdx) {
            const td = row.querySelectorAll("td")[tableColIdx];
            if (!td.hasAttribute("data-value")) {
                return td.textContent;
            }
            let value = td.dataset.value;
            if (td.hasAttribute("data-numeric")) {
                value = Number(value);
            }
            return value;
        }

        compare(rowA, rowB, tableColIdx, ascending) {
            let valA = this.getVal(rowA, tableColIdx);
            let valB = this.getVal(rowB, tableColIdx);
            // NaNs go at the bottom regardless of sorting order
            if (typeof(valA) === "number" && typeof(valB) === "number") {
                if (isNaN(valA) && !isNaN(valB)) {
                    return 1;
                }
                if (isNaN(valB) && !isNaN(valA)) {
                    return -1;
                }
            }
            // When the values are equal, keep the original dataframe column
            // order
            if (!(valA > valB || valB > valA)) {
                valA = Number(rowA.dataset.dataframeColumnIdx);
                valB = Number(rowB.dataset.dataframeColumnIdx);
                return valA - valB;
            }
            // Sort
            if (!ascending) {
                [valA, valB] = [valB, valA];
            }
            return valA > valB ? 1 : -1;
        }

        sort(event) {
            const colHeaders = Array.from(this.elem.querySelectorAll(
                "thead tr th"));
            const tableColIdx = colHeaders.indexOf(event.target.closest("th"));
            const body = this.elem.querySelector("tbody");
            const rows = Array.from(body.querySelectorAll("tr"));
            const ascending = event.target.dataset.direction === "ascending";

            rows.sort((a, b) => this.compare(a, b, tableColIdx, ascending));

            this.elem.querySelectorAll("button").forEach(b => b.removeAttribute(
                "data-is-active"));
            event.target.dataset.isActive = "";

            body.innerHTML = "";
            for (let r of rows) {
                body.appendChild(r);
            }
        }

    }
    SkrubTableReport.register(sortableTable);

    class SelectedColumnsDisplay extends Manager {

        constructor(elem, exchange) {
            super(elem, exchange);
            this.COLUMN_SELECTION_CHANGED();
        }

        COLUMN_SELECTION_CHANGED() {
            const allColumns = this.elem.getRootNode().querySelectorAll(
                "[data-role='selectable-column']");
            const allColumnsRepr = [];
            for (let col of allColumns) {
                if (col.querySelector("[data-role='select-column-checkbox']")
                    .checked) {
                    allColumnsRepr.push(col.dataset.nameRepr);
                }
            }
            this.elem.textContent = "[" + allColumnsRepr.join(", ") + "]";
        }
    }
    SkrubTableReport.register(SelectedColumnsDisplay);

    class SelectColumnCheckBox extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.elem.addEventListener("change", () => this.exchange.send({
                kind: "COLUMN_SELECTION_CHANGED"
            }));
        }

        DESELECT_ALL_COLUMNS() {
            this.elem.checked = false;
        }

        SELECT_ALL_VISIBLE_COLUMNS() {
            this.elem.checked = !("excludedByColumnFilter" in this.elem.closest(
                "[data-role='selectable-column']").dataset);
        }
    }
    SkrubTableReport.register(SelectColumnCheckBox);

    class DeselectAllColumns extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.elem.addEventListener("click", () => {
                this.exchange.send({
                    kind: "DESELECT_ALL_COLUMNS"
                });
                this.exchange.send({
                    kind: "COLUMN_SELECTION_CHANGED"
                });
            });
        }
    }
    SkrubTableReport.register(DeselectAllColumns);

    class SelectAllVisibleColumns extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.elem.addEventListener("click", () => {
                this.exchange.send({
                    kind: "SELECT_ALL_VISIBLE_COLUMNS"
                });
                this.exchange.send({
                    kind: "COLUMN_SELECTION_CHANGED"
                });
            });
        }
    }
    SkrubTableReport.register(SelectAllVisibleColumns);


    class Toggletip extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.elem.querySelector("button").addEventListener("keydown", e => {
                if (e.key === "Escape") {
                    e.target.blur();
                }
            });
        }
    }
    SkrubTableReport.register(Toggletip);

    function initReport(reportId) {
        const report = document.getElementById(reportId);
        report.init();
    }

    function copyButtonClick(event) {
        const button = event.target;
        button.dataset.showCheckmark = "";
        setTimeout(() => button.removeAttribute("data-show-checkmark"), 2000);
        const textElem = button.getRootNode().getElementById(button.dataset.targetId);
        copyTextToClipboard(textElem);
    }

    function copyTextToClipboard(elem) {
        if (elem.hasAttribute("data-shows-placeholder")) {
            return;
        }
        if (navigator.clipboard) {
            navigator.clipboard.writeText(elem.dataset.copyText || elem.textContent ||
                "");
        } else {
            // fallback when navigator not available. in this case we just copy
            // the text content of the element (we could create a hidden one to
            // fill it with the data-copy-text and select that but this is
            // probably good enough)
            const selection = window.getSelection();
            if (selection == null) {
                return;
            }
            selection.removeAllRanges();
            const range = document.createRange();
            range.selectNodeContents(elem);
            selection.addRange(range);
            document.execCommand("copy");
            selection.removeAllRanges();
        }
    }

    function hasModifier(event) {
        return event.ctrlKey || event.metaKey || event.shiftKey || event.altKey;
    }

    /* Jupyter notebooks and vscode stop the propagation of some keyboard events
    during the capture phase to implement their keyboard shortcuts etc. When an
    element inside the TableReport has the keyboard focus and wants to react on
    that event (eg in the sample table arrow keys allow selecting a neighbouring
    cell) we need to override that behavior.

    We do that by adding an event listener on the window that is triggered
    during the capture phase. If it can make sure the key press is for a report
    element that will react to it, we stop its propagation (to avoid eg the
    notebook jumping to the next jupyter code cell) and dispatch an event on the
    targeted element. To make sure it does not get handled again by the listener
    on the window and cause infinite recursion, we dispatch a custom event
    instead of a KeyDown event.

    This capture is only enabled if we detect the report is inserted in a page
    where it is needed, by checking if there are elements in the page with class
    names that are used by jupyter or vscode.
    */
    function forwardKeyboardEvent(e) {
        if (e.eventPhase !== 1) {
            return;
        }
        if (e.target.tagName !== "SKRUB-TABLE-REPORT") {
            return;
        }
        if (hasModifier(e)) {
            return;
        }
        const target = e.target.shadowRoot.activeElement;
        // only capture the event if the element lists the key in the keys it
        // wants to capture ie in captureKeys
        const wantsKey = target?.dataset.captureKeys?.split(/\s+/).includes(e.key);
        if (!wantsKey) {
            return;
        }
        const newEvent = new CustomEvent('skrub-keydown', {
            bubbles: true,
            cancelable: true,
            detail: {
                key: e.key,
                code: e.code,
                shiftKey: e.shiftKey,
                altKey: e.altKey,
                ctrlKey: e.ctrlKey,
                metaKey: e.metaKey,
            }
        });
        target.dispatchEvent(newEvent);
        e.stopImmediatePropagation();
        e.preventDefault();
    }

    /* Helper to unpack the custom event (see forwardKeyboardEvent above) and
    make it look like a regular KeyDown event. */
    function unwrapSkrubKeyDown(e) {
        return {
            preventDefault: () => {},
            stopPropagation: () => {},
            target: e.target,
            ...e.detail
        };
    }

    if (document.querySelector(".jp-Cell, .widgetarea")) {
        window.addEventListener("keydown", forwardKeyboardEvent, true);
    }

}
