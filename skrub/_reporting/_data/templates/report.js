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

        matchesColumnFilter({
            acceptedColumns
        }) {
            return acceptedColumns.has(this.elem.dataset.columnName);
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
            if (msg.columnName === this.elem.dataset.columnName) {
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
                columnName: this.elem.dataset.columnName,
                columnIdx: this.elem.dataset.columnIdx,
                valueStr: this.elem.dataset.valueStr,
                valueRepr: this.elem.dataset.valueRepr,
                columnNameStr: this.elem.dataset.colNameStr,
                columnNameRepr: this.elem.dataset.colNameRepr,
                dataframeModule: this.elem.dataset.dataframeModule,
                valueIsNone: "valueIsNone" in this.elem.dataset,
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
            if (msg.columnName === this.elem.dataset.columnName) {
                this.elem.dataset.isInActiveColumn = "";
            } else {
                delete this.elem.dataset.isInActiveColumn;
            }
        }

        SAMPLE_COLUMN_SUMMARY_CLOSED() {
            this.deactivate();
        }

        COLUMN_FILTER_CHANGED(msg) {
            if (!this.matchesColumnFilter(msg)) {
                this.deactivate();
            }
        }
    }
    SkrubTableReport.register(SampleTableCell);

    class SampleTable extends Manager {
        constructor(elem, exchange) {
            super(elem, exchange);
            this.root = this.elem.getRootNode();
            this.nHeadRows = this.elem.dataset.nHeadRows;
            this.nTailRows = this.elem.dataset.nTailRows;
            this.nCols = this.elem.dataset.nCols;
            this.elem.addEventListener('keydown', (e) => this.onKeyDown(e));
        }

        onKeyDown(event) {
            if (hasModifier(event)) {
                return;
            }
            const cell = event.target;
            let {
                tablePart,
                rowIdxInTablePart: row,
                columnIdx: col
            } = cell.dataset;
            if (tablePart === undefined || row === undefined || col === undefined) {
                return;
            }
            [row, col] = [Number(row), Number(col)];
            let newCellId = null;
            switch (event.key) {
                case "ArrowLeft":
                    newCellId = this.findCellLeft(tablePart, row, col);
                    break;
                case "ArrowRight":
                    newCellId = this.findCellRight(tablePart, row, col);
                    break;
                case "ArrowUp":
                    newCellId = this.findCellUp(tablePart, row, col);
                    break;
                case "ArrowDown":
                    newCellId = this.findCellDown(tablePart, row, col);
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

        rowName(row) {
            return row === -1 ? "header" : String(row);
        }

        findCellLeft(tablePart, row, col) {
            let newCol = col;
            while (newCol > 0) {
                newCol -= 1;
                let newCellId =
                    `sample-table-cell-${tablePart}-${this.rowName(row)}-${newCol}`;
                let newCell = this.root.getElementById(newCellId);
                if (newCell === null) {
                    return null;
                }
                if ("excludedByColumnFilter" in newCell.dataset) {
                    continue;
                }
                return newCell.id;
            }
            return null;
        }

        findCellRight(tablePart, row, col) {
            let newCol = col;
            while (newCol < this.nCols - 1) {
                newCol += 1;
                let newCellId =
                    `sample-table-cell-${tablePart}-${this.rowName(row)}-${newCol}`;
                let newCell = this.root.getElementById(newCellId);
                if (newCell === null) {
                    return null;
                }
                if ("excludedByColumnFilter" in newCell.dataset) {
                    continue;
                }
                return newCell.id;
            }
            return null;
        }

        findCellDown(tablePart, row, col) {
            let newRow = row;
            let newTablePart = tablePart;
            while (newTablePart === "head" || newRow < this.nTailRows - 1) {
                if (newTablePart === "head" && newRow === this.nHeadRows - 1) {
                    if (this.nTailRows === 0) {
                        return null;
                    }
                    newTablePart = "tail";
                    newRow = 0;
                } else {
                    newRow += 1;
                }
                let newCellId =
                    `sample-table-cell-${newTablePart}-${this.rowName(newRow)}-${col}`;
                let newCell = this.root.getElementById(newCellId);
                if (newCell === null) {
                    return null;
                }
                return newCell.id;
            }
            return null;
        }

        findCellUp(tablePart, row, col) {
            let newRow = row;
            let newTablePart = tablePart;
            while (newTablePart === "tail" || newRow > -1) {
                if (newTablePart === "tail" && newRow === 0) {
                    newTablePart = "head";
                    newRow = this.nHeadRows - 1;
                } else {
                    newRow -= 1;
                }
                let newCellId =
                    `sample-table-cell-${newTablePart}-${this.rowName(newRow)}-${col}`;
                let newCell = this.root.getElementById(newCellId);
                if (newCell === null) {
                    return null;
                }
                return newCell.id;
            }
            return null;
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
                    tab.addEventListener("keydown", (event) => this.onKeyDown(
                        event));
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
}
