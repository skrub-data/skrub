if (customElements.get('skrub-table-report') === undefined) {

    class SkrubTableReport extends HTMLElement {
        constructor() {
            super();
        }

        connectedCallback() {
            const template = document.getElementById(`${this.id}-template`);
            this.attachShadow({
                mode: "open"
            });
            this.shadowRoot.appendChild(template.content.cloneNode(true));
        }

        updateSelectedColsSnippet() {
            const allCols = this.shadowRoot.querySelectorAll(
                ".column-summary");
            const selectedCols = Array.from(allCols).filter(c => this.isSelectedCol(
                c));
            const snippet = selectedCols.map(col => col.dataset.nameRepr).join(
                ", ");
            const bar = this.shadowRoot.querySelector(".selected-columns-box");
            bar.textContent = "[" + snippet + "]";
        }

        isSelectedCol(columnElem) {
            const checkboxElem = columnElem.querySelector(
                "input.select-column-checkbox[type='checkbox']");
            return checkboxElem && checkboxElem.checked;
        }

        updateBarContent(barId) {
            const bar = this.shadowRoot.getElementById(barId);
            const select = this.shadowRoot.getElementById(bar.dataset.selectorId);
            const selectedOption = select.options[select.selectedIndex];
            const selectedOptionValue = selectedOption.value;
            const contentAttribute = `data-content-${selectedOptionValue}`;
            if (!bar.hasAttribute(contentAttribute)) {
                bar.textContent = selectedOption.dataset.placeholder;
                bar.dataset.showsPlaceholder = "";
            } else {
                bar.textContent = bar.getAttribute(contentAttribute);
                bar.removeAttribute("data-shows-placeholder");
            }
        }

        clearSelectedCols() {
            this.shadowRoot.querySelectorAll(
                "input.select-column-checkbox[type='checkbox']").forEach(
                box => {
                    box.checked = false;
                }
            );
            this.updateSelectedColsSnippet();
        }

        selectAllCols() {
            this.shadowRoot.querySelectorAll(".column-summary").forEach(
                elem => {
                    const box = elem.querySelector(
                        "input.select-column-checkbox[type='checkbox']"
                    );
                    if (!(box === null)) {
                        box.checked = !elem.hasAttribute(
                            "data-is-excluded-by-filter");
                    }
                }
            );
            this.updateSelectedColsSnippet();
        }

        displayTab(tabButtonId, removeWarnings = false) {
            const tabButton = this.shadowRoot.getElementById(tabButtonId);
            const tab = this.shadowRoot.getElementById(tabButton.dataset.targetTab);
            tabButton.parentElement.querySelectorAll("button").forEach(elem => {
                elem.removeAttribute("data-is-selected");
            });
            tab.parentElement.querySelectorAll(".tab").forEach(elem => {
                elem.removeAttribute("data-is-displayed");
            });
            tabButton.setAttribute("data-is-selected", "");
            tab.setAttribute("data-is-displayed", "");
            if (removeWarnings && tabButton.hasAttribute("data-has-warning")) {
                tabButton.removeAttribute("data-has-warning");
            }
            const filterSelect = this.shadowRoot.getElementById("col-filter-select-wrapper");
            if (tab.id === "interactions-tab"){
                filterSelect.dataset.notVisible = "";
            } else {
                delete filterSelect.dataset.notVisible;
            }
        }

        displayValue(event) {
            const elem = event.target;
            const table = elem.closest("table");
            table.setAttribute("data-selected-column", elem.dataset.colNameStr);
            table.querySelectorAll(".table-cell").forEach(cell => {
                cell.removeAttribute("data-is-selected");
                if (cell.dataset.columnIdx === elem.dataset.columnIdx) {
                    cell.setAttribute("data-is-in-selected-column", "");
                } else {
                    cell.removeAttribute("data-is-in-selected-column");
                }
            });
            table.querySelectorAll("th").forEach(head => {
                if (head.dataset.columnIdx === elem.dataset.columnIdx) {
                    head.setAttribute("data-is-in-selected-column", "");
                } else {
                    head.removeAttribute("data-is-in-selected-column");
                }
            });
            elem.setAttribute("data-is-selected", "");

            const bar = this.shadowRoot.getElementById("top-bar");
            const barToggle = bar.closest(".top-bar-toggle");
            barToggle.setAttribute("data-predicate", "true");
            bar.setAttribute("data-content-table-cell-value", elem.dataset
                .valueStr);
            bar.setAttribute("data-content-table-cell-repr", elem.dataset
                .valueRepr);
            bar.setAttribute("data-content-table-column-name", elem.dataset
                .colNameStr);
            bar.setAttribute("data-content-table-column-name-repr", elem.dataset
                .colNameRepr);

            const snippet = filterSnippet(elem.dataset.colNameRepr,
                elem.dataset.valueRepr,
                elem.hasAttribute("data-value-is-none"),
                elem.dataset.dataframeModule);
            bar.setAttribute(`data-content-table-cell-filter`, snippet);

            this.revealColCard(elem.dataset.columnIdx);
            this.updateBarContent("top-bar");
        }

        revealColCard(colIdx) {
            const allCols = this.shadowRoot.querySelectorAll(
                ".columns-in-sample-tab .column-summary");
            allCols.forEach(col => {
                col.removeAttribute("data-is-selected-in-table");
            });
            if (colIdx === null) {
                return;
            }
            const targetCol = this.shadowRoot.getElementById(
                `col_${colIdx}_in_sample_tab`);
            targetCol.dataset.isSelectedInTable = "";

        }

        clearTableCellSelection() {
            const tableElem = this.shadowRoot.getElementById("sample-table");
            tableElem.querySelectorAll("th, td").forEach(
                cell => {
                    cell.removeAttribute("data-is-selected");
                    cell.removeAttribute("data-is-in-selected-column");
                });
            tableElem.removeAttribute("data-selected-cell");
            const bar = this.shadowRoot.getElementById("top-bar");
            const barToggle = bar.closest(".top-bar-toggle");
            barToggle.setAttribute("data-predicate", "false");
            bar.removeAttribute("data-content-table-cell-value");
            bar.removeAttribute("data-content-table-cell-repr");
            bar.removeAttribute("data-content-table-column-name");
            bar.removeAttribute("data-content-table-column-name-repr");
            bar.removeAttribute("data-content-table-cell-filter");
            this.updateBarContent("top-bar");
            this.revealColCard(null);
        }

        onFilterChange() {
            const selectElem = this.shadowRoot.getElementById("col-filter-select");
            const colFilters = window[`columnFiltersForReport${this.id}`];
            const filterName = selectElem.value;
            const acceptedCols = colFilters[filterName]["columns"];
            const colElements = this.shadowRoot.querySelectorAll(
                ".filterable-column");
            colElements.forEach(elem => {
                if (acceptedCols.includes(elem.dataset.columnName)) {
                    elem.removeAttribute("data-is-excluded-by-filter");
                } else {
                    elem.dataset.isExcludedByFilter = "";
                }
            });
            this.shadowRoot.getElementById("display-n-columns").textContent =
                acceptedCols
                .length.toString();
            const tableElem = this.shadowRoot.getElementById("sample-table");
            if (!acceptedCols.includes(tableElem.dataset.selectedColumn)) {
                this.clearTableCellSelection();
            }
            for (let toggleSelector of [".table-sample-toggle",
                    ".column-summaries-toggle"
                ]) {
                const toggle = this.shadowRoot.querySelector(toggleSelector);
                toggle.dataset.predicate = acceptedCols.length === 0 ? "false" :
                    "true";
                const filterDisplay = toggle.querySelector(
                    ".selected-filter-display");
                filterDisplay.textContent = '"' + colFilters[filterName][
                    "display_name"
                ] + '"';
            }
        }


        clearColFilter(event) {
            const selectElem = this.shadowRoot.getElementById("col-filter-select");
            selectElem.value = "all()";
            this.onFilterChange();
        }

    }

    customElements.define("skrub-table-report", SkrubTableReport);

    function initReport(reportId) {
        const report = document.getElementById(reportId);
        report.updateBarContent("top-bar");
        report.updateSelectedColsSnippet();
        report.displayTab("sample-tab-button");
        report.onFilterChange();
    }

    function displayTab(event) {
        const button = event.target;
        button.getRootNode().host.displayTab(button.id);
    }


    function copyButtonClick(event) {
        const button = event.target;
        const textElem = button.getRootNode().getElementById(button.dataset.targetId);
        copyTextToClipboard(textElem);
    }

    function copyTextToClipboard(elem) {
        if (elem.hasAttribute("data-shows-placeholder")) {
            return;
        }
        elem.setAttribute("data-is-being-copied", "");
        if (navigator.clipboard) {
            navigator.clipboard.writeText(elem.textContent || "");
        } else {
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

        setTimeout(() => {
            elem.removeAttribute("data-is-being-copied");
        }, 200);
    }

    function pandasFilterSnippet(colName, value, valueIsNone) {
        if (valueIsNone) {
            return `df.loc[df[${colName}].isnull()]`;
        }
        return `df.loc[df[${colName}] == ${value}]`;
    }

    function polarsFilterSnippet(colName, value, valueIsNone) {
        if (valueIsNone) {
            return `df.filter(pl.col(${colName}).is_null())`;
        }
        return `df.filter(pl.col(${colName}) == ${value})`;
    }

    function filterSnippet(colName, value, valueIsNone, dataframeModule) {
        if (dataframeModule === "polars") {
            return polarsFilterSnippet(colName, value, valueIsNone);
        }
        if (dataframeModule === "pandas") {
            return pandasFilterSnippet(colName, value, valueIsNone);
        }
        return `Unknown dataframe library: ${dataframeModule}`;
    }

    function updateSiblingBarContents(event) {
        const select = event.target;
        const report = select.getRootNode().host;
        select.parentElement.querySelectorAll(`*[data-selector-id=${select.id}]`)
            .forEach(
                elem => {
                    report.updateBarContent(elem.id);
                });
    }

    function displayFirstCellValue(event) {
        const header = event.target;
        const idx = header.dataset.columnIdx;
        const firstCell = header.closest("table").querySelector(
            `.table-cell[data-column-idx="${idx}"]`);
        if (firstCell) {
            firstCell.click();
        }
    }

    function getReport(event) {
        return event.target.getRootNode().host;
    }

}
