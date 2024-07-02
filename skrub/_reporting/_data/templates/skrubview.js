function updateColSelection(event) {
    updateSelectedColsSnippet(event.target.dataset.reportId);
}

function isSelectedCol(columnElem) {
    const checkboxElem = columnElem.querySelector(
        "input.skrubview-select-column-checkbox[type='checkbox']");
    return checkboxElem && checkboxElem.checked;
}

function updateSelectedColsSnippet(reportId) {
    const reportElem = document.getElementById(reportId);
    const allCols = reportElem.querySelectorAll(".skrubview-column-summary");
    const selectedCols = Array.from(allCols).filter(c => isSelectedCol(c));
    const snippet = selectedCols.map(col => col.dataset.nameRepr).join(", ");
    const bar = reportElem.querySelector(".selected-columns-box");
    bar.textContent = "[" + snippet + "]";
}

function clearSelectedCols(reportId) {
    const reportElem = document.getElementById(reportId);
    reportElem.querySelectorAll(
        "input.skrubview-select-column-checkbox[type='checkbox']").forEach(
        box => {
            box.checked = false;
        }
    );
    updateSelectedColsSnippet(reportId);
}

function selectAllCols(reportId) {
    const reportElem = document.getElementById(reportId);
    reportElem.querySelectorAll(".skrubview-column-summary").forEach(
        elem => {
            const box = elem.querySelector(
                "input.skrubview-select-column-checkbox[type='checkbox']");
            if (!(box === null)) {
                box.checked = !elem.hasAttribute("data-is-excluded-by-filter");
            }
        }
    );
    updateSelectedColsSnippet(reportId);
}

function copyTextToClipboard(elementID) {
    const elem = document.getElementById(elementID);
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

function updateBarContent(barId) {
    const bar = document.getElementById(barId);
    const select = document.getElementById(bar.dataset.selectorId);
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

function updateSiblingBarContents(event) {
    const select = event.target;
    select.parentElement.querySelectorAll(`*[data-selector-id=${select.id}]`).forEach(
        elem => {
            updateBarContent(elem.id);
        })
}

function displayValue(event) {
    const elem = event.target;
    const table = document.getElementById(elem.dataset.parentTableId);
    table.setAttribute("data-selected-column", elem.dataset.colNameStr);
    table.querySelectorAll(".skrubview-table-cell").forEach(cell => {
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

    const topBarId = table.dataset.topBarId;
    const bar = document.getElementById(topBarId);
    const barToggle = bar.closest(".skrubview-top-bar-toggle");
    barToggle.setAttribute("data-predicate", "true");
    bar.setAttribute("data-content-table-cell-value", elem.dataset.valueStr);
    bar.setAttribute("data-content-table-cell-repr", elem.dataset.valueRepr);
    bar.setAttribute("data-content-table-column-name", elem.dataset.colNameStr);
    bar.setAttribute("data-content-table-column-name-repr", elem.dataset.colNameRepr);

    const snippet = filterSnippet(elem.dataset.columnNameRepr,
        elem.dataset.valueRepr,
        elem.hasAttribute("data-value-is-none"),
        elem.dataset.dataframeModule);
    bar.setAttribute(`data-content-table-cell-filter`, snippet);

    revealColCard(table.dataset.reportId, elem.dataset.columnIdx);

    updateBarContent(topBarId);
}

function clearTableCellSelection(tableElem) {
    tableElem.querySelectorAll("th, td").forEach(
        cell => {
            cell.removeAttribute("data-is-selected");
            cell.removeAttribute("data-is-in-selected-column");
        });
    tableElem.removeAttribute("data-selected-cell");
    const topBarId = tableElem.dataset.topBarId;
    const bar = document.getElementById(topBarId);
    const barToggle = bar.closest(".skrubview-top-bar-toggle");
    barToggle.setAttribute("data-predicate", "false");
    bar.removeAttribute("data-content-table-cell-value");
    bar.removeAttribute("data-content-table-cell-repr");
    bar.removeAttribute("data-content-table-column-name");
    bar.removeAttribute("data-content-table-column-name-repr");
    bar.removeAttribute("data-content-table-cell-filter");
    updateBarContent(topBarId);
    revealColCard(tableElem.closest(".skrubview-report").id, null);
}

function displayFirstCellValue(event) {
    const header = event.target;
    const idx = header.dataset.columnIdx;
    const firstCell = header.closest("table").querySelector(
        `.skrubview-table-cell[data-column-idx="${idx}"]`);
    if (firstCell) {
        firstCell.click();
    }
}

function revealColCard(reportId, colIdx) {
    const reportElem = document.getElementById(reportId);
    const allCols = reportElem.querySelectorAll(
        ".skrubview-columns-in-sample-tab .skrubview-column-summary");
    allCols.forEach(col => {
        col.removeAttribute("data-is-selected-in-table");
    });
    if (colIdx === null) {
        return;
    }
    const targetCol = document.getElementById(
        `${reportId}_col_${colIdx}_in_sample_tab`);
    targetCol.dataset.isSelectedInTable = "";

}

function displayTab(event) {
    const elem = event.target;
    elem.parentElement.querySelectorAll("button").forEach(elem => {
        elem.removeAttribute("data-is-selected");
    });
    elem.setAttribute("data-is-selected", "");
    const tab = document.getElementById(elem.dataset.targetTab);
    tab.parentElement.querySelectorAll(".skrubview-tab").forEach(elem => {
        elem.removeAttribute("data-is-displayed");
    });
    tab.setAttribute("data-is-displayed", "");
    if (elem.hasAttribute("data-has-warning")) {
        elem.removeAttribute("data-has-warning");
    }
}

function onFilterChange(colFilterId) {
    const selectElem = document.getElementById(colFilterId);
    const reportId = selectElem.dataset.reportId;
    const colFilters = window[`columnFiltersForReport${reportId}`];
    const filterName = selectElem.value;
    const acceptedCols = colFilters[filterName]["columns"];
    const reportElem = document.getElementById(reportId);
    const colElements = reportElem.querySelectorAll(".skrubview-filterable-column");
    colElements.forEach(elem => {
        if (acceptedCols.includes(elem.dataset.columnName)) {
            elem.removeAttribute("data-is-excluded-by-filter");
        } else {
            elem.dataset.isExcludedByFilter = "";
        }
    });
    document.getElementById(`${reportId}_display_n_columns`).textContent = acceptedCols
        .length.toString();
    const tableElem = reportElem.querySelector(".skrubview-dataframe-sample-table");
    if (!acceptedCols.includes(tableElem.dataset.selectedColumn)) {
        clearTableCellSelection(tableElem);
    }
    for (let toggleSelector of [".skrubview-table-sample-toggle",
            ".skrubview-column-summaries-toggle"
        ]) {
        const toggle = reportElem.querySelector(toggleSelector);
        toggle.dataset.predicate = acceptedCols.length === 0 ? "false" : "true";
        const filterDisplay = toggle.querySelector(
            ".skrubview-selected-filter-display");
        filterDisplay.textContent = '"' + colFilters[filterName]["display_name"] + '"';
    }
}

function clearColFilter(event){
    const reportElem = event.target.closest(".skrubview-report");
    const selectElem = reportElem.querySelector(".skrubview-col-filter-select");
    selectElem.value = "all()";
    onFilterChange(selectElem.id);
}
