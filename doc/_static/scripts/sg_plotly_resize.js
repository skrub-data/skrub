// Related to https://github.com/scikit-learn/scikit-learn/issues/30279
// There an interaction between plotly and bootstrap/pydata-sphinx-theme
// that causes plotly figures to not detect the right-hand sidebar width

// plotly figures are responsive so we just trigger a resize event when the dom
// loads and they will resize themselves.

document.addEventListener("DOMContentLoaded", () => {
    window.dispatchEvent(new Event('resize'));
});
