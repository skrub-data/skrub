import importlib

from ._table_report import TableReport

_METHODS_TO_PATCH = ["_repr_mimebundle_", "_repr_html_"]


def _stashed_name(method_name):
    return f"_skrub_{method_name}"


def _patch(cls, method_name, verbose, max_plot_columns, max_association_columns):
    if (original_method := getattr(cls, method_name, None)) is None:
        return
    stashed_name = _stashed_name(method_name)
    if not hasattr(cls, stashed_name):
        setattr(cls, stashed_name, original_method)
    setattr(
        cls,
        method_name,
        lambda df: getattr(
            TableReport(
                df,
                verbose=verbose,
                max_plot_columns=max_plot_columns,
                max_association_columns=max_association_columns,
            ),
            method_name,
        )(),
    )


def _unpatch(cls, method_name):
    stashed_name = _stashed_name(method_name)
    if (original_method := getattr(cls, stashed_name, None)) is None:
        return
    setattr(cls, method_name, original_method)


def _change_display(transform, to_patch, **transform_kwargs):
    for module_name, class_names in to_patch:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        for cls_name in class_names:
            cls = getattr(mod, cls_name)
            for method_name in _METHODS_TO_PATCH:
                transform(cls, method_name, **transform_kwargs)


def _get_to_patch(pandas, polars):
    to_patch = []
    if pandas:
        to_patch.append(("pandas", ["DataFrame"]))
    if polars:
        to_patch.append(("polars", ["DataFrame"]))
    return to_patch


def _patch_display(
    pandas=True, polars=True, verbose=1, max_plot_columns=30, max_association_columns=30
):
    _change_display(
        _patch,
        _get_to_patch(pandas=pandas, polars=polars),
        verbose=verbose,
        max_plot_columns=max_plot_columns,
        max_association_columns=max_association_columns,
    )


def _unpatch_display(pandas=True, polars=True):
    _change_display(_unpatch, _get_to_patch(pandas=pandas, polars=polars))
