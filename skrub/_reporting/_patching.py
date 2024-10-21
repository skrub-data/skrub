import importlib

from ._table_report import TableReport

__all__ = ["patch_display", "unpatch_display"]

_METHODS_TO_PATCH = ["_repr_mimebundle_", "_repr_html_"]


def _stashed_name(method_name):
    return f"_skrub_{method_name}"


def _patch(cls, method_name):
    if (original_method := getattr(cls, method_name, None)) is None:
        return
    stashed_name = _stashed_name(method_name)
    if not hasattr(cls, stashed_name):
        setattr(cls, stashed_name, original_method)
    setattr(cls, method_name, lambda df: getattr(TableReport(df), method_name)())


def _unpatch(cls, method_name):
    stashed_name = _stashed_name(method_name)
    if (original_method := getattr(cls, stashed_name, None)) is None:
        return
    setattr(cls, method_name, original_method)


def _change_display(transform, to_patch):
    for module_name, class_names in to_patch:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        for cls_name in class_names:
            cls = getattr(mod, cls_name)
            for method_name in _METHODS_TO_PATCH:
                transform(cls, method_name)


def _get_to_patch(pandas, polars):
    to_patch = []
    if pandas:
        to_patch.append(("pandas", ["DataFrame"]))
    if polars:
        to_patch.append(("polars", ["DataFrame"]))
    return to_patch


def patch_display(pandas=True, polars=True):
    """Replace the default DataFrame HTML displays with ``skrub.TableReport``.

    This function replaces the HTML displays (what is shown when an object is
    the output of a jupyter notebook cell) of pandas and polars DataFrames
    with a TableReport.

    It can be undone with ``skrub.unpatch_display()``.

    Parameters
    ----------
    pandas : bool, optional (default=True)
        If False, do not override the displays for pandas dataframes.
    polars : bool, optional (default=True)
        If False, do not override the displays for polars dataframes.

    See Also
    --------
    unpatch_display :
        Undo the change made by this function.

    TableReport :
        Directly create a report from a dataframe.
    """
    _change_display(_patch, _get_to_patch(pandas=pandas, polars=polars))


def unpatch_display(pandas=True, polars=True):
    """Undo the effect of ``skrub.patch_display()``.

    This function restores the default HTML displays of pandas and polars
    DataFrames.

    Parameters
    ----------
    pandas : bool, optional (default=True)
        If False, do not restore the displays for pandas dataframes.
    polars : bool, optional (default=True)
        If False, do not restore the displays for polars dataframes.

    See Also
    --------
    patch_display :
        Replace the default dataframe display with a TableReport.

    TableReport :
        Directly create a report from a dataframe.
    """
    _change_display(_unpatch, _get_to_patch(pandas=pandas, polars=polars))
