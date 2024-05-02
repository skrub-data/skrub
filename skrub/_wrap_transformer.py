from ._on_each_column import OnEachColumn
from ._on_subframe import OnSubFrame
from ._selectors import make_selector


def wrap_transformer(
    transformer,
    selector,
    keep_original=False,
    rename_columns="{}",
    n_jobs=None,
    columnwise="auto",
):
    selector = make_selector(selector)

    if isinstance(columnwise, str) and columnwise == "auto":
        columnwise = hasattr(transformer, "__single_column_transformer__")

    if columnwise:
        return OnEachColumn(
            transformer,
            keep_original=keep_original,
            rename_columns=rename_columns,
            cols=selector,
            n_jobs=n_jobs,
        )
    return OnSubFrame(
        transformer,
        keep_original=keep_original,
        rename_columns=rename_columns,
        cols=selector,
    )
