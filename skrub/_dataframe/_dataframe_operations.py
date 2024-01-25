from ._dataframe_api import asdfapi, asnative, dfapi_ns

__all__ = [
    "is_numeric",
    "is_temporal",
    "numeric_column_names",
    "temporal_column_names",
    "select",
    "set_column_names",
    "collect",
]


def is_numeric(column, include_bool=True):
    column = asdfapi(column)
    ns = dfapi_ns(column)
    if ns.is_dtype(column.dtype, "numeric"):
        return True
    if not include_bool:
        return False
    return ns.is_dtype(column.dtype, "bool")


def is_temporal(column):
    column = asdfapi(column)
    ns = dfapi_ns(column)
    for dtype in [ns.Date, ns.Datetime]:
        if isinstance(column.dtype, dtype):
            return True
    return False


def _select_column_names(df, predicate):
    df = asdfapi(df)
    return [col_name for col_name in df.column_names if predicate(df.col(col_name))]


def numeric_column_names(df):
    return _select_column_names(df, is_numeric)


def temporal_column_names(df):
    return _select_column_names(df, is_temporal)


def select(df, column_names):
    return asnative(asdfapi(df).select(*column_names))


def collect(df):
    if hasattr(df, "collect"):
        return df.collect()
    return df


def set_column_names(df, new_column_names):
    df = asdfapi(df)
    ns = dfapi_ns(df)
    new_columns = (
        df.col(col_name).rename(new_name).persist()
        for (col_name, new_name) in zip(df.column_names, new_column_names)
    )
    return asnative(ns.dataframe_from_columns(*new_columns))
