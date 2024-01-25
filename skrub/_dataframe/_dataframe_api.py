def asdfapi(obj):
    if hasattr(obj, "__dataframe_namespace__"):
        return obj
    if hasattr(obj, "__column_namespace__"):
        return obj
    if hasattr(obj, "__dataframe_consortium_standard__"):
        return obj.__dataframe_consortium_standard__()
    if hasattr(obj, "__column_consortium_standard__"):
        return obj.__column_consortium_standard__()
    try:
        return _asdfapi_old_pandas(obj)
    except (ImportError, TypeError):
        pass
    try:
        return _asdfapi_old_polars(obj)
    except (ImportError, TypeError):
        pass
    raise TypeError(
        f"{obj} cannot be converted to DataFrame Consortium Standard object."
    )


def _asdfapi_old_pandas(obj):
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        from dataframe_api_compat.pandas_standard import (
            convert_to_standard_compliant_dataframe,
        )

        return convert_to_standard_compliant_dataframe(obj)
    if isinstance(obj, pd.Series):
        from dataframe_api_compat.pandas_standard import (
            convert_to_standard_compliant_column,
        )

        return convert_to_standard_compliant_column(obj)
    raise TypeError()


def _asdfapi_old_polars(obj):
    import polars as pl

    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        from dataframe_api_compat.polars_standard import (
            convert_to_standard_compliant_dataframe,
        )

        return convert_to_standard_compliant_dataframe(obj)
    if isinstance(obj, pl.Series):
        from dataframe_api_compat.polars_standard import (
            convert_to_standard_compliant_column,
        )

        return convert_to_standard_compliant_column(obj)
    raise TypeError()


def asnative(obj):
    if hasattr(obj, "__dataframe_namespace__"):
        return obj.dataframe
    if hasattr(obj, "__column_namespace__"):
        return obj.column
    if hasattr(obj, "__scalar_namespace__"):
        return obj.scalar
    return obj


def dfapi_ns(obj):
    obj = asdfapi(obj)
    for attr in [
        "__dataframe_namespace__",
        "__column_namespace__",
        "__scalar_namespace__",
    ]:
        if hasattr(obj, attr):
            return getattr(obj, attr)()
    raise TypeError(f"{obj} is not a Dataframe Consortium Standard object.")
