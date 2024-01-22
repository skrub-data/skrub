import functools

import pandas as pd

try:
    import polars as pl

    POLARS_INSTALLED = True
except ImportError:
    POLARS_INSTALLED = False

from . import _pandas

if POLARS_INSTALLED:
    from . import _polars


def asdfapi(obj):
    if hasattr(obj, "__dataframe_namespace__"):
        return obj
    if hasattr(obj, "__column_namespace__"):
        return obj
    if hasattr(obj, "__dataframe_consortium_standard__"):
        return obj.__dataframe_consortium_standard__()
    if hasattr(obj, "__column_consortium_standard__"):
        return obj.__column_consortium_standard__()
    raise TypeError(
        f"{obj} cannot be converted to DataFrame Consortium Standard object."
    )


def asnative(obj):
    if hasattr(obj, "__dataframe_namespace__"):
        return obj.dataframe
    if hasattr(obj, "__column_namespace__"):
        return obj.column
    if hasattr(obj, "__scalar_namespace__"):
        return obj.scalar
    return obj


def dfns(obj):
    obj = asdfapi(obj)
    for attr in [
        "__dataframe_namespace__",
        "__column_namespace__",
        "__scalar_namespace__",
    ]:
        if hasattr(obj, attr):
            return getattr(obj, attr)()
    raise TypeError(f"{obj} is not a Dataframe Consortium Standard object.")


_COLUMN_DISPATCHED = []
_DATAFRAME_DISPATCHED = []


def _dispatch_column(fun):
    fun = functools.singledispatch(fun)
    _COLUMN_DISPATCHED.append(fun)
    return fun


def _dispatch_dataframe(fun):
    fun = functools.singledispatch(fun)
    _DATAFRAME_DISPATCHED.append(fun)
    return fun


@_dispatch_dataframe
@_dispatch_column
def skrub_namespace(obj):
    raise NotImplementedError()


@_dispatch_column
def is_numeric(column):
    column = asdfapi(column)
    ns = dfns(column)
    return ns.is_dtype(column, "numeric")


@_dispatch_column
def is_categorical(column):
    raise NotImplementedError()


@_dispatch_column
def to_categorical(column):
    raise NotImplementedError()


@_dispatch_column
def is_datetime(column):
    column = asdfapi(column)
    ns = dfns(column)
    return ns.is_dtype(column, (ns.Date, ns.Datetime, ns.Duration))


@_dispatch_column
def to_datetime(column, format):
    raise NotImplementedError()


@_dispatch_column
def unique(column):
    column = asdfapi(column)
    unique_values = column.take(column.unique_indices(skip_nulls=False))
    return asnative(unique_values)


@_dispatch_column
def native_dtype(column):
    return column.dtype


@_dispatch_column
def native_cast(column, dtype):
    raise NotImplementedError()


@_dispatch_column
def where(column, mask, other):
    raise NotImplementedError()


for fun in _COLUMN_DISPATCHED:
    if hasattr(_pandas, fun.__name__):
        fun.register(pd.Series, getattr(_pandas, fun.__name__))

if POLARS_INSTALLED:
    for fun in _COLUMN_DISPATCHED:
        if hasattr(_polars, fun.__name__):
            fun.register(pl.Series, getattr(_polars, fun.__name__))

for fun in _DATAFRAME_DISPATCHED:
    if hasattr(_pandas, fun.__name__):
        fun.register(pd.DataFrame, getattr(_pandas, fun.__name__))

if POLARS_INSTALLED:
    for fun in _DATAFRAME_DISPATCHED:
        if hasattr(_polars, fun.__name__):
            fun.register(pl.DataFrame, getattr(_polars, fun.__name__))
            fun.register(pl.LazyFrame, getattr(_polars, fun.__name__))
