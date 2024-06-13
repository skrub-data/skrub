"""
Allow specializing a function for different dataframe libraries.

Many examples of functions using this decorator can be seen in the
``skrub._dataframe._common`` module.

When this decorator is applied to a function, the function becomes a generic
function for which we can register several implementations. The implementation
to use is chosen when the function is called, based on the type of the
**first argument**.

The decorator is a thin wrapper around the standard library's
``functools.singledispatch``. The difference is that we register specializations
by providing the dataframe library's name as a string, rather than providing an
actual type. This is necessary because some backends (ATM, only polars) are
optional dependencies and may not be installed. Therefore, it may not be
possible to import the type ``polars.DataFrame`` and use it to register an
implementation for a standard ``@singledispatch`` function.

Once ``dispatch`` has been applied to a function, it has a ``specialize``
attribute that can be used to register implementations.

>>> from skrub._dispatch import dispatch

>>> @dispatch
... def my_function(df, other_arg=1.0):
...     # here we can either provide a default implementation or:
...     raise NotImplementedError()

``my_function`` now has a ``specialize`` attribute we can use to register
implementations:

>>> @my_function.specialize("pandas")
... def _my_function_pandas(df, other_arg=1.0):
...     print("calling pandas implementation")

The name ``_my_function_pandas`` is arbitrary, we could use anything such as
``_`` (unless we want to be able to call the specialized implementation
directly, e.g. for testing).

The tentative convention in skrub is to use
``f"_{generic_function_name.lstrip('_')}_{library_name}"``.

Note that we registered this implementation without needing to import pandas.
In the same way, we can define an implementation for polars even if it is not
installed.

>>> @my_function.specialize("polars")
... def _my_function_polars(df, other_arg=1.0):
...     print("calling polars implementation")

Now when we call ``my_function`` with a dataframe, the correct implementation is
chosen based on the type of the first argument.

>>> import pandas as pd
>>> df = pd.DataFrame(dict(a=[3, 4]))
>>> my_function(df)
calling pandas implementation

Inside a specialized implementation, it is safe to import the corresponding
module or to call methods on the first argument that only exist for this
specific type.


>>> @my_function.specialize("polars")
... def _my_function_polars(df, other_arg=1.0):
...     print("calling polars implementation")
...     # we know df is a polars DataFrame, LazyFrame or Series (see how to
...     # specialize for only one of those types below)
...     df.shape             # ok
...     df.to_pandas()       # ok
...     import polars as pl  # ok


Outside such specialization, skrub code **must not** directly access methods of
a container, as this would rely on backend-specific behavior:


>>> def my_other_function(df):
...     # not ok: will fail if df is a polars DataFrame
...     df.values
...     # not ok: only happens to work by chance because both polars and pandas use
...     # the same name with the same signature and semantics, but could fail for
...     # other backends added in the future
...     df.shape
...     # ok
...     import skrub._dataframe as ns
...     ns.shape(df)


We can inspect which implementations were registered in ``my_function.registry``.

It is also possible to register a specialization specifically for dataframes or
for columns:

>>> @my_function.specialize("pandas", argument_type="Column")
... def _my_function_pandas_column(df, other_arg=1.0):
...     print("calling pandas Series implementation")
...     print("               ^^^^^^               ")

>>> my_function(pd.Series([0]))
calling pandas Series implementation
               ^^^^^^
>>> my_function(pd.DataFrame())
calling pandas implementation

Note that the last registered implementation wins -- there is no magic to
prioritize specializations based on how specific they are.

>>> @my_function.specialize("pandas")
... def _override_all_pandas_specializations(df, other_arg=1.0):
...     print("Any pandas (Series or DataFrame)")

>>> my_function(pd.Series([0]))
Any pandas (Series or DataFrame)
>>> my_function(pd.DataFrame())
Any pandas (Series or DataFrame)

Importantly, single dispatch is done based on the type of the
**first argument**. This means our generic function should be written in such a way
that the first argument is the dataframe or column that will determine which
implementation to use. For example the following would not work:

>>> @dispatch
... def sample(n_rows, df):
...     ...

Instead we must rewrite it to be:

>>> @dispatch
... def sample(df, n_rows):
...     ...
"""

from dataclasses import dataclass
from functools import singledispatch
from types import MappingProxyType, ModuleType
from typing import Any, Dict, Tuple


@dataclass
class DataFrameModuleInfo:
    name: str
    module: ModuleType
    types: Dict[str, Tuple[Any]]


def _load_dataframe_module_info(name):
    # if the module is not installed, import errors are propagated
    if name == "pandas":
        import pandas

        return DataFrameModuleInfo(
            **{
                "name": "pandas",
                "module": pandas,
                "types": MappingProxyType(
                    {
                        "DataFrame": (pandas.DataFrame,),
                        "Column": (pandas.Series,),
                    }
                ),
            }
        )
    if name == "polars":
        import polars

        return DataFrameModuleInfo(
            **{
                "name": "polars",
                "module": polars,
                "types": MappingProxyType(
                    {
                        "DataFrame": (polars.DataFrame, polars.LazyFrame),
                        "LazyFrame": (polars.LazyFrame,),
                        "EagerFrame": (polars.DataFrame,),
                        "Column": (polars.Series,),
                    }
                ),
            }
        )
    raise KeyError(
        f"Unknown dataframe module: {name}. "
        "Available modules are ['pandas' and 'polars']."
    )


def dispatch(function):
    """Make a generic function that performs dispatch on the first argument's type.

    The returned value is a generic function whose ``specialize`` attribute can
    be used to register implementations specialized for different dataframe
    modules (pandas, polars). See this module's docstring for more details and
    examples.
    """
    # Apply functools.singledispatch
    dispatched = singledispatch(function)

    # Now we can register implementations with ``dispatched.register``.
    # However that requires a type, and some of the types we want to register
    # may not be importable (eg polars.DataFrame if polars is not installed).
    # Therefore for convenience we add a ``specialize`` attribute that accepts
    # strings instead of types and calls ``register`` with the appropriate
    # types if they can be imported.

    def specialize(module_name, *, argument_type=None):
        # Build a decorator responsible for adding a new specialized
        # implementation to ``dispatched``'s registry. It accepts the module
        # name and an optional list of generic type names (as strings) to
        # register. If ``argument_type`` is ``None``, all type names for the
        # module are used.
        try:
            module_info = _load_dataframe_module_info(module_name)
        except ImportError:
            # The module cannot be imported. We return a decorator that does
            # not do anything. The implementations it is applied to will not be
            # registered and they will not be called; they are written for a
            # module that is not installed in the current environment.
            def decorator(specialized_impl):
                return specialized_impl

            return decorator

        if argument_type is None:
            # Use all type names in the module's description by default.
            argument_type = list(module_info.types.keys())
        elif isinstance(argument_type, str):
            argument_type = (argument_type,)

        # Define a decorator that adds specialized implementations to
        # ``dispatched``'s registry. When the decorator is applied to a
        # function, this function is registered as the implementation to use
        # for all the types listed in ``argument_type`` for the module
        # specified by ``module_name``.
        def decorator(specialized_impl):
            types_to_register = set()
            for type_name in argument_type:
                types_to_register.update(module_info.types[type_name])
            for module_type in types_to_register:
                dispatched.register(module_type, specialized_impl)
            return specialized_impl

        return decorator

    # tack the decorator factory onto the generic function as an attribute
    dispatched.specialize = specialize

    # return the generic function
    return dispatched
