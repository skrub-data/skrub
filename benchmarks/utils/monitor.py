import tracemalloc
from collections import defaultdict
from datetime import datetime
from itertools import product as _product
from pathlib import Path
from time import perf_counter
from collections.abc import Callable, Collection
from warnings import warn
from typing import Any

import pandas as pd
from tqdm import tqdm


def repr_func(f: Callable, args: tuple, kwargs: dict) -> str:
    """
    Takes a function (f) and its arguments (args, kwargs), and
    returns it represented as "f(*args, **kwargs)".
    For example, with ``f=do_smth``, ``args=(10, 5)`` and
    ``kwargs={"keyboard": "qwerty"}``,
    returns "do_smth(10, 5, keyboard=qwerty)".
    """
    str_args = ", ".join(args)
    str_kwargs = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return f"{f.__name__}({', '.join(st for st in [str_args, str_kwargs] if st)})"


def monitor(
    *,
    parametrize: Collection[Collection[Any]] | dict[str, Collection[Any]] | None = None,
    memory: bool = True,
    time: bool = True,
    repeat: int = 1,
    save_as: str | None = None,
) -> Callable[..., Callable[..., pd.DataFrame]]:
    """Decorator used to monitor the execution of a function.

    The decorated function should return either None, or a dictionary,
    which will be added to the results.
    Executions are sequential, so it's usually pretty long to run!

    Parameters
    ----------
    parametrize : a collection of collections of parameters, optional
        Specifies the parameter matrix to be used on the function.
        These can either be passed as positional arguments (e.g., list of list
        of parameters), or as keyword arguments (e.g., dictionary of list of
        parameters).
        Note: when specified, ignores the parameters passed to the function.
    memory : bool, optional, default=True
        Whether the RAM usage should be monitored throughout the function execution.
        Note: monitored in the main thread.
    time : bool, optional, default=True
        Whether the time the function took to execute should be measured.
        Note: if `memory` is also set, consider that as the memory profiler runs
        in the main thread, the timings will be different from an execution
        without the memory monitoring.
    repeat : int, optional, default=1
        How many times we want to repeat the execution of the function for more
        representative time and memory extracts.
    save_as : str, optional
        Can be specified as a benchmark name for the results to be automatically
        saved on disk.
        E.g. "table_vectorizer_tuning"
        If None (the default), results are not saved on disk, only returned.

    Returns
    -------
    Callable[..., Callable[..., pd.DataFrame]]
        A double-nested callable that returns a DataFrame of the results.

    Examples
    --------
    When `parametrize` is not passed, the values the function is called with
    are used.
    >>> @monitor()
    >>> def function(choice: typing.Literal["yes", "no", number: int):
    >>>     ...
    >>> function("yes", 15)

    For more complex combinations, the `parametrize` parameter can be used:
    >>> @monitor(
    >>>     parametrize={
    >>>         "choice": ["yes", "no"],
    >>>         "number": [10, 66, 0],
    >>>     },
    >>> )
    >>> def function(choice: typing.Literal["yes", "no"], number: int):
    >>>     ...
    >>> function()  # Called without any parameter

    For benchmarking specific combinations, they can be passed as a list:
    >>> @monitor(
    >>>     parametrize=[
    >>>         ("yes", 10),
    >>>         ("no", 20),
    >>>     ],
    >>> )
    >>> def function(choice: typing.Literal["yes", "no"], number: int):
    >>>     ...
    >>> function()  # Called without any parameter
    """

    def decorator(func: Callable[[Any], None]):
        """Only catches the decorated function."""

        def wrapper(*call_args, **call_kwargs) -> pd.DataFrame:
            """
            Parameters
            ----------
            call_args : tuple of any values
                Arguments passed by the function call.
            call_kwargs : mapping of str to any values
                Keyword arguments passed by the function call.
            """

            def product(
                iterables: Collection[Collection[Any]] | dict[str, Collection[Any]],
            ):
                """
                Wrapper for ``itertools.product`` in order to better
                handle dictionaries.
                """
                if isinstance(iterables, dict):
                    for pair in _product(*iterables.values()):
                        yield (), dict(zip(iterables.keys(), pair))
                else:
                    for params in _product(*iterables):
                        return params, {}

            def exec_func(*args, **kwargs) -> dict[str, list[float]]:
                """
                Parameters
                ----------
                *args : tuple of any values
                    Arguments to pass to the function.
                **kwargs : mapping of str to any values
                    Keyword arguments to pass to the function.

                Returns
                -------
                dictionary of str to list of float
                    a mapping of monitored resource name to a list of values:
                    - "time": how long the execution took, in seconds.
                      Only present if ``time=True``.
                    - "memory": The number of MB of memory used.
                      Only present if ``memory=True``.
                    The size of these lists is equal to `repeat`.
                """
                _monitored = defaultdict(lambda: [])

                # Global initialization of monitored values
                if memory:
                    tracemalloc.start()

                for n in range(repeat):
                    _monitored["iter"].append(n)
                    # Initialize loop monitored values
                    if memory:
                        tracemalloc.reset_peak()
                    if time:
                        t0 = perf_counter()

                    res_dic = func(*args, **kwargs)

                    if res_dic is not None:
                        for key, value in res_dic.items():
                            _monitored[key].append(value)

                    # Collect and store loop monitored values
                    if time:
                        _monitored["time"].append(perf_counter() - t0)
                    if memory:
                        _, peak = tracemalloc.get_traced_memory()
                        _monitored["memory"].append(peak / (1024**2))

                # Global cleanup of monitored values
                if memory:
                    tracemalloc.stop()
                return _monitored

            if parametrize is None:
                # Use the parameters passed by the call
                parametrization = (call_args, call_kwargs)
            elif isinstance(parametrize, list):
                parametrization = (parametrize, ())
            else:
                parametrization = list(product(parametrize))

            df = pd.DataFrame()
            for args, kwargs in tqdm(parametrization):
                call_repr = repr_func(func, args, kwargs)
                res_dic = exec_func(*args, **kwargs)
                if not res_dic:  # Dict is empty
                    warn(
                        "Nothing was returned during the execution, "
                        "there is therefore nothing to monitor for. ",
                        stacklevel=2,
                    )
                    return df

                # Add arguments to the results in wide format
                for index, arg in enumerate(args):
                    res_dic[f"arg{index}"] = arg
                for key, value in kwargs.items():
                    if isinstance(value, (list, set, tuple, dict)):
                        # Prevent creating new lines
                        value = str(value)
                    res_dic[key] = value
                res_dic["call"] = call_repr
                df = pd.concat((df, pd.DataFrame(res_dic)), ignore_index=True)

            if save_as is not None:
                save_dir = Path(__file__).parent.parent / "results"
                save_dir.mkdir(exist_ok=True)
                now = datetime.now()
                file = f"{save_as}-{now.year}{now.month}{now.day}.parquet"
                df.to_parquet(save_dir / file)

            return df

        return wrapper

    return decorator
