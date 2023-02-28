import tracemalloc
from collections import defaultdict
from datetime import datetime
from itertools import product as _product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Collection, Dict, List, Optional, Union
from warnings import warn

import pandas as pd
from tqdm import tqdm


def repr_func(f: Callable, args: tuple, kwargs: dict) -> str:
    """
    Takes a function (f) and its arguments (args, kwargs), and
    returns it represented as "f(*args, **kwargs)".
    For example, with ``f=do_smth``, ``args=(10, 5)`` and
    ``kwargs={"keyboard": "qwerty"}``,
    returns "do_smth(10, 5, keyboard=qwerty)".
    Can be parsed with the function `parse_func_repr` below.
    """
    str_args = ", ".join(args)
    str_kwargs = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return f"{f.__name__}({', '.join(st for st in [str_args, str_kwargs] if st)})"


def monitor(
    memory: bool,
    time: bool,
    parametrize: Union[Collection[Collection[Any]], Dict[str, Collection[Any]]],
    repeat: int = 1,
    save_as: Optional[str] = None,
) -> Callable[..., Callable[..., pd.DataFrame]]:
    """Decorator used to monitor the execution of a function.

    The decorated function should return either None, or a dictionary,
    which will be added to the results.
    Executions are sequential, so it's usually pretty long to run!

    Parameters
    ----------

    memory : bool
        Whether the RAM usage should be monitored throughout the function execution.
        Note: monitored in the main thread.
    time : bool
        Whether the time the function took to execute should be measured.
        Note: if `memory` is also set, consider that as the memory profiler runs
        in the main thread, the timings will be different from an execution
        without the memory monitoring.
    parametrize : a collection of a collection of parameters
        Specifies the parameter matrix to be used on the function.
        These can either be passed as positional arguments (e.g., list of list
        of parameters), or as keyword arguments (e.g., dictionary of list of
        parameters).
        Note: when specified, ignores the parameters passed to the function.
    repeat : int
        How many times we want to repeat the execution of the function for more
        representative time and memory extracts.
    save_as : str, optional
        Can be specified as a benchmark name for the results to be automatically
        saved on disk.
        E.g. "supervectorizer_tuning"
        If None (the default), results are not saved on disk, only returned.

    Returns
    -------

    Callable[..., Callable[..., pd.DataFrame]]
        A double-nested callable that returns a DataFrame of the results.

    """
    if not memory and not time:
        warn(
            "Parameters 'memory' and 'time' are both set to False ; "
            "there is therefore nothing to monitor for, returning empty. ",
            stacklevel=2,
        )
        return lambda *_, **__: lambda *_, **__: pd.DataFrame()

    def decorator(func: Callable[[Any], None]):
        """Only catches the decorated function."""

        def wrapper(*_, **__) -> pd.DataFrame:
            """
            Parameters
            ----------

            *_ : Tuple[Any]
                Arguments passed by the function call. Ignored.
                Use `parametrize` instead.
            **__ : Dict[str, Any]
                Keyword arguments passed by the function call. Ignored.
                Use `parametrize` instead.

            """

            def product(
                iterables: Union[
                    Collection[Collection[Any]], Dict[str, Collection[Any]]
                ],
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

            def exec_func(*args, **kwargs) -> Dict[str, List[float]]:
                """
                Parameters
                ----------

                *args : Tuple[Any]
                    Arguments to pass to the function.
                **kwargs : Dict[str, Any]
                    Keyword arguments to pass to the function.

                Returns
                -------
                Dict[str, List[float]]
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

            df = pd.DataFrame()
            for args, kwargs in tqdm(list(product(parametrize))):
                call_repr = repr_func(func, args, kwargs)
                res_dic = exec_func(*args, **kwargs)
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
                file = f"{save_as}-{now.year}{now.month}{now.day}.csv"
                df.to_csv(save_dir / file)

            return df

        return wrapper

    return decorator
