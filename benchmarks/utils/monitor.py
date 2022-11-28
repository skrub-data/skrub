import tracemalloc
import pandas as pd

from typing import Callable, Collection, Any, Optional, Tuple, Dict, List, Union
from time import perf_counter
from itertools import product as _product
from collections import defaultdict
from datetime import datetime
from warnings import warn
from pathlib import Path
from git import Repo


def monitor(
    memory: bool,
    time: bool,
    parametrize: Union[Collection[Collection[Any]], Dict[str, Collection[Any]]],
    repeat: int = 1,
    save_benchmark_as: Optional[str] = None,
):
    """Decorator used to monitor the execution of a function.

    The decorated function should return nothing (even if it does, nothing will
    be passed through by this decorator).
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
    save_benchmark_as : str, optional
        Can be specified as a benchmark name for the results to be automatically
        saved on disk.
        E.g. "supervectorizer_tuning"
        If None (the default), results are not saved on disk, only returned.

    Returns
    -------

    Dict[str, Dict[str, List[float]]]
        A mapping of parameters (parametrized value) to their results:
        a mapping of monitored ressource name to a list of values:
        - "time": how long the execution took, in seconds.
        - "memory": The number of MB of memory used.
        The size of these lists is equal to `repeat`.

    """
    if not memory and not time:
        warn(
            "Parameters 'memory' and 'time' are both set to False ; "
            "there is therefore nothing to monitor for, returning empty. ",
            stacklevel=2,
        )
        return lambda: lambda: {}

    def decorator(func: Callable[[Any], None]):
        """Only catches the decorated function."""

        def wrapper(*call_args, **call_kwargs) -> Dict[str, Dict[str, List[float]]]:
            """
            Parameters
            ----------

            *call_args : Tuple[Any]
                Arguments passed by the function call. Ignored.
                Use `parametrize` instead.
            **call_kwargs : Dict[str, Any]
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

                """
                _monitored = defaultdict(lambda: [])

                if memory:
                    tracemalloc.start()

                for n in range(repeat):
                    if memory:
                        tracemalloc.reset_peak()
                    if time:
                        t0 = perf_counter()
                    func(*args, **kwargs)

                    if time:
                        _monitored["time"].append(perf_counter() - t0)
                    if memory:
                        _, peak = tracemalloc.get_traced_memory()
                        _monitored["memory"].append(peak / (1024**2))

                if memory:
                    tracemalloc.stop()

                return dict(_monitored)

            def repr_func(f: Callable, args: tuple, kwargs: dict) -> str:
                """
                Takes a function (f) and its arguments (args, kwargs), and
                returns it represented as "f(*args, **kwargs)".
                For exemple, with ``f=do_smth``, ``args=(10, 5)`` and
                ``kwargs={"azerty": "qwerty"}``,
                returns "do_smth(10, 5, azerty=qwerty)".
                """
                str_args = ", ".join(args)
                str_kwargs = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                return (
                    f"{f.__name__}"
                    f"({', '.join(st for st in [str_args, str_kwargs] if st)})"
                )

            def to_df(results: Dict[str, Dict[str, List[float]]]):
                """
                Converts the result of the benchmark to a pandas DataFrame.
                """
                index = list(results.keys())
                columns = results[index[0]].keys()  # Take the first one for sample
                data = [[results[idx][col] for col in columns] for idx in index]
                df = pd.DataFrame(data, index=index, columns=columns)
                return df

            results = {
                repr_func(func, args, kwargs): exec_func(*args, **kwargs)
                for args, kwargs in product(parametrize)
            }

            if save_benchmark_as is not None:
                save_dir = Path(__file__).parent.parent / "results"
                save_dir.mkdir(exist_ok=True)
                now = datetime.now()
                commit = Repo(Path(__file__).parent.parent.parent).active_branch.commit
                file = (
                    f"{save_benchmark_as}-{now.year}{now.month}{now.day}-{commit}.csv"
                )
                to_df(results).to_csv(save_dir / file, index_label="call")

            return results

        return wrapper

    return decorator
