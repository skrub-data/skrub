import tracemalloc
from collections import defaultdict
from datetime import datetime
from itertools import product as _product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Collection, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm


def monitor(
    *,
    parametrize: Union[Collection[Collection[Any]], Dict[str, Collection[Any]]] = None,
    memory: bool = True,
    time: bool = True,
    repeat: int = 1,
    save_as: Optional[str] = None,
) -> Callable[..., Callable[..., pd.DataFrame]]:
    """Decorator used to monitor the execution of a function.

    The decorated function should return either:
    - ``None``, when the goal is only to monitor time of exection and/or memory
      (parameters ``time`` and/or ``memory`` should be ``True`` (the default));
    - a dictionary, which will be added to the results. The keys are going
      to be the columns of the resulting pandas DataFrame.
    - a list of dictionaries, especially useful when there is an interative
      process within the function, and we want to monitor the results at each
      step. There is however a caveat with this method: the time and memory
      results are representative of the whole process.
      If you wanted step-by-step results, you should implement the time and/or
      memory monitoring within your function (and disable global monitoring
      i.e. ``@monitor(time=False, memory=False)`` if relevant).

    The result of the call of the decorated function is a pandas DataFrame,
    containing the results of the function for each parameter combination.
    The columns are the keys of the dictionar(y/ies) returned by the function,
    and the optional columns ``iter``, ``time`` and ``memory``.

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
        Can be specified as a benchmark name for the results to be
        automatically saved on disk (directory `benchmarks/results/`).
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

    For benchmarking specific combinations, they can be passed as a list of tuples:
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

    reserved_column_names = {"iter", "time", "memory"}

    def decorator(
        func: Callable[[Any], Union[None, Dict[str, Any], List[Dict[str, Any]]]]
    ):
        """
        Catches the decorated function.

        Parameters
        ----------
        func : callable returning none, a dictionary or a list of dictionary
            The decorated function callable object.
        """

        def wrapper(*call_args, **call_kwargs) -> pd.DataFrame:
            """
            Catches the decorated function's call arguments.

            Parameters
            ----------
            call_args : tuple of any values
                Arguments passed by the function call.
            call_kwargs : mapping of str to any
                Keyword arguments passed by the function call.
            """

            def product(
                iterables: Union[
                    Collection[Collection[Any]], Dict[str, Collection[Any]]
                ],
            ):
                """``itertools.product`` with better dictionary support."""
                if isinstance(iterables, dict):
                    for pair in _product(*iterables.values()):
                        yield (), dict(zip(iterables.keys(), pair))
                else:
                    for params in _product(*iterables):
                        return params, {}

            def exec_func(*args, **kwargs) -> pd.DataFrame:
                """
                Wraps the decorated function call with a single set of
                parameters, and pre-process the returned values.

                Parameters
                ----------
                *args : Tuple[Any]
                    Arguments to pass to the function.
                **kwargs : Dict[str, Any]
                    Keyword arguments to pass to the function.

                Returns
                -------
                pd.DataFrame
                    A DataFrame containing for each column one or more
                    (depending on the number of repeats and number of
                    dictionaries returned by the function).
                    The columns names are the monitored values, and
                    the optional ``iter``, ``time`` and ``memory``.
                """
                results = defaultdict(lambda: [])

                # Global initialization of monitored values
                if memory:
                    tracemalloc.start()

                for n in range(repeat):
                    # Initialize loop monitored values
                    if memory:
                        tracemalloc.reset_peak()
                    if time:
                        t0 = perf_counter()

                    result = func(*args, **kwargs)

                    # To avoid repeating code, we move the result
                    # dictionar(y/ies) to a list.
                    result_dictionaries = []
                    if result is None:
                        pass
                    elif isinstance(result, dict):
                        result_dictionaries = [result]
                    elif isinstance(result, list):
                        result_dictionaries = result

                    # and iterate over that list, saving monitored values,
                    # as well as results returned by the function call.
                    for result_dict in result_dictionaries:
                        results["iter"].append(n)
                        if time:
                            results["time"].append(perf_counter() - t0)
                        if memory:
                            _, peak = tracemalloc.get_traced_memory()
                            results["memory"].append(peak / (1024**2))
                        for key, value in result_dict.items():
                            results[key].append(value)

                # Global cleanup of monitored values
                if memory:
                    tracemalloc.stop()

                from pprint import pprint

                pprint(results)

                # Now that all the loops have been done, cast some values that
                # should be the same for all instances.
                # To simplify this process, we'll convert the results to
                # a DataFrame
                df_results = pd.DataFrame(results)
                # Add arguments to the results in wide format
                for index, arg in enumerate(args):
                    df_results[f"arg{index}"] = arg
                for key, value in kwargs.items():
                    if isinstance(value, (list, set, tuple, dict)):
                        # Prevent creating new lines
                        value = str(value)
                    df_results[key] = value

                print(df_results)

                return df_results

            if parametrize is None:
                # Use the parameters passed by the call
                parametrization = (call_args, call_kwargs)
            elif isinstance(parametrize, list):
                parametrization = (parametrize, ())
            else:
                parametrization = list(product(parametrize))

            df = pd.DataFrame()
            for args, kwargs in tqdm(parametrization):
                res_df = exec_func(*args, **kwargs)
                df = pd.concat((df, res_df), ignore_index=True)

            if save_as is not None:
                save_dir = Path(__file__).parent.parent / "results"
                save_dir.mkdir(exist_ok=True)
                now = datetime.now()
                file = f"{save_as}-{now.year}{now.month}{now.day}.parquet"
                df.to_parquet(save_dir / file)

            return df

        return wrapper

    return decorator
