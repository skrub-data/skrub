import tracemalloc
import os

from collections import defaultdict
from collections.abc import Callable, Collection, Mapping
from datetime import datetime
from itertools import product
from pathlib import Path
from random import choice
from string import ascii_letters, digits
from time import perf_counter, time as get_time
from typing import Any
from warnings import warn

import pandas as pd
from tqdm import tqdm


def monitor(
    *,
    parametrize: Collection[Mapping[Any]] | Mapping[str, Collection[Any]] | None = None,
    save_as: str | None = None,
    memory: bool = True,
    time: bool = True,
    repeat: int = 1,
    hot_load: str | None = None,
) -> Callable[..., Callable[..., pd.DataFrame]]:
    """Decorator used to monitor the execution of a function.

    The decorated function should return either:
    - ``None``, when the goal is only to monitor time of exection and/or memory
      (parameters ``time`` and/or ``memory`` should be ``True`` (the default));
    - a mapping (dict), which will be added to the results. The keys are going
      to be the columns of the resulting pandas DataFrame.
    - a list of mappings, especially useful when there is an iterative
      process within the function, and we want to monitor the results at each
      step. There is however a caveat with this method: the time and memory
      results are representative of the whole execution.
      If you wanted step-by-step results, you should implement the time and/or
      memory monitoring within your function (and disable global monitoring
      i.e. ``@monitor(time=False, memory=False)`` if relevant).

    The result of the call of the decorated function is a pandas DataFrame,
    containing the results of the function for each parameter combination.
    The columns are the keys of the mapping(s) returned by the function,
    and the optional columns ``iter``, ``time`` and ``memory``.

    To avoid losing data upon error, especially because benchmarks tend to be
    long and computationally expensive, the results are saved after each
    execution and can be hot-loaded later on with parameter ``hot-load``.

    Executions are sequential, so it's usually pretty long to run!

    Parameters
    ----------
    parametrize : a collection of collections of parameters, optional
        Specifies the parameter matrix to be used on the function.
        These can only be passed as keyword arguments using two formats:
        - mapping of parameter name to a list of possible values ; in this
          configuration, all combinations will be executed, meaning the number
          of possibilities grows exponentially
        - list of mapping of parameter name to a single possible value ; in
          this configuration, each mapping is executed independently, meaning
          the number of possibilities grows linearly
        See the example section for illustrations.
        Note: when `parametrize` is specified, the parameters passed to the
        function are ignored.
    save_as : str
        Specifies a benchmark name for the results to be automatically
        saved on disk (directory `benchmarks/results/`).
        E.g. "table_vectorizer_tuning"
        Note: results are also returned.
    memory : bool, default=True
        Whether the RAM usage should be monitored throughout the function execution.
        Note: monitored in the main thread.
    time : bool, default=True
        Whether the time the function took to execute should be measured.
        Note: if `memory` is also set, consider that as the memory profiler
        runs in the main thread, the timings will be different from an
        execution without the memory monitoring.
    hot_load : str, optional
        Name of the file to hot-load (meaning, recovering partial results
        from a previous run that was interupted).
        The name of the file is random (created at runtime), and printed before
        the run. Grab it from the stdout of your interrupted run.
    repeat : int, default=1
        How many times we want to repeat the execution of the function for more
        representative time and memory extracts.
        These additional runs are stored in the returned DataFrame, with a
        different value in the ``iter`` column.

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
    >>> function(choice="yes", number=15)

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

    For benchmarking specific combinations, they can be passed as a list of dicts:
    >>> @monitor(
    >>>     parametrize=[
    >>>         dict(choice="yes", number=10),
    >>>         dict(choice="no", number=20),
    >>>     ],
    >>> )
    >>> def function(choice: typing.Literal["yes", "no"], number: int):
    >>>     ...
    >>> function()  # Called without any parameter
    """

    reserved_column_names = {"iter", "time", "memory"}

    def decorator(
        func: Callable[..., Mapping[str, Any] | list[Mapping[str, Any]] | None]
    ):
        """
        Catches the decorated function.

        Parameters
        ----------
        func : callable returning none, a mapping or a list of mappings
            The decorated function callable object.
        """

        def wrapper(*call_args, **call_kwargs) -> pd.DataFrame:
            """
            Catches the decorated function's call arguments.

            Parameters
            ----------
            call_args : tuple of any values
                Arguments passed by the function call (should be empty, i.e.
                no positional arguments should be passed).
            call_kwargs : mapping of str to any
                Keyword arguments passed by the function call.
            """

            # Instead of just not catching positional arguments,
            # we get them and raise a clean error, otherwise it's
            # kind of unclear what's happening.
            if call_args:
                raise ValueError(
                    "All arguments should be passed by keyword, got"
                    f"positional values: {call_args!r}"
                )

            def get_random_file_name() -> str:
                """
                Returns a random file name, used by hot-loading.
                Format is ``{time}-{random_string}.parquet``.
                """
                name = "".join(choice(ascii_letters + digits) for _ in range(8))
                time = int(get_time())
                return f"{time}-{name}.parquet"

            def load_intermediate_results(file_name: str) -> pd.DataFrame:
                """
                Loads the results from the file passed.
                If the file is not found, and to avoid unexpected behavior,
                we raise an error.
                """
                file_name = os.path.abspath(file_name)

                if not os.path.isfile(file_name):
                    raise FileNotFoundError(f"Could not hot-load file {file_name!r}")

                return pd.read_parquet(file_name)

            def product_map(iterables: Mapping[str, Any]):
                """``itertools.product`` with mapping support."""
                for combination in product(*iterables.values()):
                    yield dict(zip(iterables.keys(), combination))

            def exec_func(**kwargs) -> pd.DataFrame:
                """
                Wraps the decorated function call with a single set of
                parameters, and pre-process the returned values.

                Parameters
                ----------
                **kwargs : mapping of str to any values
                    Keyword arguments to pass to the function.

                Returns
                -------
                pd.DataFrame
                    A DataFrame containing for each column one or more
                    (depending on the number of repeats and number of
                    mappings returned by the function).
                    The columns names are the monitored values, and
                    the optional ``iter``, ``time`` and ``memory``.
                """
                results = defaultdict(lambda: [])

                # Global initialization of monitored values
                if memory:
                    tracemalloc.start()

                for n in tqdm(range(repeat)):
                    # Initialize loop monitored values
                    if memory:
                        tracemalloc.reset_peak()
                    if time:
                        t0 = perf_counter()

                    result = func(**kwargs)

                    # To avoid repeating code, we move the result
                    # mapping(s) to a list.
                    result_mappings = []
                    if result is None:
                        pass
                    elif isinstance(result, dict):
                        result_mappings = [result]
                    elif isinstance(result, list):
                        result_mappings = result

                    # and iterate over that list, saving monitored values,
                    # as well as results returned by the function call.
                    for result_mapping in result_mappings:
                        results["iter"].append(n)
                        if time:
                            results["time"].append(perf_counter() - t0)
                        if memory:
                            _, peak = tracemalloc.get_traced_memory()
                            results["memory"].append(peak / (1024**2))
                        for key in reserved_column_names:
                            if key in result_mapping:
                                warn(
                                    f"Column name {key!r} is reserved. "
                                    "Results will be overwritten. "
                                )
                        for key, value in result_mapping.items():
                            results[key].append(value)

                else:
                    # Accessing the upper-level variable, kind of hackish,
                    # but cleanest in terms of code.
                    progress_bar.update(1)

                # Global cleanup of monitored values
                if memory:
                    tracemalloc.stop()

                # Now that all the loops have been done, cast some values that
                # should be the same for all instances.
                # To simplify this process, we'll convert the results to
                # a DataFrame
                df_results = pd.DataFrame(results)
                # Add arguments to the results in wide format
                for key, value in kwargs.items():
                    if isinstance(value, (list, set, tuple, dict)):
                        # Prevent creating new lines
                        value = str(value)
                    df_results[key] = value

                return df_results

            parametrization: list[Mapping]
            if parametrize is None:
                # Use the parameters passed by the call
                parametrization = [call_kwargs]
            elif isinstance(parametrize, list):
                parametrization = parametrize
            elif isinstance(parametrize, Mapping):
                parametrization = list(product_map(parametrize))
            else:
                raise ValueError(
                    f"Invalid parametrize type: {type(parametrize)} ({parametrize}). "
                )

            if hot_load:
                df = load_intermediate_results(hot_load)
                intermediate_results_file = Path(hot_load).absolute()
            else:
                df = pd.DataFrame(columns=list(parametrization[0].keys()))
                intermediate_results_file = Path(get_random_file_name()).absolute()
                intermediate_results_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(intermediate_results_file)
            print(
                "Intermediate results will be saved in file "
                f"{intermediate_results_file}. If this instance crashes, use "
                "its path with the `hot_load` parameter of `monitor` in the "
                "next run to continue the benchmark where it left off."
            )

            with tqdm(total=len(parametrization) * repeat) as progress_bar:
                for kwargs in tqdm(parametrization):
                    kwargs_s = pd.DataFrame(
                        data=[list(kwargs.values())],
                        columns=list(kwargs.keys()),
                    )
                    if kwargs_s.isin(df[kwargs.keys()]).all(axis=1).any():
                        # Argument combination already ran before, skipping
                        progress_bar.update(1)
                        continue
                    res_df = exec_func(**kwargs)
                    df = pd.concat((df, res_df), ignore_index=True)
                    # Save progress
                    df.to_parquet(intermediate_results_file)

            save_dir = Path(__file__).parent.parent / "results"
            save_dir.mkdir(exist_ok=True)
            now = datetime.now()
            file = f"{save_as}-{now.year}{now.month:02d}{now.day:02d}.parquet"
            save_file = save_dir / file
            df.to_parquet(save_file)
            print(f"Final results were saved to {save_file}")

            # Remove the intermediate results file
            intermediate_results_file.unlink()
            print(f"Intermediate results ({intermediate_results_file}) were deleted. ")

            return df

        return wrapper

    return decorator
