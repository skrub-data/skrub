import numbers
import os
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def _get_default_data_dir():
    """Get the default data directory path.

    Returns the path to SKB_DATA_DIRECTORY if set and absolute,
    otherwise defaults to ~/skrub_data.

    Deprecated env var SKRUB_DATA_DIRECTORY is still supported with a warning.
    """
    # Check for the new env var first
    data_home_envar = os.environ.get("SKB_DATA_DIRECTORY")

    # Check for deprecated env var
    if not data_home_envar:
        deprecated_envar = os.environ.get("SKRUB_DATA_DIRECTORY")
        if deprecated_envar:
            warnings.warn(
                "The environment variable 'SKRUB_DATA_DIRECTORY' is deprecated. "
                "Please use 'SKB_DATA_DIRECTORY' instead.",
                DeprecationWarning,
            )
            data_home_envar = deprecated_envar

    if data_home_envar and (path := Path(data_home_envar)).is_absolute():
        data_home = path
    else:
        data_home = Path.home() / "skrub_data"

    data_home.mkdir(parents=True, exist_ok=True)

    return str(data_home)


def _parse_env_bool(env_variable_name, default):
    value = os.getenv(env_variable_name, default)
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise ValueError(
            f"{env_variable_name!r} must be either 'True' or 'False', got {value}."
        )


_global_config = {
    "use_table_report_data_ops": _parse_env_bool("SKB_USE_TABLE_REPORT_DATA_OPS", True),
    "table_report_plots_threshold": int(
        os.environ.get("SKB_TABLE_REPORT_PLOTS_THRESHOLD", 30)
    ),
    "table_report_associations_threshold": int(
        os.environ.get("SKB_TABLE_REPORT_ASSOCIATIONS_THRESHOLD", 30)
    ),
    "table_report_verbosity": int(os.environ.get("SKB_TABLE_REPORT_VERBOSITY", 1)),
    "subsampling_seed": int(os.environ.get("SKB_SUBSAMPLING_SEED", 0)),
    "enable_subsampling": os.environ.get("SKB_ENABLE_SUBSAMPLING", "default"),
    "float_precision": int(os.environ.get("SKB_FLOAT_PRECISION", 3)),
    "cardinality_threshold": int(os.environ.get("SKB_CARDINALITY_THRESHOLD", 40)),
    "data_dir": _get_default_data_dir(),
    "eager_data_ops": _parse_env_bool("SKB_EAGER_DATA_OPS", True),
    "data_ops_open_graph_dropdown": _parse_env_bool(
        "SKB_DATA_OPS_OPEN_GRAPH_DROPDOWN", False
    ),
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    """Return a thread-local copy of the global configuration.

    This is used to ensure that each thread has its own configuration
    without affecting the global configuration.
    """
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global skrub configuration.
    set_config : Set global skrub configuration.

    Examples
    --------
    >>> import skrub
    >>> config = skrub.get_config()
    >>> config.keys()
    dict_keys([...])
    """
    return _get_threadlocal_config().copy()


def set_config(
    use_table_report_data_ops=None,
    table_report_plots_threshold=None,
    table_report_associations_threshold=None,
    table_report_verbosity=None,
    subsampling_seed=None,
    enable_subsampling=None,
    float_precision=None,
    cardinality_threshold=None,
    data_dir=None,
    eager_data_ops=None,
    data_ops_open_graph_dropdown=None,
):
    """Set global skrub configuration.

    Parameters
    ----------
    use_table_report_data_ops : bool, default=None
        The type of HTML representation used for the dataframes preview in skrub
        DataOps. If ``None``, falls back to the current configuration, which is ``True``
        by default.

        - If ``True``, :class:`~skrub.TableReport` will be used.
        - If ``False``, the original Pandas or Polars dataframe display will be
          used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT_DATA_OPS``
        environment variable.

    table_report_plots_threshold : int, default=None
        Maximum number of columns for which distribution plots are generated
        in :class:`~skrub.TableReport` when ``plot_distributions="auto"``
        (the default). Dataframes with more columns will skip plots.
        Default is 30.

        This configuration can also be set with the ``SKB_TABLE_REPORT_PLOTS_THRESHOLD``
        environment variable.

    table_report_associations_threshold : int, default=None
        Maximum number of columns for which associations are computed
        in :class:`~skrub.TableReport` when ``compute_associations="auto"``
        (the default). Dataframes with more columns will skip associations.
        Default is 30.

        This configuration can also be set with the
        ``SKB_TABLE_REPORT_ASSOCIATIONS_THRESHOLD`` environment variable.

    table_report_verbosity : int, default=None
        Set the level of verbosity of the :class:`~skrub.TableReport`.
        Default is 1 (print the progress bar). Refer to the ``TableReport``
        documentation for more details.

    subsampling_seed : int, default=None
        Set the random seed of subsampling in skrub DataOps
        :func:`skrub.DataOp.skb.subsample`, when ``how="random"`` is passed.

        This configuration can also be set with the ``SKB_SUBSAMPLING_SEED`` environment
        variable.

    enable_subsampling : {'default', 'disable', 'force'}, default=None
        Control the activation of subsampling in skrub DataOps
        :func:`skrub.DataOp.skb.subsample`. Default is ``"default"``.

        - If ``"default"``, the behavior of :func:`skrub.DataOp.skb.subsample` is used.
        - If ``"disable"``, subsampling is never used, so ``skb.subsample`` becomes a
          no-op.
        - If ``"force"``, subsampling is used in all DataOps evaluation modes
          (:func:`~skrub.DataOp.skb.eval`, fit_transform, etc.).

        This configuration can also be set with the ``SKB_ENABLE_SUBSAMPLING``
        environment variable.

    float_precision : int, default=3
        Control the number of significant digits shown when formatting floats.
        Applies overall precision rather than fixed decimal places. Default is 3.

        This configuration can also be set with the ``SKB_FLOAT_PRECISION``
        environment variable.

    cardinality_threshold : int, default=40
        Set the ``cardinality_threshold`` argument of :class:`~skrub.TableVectorizer`.
        Control the threshold value used to warn user if they have
        high cardinality columns in there dataset.

        This configuration can also be set with the ``SKB_CARDINALITY_THRESHOLD``
        environment variable.

    data_dir : str or pathlib.Path, default=None
        Set the data directory path for skrub datasets. If ``None``, falls back to
        the current configuration.

        - If the ``SKB_DATA_DIRECTORY`` environment variable is set to an absolute
          path, that path will be used.
        - Otherwise, the default is ``~/skrub_data``.

        This configuration can also be set with the ``SKB_DATA_DIRECTORY``
        environment variable. The deprecated ``SKRUB_DATA_DIRECTORY`` is still
        supported with a deprecation warning.

    eager_data_ops : bool, default=True
        Eagerly perform checks on the DataOps as soon they are created, and
        compute previews if preview data is available. If disabled, those
        checks are delayed until the DataOp is actually used (e.g. by calling
        ``.skb.eval()`` or ``make_learner()``), and previews are not computed.

        This option is used to speed-up the creation of large DataOps
        containing many nodes. It can also be useful in rare cases where a
        DataOp needs no inputs (for example it relies on a hard-coded filename
        to load data) but we want to prevent it from computing preview results
        as soon as it is constructed and delay computation until we explicitly
        request it. For most DataOps that do need inputs (contain
        ``skrub.var()`` nodes), previews can also be disabled simply by not
        providing preview data to ``skrub.var()``.

        This configuration can also be set with the ``SKB_EAGER_DATA_OPS``
        environment variable.

    data_ops_open_graph_dropdown : bool, default=False
        When displaying a DataOp that has a preview value in a jupyter
        notebook, should the dropdown that reveals the computational graph
        drawing be open (if True) or close (if False). This option mostly
        exists to control the display of DataOps in the skrub documentation
        examples. This configuration can also be set with the
        ``SKB_DATA_OPS_OPEN_GRAPH_DROPDOWN`` environment variable.



    See Also
    --------

    get_config : Retrieve current values for global configuration.
    config_context : Context manager for global skrub configuration.

    Examples
    --------
    >>> from skrub import set_config
    >>> set_config(use_table_report_data_ops=True)  # doctest: +SKIP
    """
    local_config = _get_threadlocal_config()
    if use_table_report_data_ops is not None:
        if not isinstance(use_table_report_data_ops, bool):
            raise ValueError(
                "'use_table_report_data_ops' must be a boolean, got "
                f"{use_table_report_data_ops!r}."
            )
        local_config["use_table_report_data_ops"] = use_table_report_data_ops

    if table_report_plots_threshold is not None:
        if (
            not isinstance(table_report_plots_threshold, numbers.Integral)
            or table_report_plots_threshold < 0
        ):
            raise ValueError(
                "'table_report_plots_threshold' must be a non-negative integer, got"
                f" {table_report_plots_threshold!r}"
            )
        local_config["table_report_plots_threshold"] = table_report_plots_threshold

    if table_report_associations_threshold is not None:
        if (
            not isinstance(table_report_associations_threshold, numbers.Integral)
            or table_report_associations_threshold < 0
        ):
            raise ValueError(
                "'table_report_associations_threshold' must be a non-negative integer,"
                f" got {table_report_associations_threshold!r}"
            )
        local_config["table_report_associations_threshold"] = (
            table_report_associations_threshold
        )

    if table_report_verbosity is not None:
        if (
            not isinstance(table_report_verbosity, numbers.Integral)
            or table_report_verbosity < 0
        ):
            raise ValueError(
                "'table_report_verbosity' must be a non-negative integer, got"
                f" {table_report_verbosity!r}"
            )
        local_config["table_report_verbosity"] = table_report_verbosity

    if subsampling_seed is not None:
        np.random.RandomState(subsampling_seed)  # check seed
        local_config["subsampling_seed"] = subsampling_seed

    if enable_subsampling is not None:
        if enable_subsampling not in (options := ("default", "force", "disable")):
            raise ValueError(
                f"'enable_subsampling' options are {options!r}, got "
                f"{enable_subsampling!r}"
            )
        local_config["enable_subsampling"] = enable_subsampling

    if float_precision is not None:
        if not isinstance(float_precision, numbers.Integral) or float_precision <= 0:
            raise ValueError(
                f"'float_precision' must be a positive integer, got {float_precision!r}"
            )
        local_config["float_precision"] = float_precision

    if cardinality_threshold is not None:
        if (
            not isinstance(cardinality_threshold, numbers.Integral)
            or cardinality_threshold < 0
        ):
            raise ValueError(
                "'cardinality_threshold' must be a positive"
                f"integer, got {cardinality_threshold!r}"
            )

    if data_dir is not None:
        data_dir = Path(data_dir).expanduser().resolve()
        local_config["data_dir"] = str(data_dir)

    if eager_data_ops is not None:
        local_config["eager_data_ops"] = eager_data_ops

    if data_ops_open_graph_dropdown is not None:
        local_config["data_ops_open_graph_dropdown"] = bool(
            data_ops_open_graph_dropdown
        )


@contextmanager
def config_context(
    *,
    use_table_report_data_ops=None,
    table_report_plots_threshold=None,
    table_report_associations_threshold=None,
    table_report_verbosity=None,
    subsampling_seed=None,
    enable_subsampling=None,
    float_precision=None,
    cardinality_threshold=None,
    data_dir=None,
    eager_data_ops=None,
    data_ops_open_graph_dropdown=None,
):
    """Context manager for global skrub configuration.

    Parameters
    ----------
    use_table_report_data_ops : bool, default=None
        The type of HTML representation used for the dataframes preview in skrub
        DataOps. Default is ``True``.

        - If ``True``, :class:`~skrub.TableReport` will be used.
        - If ``False``, the original Pandas or Polars dataframe display will be
          used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT_DATA_OPS``
        environment variable.

    table_report_verbosity : int, default=None
        Set the level of verbosity of the :class:`~skrub.TableReport`.
        Default is 0 (no verbosity). Refer to the ``TableReport`` documentation for
        more details.

    table_report_plots_threshold : int, default=None
        Maximum number of columns for which distribution plots are generated
        when ``plot_distributions="auto"`` (the default). Default is 30.

        This configuration can also be set with the ``SKB_TABLE_REPORT_PLOTS_THRESHOLD``
        environment variable.

    table_report_associations_threshold : int, default=None
        Maximum number of columns for which associations are computed
        when ``compute_associations="auto"`` (the default). Default is 30.

        This configuration can also be set with the
        ``SKB_TABLE_REPORT_ASSOCIATIONS_THRESHOLD``environment variable.

    subsampling_seed : int, default=None
        Set the random seed of subsampling in skrub DataOps
        :func:`skrub.DataOp.skb.subsample`, when ``how="random"`` is passed.

        This configuration can also be set with the ``SKB_SUBSAMPLING_SEED`` environment
        variable.

    enable_subsampling : {'default', 'disable', 'force'}, default=None
        Control the activation of subsampling in skrub DataOps
        :func:`skrub.DataOp.skb.subsample`. Default is ``"default"``.

        - If ``"default"``, the behavior of :func:`skrub.DataOp.skb.subsample` is used.
        - If ``"disable"``, subsampling is never used, so ``skb.subsample`` becomes a
          no-op.
        - If ``"force"``, subsampling is used in all DataOps evaluation modes
          (preview, fit_transform, etc.).

        This configuration can also be set with the ``SKB_ENABLE_SUBSAMPLING``
        environment variable.

    float_precision : int, default=3
        Control the number of significant digits shown when formatting floats.
        Applies overall precision rather than fixed decimal places. Default is 3.

        This configuration can also be set with the ``SKB_FLOAT_PRECISION``
        environment variable.

    cardinality_threshold : int, default=40
        Set the ``cardinality_threshold`` argument of :class:`~skrub.TableVectorizer`.
        Control the threshold value used to warn user if they have
        high cardinality columns in their dataset.

        This configuration can also be set with the ``SKB_CARDINALITY_THRESHOLD``
        environment variable.

    data_dir : str or pathlib.Path, default=None
        Set the data directory path for skrub datasets. If ``None``, falls back to
        the current configuration.

        - If the ``SKB_DATA_DIRECTORY`` environment variable is set to an absolute
          path, that path will be used.
        - Otherwise, the default is ``~/skrub_data``.

        This configuration can also be set with the ``SKB_DATA_DIRECTORY``
        environment variable. The deprecated ``SKRUB_DATA_DIRECTORY`` is still
        supported with a deprecation warning.

    eager_data_ops : bool, default=True
        Eagerly perform checks on the DataOps as soon they are created, and
        compute previews if preview data is available. If disabled, those
        checks are delayed until the DataOp is actually used (e.g. by calling
        ``.skb.eval()`` or ``make_learner()``), and previews are not computed.

        This option is used to speed-up the creation of large DataOps
        containing many nodes. It can also be useful in rare cases where a
        DataOp needs no inputs (for example it relies on a hard-coded filename
        to load data) but we want to prevent it from computing preview results
        as soon as it is constructed and delay computation until we explicitly
        request it. For most DataOps that do need inputs (contain
        ``skrub.var()`` nodes), previews can also be disabled simply by not
        providing preview data to ``skrub.var()``.

        This configuration can also be set with the ``SKB_EAGER_DATA_OPS``
        environment variable.

    data_ops_open_graph_dropdown : bool, default=False
        When displaying a DataOp that has a preview value in a jupyter
        notebook, should the dropdown that reveals the computational graph
        drawing be open (if True) or close (if False). This option mostly
        exists to control the display of DataOps in the skrub documentation
        examples. This configuration can also be set with the
        ``SKB_DATA_OPS_OPEN_GRAPH_DROPDOWN`` environment variable.

    Yields
    ------
    None.

    See Also
    --------
    get_config : Retrieve current values for global configuration.
    set_config : Set global skrub configuration.

    Examples
    --------
    >>> import skrub
    >>> with skrub.config_context(table_report_plots_threshold=1):
    ...     ...  # doctest: +SKIP
    """
    original_config = get_config()
    set_config(
        use_table_report_data_ops=use_table_report_data_ops,
        table_report_plots_threshold=table_report_plots_threshold,
        table_report_associations_threshold=table_report_associations_threshold,
        table_report_verbosity=table_report_verbosity,
        subsampling_seed=subsampling_seed,
        enable_subsampling=enable_subsampling,
        float_precision=float_precision,
        cardinality_threshold=cardinality_threshold,
        data_dir=data_dir,
        eager_data_ops=eager_data_ops,
        data_ops_open_graph_dropdown=data_ops_open_graph_dropdown,
    )

    try:
        yield
    finally:
        set_config(**original_config)
