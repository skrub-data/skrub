import numbers
import os
import threading
from contextlib import contextmanager

import numpy as np

from ._reporting import _patching


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
    "use_table_report": _parse_env_bool("SKB_USE_TABLE_REPORT", False),
    "use_table_report_data_ops": _parse_env_bool("SKB_USE_TABLE_REPORT_DATA_OPS", True),
    "plot_distributions": _parse_env_bool("SKB_PLOT_DISTRIBUTIONS", True),
    "compute_associations": _parse_env_bool("SKB_COMPUTE_ASSOCIATIONS", True),
    "columns_threshold": int(os.environ.get("SKB_COLUMNS_THRESHOLD", 30)),
    "subsampling_seed": int(os.environ.get("SKB_SUBSAMPLING_SEED", 0)),
    "enable_subsampling": os.environ.get("SKB_ENABLE_SUBSAMPLING", "default"),
    "float_precision": int(os.environ.get("SKB_FLOAT_PRECISION", 3)),
    "cardinality_threshold": int(os.environ.get("SKB_CARDINALITY_THRESHOLD", 40)),
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


def _apply_external_patches(config):
    if config["use_table_report"]:
        _patching._patch_display(
            plot_distributions=config["plot_distributions"],
            compute_associations=config["compute_associations"],
            columns_threshold=config["columns_threshold"],
        )
    else:
        # No-op if dispatch haven't been previously enabled
        _patching._unpatch_display()


def set_config(
    use_table_report=None,
    use_table_report_data_ops=None,
    plot_distributions=None,
    compute_associations=None,
    columns_threshold=None,
    subsampling_seed=None,
    enable_subsampling=None,
    float_precision=None,
    cardinality_threshold=None,
):
    """Set global skrub configuration.

    Parameters
    ----------
    use_table_report : bool, default=None
        The type of display used for dataframes. Default is ``True``.

        - If ``True``, replace the default DataFrame HTML displays with
          :class:`~skrub.TableReport`.
        - If ``False``, the original Pandas or Polars dataframe HTML representation
          will be used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT``
        environment variable.

    use_table_report_data_ops : bool, default=None
        The type of HTML representation used for the dataframes preview in skrub
        DataOps. Default is ``False``.

        - If ``True``, :class:`~skrub.TableReport` will be used.
        - If ``False``, the original Pandas or Polars dataframe display will be
          used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT_DATA_OPS``
        environment variable.

    plot_distributions : bool, default=None
        Control whether to plot distributions in :class:`~skrub.TableReport`.
        Default is ``True``.

        This configuration can also be set with the ``SKB_PLOT_DISTRIBUTIONS``
        environment variable.

    compute_associations : bool, default=None
        Control whether to compute associations in :class:`~skrub.TableReport`.
        Default is ``True``.

        This configuration can also be set with the ``SKB_COMPUTE_ASSOCIATIONS``
        environment variable.

    columns_threshold : int, default=None
        If a dataframe has more columns than the value set here, the
        :class:`~skrub.TableReport` will skip generating the plots and computing
        the associations.
        Default is 30.

        This configuration can also be set with the ``SKB_COLUMNS_THRESHOLD``
        environment variable.

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

    See Also
    --------
    get_config : Retrieve current values for global configuration.
    config_context : Context manager for global skrub configuration.

    Examples
    --------
    >>> from skrub import set_config
    >>> set_config(use_table_report=True)  # doctest: +SKIP
    """
    local_config = _get_threadlocal_config()
    if use_table_report is not None:
        if not isinstance(use_table_report, bool):
            raise ValueError(
                f"'use_table_report' must be a boolean, got {use_table_report!r}."
            )
        local_config["use_table_report"] = use_table_report

    if use_table_report_data_ops is not None:
        if not isinstance(use_table_report_data_ops, bool):
            raise ValueError(
                "'use_table_report_data_ops' must be a boolean, got "
                f"{use_table_report_data_ops!r}."
            )
        local_config["use_table_report_data_ops"] = use_table_report_data_ops

    if plot_distributions is not None:
        if not isinstance(plot_distributions, bool):
            raise ValueError(
                f"'plot_distributions' must be a boolean, got {plot_distributions!r}."
            )
        local_config["plot_distributions"] = plot_distributions

    if compute_associations is not None:
        if not isinstance(compute_associations, bool):
            raise ValueError(
                "'compute_associations' must be a boolean, got"
                f" {compute_associations!r}."
            )
        local_config["compute_associations"] = compute_associations

    if columns_threshold is not None:
        if not isinstance(columns_threshold, numbers.Integral) or columns_threshold < 0:
            raise ValueError(
                "'columns_threshold' must be a non-negative integer, got"
                f" {columns_threshold!r}"
            )
        local_config["columns_threshold"] = columns_threshold

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

    _apply_external_patches(local_config)


@contextmanager
def config_context(
    *,
    use_table_report=None,
    use_table_report_data_ops=None,
    plot_distributions=None,
    compute_associations=None,
    columns_threshold=None,
    subsampling_seed=None,
    enable_subsampling=None,
    float_precision=None,
    cardinality_threshold=None,
):
    """Context manager for global skrub configuration.

    Parameters
    ----------
    use_table_report : bool, default=None
        The type of display used for dataframes. Default is ``False``.

        - If ``True``, replace the default DataFrame HTML displays with
          :class:`~skrub.TableReport`.
        - If ``False``, the original Pandas or Polars dataframe HTML representation
          will be used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT``
        environment variable.

    use_table_report_data_ops : bool, default=None
        The type of HTML representation used for the dataframes preview in skrub
        DataOps. Default is ``True``.

        - If ``True``, :class:`~skrub.TableReport` will be used.
        - If ``False``, the original Pandas or Polars dataframe display will be
          used.

        This configuration can also be set with the ``SKB_USE_TABLE_REPORT_DATA_OPS``
        environment variable.

    plot_distributions : bool, default=None
        Control whether to plot distributions in :class:`~skrub.TableReport`.
        Default is ``True``.

        This configuration can also be set with the ``SKB_PLOT_DISTRIBUTIONS``
        environment variable.

    compute_associations : bool, default=None
        Control whether to compute associations in :class:`~skrub.TableReport`.
        Default is ``True``.

        This configuration can also be set with the ``SKB_COMPUTE_ASSOCIATIONS``
        environment variable.

    columns_threshold : int, default=None
        If a dataframe has more columns than the value set here, the
        :class:`~skrub.TableReport` will skip generating the plots and computing
        the associations.
        Default is 30.

        This configuration can also be set with the ``SKB_COLUMNS_THRESHOLD``
        environment variable.

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
    >>> with skrub.config_context(columns_threshold=1):
    ...     ...  # doctest: +SKIP
    """
    original_config = get_config()
    set_config(
        use_table_report=use_table_report,
        use_table_report_data_ops=use_table_report_data_ops,
        plot_distributions=plot_distributions,
        compute_associations=compute_associations,
        columns_threshold=columns_threshold,
        subsampling_seed=subsampling_seed,
        enable_subsampling=enable_subsampling,
        float_precision=float_precision,
        cardinality_threshold=cardinality_threshold,
    )

    try:
        yield
    finally:
        set_config(**original_config)


# Apply patching set by environment variables. Without it, setting SKB_USE_TABLE_REPORT
# or SKB_USE_TABLE_REPORT_DATA_OPS would not have an effect.
_apply_external_patches(get_config())
