import numbers
import os
import threading
from contextlib import contextmanager

import numpy as np

from ._reporting import _patching

_global_config = {
    "expression_display": os.environ.get("SKB_EXPRESSION_DISPLAY", "tablereport"),
    "dataframe_display": os.environ.get("SKB_DATAFRAME_DISPLAY", "original"),
    "tablereport_threshold": int(os.environ.get("SKB_TABLEREPORT_THRESHOLD", 30)),
    "subsampling_seed": int(os.environ.get("SKB_SUBSAMPLING_SEED", 0)),
    "enable_subsampling": os.environ.get("SKB_ENABLE_SUBSAMPLING", "default"),
}
_threadlocal = threading.local()


def _get_threadlocal_config():
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
    >>> import sklearn
    >>> config = sklearn.get_config()
    >>> config.keys()
    dict_keys([...])
    """
    return _get_threadlocal_config().copy()


def _apply_external_patches(config):
    """
    TODO
    """
    if config["dataframe_display"] == "tablereport":
        _patching.patch_display(
            max_plot_columns=config["tablereport_threshold"],
            max_association_columns=config["tablereport_threshold"],
        )
    else:
        # No-op if dispatch haven't been previously enabled
        _patching.unpatch_display()


def set_config(
    expression_display=None,
    dataframe_display=None,
    tablereport_threshold=None,
    subsampling_seed=None,
    enable_subsampling=None,
):
    """Set global skrub configuration.

    Parameters
    ----------
    expression_display : {'tablereport', 'original'}, default=None
        The type of HTML representation used for the dataframes preview in
        skrub expressions. Default is 'tablereport'.

        - If 'tablereport', :class:`~skrub.TableReport` will be used.
        - If 'original', the original Pandas or Polars dataframe display will be used.

    dataframe_display : {'tablereport', 'original'}, default=None
        The type of display used for dataframes. Default is 'original'.

        - If 'tablereport', replace the default DataFrame HTML displays with
          :class:`~skrub.TableReport`.
        - If 'original', the original Pandas or Polars dataframe HTML representation
          will be used.

    tablereport_threshold : int, default=None
        Set both the 'max_plot_columns' and 'max_association_columns' argument of
        :class:`~skrub.TableReport`. Default is 30.

    subsampling_seed : int, default=None
        Set the random seed of subsampling in skrub expressions
        :func:`skrub.Expr.skb.subsample`, when how='random' is passed.

    enable_subsampling : {'default', 'disable', 'force'}, default=None
        Control the activation of subsampling in skrub expressions
        :func:`skrub.Expr.skb.subsample`. Default is 'default'.

        - If 'default', the behavior of :func:`skrub.Expr.skb.subsample` is used.
        - If 'disable', subsampling is never used, so `skb.subsample` becomes a no-op.
        - If 'force', subsampling is used in all expression evaluation modes (preview,
          fit_transform, etc.).

    See Also
    --------
    get_config : Retrieve current values for global configuration.
    config_context : Context manager for global skrub configuration.

    Examples
    --------
    >>> from skrub import set_config
    >>> set_config(expression_display='tablereport')  # doctest: +SKIP
    """
    local_config = _get_threadlocal_config()
    if expression_display is not None:
        if expression_display not in (options := ("tablereport", "original")):
            raise ValueError(
                f"'expression_display' options are {options!r}, got "
                f"{expression_display!r}."
            )
        local_config["expression_display"] = expression_display

    if dataframe_display is not None:
        if dataframe_display not in (options := ("tablereport", "original")):
            raise ValueError(
                f"'dataframe_display' options are {options!r}, got "
                f"{dataframe_display!r}."
            )
        local_config["dataframe_display"] = dataframe_display

    if tablereport_threshold is not None:
        if not isinstance(tablereport_threshold, numbers.Real):
            raise ValueError(
                "'tablereport_threshold' should be a number, got "
                f"{type(tablereport_threshold)!r}"
            )
        local_config["tablereport_threshold"] = tablereport_threshold

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

    _apply_external_patches(local_config)


@contextmanager
def config_context(
    *,
    expression_display=None,
    dataframe_display=None,
    tablereport_threshold=None,
    subsampling_seed=None,
    enable_subsampling=None,
):
    """Context manager for global skrub configuration.

    Parameters
    ----------
    expression_display : {'tablereport', 'original'}, default=None
        The type of HTML representation used for the dataframes preview in
        skrub expressions. Default is 'tablereport'.

        - If 'tablereport', :class:`~skrub.TableReport` will be used.
        - If 'original', the original Pandas or Polars dataframe display will be used.

    dataframe_display : {'tablereport', 'original'}, default=None
        The type of display used for dataframes. Default is 'original'.

        - If 'tablereport', replace the default DataFrame HTML displays with
          :class:`~skrub.TableReport`.
        - If 'original', the original Pandas or Polars dataframe HTML representation
          will be used.

    tablereport_threshold : int, default=None
        Set both the 'max_plot_columns' and 'max_association_columns' argument of
        :class:`~skrub.TableReport`. Default is 30.

    subsampling_seed : int, default=None
        Set the random seed of subsampling in skrub expressions
        :func:`skrub.Expr.skb.subsample`, when how='random' is passed.

    enable_subsampling : {'default', 'disable', 'force'}, default=None
        Control the activation of subsampling in skrub expressions
        :func:`skrub.Expr.skb.subsample`. Default is 'default'.

        - If 'default', the behavior of :func:`skrub.Expr.skb.subsample` is used.
        - If 'disable', subsampling is never used, so `skb.subsample` becomes a no-op.
        - If 'force', subsampling is used in all expression evaluation modes (preview,
          fit_transform, etc.).

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
    >>> X = skrub.datasets.fetch_employee_salaries().X
    >>> with skrub.config_context(tablereport_threshold=1):
    ...     skrub.TableReport(X).open()
    """
    original_config = get_config()
    set_config(
        expression_display=expression_display,
        dataframe_display=dataframe_display,
        tablereport_threshold=tablereport_threshold,
        subsampling_seed=subsampling_seed,
        enable_subsampling=enable_subsampling,
    )

    try:
        yield
    finally:
        set_config(**original_config)
