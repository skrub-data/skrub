import numbers
import os
import threading
from contextlib import contextmanager

import numpy as np

from ._reporting import _patching

_global_config = {
    "use_tablereport": os.environ.get("SKB_USE_TABLEREPORT", False),
    "use_tablereport_expr": os.environ.get("SKB_USE_TABLEREPORT_EXPR", True),
    "tablereport_max_col": int(os.environ.get("SKB_TABLEREPORT_MAX_COL", 30)),
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
    if config["use_tablereport"]:
        _patching._patch_display(
            max_plot_columns=config["tablereport_max_col"],
            max_association_columns=config["tablereport_max_col"],
        )
    else:
        # No-op if dispatch haven't been previously enabled
        _patching._unpatch_display()


def set_config(
    use_tablereport=None,
    use_tablereport_expr=None,
    tablereport_max_col=None,
    subsampling_seed=None,
    enable_subsampling=None,
):
    """Set global skrub configuration.

    Parameters
    ----------
    use_tablereport : bool, default=None
        The type of display used for dataframes. Default is ``True``.

        - If ``True``, replace the default DataFrame HTML displays with
          :class:`~skrub.TableReport`.
        - If ``False``, the original Pandas or Polars dataframe HTML representation
          will be used.

        This configuration can also be set with the ``SKB_USE_TABLEREPORT``
        environment variable.

    use_tablereport_expr : bool, default=None
        The type of HTML representation used for the dataframes preview in skrub
        expressions. Default is ``False``.

        - If ``True``, :class:`~skrub.TableReport` will be used.
        - If ``False``, the original Pandas or Polars dataframe display will be
          used.

        This configuration can also be set with the ``SKB_USE_TABLEREPORT_EXPR``
        environment variable.

    tablereport_max_col : int, default=None
        Set both the ``max_plot_columns`` and ``max_association_columns`` argument
        of :class:`~skrub.TableReport`. Default is 30.

        This configuration can also be set with the ``SKB_TABLEREPORT_MAX_COL``
        environment variable.

    subsampling_seed : int, default=None
        Set the random seed of subsampling in skrub expressions
        :func:`skrub.Expr.skb.subsample`, when ``how="random"`` is passed.

        This configuration can also be set with the ``SKB_SUBSAMPLING_SEED`` environment
        variable.

    enable_subsampling : {'default', 'disable', 'force'}, default=None
        Control the activation of subsampling in skrub expressions
        :func:`skrub.Expr.skb.subsample`. Default is ``"default"``.

        - If ``"default"``, the behavior of :func:`skrub.Expr.skb.subsample` is used.
        - If ``"disable"``, subsampling is never used, so ``skb.subsample`` becomes a
          no-op.
        - If ``"force"``, subsampling is used in all expression evaluation modes
          (preview, fit_transform, etc.).

        This configuration can also be set with the ``SKB_ENABLE_SUBSAMPLING``
        environment variable.

    See Also
    --------
    get_config : Retrieve current values for global configuration.
    config_context : Context manager for global skrub configuration.

    Examples
    --------
    >>> from skrub import set_config
    >>> set_config(use_tablereport=True)  # doctest: +SKIP
    """
    local_config = _get_threadlocal_config()
    if use_tablereport is not None:
        if not isinstance(use_tablereport, bool):
            raise ValueError(
                f"'use_tablereport' must be a boolean, got {use_tablereport!r}."
            )
        local_config["use_tablereport"] = use_tablereport

    if use_tablereport_expr is not None:
        if not isinstance(use_tablereport_expr, bool):
            raise ValueError(
                "'use_tablereport_expr' must be a boolean, got "
                f"{use_tablereport_expr!r}."
            )
        local_config["use_tablereport_expr"] = use_tablereport_expr

    if tablereport_max_col is not None:
        if not isinstance(tablereport_max_col, numbers.Real):
            raise ValueError(
                "'tablereport_max_col' should be a number, got "
                f"{type(tablereport_max_col)!r}"
            )
        local_config["tablereport_max_col"] = tablereport_max_col

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
    use_tablereport=None,
    use_tablereport_expr=None,
    tablereport_max_col=None,
    subsampling_seed=None,
    enable_subsampling=None,
):
    """Context manager for global skrub configuration.

    Parameters
    ----------
    use_tablereport : bool, default=None
        The type of display used for dataframes. Default is ``False``.

        - If ``True``, replace the default DataFrame HTML displays with
          :class:`~skrub.TableReport`.
        - If ``False``, the original Pandas or Polars dataframe HTML representation
          will be used.

        This configuration can also be set with the ``SKB_USE_TABLEREPORT``
        environment variable.

    use_tablereport_expr : bool, default=None
        The type of HTML representation used for the dataframes preview in skrub
        expressions. Default is ``True``.

        - If ``True``, :class:`~skrub.TableReport` will be used.
        - If ``False``, the original Pandas or Polars dataframe display will be
          used.

        This configuration can also be set with the ``SKB_USE_TABLEREPORT_EXPR``
        environment variable.

    tablereport_max_col : int, default=None
        Set both the ``max_plot_columns`` and ``max_association_columns`` argument
        of :class:`~skrub.TableReport`. Default is 30.

        This configuration can also be set with the ``SKB_TABLEREPORT_MAX_COL``
        environment variable.

    subsampling_seed : int, default=None
        Set the random seed of subsampling in skrub expressions
        :func:`skrub.Expr.skb.subsample`, when ``how="random"`` is passed.

        This configuration can also be set with the ``SKB_SUBSAMPLING_SEED`` environment
        variable.

    enable_subsampling : {'default', 'disable', 'force'}, default=None
        Control the activation of subsampling in skrub expressions
        :func:`skrub.Expr.skb.subsample`. Default is ``"default"``.

        - If ``"default"``, the behavior of :func:`skrub.Expr.skb.subsample` is used.
        - If ``"disable"``, subsampling is never used, so ``skb.subsample`` becomes a
          no-op.
        - If ``"force"``, subsampling is used in all expression evaluation modes
          (preview, fit_transform, etc.).

        This configuration can also be set with the ``SKB_ENABLE_SUBSAMPLING``
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
    >>> with skrub.config_context(tablereport_max_col=1):
    ...     ...  # doctest: +SKIP
    """
    original_config = get_config()
    set_config(
        use_tablereport=use_tablereport,
        use_tablereport_expr=use_tablereport_expr,
        tablereport_max_col=tablereport_max_col,
        subsampling_seed=subsampling_seed,
        enable_subsampling=enable_subsampling,
    )

    try:
        yield
    finally:
        set_config(**original_config)
