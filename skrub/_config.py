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
    """
    TODO
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
    """
    TODO
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
    """
    TODO
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
