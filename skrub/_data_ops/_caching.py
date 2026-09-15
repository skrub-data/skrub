"""
Helper to cache functions and estimator methods according to the config.
"""

import subprocess
import sys
from pathlib import Path

import joblib

from .. import _config


def _call_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    result = getattr(estimator, method_name)(*args, **kwargs)
    return estimator, result


def _call_non_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    return getattr(estimator, method_name)(*args, **kwargs)


# Note: the config is stored in a thread-local variable (mostly because it has
# been copy-pasted from scikit-learn) but there is probably no valid use-case
# for setting the cache dir concurrently in different threads; for simplicity
# we do not handle it here and thus the Memory is not thread-safe. The config
# cache_dir / data_dir should not be modified concurrently by different
# threads.


class Memory:
    """
    Wrapper around joblib.Memory to handle the functions we need to cache and
    take the skrub config into account.

    For caching estimator methods,

    - When fitting an estimator_id is generated from the class and arguments
    - When predicting the estimator_id is used for hashing, rather than the
      fitted estimator itself (which could cause spurious cache misses,
      serialization errors and hashing computation time).

    The cached function wrapped by call_deferred_func takes care of recompiling
    a function with the evaluated globals, defaults and closure if needed.
    """

    def __init__(self):
        self.cache_dir = None
        self.memory = None
        self.cached_func = {}
        self._ran_reduce_cache = False

    def _reduce_cache_size(self):
        target_size = _config.get_config()["target_cache_size"]
        if str(target_size).lower() in ("none", ""):
            return
        script = (Path(__file__).parent / "_reduce_cache_size.py").resolve()
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            kwargs["start_new_session"] = True
        # keep a reference to the handle to avoid ResourceWarning
        self._pruning_subprocess = subprocess.Popen(
            [sys.executable, str(script), str(self.cache_dir), str(target_size)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )

    def _check_cache_dir(self):
        """
        Recreate the joblib Memory if the cache configuration has changed.
        """
        cache_dir = _config.get_cache_dir()
        if cache_dir == self.cache_dir:
            return
        self.cached_func = {}
        self.cache_dir = cache_dir
        if self.cache_dir is None:
            self.memory = None
        else:
            # Always pass a string to Memory because it behaves differently if
            # we pass a Path (adds joblib/ for strings but not for Paths)
            # https://github.com/joblib/joblib/issues/1684
            self.memory = joblib.Memory(str(cache_dir), verbose=0)
            if not self._ran_reduce_cache:
                self._ran_reduce_cache = True
                try:
                    self._reduce_cache_size()
                except Exception:
                    pass

    def has_memory(self):
        """
        Update self.memory based on config and return True if caching is enabled.
        """
        self._check_cache_dir()
        return self.memory is not None

    def cache(self, func, ignore=()):
        # ignore is passed to joblib.Memory.cache: parameters that are not
        # hashed / taken into account for caching
        self._check_cache_dir()
        if self.memory is None:
            return func
        key = (id(func), ignore)
        try:
            # The value stored in `cached_func` keeps the func object
            # alive, so if it is found we know the id cannot have been reused
            # and it is the same object.
            return self.cached_func[key]
        except KeyError:
            pass
        result = self.memory.cache(func, ignore=ignore)
        self.cached_func[key] = result
        return result

    def call_func(self, func, args, kwargs, *, no_cache):
        if (
            no_cache
            or not self.has_memory()
            or getattr(func, "__name__", None) == "<lambda>"
            or getattr(func, "__closure__", None)
        ):
            return func(*args, **kwargs)
        try:
            return self.cache(func)(*args, **kwargs)
        except Exception:
            # It could be caused by caching (e.g. unhashable args) so try again
            # without.
            pass
        return func(*args, **kwargs)

    def call_fitting_method(self, estimator, method_name, args, kwargs, *, no_cache):
        if no_cache or not self.has_memory():
            result = getattr(estimator, method_name)(*args, **kwargs)
            return estimator, result, None
        try:
            estimator_id = joblib.hash((estimator, method_name, args, kwargs))
            estimator, result = self.cache(
                _call_fitting_method,
                # those arguments are ignored for caching because their hash is
                # already captured in the estimator_id.
                ignore=("estimator", "method_name", "args", "kwargs"),
            )(estimator, method_name, args, kwargs, estimator_id)
            return estimator, result, estimator_id
        except Exception:
            # It could be caused by caching (e.g. unhashable args) so try again
            # without.
            pass
        # Fall back to non-cached call if arguments cannot be serialized
        result = getattr(estimator, method_name)(*args, **kwargs)
        return estimator, result, None

    def call_non_fitting_method(
        self, estimator, method_name, args, kwargs, estimator_id, *, no_cache
    ):
        if no_cache or not self.has_memory() or estimator_id is None:
            return getattr(estimator, method_name)(*args, **kwargs)
        try:
            # ignore estimator, rely on the hash of estimator_id instead
            return self.cache(_call_non_fitting_method, ignore=("estimator",))(
                estimator, method_name, args, kwargs, estimator_id
            )
        except Exception:
            # It could be caused by caching (e.g. unhashable args) so try again
            # without.
            pass
        # Fall back to non-cached call if arguments cannot be serialized
        return getattr(estimator, method_name)(*args, **kwargs)
