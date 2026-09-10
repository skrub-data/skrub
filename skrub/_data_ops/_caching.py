"""
Helper to cache functions and estimator methods according to the config.
"""

import pickle
import types

import joblib

from .. import _config


def _call_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    result = getattr(estimator, method_name)(*args, **kwargs)
    return estimator, result


def _call_non_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    return getattr(estimator, method_name)(*args, **kwargs)


def _call_deferred_func(func, args, kwargs, globals, closure, defaults, kwdefaults):
    if globals or closure or defaults:
        # The deferred function has skrub DataOps (that need to be
        # evaluated) in its global variables, free variables or default
        # arguments. In this case after those are evaluated, we recompile a
        # new function in which the DataOps have been replaced by their
        # computed value. More details in the docstring of
        # `skrub.deferred`.
        func = types.FunctionType(
            func.__code__,
            globals={**func.__globals__, **globals},
            argdefs=defaults,
            closure=tuple(types.CellType(c) for c in closure),
        )
    kwargs = (kwdefaults or {}) | kwargs
    return func(*args, **kwargs)


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

    def _check_cache_dir(self):
        """
        Recreate the joblib Memory if the cache configuration has changed.
        """
        cache_dir = _config.get_cache_dir()
        if cache_dir == self.cache_dir:
            return
        self.cached_func = {}
        self.memory = joblib.Memory(cache_dir, verbose=0)
        self.cache_dir = cache_dir

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
        key = (func, ignore)
        try:
            return self.cached_func[key]
        except KeyError:
            pass
        result = self.memory.cache(func, ignore=ignore)
        self.cached_func[key] = result
        return result

    def call_deferred_func(
        self, func, args, kwargs, globals, closure, defaults, kwdefaults, *, no_cache
    ):
        all_args = (func, args, kwargs, globals, closure, defaults, kwdefaults)
        if no_cache or not self.has_memory():
            return _call_deferred_func(*all_args)
        try:
            return self.cache(_call_deferred_func)(*all_args)
        except pickle.PicklingError:
            pass
        # Fall back to non-cached call if arguments cannot be serialized
        return _call_deferred_func(*all_args)

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
        except pickle.PicklingError:
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
        except pickle.PicklingError:
            pass
        # Fall back to non-cached call if arguments cannot be serialized
        return getattr(estimator, method_name)(*args, **kwargs)
