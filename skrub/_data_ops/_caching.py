"""
Helper to cache functions and estimator methods according to the config.
"""

import pickle
import types

import joblib

from .. import _config

# The functions meant to be cached. They are at the top because joblib
# invalidates the cache when the line number changes.


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


class Memory:
    def __init__(self):
        self.cache_dir = None
        self.memory = None
        self.cached_func = {}

    def _check_cache_dir(self):
        cache_dir = _config.get_cache_dir()
        if cache_dir == self.cache_dir:
            return
        self.cached_func = {}
        self.memory = joblib.Memory(cache_dir, verbose=0)
        self.cache_dir = cache_dir

    def has_memory(self):
        self._check_cache_dir()
        return self.memory is not None

    def cache(self, func, ignore=()):
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
            return self.cache(_call_non_fitting_method, ignore=("estimator",))(
                estimator, method_name, args, kwargs, estimator_id
            )
        except pickle.PicklingError:
            pass
        # Fall back to non-cached call if arguments cannot be serialized
        return getattr(estimator, method_name)(*args, **kwargs)
