"""
functions meant to be cached with joblib.

They are in their own module so the cache is less likely to be invalidated due
to the line number of the function definition changing.
"""

import joblib


def _call_fitting_method(estimator, method_name, args, kwargs):
    # we could also just generate a str(uuid.uuid4()) 🤔
    estimator_id = joblib.hash((estimator, method_name, args, kwargs))
    result = getattr(estimator, method_name)(*args, **kwargs)
    return estimator, result, estimator_id


def _call_non_fitting_method(estimator, method_name, args, kwargs, estimator_id):
    return getattr(estimator, method_name)(*args, **kwargs)
