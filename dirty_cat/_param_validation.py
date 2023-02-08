"""
Provides two decorators for validating input arguments:
- `validate_types` for all functions. Is customizable.
- `validate_types_with_inspect` for the class' `__init__`.

This is private API dedicated to the dirty_cat developers.
It may be used by other projects, without guarantees - we implement only
what we need in dirty_cat.
Improvements ideas are welcome.
"""

import inspect
import typing
from functools import wraps
from typing import Any, Callable
from warnings import warn


class InvalidParameterError(ValueError, TypeError):
    """
    Custom exception raised when the parameter of a
    class/method/function does not have a valid type or value.
    """


class InvalidDefaultWarning(UserWarning):
    """
    Custom warning raised when the default value of a signature does not
    match the annotation.
    """


def _validate_value(name: str, value: Any, annotation) -> typing.Optional[bool]:
    """
    Takes a value and an annotation, and validates that they match.
    If they don't, a clean `InvalidParameterError` is raised.
    """
    # Special case for bool
    if annotation is bool:
        if not (value is True or value is False):
            raise InvalidParameterError(
                f"Expected {name!r} to be an instance of {annotation}, "
                f"got {value!r} (type {type(annotation)}) instead."
            )
        return

    if type(annotation) is type:
        if not isinstance(value, annotation):
            raise InvalidParameterError(
                f"Expected {name!r} to be an instance of {annotation}, "
                f"got {value!r} (type {type(annotation)}) instead."
            )
        return

    if issubclass(type(annotation), typing._GenericAlias):
        contained_types = annotation.__args__

        if annotation.__name__ == "Literal":
            # For the comparisons to make sense,
            # we'll divide the values in 2 categories:
            #  - instanced objects, we'll compare them with `==`
            #  - classes, we'll compare them with `is`
            inst_objs = []
            classes = []
            for v in contained_types:
                if not hasattr(v, "__class__") or isinstance(v, bool):
                    classes.append(v)
                else:
                    inst_objs.append(v)

            if not any([value is cls for cls in classes]) and not any(
                [value == val for val in inst_objs]
            ):
                raise InvalidParameterError(
                    f"Expected {name!r} to be any of {contained_types}, "
                    f"got {value!r} instead."
                )

            return

        elif annotation.__name__ in ["Union", "Optional"]:
            if not any(isinstance(value, cls) for cls in contained_types):
                raise InvalidParameterError(
                    f"Expected {name!r} to be an instance of "
                    f"{contained_types}, got {value!r} instead."
                )


def _validate_class_parameters(instance):
    """
    Takes the instance of class, and validates that the parameters taken
    at initialization are set as instance attributes
    (could be class attributes, we don't really care in this case).
    """
    sig = inspect.signature(instance.__init__)
    # Gets a mapping of parameter name to the instance attribute value
    values = {
        name: instance.__getattr__(name)
        for name in sig.parameters
        if name not in {"self", "args", "kwargs"}
    }
    bound = sig.bind(**values)
    for name, value in bound.arguments.items():
        annotation = sig.parameters[name].annotation
        if annotation is not inspect.Parameter.empty:
            _validate_value(name, value, annotation)


def _validate_default_value(*args, **kwargs):
    """
    Simple wrapper around the value validation, that raises a warning instead
    of an error when validating a default value.
    """
    try:
        _validate_value(*args, **kwargs)
    except InvalidParameterError as e:
        warn(str(e), InvalidDefaultWarning)


def _validate_parameters(func: Callable, args: tuple, kwargs: dict):
    """
    Takes a function, and validates that the arguments passed match their
    annotations.
    """
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    for name, value in bound.arguments.items():
        parameter = sig.parameters[name]
        if parameter.annotation is not inspect.Parameter.empty:
            # If the parameter has a default, check it is valid.
            # If it isn't, raise a warning. If it is used as the value,
            # an error will be raised anyway afterwards.
            if parameter.default is not inspect.Parameter.empty:
                _validate_default_value(name, parameter.default, parameter.annotation)
            _validate_value(name, value, parameter.annotation)


def _validate_return(func: Callable, returned_value: Any):
    """
    Given a function, checks that the returned value matches the annotation.
    """
    sig = inspect.signature(func)
    if sig.return_annotation is not inspect.Parameter.empty:
        _validate_value("returned", returned_value, sig.return_annotation)


def validate_types_with_inspect(func):
    """
    Special validation decorator only used when there is a need for
    inspecting the decorated function's signature.
    The only drawback is the
    """

    @wraps(func)
    def decorator(*args, **kwargs):
        validate_types(class_parameters=True, parameters=True, returned=True,)(
            func
        )(*args, **kwargs)

    return decorator


def validate_types(
    class_parameters: bool = True,
    parameters: bool = True,
    returned: bool = True,
):
    """
    Decorator that can be used around any function.

    If used as part of a class, `class_parameters` is relevant:
    we assume the arguments taken in `__init__` are set as instance attributes
    as-is.

    For instance, take this class:
    >>> class Encoder:
    >>>     def __init__(self, x, y, z):
    >>>         self.x = x
    >>>         self.y = y
    >>>         self.z = z
    notice the names are the same between the signature and the attributes.
    This is important!

    Then, an instance method can be decorated, and the options tweaked.

    The decorator also works on standalone functions and static/class methods,
    in which case `class_parameters` is set to False automatically.

    Uses the type hints to check the values passed are correct.
    If an inconsistency is found, a clean error is raised.

    Parameters
    ----------
    class_parameters: bool, default=True
        Whether we should check that the attributes set during `__init__`
        (see note above) are of the correct type.
    parameters: bool, default=True
        Whether we should check the parameters
        passed to the decorated function.
    returned: bool, default=True
        Whether we should check the value returned by the decorated function.

    Notes
    -----
    Uses `inspect` to parse signatures.
    `class_parameters` and `returned` are automatically set to False when
    the decorated function is the constructor.
    Type hints are represented differently depending on the version of Python.
    For example:
    >>> from typing import List, Optional
    >>> l: Optional[List[str]]  # Python >=3.7
    >>> l: Optional[list[str]]  # Python >=3.9
    >>> l: list[str] | None  # >=Python 3.10

    Examples
    --------
    Here's an example of the usage of this decorator.
    See `dirty_cat/tests/test_param_validation.py` for more.

    >>> from typing import Optional
    >>>
    >>> class Encoder:
    >>>
    >>>     @validate_types
    >>>     def __init__(self, handle_error: Literal["raise", "ignore"],
    >>>                  n_jobs: Optional[int]):
    >>>         self.handle_error = handle_error
    >>>         self.n_jobs = n_jobs
    >>>
    >>>     @validate_types
    >>>     def fit(self, X: pd.DataFrame) -> np.ndarray:
    >>>         ...
    """

    def decorator(func):
        nonlocal class_parameters
        nonlocal parameters
        nonlocal returned

        # Modify some parameters depending on the context
        if "." not in func.__qualname__:
            # Function is standalone
            class_parameters = False
        else:
            # Function is part of a class
            if isinstance(
                inspect.getattr_static(func.__class__, func.__name__),
                (staticmethod, classmethod),
            ):
                # Function is a static/class method
                class_parameters = False
            if func.__name__ == "__init__":
                # Function is the constructor
                class_parameters = False
                returned = False

        @wraps(func)
        def wrapper(*args, **kwargs):
            if class_parameters:
                _validate_class_parameters(args[0])  # args[0] = self
            if parameters:
                _validate_parameters(func, args, kwargs)
            returned_value = func(*args, **kwargs)
            if returned:
                _validate_return(func, returned_value)
            return returned_value

        return wrapper

    return decorator
