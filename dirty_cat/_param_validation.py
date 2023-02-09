"""
Provides the main decorator `validate_types` for validating the input/returned
types, as well as class attributes.

Another decorator `validate_types_with_inspect` is available if
inspection is a required feature. Only use if necessary.
Eventually, this feature should be implemented in the main decorator.

This is private API dedicated to the dirty_cat developers.
It may be used by other projects, without guarantees - we implement only
what we need in dirty_cat.
Improvement ideas are welcome.
"""

import inspect
import re
import sys
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
    Custom warning raised when the default value of a parameter does not
    match the annotation.
    """


def _validate_value(
    *,
    name: typing.Optional[str] = None,
    value: Any,
    annotation: Any,
) -> None:
    """
    Takes a value and its annotation, and validate they match.
    Works recursively if types are nested.

    Parameters
    ----------
    name : str, optional
        Name given to the value. Only relevant if `action="raise"` at it is
        used in the error message.
    value : Any
        Value to check the annotation against.
    annotation : Any
        Annotation to check the value against.
        May contain types from the `typing` module, classes and built-in types.

    Raises
    ------
    InvalidParameterError
        If the annotation and the value do not match.

    Returns
    -------
    None
        See section "Raises" for more information

    """

    def get_typing_alias_name(alias) -> str:
        """
        Helper function to get the name of a typing alias.
        """
        if sys.version_info[1] <= 9:
            # Quirk of this regex: it assumes it always ends with brackets.
            # `Literal` would not be caught in `typing.Literal`,
            # but would be with `typing.Literal[...]`
            # (even empty brackets work).
            return re.findall(r"^typing\.([^\[]+)\[.*$", repr(alias))[0]
        else:
            return alias.__name__

    # Special case for bool
    if annotation is bool:
        if not (value is True or value is False):
            raise InvalidParameterError(
                f"Expected {name!r} to be an instance of {annotation}, "
                f"got {value!r} (type {type(annotation)}) instead."
            )

    # Special case for None
    if isinstance(annotation, type(None)):
        if not (value is None or isinstance(value, type(None))):
            raise InvalidParameterError(
                f"Expected {name!r} to be None, "
                f"got {value!r} (type {type(annotation)}) instead."
            )

    elif type(annotation) is type:
        if not isinstance(value, annotation):
            raise InvalidParameterError(
                f"Expected {name!r} to be an instance of {annotation}, "
                f"got {value!r} (type {type(annotation)}) instead."
            )

    elif issubclass(type(annotation), (typing._GenericAlias, typing._SpecialForm)):
        # Types from the `typing` module

        # Get the type(s) contained in the brackets
        # (e.g. `(int, float)` in `Union[int, float]`).
        contained_types = annotation.__args__
        annotation_name = get_typing_alias_name(annotation)
        print(f"Found typing annotation: {annotation} with name {annotation_name}")

        if annotation_name == "Literal":
            # Literal should not contain nested types, so we won't recurse.
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
                    f"Expected {name!r} to be any of {contained_types} but "
                    f"got {value!r} instead."
                )

        elif annotation_name in ["Union", "Optional"]:
            # Can contain nested types, we will recurse.
            if not any(
                _validate_value_with_signal(
                    value=value,
                    annotation=contained_type,
                )
                for contained_type in contained_types
            ):
                raise InvalidParameterError(
                    f"Expected {name!r} to be any of {contained_types} "
                    f"but got {value!r} (type {type(value)}) instead."
                )

        elif annotation_name in ["List", "Set"]:
            # Unordered, we only want each item to be of the specified type
            # (there can only be one in the type definition).
            # E.g. `List[str]` is valid, `List[float, int]` isn't.
            inner_annotation = contained_types[0]
            # Just a cleaner if-else alternative for the type selection
            expected_type_map: typing.Dict[str, type] = {
                "List": list,
                "Set": set,
            }
            expected_type = expected_type_map[annotation_name]
            if not isinstance(value, expected_type):
                raise InvalidParameterError(
                    f"Expected {name!r} to be an instance of {expected_type!r} "
                    f"but got {value!r} (type {type(value)}) instead."
                )
            # Check that all values in the iterable are of the expected type
            # (recursive).
            for element in value:
                if not _validate_value_with_signal(
                    value=element,
                    annotation=inner_annotation,
                ):
                    raise InvalidParameterError(
                        f"Expected {name!r} to be of type {annotation} but "
                        f"got {value!r}, which contains the invalid element "
                        f"{element!r} (type {type(element)})."
                    )

        elif annotation_name == "Tuple":
            # Check that the value is actually a tuple.
            if not isinstance(value, tuple):
                raise InvalidParameterError(
                    f"Expected {name!r} to be of type {tuple} but "
                    f"got {value!r} (type {type(value)}) instead."
                )
            # Size matters
            if len(value) != len(contained_types):
                raise InvalidParameterError(
                    f"Expected {name!r} to be a tuple of length "
                    f"{len(contained_types)} (annotated {annotation}) but "
                    f"got {value!r} (length {len(value)})."
                )
            # Order matters
            for element, contained_type in zip(value, contained_types):
                if not _validate_value_with_signal(
                    value=element,
                    annotation=contained_type,
                ):
                    raise InvalidParameterError(
                        f"Expected {name!r} to be a tuple containing "
                        f"{contained_types}, but element {element!r} (type "
                        f"{type(element)}) does not match {contained_type}."
                    )

        elif annotation_name == "Dict":
            # Check that it's actually a dictionary
            if not isinstance(value, dict):
                raise InvalidParameterError(
                    f"Expected {name!r} to be of type {annotation} but "
                    f"got {value!r} (type {type(value)}) instead."
                )
            keys_type, values_type = contained_types
            # Validate keys
            for key in value.keys():
                if not _validate_value_with_signal(
                    value=key,
                    annotation=keys_type,
                ):
                    raise InvalidParameterError(
                        f"Expected {name!r} to be of type {annotation} but "
                        f"got {value!r}, which contains the invalid key "
                        f"{key!r} (type {type(key)})."
                    )
            # Validate values
            for dict_value in value.values():
                if not _validate_value_with_signal(
                    value=dict_value,
                    annotation=values_type,
                ):
                    raise InvalidParameterError(
                        f"Expected {name!r} to be of type {annotation} but "
                        f"got {value!r}, which contains the invalid value "
                        f"{dict_value!r} (type {type(dict_value)})."
                    )

        elif annotation_name == "Any":
            pass


def _validate_value_with_signal(**kwargs) -> bool:
    """
    Wrapper for `_validate_value`, which, instead of raising an error,
    returns a boolean value to indicate whether the type matches.
    DO NOT use directly, it's an internal function used as part of the recursion.

    Returns
    -------
    bool
        True if the value and the annotation match, False if they don't.

    """
    try:
        _validate_value(**kwargs)
    except InvalidParameterError:
        return False
    else:
        return True


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
            _validate_value(
                name=f"{instance.__class__.__name__}.{name}",
                value=value,
                annotation=annotation,
            )


def _validate_default_value(**kwargs):
    """
    Simple wrapper around the value validation, that raises a warning instead
    of an error when validating a default value.
    """
    try:
        _validate_value(**kwargs)
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
                _validate_default_value(
                    name=name,
                    value=parameter.default,
                    annotation=parameter.annotation,
                )
            _validate_value(
                name=name,
                value=value,
                annotation=parameter.annotation,
            )


def _validate_return(func: Callable, returned_value: Any):
    """
    Given a function, checks that the returned value matches the annotation.
    """
    sig = inspect.signature(func)
    if sig.return_annotation is not inspect.Parameter.empty:
        _validate_value(
            name="returned value",
            value=returned_value,
            annotation=sig.return_annotation,
        )


def validate_types_with_inspect(func):
    """
    Special validation decorator only used when there is a need for
    inspecting the decorated function's signature.
    The only drawback is that the parameters cannot be set,
    and are enforced as True.
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
    Decorator that can be used around any function to check the types of
    - the class attributes
    - the function's arguments
    - the function's return value

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
    in which case `class_parameters` is set to False automatically:
    >>> @validate_types()
    >>> def my_function(identifier: int, connections: typing.List[int]) -> bool:
    >>>     ...
    Notice the decorator is called: it's required when decorating a
    standalone function, but not for a class function (regardless if it's a
    static/class/instance method).

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
