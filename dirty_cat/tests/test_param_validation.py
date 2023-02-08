import inspect
import typing
from pathlib import Path

import pytest

from dirty_cat._param_validation import (
    InvalidParameterError,
    validate_types,
    validate_types_with_inspect,
)


class GenericThing:
    """
    Example class used to test the parameters validation system.
    """

    @validate_types_with_inspect
    def __init__(
        self,
        bool_arg: bool,
        str_arg: str,
        int_arg: int,
        float_arg: float,
        class_arg: Path,
        literal_arg: typing.Literal["yes", "no"],
        optional_arg: typing.Optional[str],
        union_arg: typing.Union[str, int],
    ):
        self.bool_arg = bool_arg
        self.str_arg = str_arg
        self.int_arg = int_arg
        self.float_arg = float_arg
        self.class_arg = class_arg
        self.literal_arg = literal_arg
        self.optional_arg = optional_arg
        self.union_arg = union_arg

    @validate_types
    def exec_correct(self, arg1: str, arg2: typing.Optional[int]) -> str:
        return "Worked!"

    @validate_types
    def exec_incorrect(self, arg1: str, arg2: typing.Optional[int]) -> int:
        pass


valid_confs = [
    (False, "wool", 17, 0.52, Path("./"), "yes", "lab", "culture"),
    (False, "Romeo", 190, 0.1, Path("lib/"), "no", None, 44),
]


@validate_types()
def independent_function(
    arg1: str, arg2: int, arg3: typing.Optional[Path], malfunction: bool = False
) -> bool:
    """
    Standalone function to test.
    If malfunction is passed, returns an invalid return value.
    """
    if malfunction:
        return 123
    else:
        return False


@pytest.mark.parametrize(
    "args",
    valid_confs,
)
def test_valid_combinations(args) -> None:
    """
    Takes positional parameters for the example class, and initializes it
    with them.
    All passed combinations should be valid.
    """
    GenericThing(*args)


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("bool_arg", "day"),
        ("bool_arg", None),
        ("bool_arg", 1),
        ("str_arg", 90),
        ("int_arg", 0.6),
        ("float_arg", "night"),
        ("class_arg", ""),
        ("literal_arg", "nop"),
        ("literal_arg", 10),
        ("literal_arg", None),
        ("optional_arg", 1940),
        ("union_arg", 0.0),
        ("union_arg", None),
    ],
)
def test_invalid_combinations(param: str, value: typing.Any) -> None:
    """
    Given the index of a positional argument for the example class and an
    invalid value for this parameter, checks that passing this value
    actually raises an error.
    """
    # Provide a default VALID configuration that we're going to modify one
    # element at a time to validate that each validation works.
    default_args = valid_confs[0]
    sig = inspect.signature(GenericThing)
    bound = sig.bind(*default_args)
    # Replace the valid value with the invalid one passed
    bound.arguments[param] = value
    with pytest.raises(InvalidParameterError):
        assert GenericThing(*bound.args, **bound.kwargs)


def test_independent_function():
    independent_function("test", 55, None, malfunction=False)

    with pytest.raises(InvalidParameterError):
        independent_function("test", 0.5, None, malfunction=False)
        independent_function(5, 55, None, malfunction=False)
        independent_function("test", 55, "path", malfunction=False)
        independent_function("test", 55, None, malfunction=True)
