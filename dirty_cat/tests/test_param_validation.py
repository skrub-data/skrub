import inspect
import typing
from pathlib import Path

import pytest

from dirty_cat._param_validation import InvalidParameterError, validate_types


class GenericThing:
    """
    Example class used to test the parameters validation system.
    """

    @validate_types()
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
        nested_arg1: typing.Union[typing.Literal[1, 0, "yes", "no"], bool],
        nester_arg2: typing.Optional[typing.Union[int, float]],
        tuple_arg: typing.Tuple[int, float, typing.List[str]],
        list_arg: typing.List[int],
        set_arg: typing.Set[typing.Union[int, float]],
        dict_arg: typing.Dict[str, int],
    ):
        self.bool_arg = bool_arg
        self.str_arg = str_arg
        self.int_arg = int_arg
        self.float_arg = float_arg
        self.class_arg = class_arg
        self.literal_arg = literal_arg
        self.optional_arg = optional_arg
        self.union_arg = union_arg
        self.nested_arg1 = nested_arg1
        self.nester_arg2 = nester_arg2
        self.tuple_arg = tuple_arg
        self.list_arg = list_arg
        self.set_arg = set_arg
        self.dict_arg = dict_arg

    @validate_types()
    def exec_correct(self, _: str, __: typing.Optional[int]) -> str:
        return "Works!"

    @validate_types()
    def exec_incorrect(self, _: str, __: typing.Optional[int]) -> int:
        pass


valid_confs = [
    (
        False,
        "wool",
        17,
        0.52,
        Path("./"),
        "yes",
        "lab",
        "culture",
        1,
        None,
        (5, 0.5, ["5", "55"]),
        [1, 2, 3],
        {15, 0.6, 0},
        {"t": 5},
    ),
    (
        False,
        "Romeo",
        190,
        0.1,
        Path("lib/"),
        "no",
        None,
        44,
        True,
        0.5,
        (1, 0.8, [""]),
        [50],
        {0.5},
        {"e": 1},
    ),
]


@validate_types()
def independent_function(
    _: str, __: int, ___: typing.Optional[Path], malfunction: bool = False
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
        ("nested_arg1", "hell yeah"),
        ("nested_arg1", None),
        ("nester_arg2", [5]),
        ("nester_arg2", "5"),
        ("tuple_arg", ()),
        ("tuple_arg", (0.5, 5, [])),
        ("tuple_arg", (5, 0.5, ())),
        ("list_arg", [0.5]),
        ("list_arg", ()),
        ("set_arg", dict()),
        ("dict_arg", ["str", 1]),
        ("dict_arg", {"str": 0.5}),
    ],
)
def test_invalid_combinations(param: str, value: typing.Any) -> None:
    """
    Given the name of a keyword argument for the example class and a value
    not matching the annotation for this parameter, checks that passing it
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
        GenericThing(*bound.args, **bound.kwargs)


def test_independent_function():
    independent_function("test", 55, None, malfunction=False)

    with pytest.raises(InvalidParameterError):
        independent_function(5, 55, None, malfunction=False)
        independent_function("test", 0.5, None, malfunction=False)
        independent_function("test", 55, "path", malfunction=False)
        independent_function("test", 55, None, malfunction=True)
