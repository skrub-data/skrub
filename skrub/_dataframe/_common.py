import enum


class Selector(enum.Enum):
    ALL = enum.auto()
    NONE = enum.auto()
    NUMERIC = enum.auto()
    CATEGORICAL = enum.auto()
    STRING = enum.auto()
