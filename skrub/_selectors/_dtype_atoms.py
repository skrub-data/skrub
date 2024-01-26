from .. import _dataframe as sbd
from ._atoms import Filter
from ._base import Selector


class Numeric(Filter):
    def __init__(self):
        super().__init__(sbd.is_numeric)

    def __repr__(self):
        return "numeric()"


def numeric():
    return Numeric()


class AnyDate(Filter):
    def __init__(self):
        super().__init__(sbd.is_anydate)

    def __repr__(self):
        return "anydate()"


def anydate():
    return AnyDate()


class Categorical(Filter):
    def __init__(self):
        super().__init__(sbd.is_categorical)

    def __repr__(self):
        return "categorical()"


def categorical():
    return Categorical()


class String(Filter):
    def __init__(self):
        super().__init__(sbd.is_string)

    def __repr__(self):
        return "string()"


def string():
    return String()
