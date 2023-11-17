import enum


class Selector(enum.Enum):
    ALL = enum.auto()
    NONE = enum.auto()
    NUMERIC = enum.auto()
    CATEGORICAL = enum.auto()


def std(obj):
    try:
        return obj.__dataframe_consortium_standard__()
    except AttributeError:
        return obj.__column_consortium_standard__()


def stdns(dataframe):
    return dataframe.__dataframe_consortium_standard__().__dataframe_namespace__()
