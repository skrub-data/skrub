from ._atoms import custom, filter, filter_names, glob, regex
from ._base import all, cols, inv, make_selector, select
from ._dtype_atoms import anydate, categorical, numeric, string

__all__ = [
    "make_selector",
    "all",
    "cols",
    "inv",
    "glob",
    "regex",
    "filter",
    "filter_names",
    "custom",
    "numeric",
    "anydate",
    "categorical",
    "string",
]

for name in __all__:
    setattr(select, name, globals()[name])

__all__.append("select")
