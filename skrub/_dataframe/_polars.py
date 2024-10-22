"""
Polars specialization of the aggregate and join operations.
"""

# TODO: _dataframe._polars is temporary; all code in this module should be moved
# elsewhere and use the dispatch mechanism.


def rename_columns(dataframe, renaming_function):
    return dataframe.rename({c: renaming_function(c) for c in dataframe.columns})
