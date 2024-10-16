"""
Pandas specialization of the aggregate and join operations.
"""

# TODO: _dataframe._pandas is temporary; all code in this module should be moved
# elsewhere and use the dispatch mechanism.


def rename_columns(dataframe, renaming_function):
    return dataframe.rename(
        columns={c: renaming_function(c) for c in dataframe.columns}
    )
