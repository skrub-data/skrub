"""Utilities specific to the JOIN operations."""

from skrub import _utils


def add_column_name_suffix(dataframes, suffix):
    """Add a suffix to all column names in a list of dataframes.

    This function does not modify the provided dataframes; it returns a list of
    new dataframes (but it does not copy the underlying data).

    Parameters
    ----------
    dataframes : list of DataFrames
        The dataframes whose columns must be renamed.

    suffix : str
        The suffix to add to all column names.

    Returns
    -------
    renamed : list of DataFrames
        The same dataframes, with their columns renamed.
    """
    if suffix == "":
        return dataframes
    renamed = []
    for df in dataframes:
        renamed.append(df.rename(columns={c: f"{c}{suffix}" for c in df.columns}))
    return renamed


def check_key(main_key, aux_key, key):
    """Find the correct main and auxiliary keys (matching column names).

    They can be provided either as `key` when the names are the same in
    both tables, or as `main_key` and `aux_key` when they differ. This
    function checks that only one of those options is used and returns
    the `main_key` and `aux_key` to use, as lists of strings.

    Parameters
    ----------
    main_key : list of str, str, or None
        Matching columns in the main table. Can be a single column name (str)
        if matching on a single column: ``"User_ID"`` is the same as
        ``"[User_ID]"``.
    aux_key : list of str, str, or None
        Matching columns in the auxiliary table. Can be a single column name (str)
        if matching on a single column: ``"User_ID"`` is the same as
        ``"[User_ID]"``.
    key : list of str, str, or None
        Can be provided in place of `main_key` and `aux_key` when they are the
        same. We must provide non-``None`` values for either `key` or both
        `main_key` and `aux_key`.

    Returns
    -------
    main_key, aux_key : pair (tuple) of lists of str
        The correct sets of matching columns to use, each provided as a list of
        column names.
    """
    if key is not None:
        if aux_key is not None or main_key is not None:
            raise ValueError(
                "Can only pass argument 'key' OR 'main_key' and "
                "'aux_key', not a combination of both."
            )
        main_key, aux_key = key, key
    else:
        if aux_key is None or main_key is None:
            raise ValueError("Must pass EITHER 'key', OR ('main_key' AND 'aux_key').")
    main_key = _utils.atleast_1d_or_none(main_key)
    aux_key = _utils.atleast_1d_or_none(aux_key)
    return main_key, aux_key
