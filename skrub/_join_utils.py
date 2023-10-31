"""Utilities specific to the JOIN operations."""

from skrub import _utils


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


def check_missing_columns(table, key, table_name):
    """Check that all the columns in `key` can be found in `table`.

    Parameters
    ----------
    table : DataFrame
        The table that should contain the columns listed in `key`.

    key : list of str
        List of column names, all of which must be found in `table`.

    table_name : str
        Name by which to refer to `table` in the error message if necessary.
    """
    missing_columns = set(key) - set(table.columns)
    if not missing_columns:
        return
    raise ValueError(
        "The following columns cannot be used for joining because they do not exist"
        f" in {table_name}:\n{missing_columns}"
    )
