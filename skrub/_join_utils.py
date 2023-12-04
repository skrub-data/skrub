"""Utilities specific to the JOIN operations."""
from collections import Counter

from skrub import _utils
from skrub._dataframe._namespace import get_df_namespace


def check_key(
    main_key,
    aux_key,
    key,
    main_key_name="main_key",
    aux_key_name="aux_key",
    key_name="key",
):
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
    main_key_name : str
        How to refer to `main_key` in error messages.
    aux_key_name : str
        How to refer to `aux_key` in error messages.
    key_name : str
        How to refer to `key` in error messages.
    Returns
    -------
    main_key, aux_key : pair (tuple) of lists of str
        The correct sets of matching columns to use, each provided as a list of
        column names.
    """
    if key is not None:
        if aux_key is not None or main_key is not None:
            raise ValueError(
                f"Can only pass argument '{key_name}' OR '{main_key_name}' and "
                f"'{aux_key_name}', not a combination of both."
            )
        main_key, aux_key = key, key
    else:
        if aux_key is None or main_key is None:
            raise ValueError(
                f"Must pass EITHER '{key_name}', OR ('{main_key_name}' AND"
                f" '{aux_key_name}')."
            )
    main_key = _utils.atleast_1d_or_none(main_key)
    aux_key = _utils.atleast_1d_or_none(aux_key)
    if len(main_key) != len(aux_key):
        raise ValueError(
            f"'{main_key_name}' and '{aux_key_name}' keys have different lengths"
            f" ({len(main_key)} and {len(aux_key)}). Cannot join on different numbers"
            " of columns."
        )
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


def check_column_name_duplicates(
    main_table,
    aux_table,
    suffix,
    main_table_name="main_table",
    aux_table_name="aux_table",
):
    """Check that there are no duplicate column names after applying a suffix.

    The suffix is applied to (a copy of) `aux_columns` before checking for
    duplicates.

    Parameters
    ----------
    main_table : dataframe
        The main table to join.
    aux_table : dataframe
        The auxiliary table to join.
    suffix : str
        The suffix that was provided by the user and will be appended to the
        auxiliary column names.
    main_table_name : str
        How to refer to ``main_table`` in error messages.
    aux_table_name : str
        How to refer to ``aux_table`` in error messages.
    Raises
    ------
    ValueError
        If any of the table has duplicate column names, or if there are column
        names that are used in both tables.
    """
    main_columns = list(main_table.columns)
    aux_columns = [f"{col}{suffix}" for col in aux_table.columns]
    for columns, table_name in [
        (main_columns, main_table_name),
        (aux_columns, aux_table_name),
    ]:
        counts = Counter(columns)
        duplicates = [k for k, v in counts.items() if v > 1]
        if duplicates:
            raise ValueError(
                f"Table '{table_name}' has duplicate column names: {duplicates}."
                " Please make sure column names are unique."
            )
    overlap = list(set(main_columns).intersection(aux_columns))
    if overlap:
        raise ValueError(
            f"After applying the suffix {suffix!r} to column names, the following"
            f" column names are found in both tables '{main_table_name}' and"
            f" '{aux_table_name}': {overlap}. Please make sure column names do not"
            " overlap by renaming some columns or choosing a different suffix."
        )


def add_column_name_suffix(dataframe, suffix):
    ns, _ = get_df_namespace(dataframe)
    return ns.rename_columns(dataframe, f"{{}}{suffix}".format)
