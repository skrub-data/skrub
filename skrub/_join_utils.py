"""Utilities specific to the JOIN operations."""

import inspect
import re

from skrub import _dataframe as sbd
from skrub import _selectors as s
from skrub import _utils
from skrub._dataframe._namespace import get_df_namespace
from skrub._dispatch import dispatch


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
        ``["User_ID"]``.
    aux_key : list of str, str, or None
        Matching columns in the auxiliary table. Can be a single column name (str)
        if matching on a single column: ``"User_ID"`` is the same as
        ``["User_ID"]``.
    key : list of str, str, or None
        Can be provided in place of `main_key` and `aux_key` when they are the
        same. We must provide non-``None`` values for either `key` or both
        `main_key` and `aux_key`.
    main_key_name : str, default="main_key"
        How to refer to `main_key` in error messages.
    aux_key_name : str, default="aux_key"
        How to refer to `aux_key` in error messages.
    key_name : str, default="key"
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
                f"Can only pass argument '{key_name}' or '{main_key_name}' and "
                f"'{aux_key_name}', not a combination of both."
            )
        main_key, aux_key = key, key
    else:
        if aux_key is None or main_key is None:
            raise ValueError(
                f"Must pass either '{key_name}', or ('{main_key_name}' and"
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
        "The following columns cannot be used because they do not exist"
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
        _utils.check_duplicated_column_names(columns, table_name=table_name)
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


def pick_column_names(suggested_names, forbidden_names=()):
    """Choose column names without duplicates.

    A tag ``__skrub_<random string>__`` is added at the end of columns that
    would otherwise be duplicates.

    If a single similar tag is present in a column name, it is shifted to the
    end of the column name. (If there are several they are removed, or replaced
    by a new one if necessary.)

    When there are duplicates, the first (leftmost) occurrence is the one left
    unchanged.

    We can pass a list of forbidden names, in which case names will also be
    tagged if they appear in the forbidden names (regardless of whether they
    have duplicates in the list and of their position).

    Parameters
    ----------
    suggested_names : list of str
        The list of column names to transform.

    forbidden_names : list of str
        A list of names that must not appear in the output.

    Returns
    -------
    list of str
        The chosen names. It has the same length as ``suggested_names`` and
        ``__skrub`` tags have been added to duplicated names.

    Examples
    --------
    >>> from skrub._join_utils import pick_column_names
    >>> pick_column_names(["A", "A", "B"])  # doctest: +SKIP
    ['A', 'A__skrub_87946836__', 'B']
    >>> pick_column_names(['A__skrub_750a0b7c__', 'B'])  # doctest: +SKIP
    ['A__skrub_750a0b7c__', 'B']
    >>> pick_column_names(
    ...     ["A__skrub_750a0b7c___year", "A__skrub_750a0b7c___month", "B"]
    ... )  # doctest: +SKIP
    ['A_year__skrub_750a0b7c__', 'A_month__skrub_750a0b7c__', 'B']
    >>> pick_column_names(["A", "B"], forbidden_names=["A", "B", "C"])  # doctest: +SKIP
    ['A__skrub_37dd63aa__', 'B__skrub_21e27e1e__']
    >>> pick_column_names(
    ...     ["concat_A__skrub_750a0b7c___A__skrub_b1eeb4f7__"]
    ... ) # doctest: +SKIP
    ['concat_A_A']
    """
    all_new_names = []
    forbidden_names = set(forbidden_names)
    for name in suggested_names:
        new_name = _get_new_name(name, forbidden_names)
        all_new_names.append(new_name)
        forbidden_names.add(new_name)
    return all_new_names


def _get_new_name(suggested_name, forbidden_names):
    tag_pattern = "__skrub_.*?__"
    tags = re.findall(tag_pattern, suggested_name)
    untagged_name = re.sub(tag_pattern, "", suggested_name)
    if len(tags) == 1:
        suggested_name = untagged_name + tags[0]
    else:
        suggested_name = untagged_name
    if suggested_name not in forbidden_names:
        return suggested_name
    token = _utils.random_string()
    return f"{untagged_name}__skrub_{token}__"


def left_join(left, right, left_on, right_on, rename_right_cols="{}"):
    """Left join two dataframes of the same type.

    The input dataframes type must agree: both `left` and `right` need to be
    pandas or polars dataframes. Mixing types will raise an error.

    `rename_right_cols` can be used to format the right dataframe columns, e.g. use
    "right_.{}" to rename all right cols with a leading "right_.".

    If duplicate column names are found between renamed right cols and left cols,
    a __skrub_<random string>__ is added at the end of columns that would otherwise
    be duplicates.

    Parameters
    ----------
    left : dataframe
        The left dataframe of the left-join.
    right : dataframe
        The right dataframe of the left-join.
    left_on : str or list of str
        Left keys to merge on.
    right_on : str or list of str
        Right keys to merge on.
    rename_right_cols : str or callable, default="{}"
        Formatting used to rename right cols. If it is a callable, it should
        accept strings as an argument. By default, no formatting is applied.

    Returns
    -------
    dataframe
        The joined output.

    Raises
    ------
    TypeError
        If either of `left` and `right` is not a dataframe, or if both types
        are not equal.
    """
    if not sbd.is_dataframe(left):
        raise TypeError(
            f"`left` must be a pandas or polars dataframe, got {type(left)}."
        )
    if not sbd.is_dataframe(right):
        raise TypeError(
            f"`right` must be a pandas or polars dataframe, got {type(right)}."
        )
    if not sbd.dataframe_module_name(left) == sbd.dataframe_module_name(right):
        raise TypeError(
            "`left` and `right` must be of the same dataframe type, got"
            f"{type(left)} and {type(right)}."
        )

    left_cols = sbd.column_names(left)
    original_right_cols = sbd.column_names(right)
    right_cols = map(_utils.renaming_func(rename_right_cols), original_right_cols)
    right_cols = pick_column_names(right_cols, forbidden_names=left_cols)
    renaming = dict(zip(original_right_cols, right_cols))
    right = sbd.set_column_names(right, right_cols)
    if isinstance(right_on, str):
        right_on = renaming[right_on]
        right_on_selector = s.cols(right_on)
    else:
        right_on = tuple(renaming[c] for c in right_on)
        right_on_selector = s.cols(*right_on)
    joined = _do_left_join(left, right, left_on, right_on)
    joined = s.select(joined, ~right_on_selector)
    return joined


@dispatch
def _do_left_join(left, right, left_on, right_on):
    raise NotImplementedError()


@_do_left_join.specialize("pandas", argument_type="DataFrame")
def _do_left_join_pandas(left, right, left_on, right_on):
    return left.merge(
        right, left_on=left_on, right_on=right_on, how="left", suffixes=("", "")
    )


@_do_left_join.specialize("polars", argument_type="DataFrame")
def _do_left_join_polars(left, right, left_on, right_on):
    if "coalesce" in inspect.signature(left.join).parameters:
        kw = {"coalesce": True}
    else:
        kw = {}
    return left.join(
        right, left_on=left_on, right_on=right_on, how="left", suffix="", **kw
    )
