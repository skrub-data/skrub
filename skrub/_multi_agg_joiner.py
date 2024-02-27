"""
The MultiAggJoiner extends AggJoiner to multiple auxiliary tables.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skrub import _join_utils
from skrub._agg_joiner import AggJoiner
from skrub._dataframe._namespace import is_pandas, is_polars
from skrub._utils import _is_array_like, atleast_2d_or_none


class MultiAggJoiner(TransformerMixin, BaseEstimator):
    """Extension of the :class:`AggJoiner` to multiple auxiliary tables.

    Apply numerical and categorical aggregation operations on the `cols`
    to aggregate, selected by dtypes. See the list of supported operations
    at the parameter `operations`.

    If `cols` is not provided, `cols` is set to a list of lists.
    For each table in `aux_tables`, the corresponding list will be all columns
    of that table, except the `aux_keys` associated with that table.

    As opposed to the :class:`AggJoiner`, here `aux_tables` is a list of tables,
    each of which will be joined on the main table. Therefore `aux_keys` is now
    a list of keys, of the same length as `aux_tables`, and each entry in `aux_keys`
    is used to join the corresponding aux table. In the same way, each entry in `cols`
    is a list of columns to aggregate in the corresponding aux table.

    Therefore if we have a single table, we could either use

    - the :class:`AggJoiner`: ``AggJoiner(aux_table, key="ID")``
    - or the :class:`MultiAggJoiner`: ``MultiAggJoiner([aux_table], keys=["ID"])``

    Note that for `keys`, `main_keys`, `aux_keys`, `cols` and `operations`,
    an input of the form ``[["a"], ["b"], ["c", "d"]]`` is valid
    while ``["a", "b", ["c", "d"]]`` is not.

    Using a column from the first aux table to join the second aux table is not
    (yet) supported.

    Accepts :obj:`pandas.DataFrame` and :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    aux_tables : iterable of DataFrameLike or "X"
        Auxiliary dataframes to aggregate then join on the base table.
        The placeholder string "X" can be provided to perform
        self-aggregation on the input data. To provide a single auxiliary table,
        ``aux_tables = [table]`` is supported, but not ``aux_tables = table``.
        It's possible to provide both the placeholder "X" and auxiliary tables,
        as in ``aux_tables = [table, "X"]``. If that's the case, the second table will
        be replaced by the input data.

    keys : iterable of str, or iterable of iterable of str, default=None
        The column names to use for both `main_keys` and `aux_key` when they
        are the same. Provide either `key` or both `main_keys` and `aux_keys`.
        If there are multiple auxiliary tables, `keys` will be used to join all
        of them.

        All `keys` must be present in the main and auxiliary tables before fit.
        It's not (yet) possible to use columns from the first joined table
        to join the second.

        If not `None`, there must be an iterable of `keys` for each table
        in `aux_tables`.

    main_keys : iterable of str, or iterable of iterable of str, default=None
        Select the columns from the main table to use as keys during
        the join operation.
        If `main_keys` contains multiple columns, we will perform a multi-column join.

        If not `None`, there must be an iterable of `main_keys` for each table
        in `aux_tables`.

    aux_keys : iterable of str, or iterable of iterable of str, default=None
        Select the columns from the auxiliary dataframes to use as keys during
        the join operation.

        All `aux_keys` must be present in respective `aux_tables` before fit.
        It's not (yet) possible to use columns from the first joined table
        to join the second.

        If not `None`, there must be an iterable of `aux_keys` for each table
        in `aux_tables`.

    cols : iterable of str, or iterable of iterable of str, default=None
        Select the columns from the auxiliary dataframes to use as values during
        the aggregation operations.

        If not `None`, there must be an iterable of `cols` for each table
        in `aux_tables`.

        If set to `None`, `cols` is set to a list of lists. For each table
        in `aux_tables`, the corresponding list will be all columns of that table,
        except the `aux_keys` associated with that table.

    operations : iterable of str, or iterable of iterable of str, default=None
        Aggregation operations to perform on the auxiliary tables.

        If not `None`, there must be an iterable of `operations` for each
        table in `aux_tables`.

        - numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}
          "hist" and "value_counts" accept an integer argument to parametrize
          the binning.
        - categorical : {"mode", "count", "value_counts"}
        - If set to `None` (the default), ["mean", "mode"] will be used
          for all auxiliary tables.

    suffixes : iterable of str, default=None
        Suffixes to append to the `aux_tables`' column names.
        If set to `None`, the table indexes in `aux_tables` are used,
        e.g. for an aggregation of 2 `aux_tables`, "_0" and "_1" would be appended
        to column names.

    See Also
    --------
    AggJoiner :
        Aggregate an auxiliary dataframe before joining it on a base dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import MultiAggJoiner
    >>> patients = pd.DataFrame({
    ...    "patient_id": [1, 2],
    ...    "age": ["72", "45"],
    ... })
    >>> hospitalizations = pd.DataFrame({
    ...    "visit_id": range(1, 7),
    ...    "patient_id": [1, 1, 1, 1, 2, 2],
    ...    "days_of_stay": [2, 4, 1, 1, 3, 12],
    ...    "hospital": ["Cochin", "Bichat", "Cochin", "Necker", "Bichat", "Bichat"],
    ... })
    >>> medications = pd.DataFrame({
    ...    "medication_id": range(1, 6),
    ...    "patient_id": [1, 1, 1, 1, 2],
    ...    "medication": ["ozempic", "ozempic", "electrolytes", "ozempic", "morphine"],
    ... })
    >>> glucose = pd.DataFrame({
    ...    "biology_id": range(1, 7),
    ...    "patientID": [1, 1, 1, 1, 2, 2],
    ...    "value": [1.4, 3.4, 1.0, 0.8, 3.1, 6.5],
    ... })
    >>> multi_agg_joiner = MultiAggJoiner(
    ...    aux_tables=[hospitalizations, medications, glucose],
    ...    main_keys=[["patient_id"], ["patient_id"], ["patient_id"]],
    ...    aux_keys=[["patient_id"], ["patient_id"], ["patientID"]],
    ...    cols=[["days_of_stay"], ["medication"], ["value"]],
    ...    operations=[["max"], ["mode"], ["mean", "std"]],
    ...    suffixes=["", "", "_glucose"],
    ... )
    >>> multi_agg_joiner.fit_transform(patients)
       patient_id  age  ...  value_mean_glucose  value_std_glucose
    0           1   72  ...                1.65           1.193035
    1           2   45  ...                4.80           2.404163

    The :class:`MultiAggJoiner` makes it convenient to aggregate multiple tables, but
    the same results could be obtained by chaining 3 separate :class:`AggJoiner`:

    >>> from skrub import AggJoiner
    >>> from sklearn.pipeline import make_pipeline
    >>> agg_joiner_1 = AggJoiner(
    ...    aux_table=hospitalizations,
    ...    key="patient_id",
    ...    cols="days_of_stay",
    ...    operations="max",
    ... )
    >>> agg_joiner_2 = AggJoiner(
    ...    aux_table=medications,
    ...    key="patient_id",
    ...    cols="medication",
    ...    operations="mode",
    ... )
    >>> agg_joiner_3 = AggJoiner(
    ...    aux_table=glucose,
    ...    main_key="patient_id",
    ...    aux_key="patientID",
    ...    cols="value",
    ...    operations=["mean", "std"],
    ...    suffix="_glucose",
    ... )
    >>> pipeline = make_pipeline(agg_joiner_1, agg_joiner_2, agg_joiner_3)
    >>> pipeline.fit_transform(patients)
       patient_id  age  ...  value_mean_glucose  value_std_glucose
    0           1   72  ...                1.65           1.193035
    1           2   45  ...                4.80           2.404163
    """

    def __init__(
        self,
        aux_tables,
        *,
        keys=None,
        main_keys=None,
        aux_keys=None,
        cols=None,
        operations=None,
        suffixes=None,
    ):
        self.aux_tables = aux_tables
        self.keys = keys
        self.main_keys = main_keys
        self.aux_keys = aux_keys
        self.cols = cols
        self.operations = operations
        self.suffixes = suffixes

    def _check_dataframes(self, X, aux_tables):
        """Check dataframes input types.

        Parameters
        ----------
        X : DataFrameLike
            The main table to augment.
        aux_tables : iterator of DataFrameLike or "X"
            The auxiliary tables.

        Raises
        ------
        TypeError
            Raises an error if all the frames don't have the same type,
            or if there is a Polars lazyframe.
        """
        if not _is_array_like(aux_tables):
            raise ValueError(
                "`aux_tables` must be an iterable containing dataframes and/or the"
                " string 'X'. If you are using a single auxiliary table, convert your"
                " current`aux_tables` into [`aux_tables`]."
            )
        # Check `X` input type
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"`X` must be a dataframe, got {type(X)}.")

        # Check `aux_tables` input types
        for i, aux_table in enumerate(aux_tables):
            if type(aux_table) == str and aux_table == "X":
                aux_tables[i] = X
            elif not hasattr(aux_table, "__dataframe__"):
                raise ValueError(
                    "`aux_tables` must be an iterable containing dataframes and/or the"
                    " string 'X'"
                )

        # Check that all input types are matching
        if is_pandas(X):
            if not all(is_pandas(aux_table) for aux_table in aux_tables):
                raise TypeError(
                    "All `aux_tables` must be Pandas dataframes"
                    " when `X` is a Pandas dataframe."
                )
        if is_polars(X):
            if not all(is_polars(aux_table) for aux_table in aux_tables):
                raise TypeError(
                    "All `aux_tables` must be Polars dataframes"
                    " when `X` is a Polars dataframe."
                )

        return X, aux_tables

    def _check_keys(self, main_keys, aux_keys, keys, aux_tables):
        """Check input keys.

        Parameters
        ----------
        main_keys : str, iterable of str, or None
            Matching columns in the main table. Can be a single column name (str)
            if matching on a single column: ``"User_ID"`` is the same as
            ``"[User_ID]"``.
        aux_keys : iterable of str, iterable of iterable of str, or None
            Matching columns in the auxiliary table. Can be a single column name (str)
            if matching on a single column: ``"User_ID"`` is the same as
            ``"[User_ID]"``.
        keys : iterable of str, or None
            Can be provided in place of `main_keys` and `aux_keys` when they are the
            same. We must provide non-``None`` values for either `keys` or both
            `main_keys` and `aux_keys`.
        aux_tables : iterable of DataFrameLike
            Auxiliary tables used to check that the length of keys match the number
            of tables to aggregate.

        Returns
        -------
        main_keys, aux_keys : iterable of str and iterable of iterable of str
            The correct sets of matching columns to use, each provided as a list of
            column names.
        """
        if keys is not None:
            if aux_keys is not None or main_keys is not None:
                raise ValueError(
                    "Can only pass argument `keys` or `main_keys` and "
                    "`aux_keys`, not a combination of both."
                )
            main_keys, aux_keys = keys, keys
        else:
            if aux_keys is None or main_keys is None:
                raise ValueError(
                    "Must pass either `keys`, or (`main_keys` and `aux_keys`)."
                )
        main_keys = atleast_2d_or_none(main_keys)
        aux_keys = atleast_2d_or_none(aux_keys)
        # Check 2d shape
        if len(aux_tables) != len(main_keys):
            raise ValueError(
                "The length of `main_keys` must match the number of `aux_tables`."
                f" Provided `main_keys` of len {len(main_keys)} and"
                f" `aux_tables` of len {len(aux_tables)}."
            )
        if len(aux_tables) != len(aux_keys):
            raise ValueError(
                "The length of `aux_keys` must match the number of `aux_tables`."
                f" Provided `aux_keys` of len {len(aux_keys)} and"
                f" `aux_tables` of len {len(aux_tables)}."
            )
        # Check each element
        for i, (main_key, aux_key) in enumerate(zip(main_keys, aux_keys)):
            if len(main_key) != len(aux_key):
                raise ValueError(
                    "`main_keys` and `aux_keys` elements have different lengths"
                    f" at position {i} ({len(main_key)} and {len(aux_key)})."
                    "  Cannot join on different numbers of columns."
                )
        return main_keys, aux_keys

    def _check_cols(self):
        """Check `cols` to aggregate.

        If None, `cols` are all columns from `aux_tables`, except `aux_keys`.

        Returns
        -------
        cols
            2-dimensional array of cols to perform aggregation on.

        Raises
        ------
        ValueError
            If the len of `cols` doesn't match the len of `aux_tables`,
            or if `cols` is not of a valid type, of if all `cols`
            are not present in the corresponding aux_table.
        """
        # If no `cols` provided, all columns but `aux_keys` are used.
        if self.cols is None:
            cols = [
                list(set(table.columns) - set(key))
                for table, key in zip(self._aux_tables, self._aux_keys)
            ]
            cols = atleast_2d_or_none(cols)
        else:
            cols = atleast_2d_or_none(self.cols)
        if len(cols) != len(self.aux_tables):
            raise ValueError(
                "The number of provided cols must match the number of"
                f" tables in `aux_tables`. Got {len(cols)} columns and"
                f" {len(self._aux_tables)} auxiliary tables."
            )
        return cols

    def _check_operations(self):
        """Check `operations` input.

        Returns
        -------
        operations
            2-dimensional array of operations to perform on columns.

        Raises
        ------
        ValueError
            If the len of `operations` doesn't match the len of `aux_tables`,
            or if `operations` is not of a valid type.
        """
        if self.operations is None:
            operations = [["mean", "mode"]] * len(self._aux_tables)
        elif _is_array_like(self.operations):
            operations = atleast_2d_or_none(self.operations)
        else:
            raise ValueError(
                "Accepted inputs for operations are None, iterable of str,"
                f" or iterable of iterable of str. Got {type(self.operations)}"
            )

        if len(operations) != len(self._aux_tables):
            raise ValueError(
                "The number of provided operations must match the number of"
                f" tables in `aux_tables`. Got {len(operations)} operations and"
                f" {len(self._aux_tables)} auxiliary tables."
            )
        return operations

    def _check_suffixes(self):
        """Check that the len of `suffixes` match the len of `aux_tables`,
        and that all suffixes are strings.

        If suffixes is None, the suffixes will default to the position of
        each auxiliary table in the list.

        Returns
        -------
        suffixes
            1-dimensional array of suffixes to append to dataframes.

        Raises
        ------
        ValueError
            If the len of `suffixes` doesn't match the len of `aux_tables`,
            or if all suffixes are not strings.
        """
        if self.suffixes is None:
            suffixes = [f"_{i}" for i in range(len(self._aux_tables))]
        else:
            suffixes = np.atleast_1d(self.suffixes).tolist()
            if len(suffixes) != len(self._aux_tables):
                raise ValueError(
                    "The number of provided suffixes must match the number of"
                    f" tables in `aux_tables`. Got {len(suffixes)} suffixes and"
                    f" {len(self._aux_tables)} aux_tables."
                )
            if not all([isinstance(suffix, str) for suffix in suffixes]):
                raise ValueError("All suffixes must be strings.")

        return suffixes

    def fit(self, X, y=None):
        """Aggregate auxiliary tables based on the main keys.

        Parameters
        ----------
        X : DataFrameLike
            Input data, based table on which to left join the
            auxiliary tables.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        MultiAggJoiner
            Fitted :class:`MultiAggJoiner` instance (self).
        """
        X, self._aux_tables = self._check_dataframes(X, self.aux_tables)
        self._main_keys, self._aux_keys = self._check_keys(
            self.main_keys, self.aux_keys, self.keys, self._aux_tables
        )
        self._cols = self._check_cols()
        self._operations = self._check_operations()
        self._suffixes = self._check_suffixes()

        self.agg_joiners_ = []
        for aux_table, main_key, aux_key, cols, operations, suffix in zip(
            self._aux_tables,
            self._main_keys,
            self._aux_keys,
            self._cols,
            self._operations,
            self._suffixes,
        ):
            agg_joiner = AggJoiner(
                aux_table=aux_table,
                main_key=main_key,
                aux_key=aux_key,
                cols=cols,
                operations=operations,
                suffix=suffix,
            )
            agg_joiner.fit(X)
            self.agg_joiners_.append(agg_joiner)

        return self

    def transform(self, X):
        """Left-join pre-aggregated tables on `X`.

        Parameters
        ----------
        X : DataFrameLike
            The input data to transform.

        Returns
        -------
        DataFrame
            The augmented input.
        """
        check_is_fitted(self, "agg_joiners_")
        X, _ = self._check_dataframes(X, self._aux_tables)
        for main_key in self._main_keys:
            _join_utils.check_missing_columns(X, main_key, "'X' (the main table)")

        for agg_joiner in self.agg_joiners_:
            X = agg_joiner.transform(X)

        return X
