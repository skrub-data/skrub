"""
The MultiJoiner and MultiAggJoiner extend Joiner and AggJoiner
to multiple auxiliary tables.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from skrub._agg_joiner import AggJoiner
from skrub._dataframe._namespace import is_pandas, is_polars
from skrub._joiner import DEFAULT_REF_DIST, DEFAULT_STRING_ENCODER  # , Joiner


def check_multi_key():
    return


class MultiJoiner(BaseEstimator, TransformerMixin):
    """Extension of the Joiner to multiple auxiliary tables.

    This transformer is initialized with auxiliary tables ``aux_tables``. It
    transforms a main table by joining it, with approximate ("fuzzy") matching,
    to the auxiliary table. The output of ``transform`` has the same rows as
    the main table (i.e. as the argument passed to ``transform``), but each row
    is augmented with values from the best match in the auxiliary table.

    To identify the best match for each row, values from the matching columns
    (``main_key`` and ``aux_key``) are vectorized, i.e. represented by vectors of
    continuous values. Then, the Euclidean distances between these vectors are
    computed to find, for each main table row, its nearest neighbor within the
    auxiliary table.

    Optionally, a maximum distance threshold, ``max_dist``, can be set. Matches
    between vectors that are separated by a distance (strictly) greater than
    ``max_dist`` will be rejected. We will consider that main table rows that
    are farther than ``max_dist`` from their nearest neighbor do not have a
    matching row in the auxiliary table, and the output will contain nulls for
    the entries that would normally have come from the auxiliary table (as in a
    traditional left join).

    To make it easier to set a ``max_dist`` threshold, the distances are
    rescaled by dividing them by a reference distance, which can be chosen with
    ``ref_dist``. The default is ``'random_pairs'``. The possible choices are:

    'random_pairs'
        Pairs of rows are sampled randomly from the auxiliary table and their
        distance is computed. The reference distance is the first quartile of
        those distances.

    'second_neighbor'
        The reference distance is the distance to the *second* nearest neighbor
        in the auxiliary table.

    'self_join_neighbor'
        Once the match candidate (i.e. the nearest neigbor from the auxiliary
        table) has been found, we find its nearest neighbor in the auxiliary
        table (excluding itself). The reference distance is the distance that
        separates those 2 auxiliary rows.

    'no_rescaling'
        The reference distance is 1.0, i.e. no rescaling of the distances is
        applied.

    Parameters
    ----------
    aux_table : :obj:`~pandas.DataFrame`
        The auxiliary table, which will be fuzzy-joined to the main table when
        calling ``transform``.
    key : str or list of str, default=None
        The column names to use for both ``main_key`` and ``aux_key`` when they
        are the same. Provide either ``key`` or both ``main_key`` and ``aux_key``.
    main_key : str or list of str, default=None
        The column names in the main table on which the join will be performed.
        Can be a string if joining on a single column.
        If ``None``, `aux_key` must also be ``None`` and `key` must be provided.
    aux_key : str or list of str, default=None
        The column names in the auxiliary table on which the join will
        be performed. Can be a string if joining on a single column.
        If ``None``, `main_key` must also be ``None`` and `key` must be provided.
    suffix : str, default=""
        Suffix to append to the ``aux_table``'s column names. You can use it
        to avoid duplicate column names in the join.
    max_dist : float, default=np.inf
        Maximum acceptable (rescaled) distance between a row in the
        ``main_table`` and its nearest neighbor in the ``aux_table``. Rows that
        are farther apart are not considered to match. By default, the distance
        is rescaled so that a value between 0 and 1 is typically a good choice,
        although rescaled distances can be greater than 1 for some choices of
        ``ref_dist``. ``None``, ``"inf"``, ``float("inf")`` or ``numpy.inf``
        mean that no matches are rejected.
    ref_dist : reference distance for rescaling, default = 'random_pairs'
        Options are {"random_pairs", "second_neighbor", "self_join_neighbor",
        "no_rescaling"}. See above for a description of each option. To
        facilitate the choice of ``max_dist``, distances between rows in
        ``main_table`` and their nearest neighbor in ``aux_table`` will be
        rescaled by this reference distance.
    string_encoder : scikit-learn transformer used to vectorize text columns
        By default a ``HashingVectorizer`` combined with a ``TfidfTransformer``
        is used. Here we use raw TF-IDF features rather than transforming them
        for example with ``GapEncoder`` or ``MinHashEncoder`` because it is
        faster, these features are only used to find nearest neighbors and not
        used by downstream estimators, and distances between TF-IDF vectors
        have a somewhat simpler interpretation.
    add_match_info : bool, default=True
        Insert some columns whose names start with `skrub_Joiner` containing
        the distance, rescaled distance and whether the rescaled distance is
        above the threshold. Those values can be helpful for an estimator that
        uses the joined features, or to inspect the result of the join and set
        a ``max_dist`` threshold.

    Attributes
    ----------
    max_dist_ : the maximum distance for a match to be accepted
        Equal to the parameter ``max_dist`` except that ``"inf"`` and ``None``
        are mapped to ``np.inf`` (i.e. accept all matches).

    vectorizer_ : scikit-learn ColumnTransformer
        The fitted transformer used to transform the matching columns into
        numerical vectors.

    See Also
    --------
    Joiner :
        Augment features in a main table by fuzzy-joining an auxiliary table to it.

    fuzzy_join :
        Join two tables (dataframes) based on approximate column matching. This
        is the same functionality as provided by the ``Joiner`` but exposed as
        a function rather than a transformer.

    Examples
    --------
    # TODO
    """

    def __init__(
        self,
        aux_table,
        *,
        key=None,
        main_key=None,
        aux_key=None,
        suffix="",
        max_dist=np.inf,
        ref_dist=DEFAULT_REF_DIST,
        string_encoder=DEFAULT_STRING_ENCODER,
        add_match_info=True,
    ):
        self.aux_table = aux_table
        self.main_key = main_key
        self.aux_key = aux_key
        self.key = key
        self.suffix = suffix
        self.max_dist = max_dist
        self.ref_dist = ref_dist
        self.string_encoder = (
            clone(string_encoder)
            if string_encoder is DEFAULT_STRING_ENCODER
            else string_encoder
        )
        self.add_match_info = add_match_info


class MultiAggJoiner(BaseEstimator, TransformerMixin):
    """Extension of the AggJoiner to multiple auxiliary tables.

    Apply numerical and categorical aggregation operations on the columns
    to aggregate, selected by dtypes. See the list of supported operations
    at the parameter `operation`.

    The grouping columns used during the aggregation are the columns used
    as keys for joining.

    Accepts :obj:`pandas.DataFrame` and :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    aux_tables : DataFrameLike or iterable of DataFrameLike or "X"
        Auxiliary dataframe to aggregate then join on the base table.
        The placeholder string "X" can be provided to perform
        self-aggregation on the input data.

    keys : str or list of str, default=None
        The column names to use for both ``main_key`` and ``aux_key`` when they
        are the same. Provide either ``key`` or both ``main_key`` and ``aux_key``.

    main_key : str or iterable of str
        Select the columns from the main table to use as keys during
        the join operation.
        If main_keys is a list, we will perform a multi-column join.

    aux_keys : str, or iterable of str, or iterable of iterable of str
        Select the columns from the auxiliary dataframe to use as keys during
        the join operation.

    cols : str, or iterable of str, or iterable of iterable of str, default=None
        Select the columns from the auxiliary dataframe to use as values during
        the aggregation operations.
        If None, cols are all columns from table, except `aux_key`.

    operations : str or iterable of str, default=None
        Aggregation operations to perform on the auxiliary table.

        numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}
            'hist' and 'value_counts' accept an integer argument to parametrize
            the binning.

        categorical : {"mode", "count", "value_counts"}

        If set to None (the default), ['mean', 'mode'] will be used.

    suffixes : str or iterable of str, default=None
        The suffixes that will be added to each table columns in case of
        duplicate column names.
        If set to None, the table index in 'aux_table' are used,
        e.g. for a duplicate columns: price (main table),
        price_1 (auxiliary table 1), price_2 (auxiliary table 2), etc.

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
    ...    main_key="patient_id",
    ...    aux_keys=["patient_id", "patient_id", "patientID"],
    ...    cols=["days_of_stay", "medication", "value"],
    ...    operations=["max", "mode", ["mean", "std"]],
    ...    suffixes=["", "", "glucose"],
    ... )
    >>> multi_agg_joiner.fit_transform(patients)
    # TODO
    """

    def __init__(
        self,
        aux_tables,
        *,
        keys=None,
        main_key=None,
        aux_keys=None,
        cols=None,
        operations=None,
        suffixes=None,
    ):
        self.aux_tables = aux_tables
        self.keys = keys
        self.main_key = main_key
        self.aux_keys = aux_keys
        self.cols = cols
        self.operations = operations
        self.suffixes = suffixes

    def _check_dataframes(self, X, aux_tables):
        """Check dataframes input types.

        Parameters
        ----------
        X : DataframeLike
            The main table to augment.
        aux_tables : DataframeLike
            The auxiliary tables.

        Raises
        ------
        TypeError
            Raises an error if all the frames don't have the same type,
            or if there is a Polars lazyframe.
        """
        # Polars lazyframes will raise an error here.
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"'X' must be a dataframe, got {type(X)}.")
        if not all(hasattr(aux_table, "__dataframe__") for aux_table in aux_tables):
            raise TypeError("'aux_tables' must all be dataframes.")

        if is_pandas(X):
            if not all(is_pandas(aux_table) for aux_table in aux_tables):
                raise TypeError("All 'aux_tables' must be Pandas dataframes.")
        if is_polars(X):
            if not all(is_polars(aux_table) for aux_table in aux_tables):
                raise TypeError("All 'aux_tables' must be Polars dataframes.")

    def _check_keys(self):
        pass

    def _check_missing_columns(self):
        pass

    def _check_column_name_duplicates(self):
        pass

    def _check_cols(self):
        pass

    def _check_operations(self):
        pass

    def fit(self, X, y=None):
        """Aggregate auxiliary tables based on the main keys.

        Parameters
        ----------
        X : DataframeLike
            Input data, based table on which to left join the
            auxiliary tables.

        y : array-like of shape (n_samples), default=None
            Prediction target. Used to compute correlations between the
            generated covariates and the target for screening purposes.

        Returns
        -------
        AggJoiner
            Fitted :class:`MultiAggJoiner` instance (self).
        """
        self._check_dataframes(X, self.aux_tables)

        self._main_key, self._aux_keys = self._check_keys(
            self.main_key, self.aux_keys, self.keys
        )

        self._check_missing_columns(X, self._main_key, "'X' (the main table)")
        self._check_missing_columns(self.aux_tables, self._aux_keys, "'aux_table'")
        self._check_column_name_duplicates(
            X, self.aux_tables, self.suffixes, main_table_name="X"
        )
        self.cols = self._check_cols()

        self.operation = self._check_operations()

        self.agg_joiners_ = []

        for aux_table, aux_key, cols, operation, suffix in zip(
            self.aux_tables,
            self._aux_keys,
            self.cols,
            self.operations,
            self.suffixes,
        ):
            agg_joiner = AggJoiner(
                aux_table=aux_table,
                main_key=self._main_key,
                aux_key=aux_key,
                cols=cols,
                operation=operation,
                suffix=suffix,
            )
            self.agg_joiners_.append(agg_joiner)

        for self.agg_joiner in self.agg_joiners_:
            self.agg_joiners_.fit(X)

        return self

    def transform(self, X):
        """Left-join pre-aggregated tables on `X`.

        Parameters
        ----------
        X : DataFrameLike
            The input data to transform.

        Returns
        -------
        X_transformed : DataFrameLike
            The augmented input.
        """
        check_is_fitted(self, "agg_joiners_")

        for agg_joiner in self.agg_joiners_:
            X = agg_joiner.transform(X)

        return X
