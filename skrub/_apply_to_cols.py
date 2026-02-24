"""
ApplyToCols selects the correct transformer between ApplyToEachCol and ApplyToSubFrame
based on the type of the transformer passed to it.
"""

from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from . import selectors
from ._apply_on_each_col import ApplyToEachCol
from ._wrap_transformer import wrap_transformer

_SELECT_ALL_COLUMNS = selectors.all()


class ApplyToCols(TransformerMixin, BaseEstimator):
    """
    Apply a transformer to columns in a dataframe.

    Columns that are not selected in the ``cols`` parameter are passed through
    without modification.


    Parameters
    ----------
    transformer : transformer instance
        The transformer to apply to the selected columns.

    cols : str, sequence of str, or skrub selector, optional
        The columns to attempt to transform. Only the selected columns will have
        the transformer applied. Columns outside this selection are passed
        through unchanged (``fit_transform`` is not called on them) and remain
        unmodified in the output. The default is to attempt transforming all
        columns.

    how : "auto", "cols" or "frame", optional
        How the transformer is applied. In most cases the default "auto"
        is appropriate.

        - "cols" means `transformer` is wrapped in a :class:`ApplyToEachCol`
          transformer, which fits a separate clone of `transformer` to each
          column in `cols`.
        - "frame" means `transformer` is wrapped in a :class:`ApplyToSubFrame`
          transformer, which fits a single clone of `transformer` to the
          selected part of the input dataframe.
        - "auto" chooses the wrapping depending on the input and transformer.
          If the transformer has a ``__single_column_transformer__`` attribute,
          "cols" is chosen. Otherwise "frame" is chosen.

    allow_reject : bool, default=False
        Whether to allow refusing to transform columns for which the provided
        transformer is not suited, for example rejecting non-datetime columns if
        transformer is a DatetimeEncoder. See the documentation of
        :class:`ApplyToEachCol` for details.

    keep_original : bool, default=False
        If ``True``, the original columns are preserved in the output. If the
        transformer produces a column with the same name, the transformation
        result is renamed so that both columns can appear in the output. If
        ``False``, when the transformer accepts a column, only the
        transformer's output is included in the result, not the original
        column. In all cases rejected columns (or columns not selected by
        ``cols``) are passed through.

    rename_columns : str, default='{}'
        Format string applied to all transformation output column names. For
        example pass ``'transformed_{}'`` to prepend ``'transformed_'`` to all
        output column names. The default value does not modify the names.
        Renaming is not applied to columns not selected by ``cols``.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a joblib ``parallel_backend`` context.
        ``-1`` means using all processors.
        Note that this parameter is only used when the transformer is wrapped in an
        ``ApplyToEachCol``.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.

    used_inputs_ : list of str
        The names of columns that were transformed.

    all_outputs_ : list of str
        All column names in the output dataframe.

    created_outputs_ : list of str
        The names of columns in the output dataframe that were created by one
        of the fitted transformers.

    input_to_outputs_ : dict
        Maps the name of each column that was transformed to the list of the
        resulting columns' names in the output.

    output_to_input_ : dict
        Maps the name of each column in the transformed output to the name of
        the input column from which it was derived.

    transformers_ : dict
        Maps the name of each column that was transformed to the corresponding
        fitted transformer.

    Notes
    -----
    All columns not listed in ``cols`` remain unmodified in the output.
    Moreover, if ``allow_reject`` is ``True`` and the transformers'
    ``fit_transform`` raises a ``RejectColumn`` exception for a particular
    column, that column is passed through unchanged. If ``allow_reject`` is
    ``False``, ``RejectColumn`` exceptions are propagated, like other errors
    raised by the transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import ApplyToCols
    >>> from sklearn.preprocessing import StandardScaler
    >>> df = pd.DataFrame(dict(A=[-10., 10.], B=[-10., 0.], C=[0., 10.]))
    >>> df
          A     B     C
    0 -10.0 -10.0   0.0
    1  10.0   0.0  10.0

    Fit a StandardScaler to each column in df:

    >>> scaler = ApplyToCols(StandardScaler(), how="cols")
    >>> scaler.fit_transform(df)
         A    B    C
    0 -1.0 -1.0 -1.0
    1  1.0  1.0  1.0
    >>> scaler.transformers_
    {'A': StandardScaler(), 'B': StandardScaler(), 'C': StandardScaler()}

    We can restrict the columns on which the transformation is applied:

    >>> scaler = ApplyToCols(StandardScaler(), cols=["A", "B"], how="cols")
    >>> scaler.fit_transform(df)
         A    B     C
    0 -1.0 -1.0   0.0
    1  1.0  1.0  10.0

    We see that the scaling has not been applied to "C", which also does not
    appear in the transformers_:

    >>> scaler.transformers_
    {'A': StandardScaler(), 'B': StandardScaler()}
    >>> scaler._wrapped_transformer.used_inputs_
    ['A', 'B']

    **Rejected columns**

    The transformer can raise ``RejectColumn`` to indicate it cannot handle a
    given column.

    >>> from skrub import ToDatetime
    >>> df = pd.DataFrame(dict(birthday=["29/01/2024"], city=["London"]))
    >>> df
         birthday    city
    0  29/01/2024  London
    >>> df.dtypes
    birthday    ...
    city        ...
    dtype: object
    >>> ToDatetime().fit_transform(df["birthday"])
    0   2024-01-29
    Name: birthday, dtype: datetime64[...]
    >>> ToDatetime().fit_transform(df["city"])
    Traceback (most recent call last):
        ...
    skrub.core._single_column_transformer.RejectColumn: Could not find a datetime format for column 'city'.

    How these rejections are handled depends on the ``allow_reject`` parameter.
    By default, no special handling is performed and rejections are considered
    to be errors:

    >>> to_datetime = ApplyToCols(ToDatetime())
    >>> to_datetime.fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: Transformer ToDatetime.fit_transform failed on column 'city'. See above for the full traceback.

    However, setting ``allow_reject=True`` gives the transformer itself some
    control over which columns it should be applied to. For example, whether a
    string column contains dates is only known once we try to parse them.
    Therefore it might be sensible to try to parse all string columns but allow
    the transformer to reject those that, upon inspection, do not contain dates.

    >>> to_datetime = ApplyToCols(ToDatetime(), allow_reject=True)
    >>> transformed = to_datetime.fit_transform(df)
    >>> transformed
        birthday    city
    0 2024-01-29  London

    Now the column 'city' was rejected but this was not treated as an error;
    'city' was passed through unchanged and only 'birthday' was converted to a
    datetime column.

    >>> transformed.dtypes
    birthday    datetime64[...]
    city                ...
    dtype: object
    >>> to_datetime.transformers_
    {'birthday': ToDatetime()}

    **Renaming outputs & keeping the original columns**

    The ``rename_columns`` parameter allows renaming output columns.

    >>> df = pd.DataFrame(dict(A=[-10., 10.], B=[0., 100.]))
    >>> scaler = ApplyToCols(StandardScaler(), rename_columns='{}_scaled', how='cols')
    >>> scaler.fit_transform(df)
       A_scaled  B_scaled
    0      -1.0      -1.0
    1       1.0       1.0

    The renaming is only applied to columns selected by ``cols`` (and not
    rejected by the transformer when ``allow_reject`` is ``True``).

    >>> scaler = ApplyToCols(StandardScaler(), cols=['A'], rename_columns='{}_scaled', how='cols')
    >>> scaler.fit_transform(df)
       A_scaled      B
    0      -1.0    0.0
    1       1.0  100.0

    ``rename_columns`` can be particularly useful when ``keep_original`` is
    ``True``. When a column is transformed, we can tell ``ApplyToCols`` to
    retain the original, untransformed column in the output. If the transformer
    produces a column with the same name, the transformation result is renamed
    to avoid a name clash.

    >>> scaler = ApplyToCols(StandardScaler(), keep_original=True, how='cols')
    >>> scaler.fit_transform(df)                                    # doctest: +SKIP
          A  A__skrub_89725c56__      B  B__skrub_81cc7d00__
    0 -10.0                 -1.0    0.0                 -1.0
    1  10.0                  1.0  100.0                  1.0

    In this case we may want to set a more sensible name for the transformer's output:

    >>> scaler = ApplyToCols(
    ...     StandardScaler(), keep_original=True, rename_columns="{}_scaled", how='cols'
    ... )
    >>> scaler.fit_transform(df)
          A  A_scaled      B  B_scaled
    0 -10.0      -1.0    0.0      -1.0
    1  10.0       1.0  100.0       1.0
    """  # noqa: E501

    def __init__(
        self,
        transformer,
        cols=_SELECT_ALL_COLUMNS,
        *,
        how="auto",
        allow_reject=False,
        keep_original=False,
        rename_columns="{}",
        n_jobs=None,
    ):
        self.transformer = transformer
        self.cols = cols
        self.how = how
        self.n_jobs = n_jobs
        self.allow_reject = allow_reject
        self.keep_original = keep_original
        self.rename_columns = rename_columns

    def fit(self, X, y=None, **kwargs):
        """Fit the transformer to the data.

        If the transformer is a SingleColumnTransformer or if how="cols",
        the transformer is wrapped in an ApplyToEachCol. Otherwise, it is wrapped
        in an ApplyToSubFrame.

        Parameters
        ----------
        X : Pandas or Polars DataFrame
            The data to transform.

        y : Pandas or Polars Series or DataFrame, default=None
            The target data.

        **kwargs
            Extra named arguments are passed to the ``fit_transform()`` method of
            the individual column transformers (the clones of ``self.transformer``).

        Returns
        -------
        ApplyToCols
            The transformer itself.
        """
        self.fit_transform(X, y, **kwargs)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """Fit the transformer on all columns and transform X.

        If the transformer is a SingleColumnTransformer or if how="cols",
        the transformer is wrapped in an ApplyToEachCol. Otherwise, it is wrapped
        in an ApplyToSubFrame.

        Parameters
        ----------
        X : Pandas or Polars DataFrame
            The data to transform.

        y : Pandas or Polars Series or DataFrame, default=None
            The target data.

        **kwargs
            Extra named arguments are passed to the ``fit_transform()`` method
            of ``self.transformer``.

        Returns
        -------
        result : Pandas or Polars DataFrame
            The transformed data.
        """
        if self.how not in ("auto", "cols", "frame"):
            raise ValueError(
                f"Invalid value for 'how': {self.how}. "
                "Expected one of 'auto', 'cols', or 'frame'."
            )

        if not isinstance(self.allow_reject, bool):
            raise TypeError(
                f"Invalid value for 'allow_reject': {self.allow_reject}. "
                "Expected a boolean."
            )
        if not isinstance(self.keep_original, bool):
            raise TypeError(
                f"Invalid value for 'keep_original': {self.keep_original}. "
                "Expected a boolean."
            )

        columnwise = {"auto": "auto", "cols": True, "frame": False}[self.how]

        self._wrapped_transformer = wrap_transformer(
            self.transformer,
            self.cols,
            allow_reject=self.allow_reject,
            keep_original=self.keep_original,
            rename_columns=self.rename_columns,
            n_jobs=self.n_jobs,
            columnwise=columnwise,
        )
        X_transformed = self._wrapped_transformer.fit_transform(X, y, **kwargs)

        if isinstance(self._wrapped_transformer, ApplyToEachCol):
            self.transformers_ = self._wrapped_transformer.transformers_
        else:
            self.transformer_ = self._wrapped_transformer.transformer_

        return X_transformed

    def transform(self, X):
        """Transform a dataframe.

        Parameters
        ----------
        X : Pandas or Polars DataFrame
            The column to transform.

        **kwargs
            Extra named arguments are passed to the ``transform()``.

        Returns
        -------
        result : Pandas or Polars DataFrame
            The transformed data.
        """

        check_is_fitted(self)

        return self._wrapped_transformer.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self)

        return self._wrapped_transformer.get_feature_names_out(input_features)
