"""
ApplyToCols selects the correct transformer between ApplyToEachCol and ApplyToSubFrame
based on the type of the transformer passed to it.
"""

from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from . import selectors
from ._apply_to_each_col import ApplyToEachCol
from ._wrap_transformer import wrap_transformer

_SELECT_ALL_COLUMNS = selectors.all()


class ApplyToCols(TransformerMixin, BaseEstimator):
    """
    Apply a transformer to selected columns in a dataframe.

    This transformer applies the given transformer to all the selected columns in
    the input dataframe; non-selected columns are passed through without modification.
    By default, all selected columns are passed to the same transformer; if the
    transformer is a :class:`SingleColumnTransformer` or if ``how="cols"``, a separate
    clone of the transformer is created for each selected column and fitted to that
    column independently.

    Refer to the documentation of :class:`SingleColumnTransformer` for more details
    on single-column transformers and how to create them.

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

    how : "auto", "cols" or "frame", optional, default="auto"
        How the transformer is applied. In most cases the default "auto"
        is appropriate.

        - "auto" chooses the wrapping depending on the input and transformer.
          If the transformer has a ``__single_column_transformer__`` attribute,
          "cols" is chosen. Otherwise "frame" is chosen.
        - "cols" means `transformer` is cloned and fitted separately to each
          column in `cols`.
        - "frame" means `transformer` is fitted on all columns in `cols` together.

    allow_reject : bool, default=False
        Whether to allow refusing to transform columns for which the provided
        transformer is not suited, for example rejecting non-datetime columns if
        transformer is a DatetimeEncoder. Only relevant if the transformer is a
        SingleColumnTransformer or if how="cols".

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
        Note that this parameter is only used when the transformer
        is a SingleColumnTransformer or when how="cols".

    Notes
    -----
    All columns not listed in ``cols`` remain unmodified in the output.
    Moreover, if ``allow_reject`` is ``True`` and the transformers'
    ``fit_transform`` raises a ``RejectColumn`` exception for a particular
    column, that column is passed through unchanged. If ``allow_reject`` is
    ``False``, ``RejectColumn`` exceptions are propagated, like other errors
    raised by the transformer.

    See also
    --------
    :class:`SingleColumnTransformer` : Base class for single-column transformers,
    which allows to define custom logic to be applied to each column independently,
    and to indicate that a column cannot be transformed by raising
    :class:`RejectColumn` exceptions.

    Examples
    --------
    Consider the following dataframe:

    >>> import pandas as pd
    >>> from skrub import ApplyToCols
    >>> from sklearn.preprocessing import StandardScaler

    >>> df = pd.DataFrame(dict(
    ...     A=[-10., 10.], B=[-10., 0.], C=["Paris", "Rome"],
    ...     D=pd.to_datetime(["2024-05-13T12:05:36", "2024-05-15T13:46:02"]))
    ... )
    >>> df
        A     B      C                   D
    0 -10.0 -10.0  Paris 2024-05-13 12:05:36
    1  10.0   0.0   Rome 2024-05-15 13:46:02


    By default, the same transformer is applied to all selected columns. It is
    possible to specify which columns to select with the ``cols`` parameter. For
    example, to apply a StandardScaler to the numeric columns:
    >>> scaler = ApplyToCols(StandardScaler(), cols=["A", "B"])
    >>> scaler.fit_transform(df)
        C                   D    A    B
    0  Paris 2024-05-13 12:05:36 -1.0 -1.0
    1   Rome 2024-05-15 13:46:02  1.0  1.0

    Note that the columns "C" and "D" were not modified since they were not selected.

    String columns can be encoded with the :class:`StringEncoder`. ``ApplyToCols``
    automatically detects that :class:`StringEncoder` is a single-column transformer
    and applies it to each selected column independently:

    >>> from skrub import StringEncoder
    >>> string_encoder = ApplyToCols(StringEncoder(n_components=2), cols=["C"])
    >>> df_enc = string_encoder.fit_transform(df)
    >>> df_enc # doctest: +SKIP
        A     B       C_0       C_1                   D
    0 -10.0 -10.0  1.414214  1.414214 2024-05-13 12:05:36
    1  10.0   0.0  0.000000  0.000000 2024-05-15 13:46:02

    It is possible to force column-wise application of a transformer by setting
    ``how="cols"``, even if the transformer is not a single-column transformer:

    >>> scaler_per_col = ApplyToCols(StandardScaler(), how="cols", cols=["A", "B"])
    >>> scaler_per_col.fit_transform(df)
        A    B      C                   D
    0 -1.0 -1.0  Paris 2024-05-13 12:05:36
    1  1.0  1.0   Rome 2024-05-15 13:46:02

    It is possible to set ``allow_reject=True`` to allow the transformer to reject
    columns it cannot handle.  In this case, the rejected columns are passed through
    unchanged:
    >>> from skrub import DatetimeEncoder
    >>> datetime = ApplyToCols(DatetimeEncoder(), allow_reject=True)
    >>> datetime.fit_transform(df)
        A     B      C  D_year  D_month  D_day  D_hour  D_total_seconds
    0 -10.0 -10.0  Paris  2024.0      5.0   13.0    12.0     1.715602e+09
    1  10.0   0.0   Rome  2024.0      5.0   15.0    13.0     1.715781e+09

    If ``allow_reject=False`` (the default), the same transformation would raise
    an error since the transformer cannot handle the columns "A", "B", and "C":
    >>> datetime = ApplyToCols(DatetimeEncoder(), allow_reject=False)
    >>> datetime.fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: Transformer DatetimeEncoder.fit_transform failed on column 'A'.
    ... See above for the full traceback.

    ** Accessing fitted transformers **
    Depending on the transformer and the value of ``how``, the fitted transformers
    are stored in different attributes. If the transformer is a single-column
    transformer or if how="cols", the fitted transformers are stored in the
    ``transformers_`` attribute as a dictionary mapping column names to fitted
    transformers.
    Otherwise, the fitted transformer is stored in the ``transformer_`` attribute.

    >>> scaler_per_col.transformers_
    {'A': StandardScaler(), 'B': StandardScaler()}

    >>> scaler.transformer_
    StandardScaler()

    Columns that were not selected or were rejected do not have a transformer:
    >>> string_encoder.transformers_
    {'C': StringEncoder(n_components=2)}

    **Renaming outputs & keeping the original columns**

    The ``rename_columns`` parameter allows renaming output columns.

    >>> df = pd.DataFrame(dict(A=[-10., 10.], B=[0., 100.]))
    >>> scaler = ApplyToCols(StandardScaler(), rename_columns='{}_scaled')
    >>> scaler.fit_transform(df)
       A_scaled  B_scaled
    0      -1.0      -1.0
    1       1.0       1.0

    The renaming is only applied to columns selected by ``cols`` (and not
    rejected by the transformer when ``allow_reject`` is ``True``).

    >>> scaler = ApplyToCols(StandardScaler(), cols=['A'], rename_columns='{}_scaled')
    >>> scaler.fit_transform(df)
        B  A_scaled
    0    0.0      -1.0
    1  100.0       1.0

    ``rename_columns`` can be particularly useful when ``keep_original`` is
    ``True``. When a column is transformed, we can tell ``ApplyToCols`` to
    retain the original, untransformed column in the output. If the transformer
    produces a column with the same name, the transformation result is renamed
    to avoid a name clash.

    >>> scaler = ApplyToCols(StandardScaler(), keep_original=True, how="cols")
    >>> scaler.fit_transform(df)                                    # doctest: +SKIP
          A  A__skrub_89725c56__      B  B__skrub_81cc7d00__
    0 -10.0                 -1.0    0.0                 -1.0
    1  10.0                  1.0  100.0                  1.0

    In this case we may want to set a more sensible name for the transformer's output:

    >>> scaler = ApplyToCols(
    ...     StandardScaler(), keep_original=True, rename_columns="{}_scaled", how="cols"
    ... )
    >>> scaler.fit_transform(df)
        A  A_scaled      B  B_scaled
    0 -10.0      -1.0    0.0      -1.0
    1  10.0       1.0  100.0       1.0
    """

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
