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
    Apply a transformer to columns in a dataframe.

    Columns that are not selected in the ``cols`` parameter are passed through
    without modification. This transformer automatically detects whether the
    provided transformer is a single-column transformer (i.e. has the
    ``__single_column_transformer__`` attribute) and applies it to each column
    independently; otherwise, it applies the transformer to the selected columns
    as a group.
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
    ** Applying a transformer to each column independently **

    >>> import pandas as pd
    >>> login = pd.to_datetime(
    ...     pd.Series(
    ...         ["2024-05-13T12:05:36", None, "2024-05-15T13:46:02"], name="login")
    ... )
    >>> login
    0   2024-05-13 12:05:36
    1                   NaT
    2   2024-05-15 13:46:02
    Name: login, dtype: datetime64[...]
    >>> from skrub import DatetimeEncoder
    >>> DatetimeEncoder().fit_transform(login)
       login_year  login_month  login_day  login_hour  login_total_seconds
    0      2024.0          5.0       13.0        12.0         1.715602e+09
    1         NaN          NaN        NaN         NaN                  NaN
    2      2024.0          5.0       15.0        13.0         1.715781e+09

    Apply a StandardScaler to each column in a dataframe:

    >>> import pandas as pd
    >>> from skrub import ApplyToEachCol
    >>> from sklearn.preprocessing import StandardScaler
    >>> df = pd.DataFrame(dict(A=[-10., 10.], B=[-10., 0.], C=[0., 10.]))
    >>> df
          A     B     C
    0 -10.0 -10.0   0.0
    1  10.0   0.0  10.0

    Fit a StandardScaler to each column in df:

    >>> scaler = ApplyToEachCol(StandardScaler())
    >>> scaler.fit_transform(df)
         A    B    C
    0 -1.0 -1.0 -1.0
    1  1.0  1.0  1.0
    >>> scaler.transformers_
    {'A': StandardScaler(), 'B': StandardScaler(), 'C': StandardScaler()}

    We can restrict the columns on which the transformation is applied:

    >>> scaler = ApplyToEachCol(StandardScaler(), cols=["A", "B"])
    >>> scaler.fit_transform(df)
         A    B     C
    0 -1.0 -1.0   0.0
    1  1.0  1.0  10.0

    We see that the scaling has not been applied to "C", which also does not
    appear in the transformers_:

    >>> scaler.transformers_
    {'A': StandardScaler(), 'B': StandardScaler()}
    >>> scaler.used_inputs_
    ['A', 'B']

    **Rejected columns**

    The transformer can raise :class:`RejectColumn` to indicate it cannot handle a
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
    skrub._single_column_transformer.RejectColumn: Could not find a datetime format for column 'city'.

    How these rejections are handled depends on the ``allow_reject`` parameter.
    By default, no special handling is performed and rejections are considered
    to be errors:

    >>> to_datetime = ApplyToEachCol(ToDatetime())
    >>> to_datetime.fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: Transformer ToDatetime.fit_transform failed on column 'city'. See above for the full traceback.

    However, setting ``allow_reject=True`` gives the transformer itself some
    control over which columns it should be applied to. For example, whether a
    string column contains dates is only known once we try to parse them.
    Therefore it might be sensible to try to parse all string columns but allow
    the transformer to reject those that, upon inspection, do not contain dates.

    >>> to_datetime = ApplyToEachCol(ToDatetime(), allow_reject=True)
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
    >>> scaler = ApplyToEachCol(StandardScaler(), rename_columns='{}_scaled')
    >>> scaler.fit_transform(df)
       A_scaled  B_scaled
    0      -1.0      -1.0
    1       1.0       1.0

    The renaming is only applied to columns selected by ``cols`` (and not
    rejected by the transformer when ``allow_reject`` is ``True``).

    >>> scaler = ApplyToEachCol(StandardScaler(), cols=['A'], rename_columns='{}_scaled')
    >>> scaler.fit_transform(df)
       A_scaled      B
    0      -1.0    0.0
    1       1.0  100.0

    ``rename_columns`` can be particularly useful when ``keep_original`` is
    ``True``. When a column is transformed, we can tell ``ApplyToEachCol`` to
    retain the original, untransformed column in the output. If the transformer
    produces a column with the same name, the transformation result is renamed
    to avoid a name clash.

    >>> scaler = ApplyToEachCol(StandardScaler(), keep_original=True)
    >>> scaler.fit_transform(df)                                    # doctest: +SKIP
          A  A__skrub_89725c56__      B  B__skrub_81cc7d00__
    0 -10.0                 -1.0    0.0                 -1.0
    1  10.0                  1.0  100.0                  1.0

    In this case we may want to set a more sensible name for the transformer's output:

    >>> scaler = ApplyToEachCol(
    ...     StandardScaler(), keep_original=True, rename_columns="{}_scaled"
    ... )
    >>> scaler.fit_transform(df)
          A  A_scaled      B  B_scaled
    0 -10.0      -1.0    0.0      -1.0
    1  10.0       1.0  100.0       1.0

    **Applying a transformer to columns as a group**

    Transformers that do not have the ``__single_column_transformer__``
    attribute are applied to the selected columns as a group, meaning that
    the transformer is fitted once on the selected columns together and can
    learn relationships between them.

    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.eye(4) * np.logspace(0, 3, 4), columns=list("abcd"))
    >>> df
         a     b      c       d
    0  1.0   0.0    0.0     0.0
    1  0.0  10.0    0.0     0.0
    2  0.0   0.0  100.0     0.0
    3  0.0   0.0    0.0  1000.0
    >>> from sklearn.decomposition import PCA
    >>> from skrub import ApplyToSubFrame
    >>> ApplyToSubFrame(PCA(n_components=2)).fit_transform(df).round(2)
         pca0   pca1
    0 -249.01 -33.18
    1 -249.04 -33.68
    2 -252.37  66.64
    3  750.42   0.22

    We can restrict the transformer to a subset of columns:

    >>> pca = ApplyToSubFrame(PCA(n_components=2), cols=["a", "b"])
    >>> pca.fit_transform(df).round(2)
           c       d  pca0  pca1
    0    0.0     0.0 -2.52  0.67
    1    0.0     0.0  7.50  0.00
    2  100.0     0.0 -2.49 -0.33
    3    0.0  1000.0 -2.49 -0.33
    >>> pca.used_inputs_
    ['a', 'b']
    >>> pca.created_outputs_
    ['pca0', 'pca1']
    >>> pca.transformer_
    PCA(n_components=2)

    It is possible to rename the output columns:

    >>> pca = ApplyToSubFrame(
    ...     PCA(n_components=2), cols=["a", "b"], rename_columns='my_tag-{}'
    ... )
    >>> pca.fit_transform(df).round(2)
           c       d  my_tag-pca0  my_tag-pca1
    0    0.0     0.0        -2.52         0.67
    1    0.0     0.0         7.50         0.00
    2  100.0     0.0        -2.49        -0.33
    3    0.0  1000.0        -2.49        -0.33

    We can also force preserving the original columns in the output:

    >>> pca = ApplyToSubFrame(PCA(n_components=2), cols=["a", "b"], keep_original=True)
    >>> pca.fit_transform(df).round(2)
         a     b      c       d  pca0  pca1
    0  1.0   0.0    0.0     0.0 -2.52  0.67
    1  0.0  10.0    0.0     0.0  7.50  0.00
    2  0.0   0.0  100.0     0.0 -2.49 -0.33
    3  0.0   0.0    0.0  1000.0 -2.49 -0.33

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
