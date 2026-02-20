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

    allow_reject : bool, default=False
        Whether to allow refusing to transform columns for which the provided
        transformer is not suited, for example rejecting non-datetime columns if
        transformer is a DatetimeEncoder. See the documentation of
        :class:`ApplyToEachCol` for details.

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

    Notes
    -----
    All columns not listed in ``cols`` remain unmodified in the output.
    Moreover, if ``allow_reject`` is ``True`` and the transformers'
    ``fit_transform`` raises a ``RejectColumn`` exception for a particular
    column, that column is passed through unchanged. If ``allow_reject`` is
    ``False``, ``RejectColumn`` exceptions are propagated, like other errors
    raised by the transformer.
    """

    def __init__(
        self,
        transformer,
        cols=_SELECT_ALL_COLUMNS,
        allow_reject=False,
        keep_original=False,
        rename_columns="{}",
        how="auto",
        n_jobs=None,
    ):
        self.transformer = transformer
        self.cols = cols
        self.allow_reject = allow_reject
        self.keep_original = keep_original
        self.rename_columns = rename_columns
        self.n_jobs = n_jobs
        self.how = how

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
        if isinstance(self._wrapped_transformer, ApplyToEachCol):
            check_is_fitted(self, "transformers_")
        else:
            check_is_fitted(self, "transformer_")

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
        return self._wrapped_transformer.get_feature_names_out(input_features)
