"""
ApplyToCols selects the correct transformer between ApplyToEachCol and ApplyToSubFrame
based on the type of the transformer passed to it.
"""

from sklearn.base import BaseEstimator, TransformerMixin

from . import selectors
from ._apply_on_each_col import ApplyToEachCol
from ._wrap_transformer import wrap_transformer

_SELECT_ALL_COLUMNS = selectors.all()


class ApplyToCols(BaseEstimator, TransformerMixin):
    """
    Map a transformer to columns in a dataframe.

    A separate clone of the transformer is applied to each column separately.

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
        See the documentation of
        :class:`ApplyToEachCol` or :class:`ApplyToSubFrame` for details.

    allow_reject : bool, default=False
        Whether to allow rejecting all columns. See the documentation of
        ``ApplyToEachCol`` for details.

    how : "auto", "cols" or "frame", optional
        How the transformer is applied. In most cases the default "auto"
        is appropriate.
        - "cols" means `transformer` is wrapped in a :class:`ApplyToEachCol`
            transformer, which fits a separate clone of `transformer` each
            column in `cols`. `transformer` must be a transformer (have a
            ``fit_transform`` method).
        - "frame" means `transformer` is wrapped in a :class:`ApplyToSubFrame`
            transformer, which fits a single clone of `transformer` to the
            selected part of the input dataframe. `transformer` must be a
            transformer.
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
    Wrapper that applies a transformer to columns selected by a selector. The
    wrapper automatically selects the correct class between ``ApplyToEachCol`` and
    ``ApplyToSubFrame`` based on the type of the input transformer, but this can
    be overridden with the ``how`` parameter.

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

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        if self.how not in ("auto", "cols", "frame"):
            raise ValueError(
                f"Invalid value for 'how': {self.how}. "
                "Expected one of 'auto', 'cols', or 'frame'."
            )

        if self.allow_reject not in (True, False):
            raise ValueError(
                f"Invalid value for 'allow_reject': {self.allow_reject}. "
                "Expected a boolean."
            )
        if self.keep_original not in (True, False):
            raise ValueError(
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
        X_transformed = self._wrapped_transformer.fit_transform(X, y)

        if isinstance(self._wrapped_transformer, ApplyToEachCol):
            self.transformers_ = self._wrapped_transformer.transformers_
        else:
            self.transformer_ = self._wrapped_transformer.transformer_

        return X_transformed

    def transform(self, X):
        return self._wrapped_transformer.transform(X)
