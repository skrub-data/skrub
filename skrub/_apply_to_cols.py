"""
ApplyToCols selects the correct transformer between ApplyToEachCol and ApplyToSubFrame
based on the type of the transformer passed to it.
"""

from sklearn.base import BaseEstimator, TransformerMixin

from ._apply_on_each_col import ApplyToEachCol
from ._wrap_transformer import wrap_transformer


class ApplyToCols(BaseEstimator, TransformerMixin):
    """
    Wrapper that applies a transformer to columns selected by a selector. The
    wrapper automatically selects the correct class between ``ApplyToEachCol`` and
    ``ApplyToSubFrame`` based on the type of the input transformer, but this can
    be overridden with the ``columnwise`` parameter.

    Parameters
    ----------
    transformer : transformer instance
        The transformer to apply to the selected columns.

    selector : selector specifier, default=s.all()
        The columns to which the transformer will be applied. See the documentation of
        ``ApplyToEachCol`` or ``ApplyToSubFrame`` for details.

    allow_reject : bool, default=False
        Whether to allow rejecting all columns. See the documentation of
        ``ApplyToEachCol`` for details.
    """

    def __init__(
        self,
        transformer,
        selector="all",
        allow_reject=False,
        keep_original=False,
        rename_columns="{}",
        columnwise="auto",
        n_jobs=None,
    ):
        self.transformer = transformer
        self.selector = selector
        self.allow_reject = allow_reject
        self.keep_original = keep_original
        self.rename_columns = rename_columns
        self.n_jobs = n_jobs
        self.columnwise = columnwise

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        self._wrapped_transformer = wrap_transformer(
            self.transformer,
            self.selector,
            allow_reject=self.allow_reject,
            keep_original=self.keep_original,
            rename_columns=self.rename_columns,
            n_jobs=self.n_jobs,
            columnwise=self.columnwise,
        )
        X_transformed = self._wrapped_transformer.fit_transform(X, y)

        if isinstance(self._wrapped_transformer, ApplyToEachCol):
            self.transformers_ = self._wrapped_transformer.transformers_
        else:
            self.transformer_ = self._wrapped_transformer.transformer_

        return X_transformed

    def transform(self, X):
        return self._wrapped_transformer.transform(X)
