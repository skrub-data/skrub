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
    Wrapper that applies a transformer to columns selected by a selector. The
    wrapper automatically selects the correct class between ``ApplyToEachCol`` and
    ``ApplyToSubFrame`` based on the type of the input transformer, but this can
    be overridden with the ``columnwise`` parameter.

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

    columnwise : 'auto' or bool, default='auto'
        Whether to apply the transformer to each selected column independently
        (equivalent to using ``ApplyToEachCol``) or to the whole sub-dataframe of
        selected columns at once (equivalent to using ``ApplyToSubFrame``). By
        default, ``ApplyToEachCol`` is used if the transformer has a
        ``__single_column_transformer__`` attribute and ``ApplyToSubFrame``
        otherwise. Pass ``columnwise=True`` to force using ``ApplyToEachCol`` and
        ``columnwise=False`` to force using ``ApplyToSubFrame``.
        Note that forcing
        ``columnwise=False`` for a single-column transformer will most likely
        cause an error during ``fit``, and forcing ``columnwise=True`` for a
        regular transformer is only appropriate if the transformer can be
        fitted on a dataframe with only one column (this is the case for most
        preprocessors such as ``OrdinalEncoder`` or ``StandardScaler``).
    """

    def __init__(
        self,
        transformer,
        cols=_SELECT_ALL_COLUMNS,
        allow_reject=False,
        keep_original=False,
        rename_columns="{}",
        columnwise="auto",
        n_jobs=None,
    ):
        self.transformer = transformer
        self.cols = cols
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
            self.cols,
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
