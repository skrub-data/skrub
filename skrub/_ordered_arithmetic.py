import pandas as pd
from sklearn.base import TransformerMixin

from . import _dataframe as sbd
from . import selectors as s
from ._base import SkrubBaseEstimator

# from ._dispatch import dispatch, raise_dispatch_unregistered_type

print(sbd.__name__)
__all__ = ["OrderedArithmetic"]


# @dispatch
# def top_N_features(modes, N):
#     raise_dispatch_unregistered_type(modes)


# @top_N_features.specialize("pandas")
# def _top_N_features_pandas(modes, N):
# # Get the column names
#     top_N = modes.iloc[0].nlargest(n).index.tolist()


# @top_N_features.specialize("polars")
# def _top_N_features_polars(df, N):
#     return "polars"


class OrderedArithmetic(TransformerMixin, SkrubBaseEstimator):
    """Transformer to perform ordered arithmetic feature engineering.

    Parameters
    ----------
    tau_card : int, default=3
        Minimum cardinality threshold for features to be considered for ordered
        arithmetic feature engineering. Features with cardinality below this
        threshold will be filtered out.
    tau_miss : float, default=0.95
        Maximum missing rate threshold for features to be considered for ordered
        arithmetic feature engineering. Features with a missing rate above this
        threshold will be filtered out.
    tau_mode : float, default=0.95
        Maximum mode frequency threshold for features to be considered for ordered
        arithmetic feature engineering. Features with a mode frequency above this
        threshold will be filtered out.
    max_base_feats : int, default=100
        Maximum number of base features to use for ordered arithmetic feature
        engineering.
        If the number of features in the input dataset exceeds this value,
        the features with the lowest mode frequency are used.
    M_features : int, default=100
        Maximum number of ordered arithmetic features to generate.
    R_order : int, default=2
        Maximum order of the ordered arithmetic features to generate.
    exclude_cols : list of str, default=None
        List of column names to exclude from ordered arithmetic feature engineering.
    """

    def __init__(
        self,
        tau_card=3,
        tau_miss=0.95,
        tau_mode=0.95,
        max_base_feats=100,
        Mfeatures=10,
        Rorder=2,
        exclude_cols=None,
        max_sample=10000,
    ):
        self.tau_card = tau_card
        self.tau_miss = tau_miss
        self.tau_mode = tau_mode
        self.max_base_feats = max_base_feats
        self.M_features = Mfeatures
        self.R_order = Rorder
        self.exclude_cols = exclude_cols
        self.max_sample = max_sample

    def fit_transform(self, X, y=None):
        """Fit transformer and transform dataframe.

            Parameters
            ----------
            X : dataframe of shape (n_samples, n_features)
                Input data to transform.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None, \
                    default=None
                Target values for supervised learning (None for unsupervised
                transformations).

            Returns
            -------
            dataframe
                The transformed input.
            """
        # sample 10% of the data up to max_sample
        n = min(int(X.shape[0] * 0.10), self.max_sample)
        self.filter = self._filter(sbd.sample(s.select(X, s.numeric()), n))
        output = "output"  # self._generate_features(X, y=y)
        # for sklearn DO I NEED THESE TOO?
        # self.feature_names_in_ = self._preprocessors[0].feature_names_out_
        # self.n_features_in_ = len(self.feature_names_in_)
        return output

    def _transform(self, X, y=None):
        output = self._generate_features(X, y=y)
        return output

    def _filter(self, X):
        modes = self._mode_frequencies(X)

        # selectors which define the filtering rules
        low_card = ~s.cardinality_below(self.tau_card)
        missing = ~s.has_nulls(self.tau_miss)
        fmode_threshold = s.select(
            modes, s.filter(lambda col: sbd.max(col) < self.tau_mode)
        )
        fmode_threshold = s.cols(*fmode_threshold)
        top_N_features = s.cols(
            *modes.iloc[0].nlargest(self.max_base_feats).index.tolist()
        )

        filter = s.numeric() & low_card & missing & fmode_threshold & top_N_features
        if self.exclude_cols:
            excluded = s.cols(*self.exclude_cols)
            filter = filter & ~excluded

        return filter
        #
        # return [s.numeric(),
        #         low_card,
        #         missing,
        #         fmode_threshold,
        #         top_N_features,
        #         ]

    def _generate_features(self, X, y=None):
        # X_filtered = s.select(X, self.filter)

        # self.nfeatures = 0
        # # combinatorics
        # #   combos = itertools.combinations(X_filtered.columns, 2)

        # # should this be a decorator?
        # def _generate_indices(columns, order):
        #     combos = itertools.combinations(columns, order)
        #     perms = itertools.permutations(columns, order)

        #     add = random.sample(perms, min(self.M_features, len(perms)))
        #     subtract = random.sample(combos, min(self.M_features, len(combos)))
        #     multiply = random.sample(perms, min(self.M_features, len(perms)))
        #     divide = random.sample(perms, min(self.M_features, len(perms)))

        #     return add, subtract, multiply, divide

        # order = 2
        # features = []
        # while self.nfeatures < self.M_features:
        #     if len(features) % 4 == 0:
        #         order += 1

        #     perms = itertools.permutations(X_filtered.columns, 2)
        #     pairs = random.sample(perms, min(self.M_features, len(perms)))
        #     features.append(pairs)
        #     self.nfeatures += len(pairs)
        pass

    def _mode_frequencies(self, X, y=None):
        N = sbd.shape(X)[0]

        modes = {
            n: sorted(
                sbd.to_list(sbd.value_counts(sbd.col(X, n))["count"] / N), reverse=True
            )[:1]
            for n in sbd.column_names(X)
        }
        return pd.DataFrame(modes)
