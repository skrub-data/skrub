from . import _dataframe as sbd
from . import selectors as s
from ._base import SkrubBaseEstimator, TransformerMixin


def _mode_frequency(X):
    """Compute the mode frequency for each feature in a dataframe.

    Parameters
    ----------
    X : dataframe of shape (n_samples, n_features)
        Input data to compute mode frequency.

    Returns
    -------
    dict
        A dictionary containing the mode frequency for each feature.
    """
    mode_freq = {}
    for col in s.select(X, s.numeric).columns:
        mode_freq[col] = X[col].value_counts(normalize=True).max()

    # Sort the dictionary by mode frequency in ascending order and return it
    return dict(sorted(mode_freq.items(), key=lambda item: item[1]))


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
        M_features=100,
        R_order=2,
        exclude_cols=None,
        max_sample=10000,
    ):
        self.tau_card = tau_card
        self.tau_miss = tau_miss
        self.tau_mode = tau_mode
        self.max_base_feats = max_base_feats
        self.M_features = M_features
        self.R_order = R_order
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
        n = min(X.shape[0] * 0.10, max_sample=self.max_sample)
        self.filter = self._filter(sbd.sample(X, n))
        output = self._generate_features(X, y=y)
        # for sklearn DO I NEED THESE TOO?
        # self.feature_names_in_ = self._preprocessors[0].feature_names_out_
        # self.n_features_in_ = len(self.feature_names_in_)

        return output

    def _transform(self, X, y=None):
        output = self._generate_features(X, y=y)
        return output

    def _filter(self, X):
        f_mode = _mode_frequency(X)
        # selectors which define the filtering rules
        low_card = ~s.cardinality_below(self.tau_card)
        missing = ~s.has_nulls(self.tau_miss)
        f_mode_threshold = s.cols(lambda col: f_mode[col] < self.tau_mode)
        top_N_features = s.cols(
            f_mode.keys()[: min(self.max_base_feats, len(f_mode.keys()))]
        )
        excluded = s.cols(self.exclude_cols) if self.exclude_cols else s.none

        filter = (
            s.numeric
            & low_card
            & missing
            & f_mode_threshold
            & top_N_features
            & ~excluded
        )

        return filter

    def _generate_features(self, X, y=None):
        # combinatorics
        # combinations = itertools.combinations(X.columns, 2)
        # permutations = itertools.permutations(X.columns, 2)

        # add = random.sample(
        #     list(combinations), min(self.M_features, len(list(combinations)))
        # )
        # subtract = random.sample(
        #     list(permutations), min(self.M_features, len(list(permutations)))
        # )
        # multiply = random.sample(
        #     list(combinations), min(self.M_features, len(list(combinations)))
        # )
        # divide = random.sample(
        #     list(permutations), min(self.M_features, len(list(permutations)))
        # )

        pass
