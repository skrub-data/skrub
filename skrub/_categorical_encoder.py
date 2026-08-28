"""
Implementation of CategoricalEncoder combining OneHotEncoder and TargetEncoder.
"""

import numpy as np
from sklearn.base import TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._single_column_transformer import SingleColumnTransformer

__all__ = ["CategoricalEncoder"]


class CategoricalEncoder(TransformerMixin, SingleColumnTransformer):
    """Encode a single categorical column combining OneHotEncoder and TargetEncoder.

    This transformer applies a :class:`~sklearn.preprocessing.OneHotEncoder`
    to encode frequent categories into binary one-hot columns, and a
    :class:`~sklearn.preprocessing.TargetEncoder` to target-encode the column.

    Parameters
    ----------
    max_categories : int or None, default=10
        Maximum number of categories for the ``OneHotEncoder``. If there are more
        categories, the remaining ones are grouped into an infrequent category.
        Ignored if a custom ``one_hot_encoder`` is provided.

    one_hot_encoder : OneHotEncoder instance or None, default=None
        Custom ``OneHotEncoder`` instance to use. If ``None``, a default
        ``OneHotEncoder(max_categories=max_categories, sparse_output=False,
        handle_unknown="ignore")`` will be used.

    target_encoder : TargetEncoder instance or None, default=None
        Custom ``TargetEncoder`` instance to use. If ``None``, a default
        ``TargetEncoder()`` will be used. Depending on the installed scikit-learn
        version and its cross-validation defaults, repeated calls to
        ``fit_transform`` may produce different encodings. Pass a configured
        ``TargetEncoder`` to control the cross-validation strategy.

    Attributes
    ----------
    one_hot_encoder_ : OneHotEncoder
        The fitted ``OneHotEncoder`` instance.

    target_encoder_ : TargetEncoder
        The fitted ``TargetEncoder`` instance.

    one_hot_outputs_ : list of str
        Feature names created by the one-hot encoder.

    target_outputs_ : list of str
        Feature names created by the target encoder. Deterministic ``_target``
        suffixes are added when names would collide with one-hot features.

    all_outputs_ : list of str
        The list of feature names created by the transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import CategoricalEncoder
    >>> s = pd.Series(["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"], name="col")
    >>> y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    >>> enc = CategoricalEncoder(max_categories=3)
    >>> enc.fit_transform(s, y)
       col_a  col_d  col_infrequent_sklearn  col
    0    1.0    0.0                     0.0  ...
    1    0.0    0.0                     1.0  ...
    """

    def __init__(
        self,
        max_categories=10,
        one_hot_encoder=None,
        target_encoder=None,
    ):
        self.max_categories = max_categories
        self.one_hot_encoder = one_hot_encoder
        self.target_encoder = target_encoder

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a categorical column.

        Parameters
        ----------
        column : Pandas or Polars Series
            The single column to transform.

        y : Pandas or Polars Series, DataFrame, or array-like
            Target values for target encoding.

        Returns
        -------
        res_df : Pandas or Polars DataFrame
            DataFrame containing one-hot and target-encoded features.
        """
        if y is None:
            raise ValueError("Target y must be provided to fit CategoricalEncoder.")

        if self.one_hot_encoder is None:
            self.one_hot_encoder_ = OneHotEncoder(
                max_categories=self.max_categories,
                sparse_output=False,
                handle_unknown="ignore",
            )
        else:
            self.one_hot_encoder_ = clone(self.one_hot_encoder)

        if self.target_encoder is None:
            self.target_encoder_ = TargetEncoder()
        else:
            self.target_encoder_ = clone(self.target_encoder)

        col_name = sbd.name(column) or "categorical_enc"
        X_pandas = sbd.to_pandas(column).to_frame()

        if sbd.is_dataframe(y):
            if sbd.shape(y)[1] != 1:
                raise ValueError(
                    "CategoricalEncoder expects y to contain exactly one column; "
                    f"got {sbd.shape(y)[1]}."
                )
            y_col = sbd.col_by_idx(y, 0)
        else:
            y_col = y

        if sbd.is_column(y_col):
            y_vec = sbd.to_numpy(y_col)
        else:
            y_vec = np.asarray(y_col)

        if y_vec.ndim == 2 and y_vec.shape[1] == 1:
            y_vec = y_vec[:, 0]
        elif y_vec.ndim != 1:
            raise ValueError(
                "CategoricalEncoder expects y to be one-dimensional or a "
                f"single-column dataframe; got an array with shape {y_vec.shape}."
            )

        if y_vec.dtype == object:
            try:
                y_vec = y_vec.astype(float)
            except (ValueError, TypeError):
                pass

        ohe_res = self.one_hot_encoder_.fit_transform(X_pandas)
        te_res = self.target_encoder_.fit_transform(X_pandas, y_vec)

        if hasattr(ohe_res, "toarray"):
            ohe_res = ohe_res.toarray()

        self.one_hot_outputs_ = list(
            self.one_hot_encoder_.get_feature_names_out([col_name])
        )
        target_outputs = list(self.target_encoder_.get_feature_names_out([col_name]))
        self.target_outputs_ = _make_target_names_unique(
            target_outputs, self.one_hot_outputs_
        )

        res_df = self._make_output(column, ohe_res, te_res)
        self.all_outputs_ = list(sbd.column_names(res_df))
        return res_df

    def transform(self, column):
        """Transform a single column using fitted OneHotEncoder and TargetEncoder.

        Parameters
        ----------
        column : Pandas or Polars Series
            The column to transform.

        Returns
        -------
        res_df : Pandas or Polars DataFrame
            Transformed features.
        """
        check_is_fitted(
            self,
            [
                "one_hot_encoder_",
                "target_encoder_",
                "one_hot_outputs_",
                "target_outputs_",
                "all_outputs_",
            ],
        )

        X_pandas = sbd.to_pandas(column).to_frame()

        ohe_res = self.one_hot_encoder_.transform(X_pandas)
        te_res = self.target_encoder_.transform(X_pandas)

        if hasattr(ohe_res, "toarray"):
            ohe_res = ohe_res.toarray()

        return self._make_output(column, ohe_res, te_res)

    def _make_output(self, column, ohe_res, te_res):
        """Build the output without coercing the encoders' individual dtypes."""
        ohe_df = sbd.make_dataframe_like(
            column, dict(zip(self.one_hot_outputs_, ohe_res.T))
        )
        ohe_df = sbd.copy_index(column, ohe_df)

        te_df = sbd.make_dataframe_like(
            column, dict(zip(self.target_outputs_, te_res.T))
        )
        te_df = sbd.copy_index(column, te_df)

        return sbd.concat(ohe_df, te_df, axis=1)

    def get_feature_names_out(self, input_features=None):
        """Return the names of all generated output features.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored.

        Returns
        -------
        list of str
            Feature names generated by the encoder.
        """
        check_is_fitted(self, "all_outputs_")
        return self.all_outputs_


def _make_target_names_unique(target_names, one_hot_names):
    """Make target-encoded names unique with deterministic suffixes."""
    used_names = set(one_hot_names)
    unique_names = []
    for name in target_names:
        candidate = name
        suffix_idx = 1
        while candidate in used_names:
            suffix = "target" if suffix_idx == 1 else f"target_{suffix_idx}"
            candidate = f"{name}_{suffix}"
            suffix_idx += 1
        unique_names.append(candidate)
        used_names.add(candidate)
    return unique_names
