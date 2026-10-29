"""
Implementation of CatEncoder combining OneHotEncoder and TargetEncoder.
"""

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._join_utils import pick_column_names
from ._single_column_transformer import SingleColumnTransformer

__all__ = ["CatEncoder"]


class CatEncoder(TransformerMixin, SingleColumnTransformer):
    """Encode a single categorical column combining OneHotEncoder and TargetEncoder.

    This transformer applies a :class:`~sklearn.preprocessing.OneHotEncoder`
    to encode frequent categories into binary one-hot columns, and a
    :class:`~sklearn.preprocessing.TargetEncoder` to target-encode the column.

    Parameters
    ----------
    max_categories : int or None, default=10
        Maximum number of categories for the ``OneHotEncoder``. If there are more
        categories, the remaining ones are grouped into an infrequent category.

    Attributes
    ----------
    one_hot_encoder_ : OneHotEncoder
        The fitted ``OneHotEncoder`` instance.

    target_encoder_ : TargetEncoder
        The fitted ``TargetEncoder`` instance.

    one_hot_outputs_ : list of str
        Feature names created by the one-hot encoder.

    target_outputs_ : list of str
        Feature names using the ``_target_sklearn`` suffix (followed by the
        class label for multiclass targets). Collisions with one-hot features
        are resolved by adding a random ``__skrub_<token>__`` suffix.

    all_outputs_ : list of str
        The list of feature names created by the transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import CatEncoder
    >>> s = pd.Series(["a", "b", "c", "d", "e"] * 4, name="col")
    >>> y = pd.Series([1, 0, 1, 0, 1] * 4)
    >>> enc = CatEncoder(max_categories=3)
    >>> enc.fit_transform(s, y).head(2)
       col_d  col_e  col_infrequent_sklearn  col_target_sklearn
    0    0.0    0.0                     1.0                 ...
    1    0.0    0.0                     1.0                 ...
    """

    def __init__(
        self,
        max_categories=10,
    ):
        self.max_categories = max_categories

    def fit(self, column, y):
        """Fit the encoder to a categorical column.

        Parameters
        ----------
        column : Pandas or Polars Series
            The single column to fit.

        y : Pandas or Polars Series, DataFrame, or array-like
            Target values for target encoding.

        Returns
        -------
        self
            The fitted encoder.
        """
        self.fit_transform(column, y)
        return self

    def fit_transform(self, column, y):
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

        self.one_hot_encoder_ = OneHotEncoder(
            max_categories=self.max_categories,
            sparse_output=False,
            handle_unknown="ignore",
        )
        self.target_encoder_ = TargetEncoder()

        col_name = sbd.name(column) or "cat_enc"
        X_arr = sbd.to_numpy(column).reshape(-1, 1)
        y_vec = _check_y(y)

        ohe_res = self.one_hot_encoder_.fit_transform(X_arr)
        te_res = self.target_encoder_.fit_transform(X_arr, y_vec)

        self.one_hot_outputs_ = list(
            self.one_hot_encoder_.get_feature_names_out([col_name])
        )
        if self.target_encoder_.target_type_ == "multiclass":
            target_outputs = [
                f"{col_name}_target_sklearn_{cls}"
                for cls in self.target_encoder_.classes_
            ]
        else:
            target_outputs = [f"{col_name}_target_sklearn"]

        self.target_outputs_ = pick_column_names(
            target_outputs, forbidden_names=self.one_hot_outputs_
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

        X_arr = sbd.to_numpy(column).reshape(-1, 1)

        ohe_res = self.one_hot_encoder_.transform(X_arr)
        te_res = self.target_encoder_.transform(X_arr)

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


def _check_y(y):
    """Validate and convert target y to a 1D numpy array."""
    if isinstance(y, np.ndarray):
        y_arr = y
    elif sbd.is_dataframe(y) or sbd.is_column(y):
        y_arr = sbd.to_numpy(y)
    else:
        y_arr = np.asarray(y)

    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]
    elif y_arr.ndim == 2:
        raise ValueError(
            f"CatEncoder expects y to contain exactly one column; got {y_arr.shape[1]}."
        )
    elif y_arr.ndim != 1:
        raise ValueError(
            "CatEncoder expects y to be one-dimensional or a single-column dataframe; "
            f"got an array with shape {y_arr.shape}."
        )

    if y_arr.dtype == object:
        # Infer numeric types from the values without parsing string labels.
        # TargetEncoder cannot infer the target type of object-typed numbers.
        inferred = np.asarray(y_arr.tolist())
        if inferred.dtype.kind in "biuf":
            y_arr = inferred
    return y_arr
