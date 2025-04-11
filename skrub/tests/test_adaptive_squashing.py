import warnings

import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._dataframe._common import _set_index
from skrub._on_each_column import RejectColumn
from skrub.adaptive_squashing import AdaptiveSquashingTransformer


@pytest.mark.parametrize(
    "values",
    [
        [0.6 * i for i in range(20)],
        [0.0] * 10 + [2.0],
        [2.0, -1.0] + [0.0] * 10 + [-1.0, 1.0],
        [],
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize(
    "params",
    [
        dict(squash_threshold=3.0, lower_quantile=0.25, upper_quantile=0.75),
        dict(squash_threshold=5.0, lower_quantile=0.1, upper_quantile=0.6),
    ],
)
def test_adaptive_squashing_output(df_module, values, dtype, params):
    with warnings.catch_warnings():
        # this is to filter the warning created by df_module.make_column
        # due to pd.Series.convert_dtypes()
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        n_train = len(values) // 2

        index = np.arange(len(values))

        X = {"train": values[:n_train], "test": values[n_train:]}
        X_df = dict()
        indexes = {"train": index[:n_train], "test": index[n_train:]}

        for part in ["train", "test"]:
            if dtype in [np.float32, np.float64]:
                X[part] = X[part] + [np.nan, np.inf, -np.inf]

            X[part] = np.asarray(X[part], dtype=dtype)
            name = part
            X_df[part] = df_module.make_column(name, X[part])
            # try to have different indexes for train and test
            indexes[part] = np.arange(len(X_df[part])) + (
                100_000 if part == "test" else 0
            )
            X_df[part] = _set_index(X_df[part], indexes[part])

        tfm = AdaptiveSquashingTransformer(**params)
        ft_out = tfm.fit_transform(X_df["train"])

        t_out = tfm.transform(X_df["test"])
        tfm_out = {"train": ft_out, "test": t_out}

        eps = 1e-30
        finite_values = X["train"][np.isfinite(X["train"])].astype(np.float32)
        if len(finite_values) > 0:
            median = np.median(finite_values)
            quantiles = [
                np.quantile(finite_values, q)
                for q in [params["lower_quantile"], params["upper_quantile"]]
            ]
            if quantiles[1] == quantiles[0]:
                min = np.min(finite_values)
                max = np.max(finite_values)
                if min == max:
                    scale = 0
                else:
                    scale = 2.0 / (max - min + eps)
            else:
                scale = 1.0 / (quantiles[1] - quantiles[0] + eps)
        else:
            median = 0.0
            scale = 1.0

        for part in ["train", "test"]:
            squash_threshold = params["squash_threshold"]
            values_np = X[part].astype(np.float32)
            result = np.copy(values_np)
            isfinite = np.isfinite(values_np)
            scaled_finite = scale * (values_np[isfinite] - median)
            result[isfinite] = scaled_finite / np.sqrt(
                1 + (scaled_finite / squash_threshold) ** 2
            )
            isinf = np.isinf(values_np)
            result[isinf] = np.sign(values_np[isinf]) * squash_threshold

            if np.any(np.abs(result) > params["squash_threshold"] + 1e-8):
                raise RuntimeError("Test target should not exceed squash_threshold")

            # todo: we test for name='train' in both cases here. Is this desired?
            result_df = sbd.to_float32(
                _set_index(df_module.make_column("train", result), indexes[part])
            )
            df_module.assert_column_equal(result_df, tfm_out[part])


@pytest.mark.parametrize(
    "config",
    [
        (
            "number",
            dict(squash_threshold=None, lower_quantile=0.25, upper_quantile=0.75),
        ),
        (
            "positive",
            dict(squash_threshold=0.0, lower_quantile=0.25, upper_quantile=0.75),
        ),
        (
            "finite",
            dict(squash_threshold=np.inf, lower_quantile=0.25, upper_quantile=0.75),
        ),
        (
            "number",
            dict(squash_threshold=3.0, lower_quantile="0.25", upper_quantile=0.75),
        ),
        (
            "number",
            dict(squash_threshold=3.0, lower_quantile=0.25, upper_quantile="0.75"),
        ),
        (
            "number",
            dict(squash_threshold=3.0, lower_quantile=0.25, upper_quantile="0.75"),
        ),
        ("need", dict(squash_threshold=3.0, lower_quantile=-0.2, upper_quantile=0.75)),
        ("need", dict(squash_threshold=3.0, lower_quantile=0.25, upper_quantile=1.1)),
        ("need", dict(squash_threshold=3.0, lower_quantile=0.8, upper_quantile=0.75)),
    ],
)
def test_adaptive_squashing_error_msgs(df_module, config):
    with warnings.catch_warnings():
        # this is to filter the warning created by df_module.make_column
        # due to pd.Series.convert_dtypes()
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        msg_match, params = config
        X = df_module.make_column("col", [-1.0, 0.0, 1.0])
        tfm = AdaptiveSquashingTransformer(**params)
        with pytest.raises(ValueError, match=msg_match):
            tfm.fit_transform(X)


def test_adaptive_squashing_no_col_name(df_module):
    with warnings.catch_warnings():
        # this is to filter the warning created by df_module.make_column
        # due to pd.Series.convert_dtypes()
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        X = df_module.make_column(None, [-1.0, 0.0, 1.0])
        tfm = AdaptiveSquashingTransformer()
        X_out = tfm.fit_transform(X)
        if sbd.name(X_out) is None:
            raise RuntimeError("Column name None did not get converted")


def test_adaptive_squashing_non_numeric(df_module):
    with warnings.catch_warnings():
        # this is to filter the warning created by df_module.make_column
        # due to pd.Series.convert_dtypes()
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        X = df_module.make_column(None, ["a", "b"])
        tfm = AdaptiveSquashingTransformer()
        with pytest.raises(RejectColumn, match="not numeric"):
            tfm.fit_transform(X)


def test_adaptive_squashing_known_values(df_module):
    with warnings.catch_warnings():
        # this is to filter the warning created by df_module.make_column
        # due to pd.Series.convert_dtypes()
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        X = df_module.make_column("test", [np.nan, -np.inf, -1.0, 0.0, 1.0, np.inf])
        tfm = AdaptiveSquashingTransformer(3.0)
        X_out = tfm.fit_transform(X)
        target_value = 1.0 / np.sqrt(1 + (1.0 / 3.0) ** 2)
        X_target = sbd.to_float32(
            df_module.make_column(
                "test", [np.nan, -3.0, -target_value, 0.0, target_value, 3.0]
            )
        )
        df_module.assert_column_equal(X_target, X_out)
