import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from skrub import ApplyToCols, CategoricalEncoder
from skrub import _dataframe as sbd


def test_categorical_encoder_pandas(df_module):
    s = pd.Series(["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"], name="col")
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    enc = CategoricalEncoder(max_categories=3, target_encoder=TargetEncoder(cv=2))
    res = enc.fit_transform(s, y)

    expected_names = ["col_a", "col_d", "col_infrequent_sklearn", "col"]
    assert sbd.is_dataframe(res)
    assert len(res) == len(s)
    assert enc.get_feature_names_out() == expected_names
    assert list(res.columns) == enc.all_outputs_

    res_trans = enc.transform(s[:5])
    assert len(res_trans) == 5
    assert list(res_trans.columns) == enc.all_outputs_


def test_categorical_encoder_polars():
    pl = pytest.importorskip("polars")
    s = pl.Series("col", ["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"])
    y = pl.Series("target", [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    enc = CategoricalEncoder(max_categories=3, target_encoder=TargetEncoder(cv=2))
    res = enc.fit_transform(s, y)

    expected_names = ["col_a", "col_d", "col_infrequent_sklearn", "col"]
    assert sbd.is_polars(res)
    assert len(res) == len(s)
    assert enc.get_feature_names_out() == expected_names

    res_trans = enc.transform(s[:5])
    assert len(res_trans) == 5
    assert list(sbd.column_names(res_trans)) == enc.all_outputs_


def test_categorical_encoder_y_none():
    s = pd.Series(["a", "b", "a"], name="col")
    enc = CategoricalEncoder()
    with pytest.raises(ValueError, match="Target y must be provided"):
        enc.fit_transform(s, y=None)


def test_categorical_encoder_custom_encoders():
    s = pd.Series(["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"], name="col")
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    custom_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    custom_te = TargetEncoder(cv=2)

    enc = CategoricalEncoder(one_hot_encoder=custom_ohe, target_encoder=custom_te)
    _ = enc.fit_transform(s, y)

    assert hasattr(enc, "one_hot_encoder_")
    assert hasattr(enc, "target_encoder_")
    # Verify original estimators were not mutated (cloned)
    assert enc.one_hot_encoder_ is not custom_ohe
    assert enc.target_encoder_ is not custom_te


def test_categorical_encoder_apply_to_cols():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"],
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    enc = CategoricalEncoder(max_categories=3, target_encoder=TargetEncoder(cv=2))
    apply = ApplyToCols(enc, cols="cat")

    res = apply.fit_transform(df, y)
    assert "cat_a" in res.columns
    assert "cat_d" in res.columns
    assert "num" in res.columns


def test_categorical_encoder_sklearn_compat():
    enc = CategoricalEncoder()
    with pytest.raises(Exception):
        check_is_fitted(enc)

    cloned = clone(enc)
    assert cloned.max_categories == enc.max_categories
