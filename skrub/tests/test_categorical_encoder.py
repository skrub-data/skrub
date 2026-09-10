import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

import skrub
from skrub import ApplyToCols, CategoricalEncoder
from skrub import _dataframe as sbd


def test_categorical_encoder(df_module):
    s = df_module.make_column("col", ["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"])
    y = df_module.make_column("target", [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    enc = CategoricalEncoder(max_categories=3, target_encoder=TargetEncoder(cv=2))
    res = enc.fit_transform(s, y)

    expected_names = ["col_a", "col_d", "col_infrequent_sklearn", "col"]
    assert sbd.shape(res) == (10, 4)
    assert enc.get_feature_names_out() == expected_names
    assert list(sbd.column_names(res)) == enc.all_outputs_

    res_trans = enc.transform(s[:5])
    assert sbd.shape(res_trans) == (5, 4)
    assert list(sbd.column_names(res_trans)) == enc.all_outputs_


def test_categorical_encoder_values_and_unknown_category(df_module):
    s = df_module.make_column("col", ["a", "b"] * 10)
    y = df_module.make_column("target", [1, 0] * 10)

    enc = CategoricalEncoder(target_encoder=TargetEncoder(cv=2))
    res = enc.fit_transform(s, y)

    expected = np.column_stack(
        [
            [1.0, 0.0] * 10,
            [0.0, 1.0] * 10,
            [1.0, 0.0] * 10,
        ]
    )
    np.testing.assert_allclose(sbd.to_numpy(res), expected)

    new = df_module.make_column("col", ["a", "unknown"])
    transformed = enc.transform(new)
    np.testing.assert_allclose(
        sbd.to_numpy(transformed),
        [[1.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
    )


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


def test_categorical_encoder_dataframe_target_and_unnamed_column():
    s = pd.Series(["a", "b", "c"] * 5, name=None)
    y = pd.DataFrame({"target": np.asarray(["0", "1", "2"] * 5, dtype=object)})

    enc = CategoricalEncoder(max_categories=2)
    res = enc.fit_transform(s, y)

    assert res.columns.tolist() == [
        "categorical_enc_c",
        "categorical_enc_infrequent_sklearn",
        "categorical_enc_0.0",
        "categorical_enc_1.0",
        "categorical_enc_2.0",
    ]
    assert res.shape == (15, 5)
    assert enc.target_encoder_.target_type_ == "multiclass"


@pytest.mark.parametrize(
    "y, expected_message",
    [
        (
            pd.DataFrame({"first": [0, 1] * 5, "second": [1, 0] * 5}),
            "exactly one column",
        ),
        (np.ones((10, 2)), "one-dimensional"),
        (np.asarray(1), "one-dimensional"),
    ],
)
def test_categorical_encoder_rejects_non_1d_target(y, expected_message):
    s = pd.Series(["a", "b"] * 5, name="col")

    with pytest.raises(ValueError, match=expected_message):
        CategoricalEncoder().fit_transform(s, y)


def test_categorical_encoder_2d_string_target_and_sparse_output():
    s = pd.Series(["a", "b", "c"] * 5, name="col")
    y = np.asarray(["one", "two", "three"] * 5, dtype=object).reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(
        sparse_output=True,
        handle_unknown="ignore",
    )

    enc = CategoricalEncoder(one_hot_encoder=one_hot_encoder)
    res = enc.fit_transform(s, y)

    assert res.columns.tolist() == [
        "col_a",
        "col_b",
        "col_c",
        "col_one",
        "col_three",
        "col_two",
    ]
    transformed = enc.transform(pd.Series(["a", "new"], name="col"))
    assert transformed.shape == (2, 6)
    assert transformed.columns.tolist() == res.columns.tolist()


def test_categorical_encoder_preserves_dtypes(df_module):
    s = df_module.make_column("col", ["a", "b"] * 10)
    y = df_module.make_column("target", [1.0, 0.0] * 10)
    one_hot_encoder = OneHotEncoder(
        dtype=np.float32,
        sparse_output=False,
        handle_unknown="ignore",
    )
    enc = CategoricalEncoder(
        one_hot_encoder=one_hot_encoder,
        target_encoder=TargetEncoder(cv=2),
    )

    fitted = enc.fit_transform(s, y)
    transformed = enc.transform(s)

    for name in enc.all_outputs_:
        assert sbd.dtype(sbd.col(fitted, name)) == sbd.dtype(sbd.col(transformed, name))
    assert sbd.to_numpy(sbd.col(fitted, "col_a")).dtype == np.float32


def test_categorical_encoder_stable_names_on_collision(df_module):
    s = df_module.make_column("col", ["a", "b", "c"] * 5)
    y = df_module.make_column("target", ["a", "b", "c"] * 5)
    expected_names = [
        "col_a",
        "col_b",
        "col_c",
        "col_a_target",
        "col_b_target",
        "col_c_target",
    ]

    first = CategoricalEncoder().fit_transform(s, y)
    second = CategoricalEncoder().fit_transform(s, y)

    assert list(sbd.column_names(first)) == expected_names
    assert list(sbd.column_names(second)) == expected_names


def test_categorical_encoder_preserves_pandas_index():
    index = pd.Index([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    s = pd.Series(["a", "b"] * 5, name="col", index=index)
    y = pd.Series([1, 0] * 5, index=index)
    enc = CategoricalEncoder(target_encoder=TargetEncoder(cv=2))

    fitted = enc.fit_transform(s, y)
    transformed = enc.transform(s)

    assert fitted.index.equals(index)
    assert transformed.index.equals(index)


def test_categorical_encoder_apply_to_cols(df_module):
    df = df_module.make_dataframe(
        {
            "cat": ["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"],
            "other": ["x", "y"] * 5,
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    y = df_module.make_column("target", [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    enc = CategoricalEncoder(max_categories=3, target_encoder=TargetEncoder(cv=2))
    apply = ApplyToCols(enc, cols=["cat", "other"])

    res = apply.fit_transform(df, y)
    assert list(sbd.column_names(res)) == [
        "cat_a",
        "cat_d",
        "cat_infrequent_sklearn",
        "cat",
        "other_x",
        "other_y",
        "other",
        "num",
    ]


def test_categorical_encoder_data_op_orders_outputs_by_input_column():
    df = pd.DataFrame(
        {
            "first": ["a", "b"] * 5,
            "second": ["x", "y"] * 5,
        }
    )
    y = pd.Series([1, 0] * 5)

    result = (
        skrub.as_data_op(df)
        .skb.apply(CategoricalEncoder(target_encoder=TargetEncoder(cv=2)), y=y)
        .skb.eval()
    )

    assert result.columns.tolist() == [
        "first_a",
        "first_b",
        "first",
        "second_x",
        "second_y",
        "second",
    ]


def test_categorical_encoder_sklearn_compat():
    enc = CategoricalEncoder()
    with pytest.raises(NotFittedError):
        enc.transform(pd.Series(["a"], name="col"))
    with pytest.raises(NotFittedError):
        enc.get_feature_names_out()

    cloned = clone(enc)
    assert cloned.max_categories == enc.max_categories
