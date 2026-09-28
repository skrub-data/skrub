import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import skrub
from skrub import ApplyToCols, CatEncoder
from skrub import _dataframe as sbd


def test_cat_encoder(df_module):
    s = df_module.make_column(
        "col", ["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"] * 2
    )
    y = df_module.make_column("target", [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 2)

    enc = CatEncoder(max_categories=3)
    res = enc.fit_transform(s, y)

    expected_names = ["col_a", "col_d", "col_infrequent_sklearn", "col"]
    assert sbd.shape(res) == (20, 4)
    assert enc.get_feature_names_out() == expected_names
    assert list(sbd.column_names(res)) == enc.all_outputs_

    res_trans = enc.transform(s[:5])
    assert sbd.shape(res_trans) == (5, 4)
    assert list(sbd.column_names(res_trans)) == enc.all_outputs_


def test_cat_encoder_values_and_unknown_category(df_module):
    s = df_module.make_column("col", ["a", "b"] * 10)
    y = df_module.make_column("target", [1, 0] * 10)

    enc = CatEncoder()
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


def test_cat_encoder_y_none():
    s = pd.Series(["a", "b", "a"], name="col")
    enc = CatEncoder()
    with pytest.raises(ValueError, match="one-dimensional"):
        enc.fit_transform(s, y=None)


def test_cat_encoder_dataframe_target_and_unnamed_column():
    s = pd.Series(["a", "b", "c"] * 5, name=None)
    y = pd.DataFrame({"target": np.asarray(["0", "1", "2"] * 5, dtype=object)})

    enc = CatEncoder(max_categories=2)
    res = enc.fit_transform(s, y)

    assert res.columns.tolist() == [
        "cat_enc_c",
        "cat_enc_infrequent_sklearn",
        "cat_enc_0.0",
        "cat_enc_1.0",
        "cat_enc_2.0",
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
def test_cat_encoder_rejects_non_1d_target(y, expected_message):
    s = pd.Series(["a", "b"] * 5, name="col")

    with pytest.raises(ValueError, match=expected_message):
        CatEncoder().fit_transform(s, y)


def test_cat_encoder_2d_string_target():
    s = pd.Series(["a", "b", "c"] * 10, name="col")
    y = np.asarray(["one", "two", "three"] * 10, dtype=object).reshape(-1, 1)

    enc = CatEncoder()
    res = enc.fit_transform(s, y)

    assert set(res.columns.tolist()) == {
        "col_a",
        "col_b",
        "col_c",
        "col_one",
        "col_three",
        "col_two",
    }
    transformed = enc.transform(pd.Series(["a", "new"], name="col"))
    assert transformed.shape == (2, 6)
    assert transformed.columns.tolist() == res.columns.tolist()


def test_cat_encoder_preserves_dtypes(df_module):
    s = df_module.make_column("col", ["a", "b"] * 10)
    y = df_module.make_column("target", [1.0, 0.0] * 10)
    enc = CatEncoder()

    fitted = enc.fit_transform(s, y)
    transformed = enc.transform(s)

    for name in enc.all_outputs_:
        assert sbd.dtype(sbd.col(fitted, name)) == sbd.dtype(sbd.col(transformed, name))


def test_cat_encoder_stable_names_on_collision(df_module):
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

    first = CatEncoder().fit_transform(s, y)
    second = CatEncoder().fit_transform(s, y)

    assert list(sbd.column_names(first)) == expected_names
    assert list(sbd.column_names(second)) == expected_names


def test_cat_encoder_preserves_pandas_index():
    index = pd.Index(list(range(100, 120)))
    s = pd.Series(["a", "b"] * 10, name="col", index=index)
    y = pd.Series([1, 0] * 10, index=index)
    enc = CatEncoder()

    fitted = enc.fit_transform(s, y)
    transformed = enc.transform(s)

    assert fitted.index.equals(index)
    assert transformed.index.equals(index)


def test_cat_encoder_apply_to_cols(df_module):
    df = df_module.make_dataframe(
        {
            "cat": ["a", "b", "a", "c", "d", "e", "a", "b", "c", "d"] * 2,
            "other": ["x", "y"] * 10,
            "num": list(range(20)),
        }
    )
    y = df_module.make_column("target", [1, 0] * 10)

    enc = CatEncoder(max_categories=3)
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


def test_cat_encoder_data_op_orders_outputs_by_input_column():
    df = pd.DataFrame(
        {
            "first": ["a", "b"] * 10,
            "second": ["x", "y"] * 10,
        }
    )
    y = pd.Series([1, 0] * 10)

    result = (
        skrub.as_data_op(df)
        .skb.apply(CatEncoder(), y=y)
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


def test_cat_encoder_sklearn_compat():
    enc = CatEncoder()
    with pytest.raises(NotFittedError):
        enc.transform(pd.Series(["a"], name="col"))
    with pytest.raises(NotFittedError):
        enc.get_feature_names_out()

    cloned = clone(enc)
    assert cloned.max_categories == enc.max_categories
