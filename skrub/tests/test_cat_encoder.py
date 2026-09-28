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

    expected_names = ["col_a", "col_d", "col_infrequent_sklearn", "col_target_sklearn"]
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


def test_cat_encoder_y_none(df_module):
    s = df_module.make_column("col", ["a", "b", "a"])
    enc = CatEncoder()
    with pytest.raises(ValueError, match="one-dimensional"):
        enc.fit_transform(s, y=None)


def test_cat_encoder_dataframe_target_and_unnamed_column(df_module):
    s = df_module.make_column(None, ["a", "b", "c"] * 5)
    y = df_module.make_dataframe({"target": ["0", "1", "2"] * 5})

    enc = CatEncoder(max_categories=2)
    res = enc.fit_transform(s, y)

    assert list(sbd.column_names(res)) == [
        "cat_enc_c",
        "cat_enc_infrequent_sklearn",
        "cat_enc_target_sklearn_0",
        "cat_enc_target_sklearn_1",
        "cat_enc_target_sklearn_2",
    ]
    assert sbd.shape(res) == (15, 5)
    assert enc.target_encoder_.target_type_ == "multiclass"


def test_cat_encoder_rejects_non_1d_target(df_module):
    s = df_module.make_column("col", ["a", "b"] * 5)
    y_df = df_module.make_dataframe({"first": [0, 1] * 5, "second": [1, 0] * 5})

    with pytest.raises(ValueError, match="exactly one column"):
        CatEncoder().fit_transform(s, y_df)

    with pytest.raises(ValueError, match="exactly one column"):
        CatEncoder().fit_transform(s, np.ones((10, 2)))

    with pytest.raises(ValueError, match="one-dimensional"):
        CatEncoder().fit_transform(s, np.asarray(1))


def test_cat_encoder_2d_string_target(df_module):
    s = df_module.make_column("col", ["a", "b", "c"] * 10)
    y = df_module.make_dataframe({"target": ["one", "two", "three"] * 10})

    enc = CatEncoder()
    res = enc.fit_transform(s, y)

    assert set(sbd.column_names(res)) == {
        "col_a",
        "col_b",
        "col_c",
        "col_target_sklearn_one",
        "col_target_sklearn_three",
        "col_target_sklearn_two",
    }
    transformed = enc.transform(df_module.make_column("col", ["a", "new"]))
    assert sbd.shape(transformed) == (2, 6)
    assert list(sbd.column_names(transformed)) == list(sbd.column_names(res))


def test_cat_encoder_preserves_dtypes(df_module):
    s = df_module.make_column("col", ["a", "b"] * 10)
    y = df_module.make_column("target", [1.0, 0.0] * 10)
    enc = CatEncoder()

    fitted = enc.fit_transform(s, y)
    transformed = enc.transform(s)

    for name in enc.all_outputs_:
        assert sbd.dtype(sbd.col(fitted, name)) == sbd.dtype(sbd.col(transformed, name))


def test_cat_encoder_column_collision_handled_by_pick_column_names(df_module):
    # A collision occurs if a category name produces a one-hot column that
    # matches the target-encoded column name: "col_target_sklearn".
    s = df_module.make_column("col", ["a", "target_sklearn"] * 10)
    y = df_module.make_column("target", [1, 0] * 10)

    res = CatEncoder().fit_transform(s, y)
    cols = list(sbd.column_names(res))

    assert len(cols) == len(set(cols))
    assert "col_a" in cols
    assert "col_target_sklearn" in cols
    assert any(col.startswith("col_target_sklearn__skrub_") for col in cols)


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
        "cat_target_sklearn",
        "other_x",
        "other_y",
        "other_target_sklearn",
        "num",
    ]


def test_cat_encoder_data_op_orders_outputs_by_input_column(df_module):
    df = df_module.make_dataframe(
        {
            "first": ["a", "b"] * 10,
            "second": ["x", "y"] * 10,
        }
    )
    y = df_module.make_column("target", [1, 0] * 10)

    result = skrub.as_data_op(df).skb.apply(CatEncoder(), y=y).skb.eval()

    assert list(sbd.column_names(result)) == [
        "first_a",
        "first_b",
        "first_target_sklearn",
        "second_x",
        "second_y",
        "second_target_sklearn",
    ]


def test_cat_encoder_sklearn_compat(df_module):
    enc = CatEncoder()
    with pytest.raises(NotFittedError):
        enc.transform(df_module.make_column("col", ["a"]))
    with pytest.raises(NotFittedError):
        enc.get_feature_names_out()

    cloned = clone(enc)
    assert cloned.max_categories == enc.max_categories


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_cat_encoder_requires_y(df_module, method):
    s = df_module.make_column("col", ["a", "b"] * 10)
    # SingleColumnTransformer's wrappers pass y=None when it is omitted.
    with pytest.raises(ValueError, match="CatEncoder expects y"):
        getattr(CatEncoder(), method)(s)


@pytest.mark.parametrize(
    "values, target_type",
    [
        ([0, 1], "binary"),
        ([False, True], "binary"),
        ([0, 1, 2], "multiclass"),
        ([0.25, 1.75], "continuous"),
    ],
)
@pytest.mark.parametrize("two_dimensional", [False, True])
def test_cat_encoder_numeric_object_target(
    df_module, values, target_type, two_dimensional
):
    y = np.asarray(values * 10)
    s = df_module.make_column("col", ["a", "b"] * (len(y) // 2))
    object_y = y.astype(object)
    if two_dimensional:
        object_y = object_y.reshape(-1, 1)

    enc = CatEncoder()
    assert enc.fit(s, object_y) is enc
    expected = CatEncoder().fit(s, y)
    assert enc.target_encoder_.target_type_ == target_type
    assert enc.get_feature_names_out() == expected.get_feature_names_out()
    np.testing.assert_allclose(
        sbd.to_numpy(enc.transform(s)), sbd.to_numpy(expected.transform(s))
    )


def test_cat_encoder_preserves_numeric_string_labels(df_module):
    s = df_module.make_column("col", ["a", "b", "c"] * 10)
    y = np.asarray(["01", "1", "2"] * 10, dtype=object)
    enc = CatEncoder().fit(s, y)
    assert enc.target_encoder_.target_type_ == "multiclass"
    np.testing.assert_array_equal(enc.target_encoder_.classes_, ["01", "1", "2"])
    assert enc.target_outputs_ == [
        "col_target_sklearn_01",
        "col_target_sklearn_1",
        "col_target_sklearn_2",
    ]


def test_cat_encoder_list_target(df_module):
    s = df_module.make_column("col", ["a", "b"] * 10)
    enc = CatEncoder().fit(s, [0, 1] * 10)
    new = df_module.make_column("col", ["a", "new"])
    np.testing.assert_allclose(
        sbd.to_numpy(enc.transform(new)), [[1, 0, 0], [0, 0, 0.5]]
    )
