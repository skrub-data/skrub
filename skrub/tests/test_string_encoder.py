import pytest
from numpy.testing import assert_almost_equal
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline

from skrub import StringEncoder, TableVectorizer
from skrub import _dataframe as sbd


@pytest.fixture
def encode_column(df_module):
    corpus = [
        "this is the first document",
        "this document is the second document",
        "and this is the third one",
        "is this the first document",
        None,
    ]

    return df_module.make_column("col1", corpus)


def test_tfidf_vectorizer(encode_column, df_module):
    ngram_range = (3, 4)
    analyzer = "char_wb"
    n_components = 2

    #### tfidf vectorizer
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)),
            ("tsvd", TruncatedSVD(n_components=n_components)),
        ]
    )
    check = pipe.fit_transform(sbd.to_numpy(sbd.fill_nulls(encode_column, "")))
    check = check.astype("float32")  # StringEncoder is float32

    names = [f"col1_{idx}" for idx in range(2)]

    check_df = df_module.make_dataframe(dict(zip(names, check.T)))

    se = StringEncoder(
        n_components=n_components,
        vectorizer="tfidf",
        ngram_range=ngram_range,
        analyzer=analyzer,
    )
    result = se.fit_transform(encode_column)

    # Converting dtypes to avoid nullable shenanigans
    check_df = sbd.pandas_convert_dtypes(check_df)
    result = sbd.pandas_convert_dtypes(result)

    df_module.assert_frame_equal(check_df, result)

    # Making coverage happy
    result_transform = se.transform(encode_column)
    result_transform = sbd.pandas_convert_dtypes(result_transform)

    for idx in range(len(check_df.columns)):
        col1 = sbd.col_by_idx(check_df, idx)
        col2 = sbd.col_by_idx(check_df, idx)

        for c1, c2 in zip(col1, col2):
            assert_almost_equal(c1, c2, decimal=6)


def test_hashing_vectorizer(encode_column, df_module):
    # Testing is less strict because HashingVectorizer is not deterministic.
    ngram_range = (3, 4)
    analyzer = "char_wb"
    n_components = 2

    #### hashing vectorizer
    pipe = Pipeline(
        [
            ("hashing", HashingVectorizer(ngram_range=ngram_range, analyzer=analyzer)),
            ("tfidf", TfidfTransformer()),
            ("tsvd", TruncatedSVD(n_components=n_components)),
        ]
    )
    check = pipe.fit_transform(sbd.to_numpy(sbd.fill_nulls(encode_column, "")))
    names = [f"col1_{idx}" for idx in range(2)]

    check_df = df_module.make_dataframe(dict(zip(names, check.T)))

    se = StringEncoder(
        n_components=n_components,
        vectorizer="hashing",
        ngram_range=ngram_range,
        analyzer=analyzer,
    )
    result = se.fit_transform(encode_column)

    # Converting dtypes to avoid nullable shenanigans
    check_df = sbd.pandas_convert_dtypes(check_df)
    result = sbd.pandas_convert_dtypes(result)

    assert check_df.shape == result.shape
    assert isinstance(check_df, type(result))

    assert all(hasattr(se, x) for x in ["tsvd_", "vectorizer"])

    for name, estimator in se.vectorizer_.named_steps.items():
        assert name in pipe.named_steps
        assert isinstance(estimator, type(pipe.named_steps[name]))


def test_error_checking(encode_column):
    n_components = -1
    vectorizer = "notavectorizer"
    ngram_range = "a"
    analyzer = "notanalyzer"

    se = StringEncoder(
        n_components=n_components,
    )
    with pytest.raises(ValueError):
        se.fit_transform(encode_column)

    se = StringEncoder(
        vectorizer=vectorizer,
    )
    with pytest.raises(ValueError):
        se.fit_transform(encode_column)

    se = StringEncoder(
        analyzer=analyzer,
    )
    with pytest.raises(ValueError):
        se.fit_transform(encode_column)

    se = StringEncoder(
        ngram_range=ngram_range,
    )
    with pytest.raises(ValueError):
        se.fit_transform(encode_column)


def test_get_feature_names_out(encode_column, df_module):
    """Test that ``get_feature_names_out`` returns the correct feature names."""
    encoder = StringEncoder(n_components=4)

    encoder.fit(encode_column)
    expected_columns = ["col1_0", "col1_1", "col1_2", "col1_3"]
    assert encoder.get_feature_names_out() == expected_columns

    # Checking that a series with an empty name generates the proper column names
    X = df_module.make_column(
        None,
        [
            "this is the first document",
            "this document is the second document",
            "and this is the third one",
            "is this the first document",
        ],
    )

    encoder = StringEncoder(n_components=4)

    encoder.fit(X)
    expected_columns = ["tsvd_0", "tsvd_1", "tsvd_2", "tsvd_3"]
    assert encoder.get_feature_names_out() == expected_columns


def test_n_components(df_module):
    ngram_range = (3, 4)
    analyzer = "char_wb"
    n_components = 2

    encoder = StringEncoder(
        n_components=n_components,
        vectorizer="tfidf",
        ngram_range=ngram_range,
        analyzer=analyzer,
    )

    X = df_module.make_column("", ["hello sir", "hola que tal"])

    encoder_2 = clone(encoder).set_params(n_components=2).fit(X)
    for meth in ("fit_transform", "transform"):
        X_out = getattr(encoder_2, meth)(X)
        assert sbd.shape(X_out)[1] == 2
        assert encoder_2.n_components_ == 2

    encoder_30 = clone(encoder).set_params(n_components=30)
    with pytest.warns(UserWarning, match="The embeddings will be truncated"):
        for meth in ("fit_transform", "transform"):
            X_out = getattr(encoder_30, meth)(X)
    assert not hasattr(encoder_30, "tsvd_")
    assert sbd.shape(X_out)[1] == 30
    assert encoder_30.n_components_ == 30


def test_n_components_equal_voc_size(df_module):
    x = df_module.make_column("x", ["aab", "bba"])
    encoder = StringEncoder(n_components=2, ngram_range=(1, 1), analyzer="char")
    out = encoder.fit_transform(x)
    assert sbd.column_names(out) == ["x_0", "x_1"]
    assert not hasattr(encoder, "tsvd_")


@pytest.mark.parametrize("vectorizer", ["tfidf", "hashing"])
def test_missing_values(df_module, vectorizer):
    col = df_module.make_column("col", ["one two", None, "", "two three"])
    encoder = StringEncoder(n_components=2, vectorizer=vectorizer)
    out = encoder.fit_transform(col)
    for c in sbd.to_column_list(out):
        assert_almost_equal(c[1], 0.0, decimal=6)
        assert_almost_equal(c[2], 0.0, decimal=6)
    out = encoder.transform(col)
    for c in sbd.to_column_list(out):
        assert_almost_equal(c[1], 0.0, decimal=6)
        assert_almost_equal(c[2], 0.0, decimal=6)
    tv = TableVectorizer(
        low_cardinality=StringEncoder(n_components=2, vectorizer=vectorizer)
    )
    df = df_module.make_dataframe({"col": col})
    out = tv.fit_transform(df)
    for c in sbd.to_column_list(out):
        assert_almost_equal(c[1], 0.0, decimal=6)
        assert_almost_equal(c[2], 0.0, decimal=6)
    out = tv.transform(df)
    for c in sbd.to_column_list(out):
        assert_almost_equal(c[1], 0.0, decimal=6)
        assert_almost_equal(c[2], 0.0, decimal=6)
