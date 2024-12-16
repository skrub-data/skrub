import pytest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline

from skrub import _dataframe as sbd
from skrub._string_encoder import StringEncoder


@pytest.fixture
def encode_column(df_module):
    corpus = [
        "this is the first document",
        "this document is the second document",
        "and this is the third one",
        "is this the first document",
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
    check = pipe.fit_transform(sbd.to_numpy(encode_column))

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

    df_module.assert_frame_equal(result, result_transform)


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
    check = pipe.fit_transform(sbd.to_numpy(encode_column))

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
    assert type(check_df) == type(result)

    assert len(se.pipe.named_steps) == len(pipe.named_steps)

    for name, estimator in se.pipe.named_steps.items():
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
