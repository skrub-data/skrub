import pytest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
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


def test_encoding(encode_column, df_module):
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("tsvd", TruncatedSVD(n_components=2)),
        ]
    )
    check = pipe.fit_transform(sbd.to_numpy(encode_column))

    names = [f"col1_{idx}" for idx in range(2)]

    check_df = df_module.make_dataframe(dict(zip(names, check.T)))

    se = StringEncoder(2)
    result = se.fit_transform(encode_column)

    # Converting dtypes to avoid nullable shenanigans
    check_df = sbd.pandas_convert_dtypes(check_df)
    result = sbd.pandas_convert_dtypes(result)

    df_module.assert_frame_equal(check_df, result)


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
