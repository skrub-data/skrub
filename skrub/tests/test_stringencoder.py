import pytest
from sklearn.decomposition import PCA
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

    return df_module.make_column("test_column", corpus)


def test_encoding(encode_column, df_module):
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("pca", PCA(n_components=2)),
        ]
    )
    check = pipe.fit_transform(sbd.to_numpy(encode_column))

    names = [f"test_column_{idx}" for idx in range(2)]

    check_df = df_module.make_dataframe(dict(zip(names, check.T)))

    se = StringEncoder(2)
    result = se.fit_transform(encode_column)

    df_module.assert_frame_equal(check_df, result)
