from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer


class StringEncoder(SingleColumnTransformer):
    """Generate a lightweight string encoding of a given column using tf-idf \
        vectorization and truncated SVD.

    First, apply a tf-idf vectorization of the text, then reduce the dimensionality
    with a truncated SVD decomposition with the given number of parameters.

    New features will be named `{col_name}_{component}` if the series has a name,
    and `tsvd_{component}` if it does not.

    Parameters
    ----------
    n_components : int
        Number of components to be used for the PCA decomposition.

    See Also
    --------
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    GapEncoder :
        Encode string columns by constructing latent topics.
    SimilarityEncoder :
        Encode string columns as a numeric array with n-gram string similarity.
    TextEncoder :
        Encode string columns using pre-trained language models.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import StringEncoder

    We will encode the comments using 2 components:

    >>> enc = StringEncoder(n_components=2)
    >>> X = pd.Series([
    ...   "The professor snatched a good interview out of the jaws of these questions.",
    ...   "Bookmarking this to watch later.",
    ...   "When you don't know the lyrics of the song except the chorus",
    ... ], name='video comments')

    >>> enc.fit_transform(X) # doctest: +SKIP
       video comments_0  video comments_1
    0      8.218069e-01      4.557474e-17
    1      6.971618e-16      1.000000e+00
    2      8.218069e-01     -3.046564e-16
    """

    def __init__(
        self,
        n_components=30,
        vectorizer="tfidf",
        ngram_range=(1, 1),
        tf_idf_followup=False,
        n_features=None,
        max_features=None,
        analyzer="word",
    ):
        self.n_components = n_components
        self.vectorizer = vectorizer
        self.ngram_range = ngram_range
        self.tf_idf_followup = tf_idf_followup
        self.n_features = n_features
        self.max_features = max_features
        self.analyzer = analyzer

    def get_feature_names_out(self):
        """Get output feature names for transformation.

        Returns
        -------
        feature_names_out : list of str objects
            Transformed feature names.
        """
        return list(self.all_outputs_)

    def fit_transform(self, X, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
        X : Pandas or Polars series
            The column to transform.
        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        X_out: Pandas or Polars dataframe with shape (len(X), tsvd_n_components)
            The embedding representation of the input.
        """
        del y

        if self.vectorizer == "tfidf":
            self.pipe = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=self.ngram_range, analyzer=self.analyzer
                        ),
                    ),
                    ("tsvd", TruncatedSVD(n_components=self.n_components)),
                ]
            )

        elif self.vectorizer == "hashing":
            pipe_elements = [
                (
                    "hashing",
                    HashingVectorizer(
                        ngram_range=self.ngram_range, analyzer=self.analyzer
                    ),
                ),
            ]
            if self.tf_idf_followup:
                pipe_elements.append(("tfidf", TfidfTransformer()))
            pipe_elements.append(("tsvd", TruncatedSVD(n_components=self.n_components)))
            self.pipe = Pipeline(pipe_elements)
        else:
            raise ValueError(f"Unknown vectorizer {self.vectorizer}.")

        name = sbd.name(X)
        if not name:
            name = "tsvd"
        self.all_outputs_ = [f"{name}_{idx}" for idx in range(self.n_components)]

        result = self.pipe.fit_transform(sbd.to_numpy(X))

        self._is_fitted = True

        return self._transform(X, result)

    def transform(self, X):
        """Transform a column.

        Parameters
        ----------
        X : Pandas or Polars series
            The column to transform.

        Returns
        -------
        X_out: Pandas or Polars dataframe with shape (len(X), tsvd_n_components)
            The embedding representation of the input.
        """
        check_is_fitted(self)

        result = self.pipe.transform(sbd.to_numpy(X))
        return self._transform(X, result)

    def _transform(self, X, result):
        result = sbd.make_dataframe_like(X, dict(zip(self.all_outputs_, result.T)))
        result = sbd.copy_index(X, result)

        return result

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
