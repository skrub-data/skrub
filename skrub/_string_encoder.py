import warnings

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline

from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer


class StringEncoder(SingleColumnTransformer):
    """Generate a lightweight string encoding of a given column using tf-idf \
        vectorization and truncated singular value decomposition (SVD).

    First, apply a tf-idf vectorization of the text, then reduce the dimensionality
    with a truncated SVD with the given number of parameters.

    New features will be named ``{col_name}_{component}`` if the series has a name,
    and ``tsvd_{component}`` if it does not.

    Parameters
    ----------
    n_components : int, default=30
        Number of components to be used for the singular value decomposition (SVD).
        Must be a positive integer.
    vectorizer : str, "tfidf" or "hashing", default="tfidf"
        Vectorizer to apply to the strings, either `tfidf` or `hashing` for
        scikit-learn TfidfVectorizer or HashingVectorizer respectively.

    ngram_range : tuple of (int, int) pairs, default=(3,4)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only unigrams,
        ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means only bigrams.

    analyzer : str, "char", "word" or "char_wb", default="char_wb"
        Whether the feature should be made of word or character n-grams.
        Option ``char_wb`` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.

    See Also
    --------
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    GapEncoder :
        Encode string columns by constructing latent topics.
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
        ngram_range=(3, 4),
        analyzer="char_wb",
    ):
        self.n_components = n_components
        self.vectorizer = vectorizer
        self.ngram_range = ngram_range
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
            self.vectorizer_ = TfidfVectorizer(
                ngram_range=self.ngram_range, analyzer=self.analyzer
            )
        elif self.vectorizer == "hashing":
            self.vectorizer_ = Pipeline(
                [
                    (
                        "hashing",
                        HashingVectorizer(
                            ngram_range=self.ngram_range, analyzer=self.analyzer
                        ),
                    ),
                    ("tfidf", TfidfTransformer()),
                ]
            )
        else:
            raise ValueError(
                f"Unknown vectorizer {self.vectorizer}. Options are 'tfidf' or"
                f" 'hashing', got {self.vectorizer!r}"
            )

        X_filled = sbd.fill_nulls(X, "")
        X_out = self.vectorizer_.fit_transform(X_filled).astype("float32")
        del X_filled  # optimizes memory: we no longer need X

        if (min_shape := min(X_out.shape)) > self.n_components:
            self.tsvd_ = TruncatedSVD(n_components=self.n_components)
            result = self.tsvd_.fit_transform(X_out)
        elif X_out.shape[1] == self.n_components:
            result = X_out.toarray()
        else:
            warnings.warn(
                f"The matrix shape is {(X_out.shape)}, and its minimum is "
                f"{min_shape}, which is too small to fit a truncated SVD with "
                f"n_components={self.n_components}. "
                "The embeddings will be truncated by keeping the first "
                f"{self.n_components} dimensions instead. "
            )
            # self.n_components can be greater than the number
            # of dimensions of result.
            # Therefore, self.n_components_ below stores the resulting
            # number of dimensions of result.
            result = X_out[:, : self.n_components].toarray()
            result = result.copy()  # To avoid a reference to X_out
        del X_out  # optimize memory: we no longer need X_out

        self._is_fitted = True
        self.n_components_ = result.shape[1]

        name = sbd.name(X)
        if not name:
            name = "tsvd"
        self.all_outputs_ = [f"{name}_{idx}" for idx in range(self.n_components_)]

        return self._post_process(X, result)

    def transform(self, X):
        """Transform a column.

        Parameters
        ----------
        X : Pandas or Polars series
            The column to transform.

        Returns
        -------
        result: Pandas or Polars dataframe with shape (len(X), tsvd_n_components)
            The embedding representation of the input.
        """

        X_filled = sbd.fill_nulls(X, "")
        X_out = self.vectorizer_.transform(X_filled).astype("float32")
        del X_filled  # optimizes memory: we no longer need X
        if hasattr(self, "tsvd_"):
            result = self.tsvd_.transform(X_out)
        else:
            result = X_out[:, : self.n_components].toarray()
            result = result.copy()
        del X_out  # optimize memory: we no longer need X_out

        return self._post_process(X, result)

    def _post_process(self, X, result):
        result = sbd.make_dataframe_like(X, dict(zip(self.all_outputs_, result.T)))
        result = sbd.copy_index(X, result)

        return result
