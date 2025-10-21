import warnings

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._apply_to_cols import SingleColumnTransformer
from ._scaling_factor import scaling_factor
from ._to_str import ToStr


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

    stop_words : {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used. There are several
        known issues with 'english' and you should consider an alternative (see `Using
        stop words <https://scikit-learn.org/stable/modules/feature_extraction.html#using-stop-words>`_).

        If a list, that list is assumed to contain stop words, all of which will be
        removed from the resulting tokens. Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used.

    random_state : int, RandomState instance or None, default=None
        Used during randomized svd. Pass an int for reproducible results across
        multiple function calls.

    Attributes
    ----------
    input_name_ : str
        Name of the fitted column, or "string_enc" if the column has no name.

    n_components_ : int
        The number of dimensions of the embeddings after dimensionality
        reduction.

    all_outputs_ : list of str
        A list that contains the name of all the features generated from the fitted
        column.

    See Also
    --------
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    GapEncoder :
        Encode string columns by constructing latent topics.
    TextEncoder :
        Encode string columns using pre-trained language models.

    Notes
    -----
    Skrub provides ``StringEncoder`` as a simple interface to perform `Latent Semantic
    Analysis (LSA) <https://scikit-learn.org/stable/modules/decomposition.html#about-truncated-svd-and-latent-semantic-analysis-(lsa)>`_.
    As such, it doesn't support all hyper-parameters exposed by the underlying
    {:class:`~sklearn.feature_extraction.text.TfidfVectorizer`,
    :class:`~sklearn.feature_extraction.text.HashingVectorizer`} and
    :class:`~sklearn.decomposition.TruncatedSVD`. If you need more flexibility than the
    proposed hyper-parameters of ``StringEncoder``, you must create your own LSA using
    scikit-learn :class:`~sklearn.pipeline.Pipeline`, such as:

    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> from sklearn.decomposition import TruncatedSVD

    >>> make_pipeline(TfidfVectorizer(max_df=300), TruncatedSVD())
    Pipeline(steps=[('tfidfvectorizer', TfidfVectorizer(max_df=300)),
                ('truncatedsvd', TruncatedSVD())])

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
    0          1.322973         -0.163070
    1          0.379688          1.659319
    2          1.306400         -0.317120
    """

    def __init__(
        self,
        n_components=30,
        vectorizer="tfidf",
        ngram_range=(3, 4),
        analyzer="char_wb",
        stop_words=None,
        random_state=None,
    ):
        self.n_components = n_components
        self.vectorizer = vectorizer
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.stop_words = stop_words
        self.random_state = random_state

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

        self.to_str = ToStr(convert_category=True)
        X_filled = self.to_str.fit_transform(X)
        X_filled = sbd.fill_nulls(X_filled, "")

        if self.vectorizer == "tfidf":
            self.vectorizer_ = TfidfVectorizer(
                ngram_range=self.ngram_range,
                analyzer=self.analyzer,
                stop_words=self.stop_words,
            )
        elif self.vectorizer == "hashing":
            self.vectorizer_ = Pipeline(
                [
                    (
                        "hashing",
                        HashingVectorizer(
                            ngram_range=self.ngram_range,
                            analyzer=self.analyzer,
                            stop_words=self.stop_words,
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

        X_out = self.vectorizer_.fit_transform(X_filled).astype("float32")
        del X_filled  # optimizes memory: we no longer need X

        if (min_shape := min(X_out.shape)) > self.n_components:
            self.tsvd_ = TruncatedSVD(
                n_components=self.n_components, random_state=self.random_state
            )
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

        # block normalize
        self.scaling_factor_ = scaling_factor(result)
        result /= self.scaling_factor_

        self.n_components_ = result.shape[1]

        self.input_name_ = sbd.name(X) or "string_enc"

        self.all_outputs_ = self.get_feature_names_out()

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
        # Error checking at fit time is done by the ToStr transformer,
        # but after ToStr is fitted it does not check the input type anymore,
        # while we want to ensure that the input column is a string or categorical
        # so we need to add the check here.
        if not (sbd.is_string(X) or sbd.is_categorical(X)):
            raise ValueError("Input column does not contain strings.")

        X_filled = self.to_str.transform(X)
        X_filled = sbd.fill_nulls(X_filled, "")
        X_out = self.vectorizer_.transform(X_filled).astype("float32")
        del X_filled  # optimizes memory: we no longer need X
        if hasattr(self, "tsvd_"):
            result = self.tsvd_.transform(X_out)
        else:
            result = X_out[:, : self.n_components].toarray()
            result = result.copy()
        del X_out  # optimize memory: we no longer need X_out

        # block normalize
        result /= self.scaling_factor_

        return self._post_process(X, result)

    def _post_process(self, X, result):
        result = sbd.make_dataframe_like(X, dict(zip(self.all_outputs_, result.T)))
        result = sbd.copy_index(X, result)

        return result

    def get_feature_names_out(self, input_features=None):
        """Return a list of features generated by the transformer.

        Each feature has format ``{input_name}_{n_component}`` where ``input_name``
        is the name of the input column, or a default name for the encoder, and
        ``n_component`` is the idx of the specific feature.

        Parameters
        ----------
        input_features : None
            The input features. Ignored, only here for compatibility.

        Returns
        -------
        list of str
            The list of feature names.
        """
        check_is_fitted(self, "n_components_")
        num_digits = len(str(self.n_components_ - 1))
        return [
            f"{self.input_name_}_{str(i).zfill(num_digits)}"
            for i in range(self.n_components_)
        ]
