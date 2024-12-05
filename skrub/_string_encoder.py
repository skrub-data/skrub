from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer


class StringEncoder(SingleColumnTransformer):
    """Generate a lightweight string encoding of a given column. First, apply a
    tf-idf vectorization of the text, then reduce the dimensionality with a PCA
    decomposition with the given number of parameters.

    Parameters
    ----------
    pca_components : int
        Number of components to be used for the PCA decomposition.

    """

    def __init__(self, pca_components=30):
        self.pca_components = pca_components

    def _transform(self, X):
        result = self.pipe.transform(sbd.to_numpy(X))

        names = self.get_feature_names_out(X)
        result = sbd.make_dataframe_like(X, dict(zip(names, result.T)))
        result = sbd.copy_index(X, result)

        return result

    def get_feature_names_out(self, X):
        name = sbd.name(X)
        if not name:
            name = "pca"
        names = [f"{name}_{idx}" for idx in range(self.pca_components)]
        return names

    def fit_transform(self, X, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
        X : Pandas or Polars series.
            The column to transform.
        y : None. Ignored

        Returns
        -------
        A Pandas or Polars dataframe (depending on input) with shape
        (len(X), pca_components). New features will be named `{col_name}_{component}`
        if the series has a name, and `pca_{component}` if it does not.
        """
        del y
        self.pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("pca", PCA(n_components=self.pca_components)),
            ]
        )

        self.pipe.fit(sbd.to_numpy(X))

        self._is_fitted = True

        return self.transform(X)

    def transform(self, X):
        """Transform a column.

        Parameters
        ----------
        X : Pandas or Polars series.
            The column to transform.

        Returns
        -------
        A Pandas or Polars dataframe (depending on input) with shape
        (len(X), pca_components). New features will be named `{col_name}_{component}`
        if the series has a name, and `pca_{component}` if it does not.
        """
        check_is_fitted(self)
        return self._transform(X)

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
