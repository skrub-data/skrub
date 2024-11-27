from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer


class StringEncoder(SingleColumnTransformer):
    """_summary_

    Parameters
    ----------

    """

    def __init__(self, pca_components=30):
        self.pca_components = pca_components

    def _transform(self, X):
        # TODO: vocabulary?

        return self.pipe.transform(X)

    def get_feature_names_out(self, X):
        name = sbd.name(X)
        if not name:
            name = "pca"
        names = [f"{name}_{idx}" for idx in range(self.pca_components)]
        return names

    def fit_transform(self, X, y=None):
        del y

        return self.transform(X)

    def transform(self, X):
        # check_is_fitted(self)

        result = self._transform(sbd.to_numpy(X))
        names = self.get_feature_names_out(X)
        result = sbd.make_dataframe_like(X, dict(zip(names, result.T)))
        result = sbd.copy_index(X, result)
        return result
