from sklearn.base import BaseEstimator, TransformerMixin, clone

from ._dataframe._namespace import get_df_namespace


class MapCols(TransformerMixin, BaseEstimator):
    def __init__(self, column_transformer):
        self.column_transformer = column_transformer

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        del y
        self.transformers_ = {}
        transformed_columns = {}
        for col_name in X.columns:
            transformer = clone(self.column_transformer)
            try:
                transformed_columns[col_name] = transformer.fit_transform(X[col_name])
                self.transformers_[col_name] = transformer
            except NotImplementedError:
                transformed_columns[col_name] = X[col_name]

        skrub_px, _ = get_df_namespace(X)
        return skrub_px.make_dataframe(
            transformed_columns, index=getattr(X, "index", None)
        )

    def transform(self, X, y=None):
        del y
        transformed_columns = {}
        for col_name in X.columns:
            if col_name in self.transformers_:
                transformed_columns[col_name] = self.transformers_[col_name].transform(
                    X[col_name]
                )
            else:
                transformed_columns[col_name] = X[col_name]
        skrub_px, _ = get_df_namespace(X)
        return skrub_px.make_dataframe(
            transformed_columns, index=getattr(X, "index", None)
        )
