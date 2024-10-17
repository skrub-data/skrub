# drop columns that contain all null values 
from ._on_each_column import SingleColumnTransformer

from . import _dataframe as sbd
from ._dispatch import dispatch
from sklearn.utils.validation import check_is_fitted


@dispatch
def _is_all_null(col):
    raise NotImplementedError()

@_is_all_null.specialize("pandas", argument_type="Column")
def _is_all_null_pandas(col):
    return bool(col.isna().all())


@_is_all_null.specialize("polars", argument_type="Column")
def _is_all_null_polars(col):
    return (col.is_null().all())

__all__ = ["DropNullColumn"]

class DropNullColumn(SingleColumnTransformer):
        
    def __init__(self):
        super().__init__()
        self._is_fitted = False
        
    def fit_transform(self, column, y=None):
        del y

        self._is_fitted = True
        return self.transform(column)
    
    def transform(self, column):
        check_is_fitted(self, )
        
        if _is_all_null(column):
            return []
        else:
            return column
        
        
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
