from sklearn.base import BaseEstimator

from . import _datetime_utils
from ._dataframe import asdfapi, is_datetime, is_numeric, native_cast, to_datetime


class ToDatetimeCol(BaseEstimator):
    def fit_transform(self, column):
        if is_numeric(column):
            raise NotImplementedError()

        if is_datetime(column):
            self.datetime_format_ = None
            return column

        # TODO downsample
        if not _datetime_utils.is_column_datetime_parsable(asdfapi(column).to_array()):
            raise NotImplementedError()

        self.datetime_format_ = _datetime_utils.guess_datetime_format(
            asdfapi(native_cast(column, str)).to_array(), random_state=0
        )
        return self.transform(column)

    def transform(self, column):
        return to_datetime(column, format=self.datetime_format_)
