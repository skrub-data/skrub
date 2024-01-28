from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from . import _datetime_utils

_SAMPLE_SIZE = 1000


class ToDatetime(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        if sbd.is_anydate(column):
            # TODO convert to UTC?
            self.datetime_format_ = None
            return column
        if not (sbd.is_string(column) or sbd.is_object(column)):
            return NotImplemented
        sample = sbd.sample(column, n=min(_SAMPLE_SIZE, sbd.shape(column)[0]))
        if not _datetime_utils.is_column_datetime_parsable(sbd.to_array(sample)):
            return NotImplemented

        self.datetime_format_ = _datetime_utils.guess_datetime_format(
            sbd.to_array(sbd.cast(sample, str)), random_state=0
        )
        return self.transform(column)

    def transform(self, column):
        if self.datetime_format_ is None:
            # TODO check dtype, unit, timezone
            return column
        return sbd.to_datetime(column, format=self.datetime_format_)
