from sklearn.base import BaseEstimator

from . import _dataframe as sb
from . import _datetime_utils

_SAMPLE_SIZE = 1000


class ToDatetime(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        if sb.is_numeric(column):
            raise NotImplementedError()

        if sb.is_anydate(column):
            self.datetime_format_ = None
            return column

        sample = sb.sample(column, n=min(_SAMPLE_SIZE, sb.shape(column)[0]))
        if not _datetime_utils.is_column_datetime_parsable(
            sb.asdfapi(sample).to_array()
        ):
            raise NotImplementedError()

        self.datetime_format_ = _datetime_utils.guess_datetime_format(
            sb.asdfapi(sb.native_cast(sample, str)).to_array(), random_state=0
        )
        return self.transform(column)

    def transform(self, column):
        return sb.to_datetime(column, format=self.datetime_format_)
