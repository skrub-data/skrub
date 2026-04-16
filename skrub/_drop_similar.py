try:
    import polars as pl
except ImportError:
    pass
import numbers

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import selectors as s
from ._column_associations import column_associations
from ._dataframe._common import raise_dispatch_unregistered_type
from ._dispatch import dispatch
from ._select_cols import DropCols


@dispatch
def _filter_associations(obj):
    raise_dispatch_unregistered_type(obj, kind="Series")


@_filter_associations.specialize("pandas")
def _filter_associations_pandas(obj, threshold):
    return obj[obj["cramer_v"] >= threshold]


@_filter_associations.specialize("polars")
def _filter_associations_polars(obj, threshold):
    return obj.filter(pl.col("cramer_v") >= threshold)


class DropSimilar(TransformerMixin, BaseEstimator):
    """Drop columns found too redundant to the rest of the dataframe,
    according to association defined by Cramér's V.

    This is done by computing Cramér's V between every possible two columns,
    and sorting these couples in descending order. Then, for every association above
    the given threshold, one of the two columns is dropped.

    Parameters
    ----------
    threshold : float, default=0.8
        The Cramér association score value above which to start dropping columns.

    Attributes
    ----------
    to_drop_ : list
        The names of columns evaluated for removal

    all_outputs_ : list
        The names of columns that the transformer keeps

    table_associations_ : dataframe
        A dataframe with columns `left_column_name', 'right_column_name' and 'cramer_v'
        listing association scores between every pair of columns.

    See Also
    --------
    DropUninformative :
        Drops columns for which various other criteria indicate that they contain
        little to no information (amount of nulls, of distinct values...)

    Cleaner :
        Runs several checks to sanitize a dataframe, including converting columns
        to standard formats or dropping certain columns.

    Examples
    --------
    >>> from skrub import DropSimilar
    >>> from skrub.datasets import fetch_employee_salaries
    >>> df = fetch_employee_salaries().X
    >>> list(df.columns)
    ['gender',
    'department',
    'department_name',
    'division',
    'assignment_category',
    'employee_position_title',
    'date_first_hired',
    'year_first_hired']
    >>> ds = DropSimilar(threshold=0.4)
    >>> clean_df = ds.fit_transform(df)

    `ds` has now removed a column for each pair with association above 0.6.
    These associations are stored in the `table_associations` attribute:

    >>> ds.table_associations
            left_column_name        right_column_name  cramer_v
    0                department          department_name  1.000000
    1                  division      assignment_category  0.601097
    2       assignment_category  employee_position_title  0.496814
    3                  division  employee_position_title  0.416034
    4           department_name  employee_position_title  0.413871
    5                department  employee_position_title  0.413871
    6           department_name      assignment_category  0.408823
    7                department      assignment_category  0.408823
    8                    gender               department  0.370436
    9                    gender          department_name  0.370436
    10               department                 division  0.362828
    11          department_name                 division  0.362828
    12  employee_position_title         date_first_hired  0.305385
    13                   gender  employee_position_title  0.263627
    14                   gender      assignment_category  0.255649
    15                   gender                 division  0.248813
    16               department         date_first_hired  0.150310
    17          department_name         date_first_hired  0.150310
    18         date_first_hired         year_first_hired  0.142581
    19  employee_position_title         year_first_hired  0.140087
    20                 division         date_first_hired  0.111298
    21                   gender         date_first_hired  0.099101
    22      assignment_category         date_first_hired  0.074086
    23          department_name         year_first_hired  0.069520
    24               department         year_first_hired  0.069520
    25                   gender         year_first_hired  0.063211
    26                 division         year_first_hired  0.060613
    27      assignment_category         year_first_hired  0.044855

    Six pairs are above the threshold, and they can be eliminated by
    dropping three columns. Therefore, these three have been marked
    as dropped by `ds`:

    >>> ds.to_drop_
    ['department_name', 'assignment_category', 'employee_position_title']

    This leaves us with the shortened employee salary database:

    >>> clean_df
        gender department                                           division date_first_hired  year_first_hired
    0         F        POL  MSB Information Mgmt and Tech Division Records...       09/22/1986              1986
    1         M        POL         ISB Major Crimes Division Fugitive Section       09/12/1988              1988
    2         F        HHS      Adult Protective and Case Management Services       11/19/1989              1989
    3         M        COR                         PRRS Facility and Security       05/05/2014              2014
    4         M        HCA                        Affordable Housing Programs       03/05/2007              2007
    ...     ...        ...                                                ...              ...               ...
    9223      F        HHS                        School Based Health Centers       11/03/2015              2015
    9224      F        FRS                           Human Resources Division       11/28/1988              1988
    9225      M        HHS  Child and Adolescent Mental Health Clinic Serv...       04/30/2001              2001
    9226      M        CCL                              Council Central Staff       09/05/2006              2006
    9227      M        DLC                Licensure, Regulation and Education       01/30/2012              2012
    """  # noqa: E501

    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.all_outputs_

    def fit(self, X, y=None):
        self.fit_transform(X, y=y)
        return self

    def fit_transform(self, X, y=None):
        # check that the threshold is correct
        if isinstance(self.threshold, bool) or not (
            isinstance(self.threshold, numbers.Real) and 0 <= self.threshold <= 1
        ):
            raise ValueError(
                f"Threshold must be a number between 0 and 1, got {self.threshold!r}."
            )

        if sbd.is_polars(X):
            try:
                import pyarrow  # noqa F401
            except ImportError:
                raise ImportError(
                    "DropSimilar requires the Pyarrow package to run on Polars"
                    " dataframes."
                )

        self.to_drop_ = []

        association_df = column_associations(X)
        self.table_associations_ = s.select(
            association_df, ["left_column_name", "right_column_name", "cramer_v"]
        )

        pairs_to_drop = _filter_associations(self.table_associations_, self.threshold)

        self.to_drop_.extend(
            sbd.to_list(sbd.unique(pairs_to_drop["right_column_name"]))
        )

        self._dropper = DropCols(self.to_drop_)
        new_X = self._dropper.fit_transform(X, y)

        self.all_outputs_ = self._dropper.kept_cols_

        return new_X

    def transform(self, X):
        check_is_fitted(self)
        return self._dropper.transform(X)
