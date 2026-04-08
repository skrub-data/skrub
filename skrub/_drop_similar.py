import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from . import DropCols, _column_associations, _config
from . import _dataframe as sbd


class DropSimilar(TransformerMixin):
    """Drop columns found too redundant to the rest of the dataframe,
    according to association defined by Cramér's V.

    This is done by computing Cramér's V between every possible two columns,
    and sorting these couples in descending order. Then, for every association above
    a preestablished threshold:
    - If one of the two columns has already been dropped, nothing happens
    - Otherwise, the column that has the highest average association score with
    every other column is dropped.

    Parameters
    ----------
    threshold : float, default=0.8
        If True, drop the column if it contains only one unique value. Missing values
        count as one additional distinct value.

    Attributes
    ----------
    to_drop : list
        The names of columns evaluated for removal

    See Also
    --------
    DropUninformative :
        Drops columns for which various other criteria indicate that they contain
        little to no information (amount of nulls, of distinct values...)

    Cleaner :
        Runs several checks to sanitize a dataframe, including converting columns
        to standard formats or dropping certain columns.
    """

    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.to_drop = []
        self._dropper = DropCols([])

    def list_similar(self, associations, col_names):
        averages = []
        asso_list = []

        for col in range(len(col_names)):
            averages.append(np.average(associations[col]))
            asso_list += [((col, i), associations[col][i]) for i in range(col)]
        asso_list.sort(key=lambda P: P[1], reverse=True)

        i = 0
        remove = []

        while i < len(asso_list) and asso_list[i][1] > self.threshold:
            (X, Y), _ = asso_list[i]
            if X not in remove and Y not in remove:
                if averages[X] > averages[Y]:
                    remove.append(X)
                else:
                    remove.append(Y)
            i += 1

        return [col_names[x] for x in remove]

    def associations_as_report(self, X):
        """Generate column associations between columns of dataframe X
        using Cramér's V.

        """
        _SUBSAMPLE_SIZE = 3000

        df = sbd.sample(
            X,
            n=min(sbd.shape(X)[0], _SUBSAMPLE_SIZE),
            seed=_config.get_config()["subsampling_seed"],
        )

        return _column_associations.column_associations(df)

    def fit_transform(self, X, y=None):
        association_df = _column_associations.column_associations(X)
        total_cols = len(pd.unique(association_df["left_column_idx"])) + 1

        col_names = []
        associations = np.zeros([total_cols, total_cols])

        for i in range(len(association_df)):
            col1, col2 = (
                association_df["left_column_name"][i],
                association_df["right_column_name"][i],
            )

            if col1 in col_names:
                i1 = col_names.index(col1)
            else:
                col_names.append(col1)
                i1 = len(col_names) - 1

            if col2 in col_names:
                i2 = col_names.index(col2)
            else:
                col_names.append(col2)
                i2 = len(col_names) - 1

            cramer = association_df["cramer_v"][i]

            associations[i1, i2] = cramer

        self.to_drop = self.list_similar(associations, col_names)
        self._dropper = DropCols(self.to_drop)
        return self._dropper.fit_transform(X, y)

    def transform(self, X):
        return self._dropper.transform(X)
