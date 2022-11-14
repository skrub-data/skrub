import pandas as pd

from dirty_cat._fuzzy_join import fuzzy_join


class FeatureAugmenter:
    """Transformer that helps augment the number of features in a table.

    Given a dictionnary of tables and key column names,
    fuzzy join them to the main table.

    Parameters
    ----------
    tables : dict
        A dictionnary containing the key column
        names and auxilliary tables that will be joined.
    main_key : str
        The key column name in the main table on which
        the join will be performed.

    Notes
    -----
    The `tables` parameter is a dictionnary. This implies
    that key column names must be different.

    Examples
    --------
    >>> main_table = pd.DataFrame(['France', 'Germany', 'Italy'], columns=['Country'])
    >>> main_table
    Country
    0   France
    1  Germany
    2    Italy

    >>> aux_table_1 = pd.DataFrame([['Germany', 84_000_000], ['France', 68_000_000], ['Italy', 59_000_000]], columns=['Country', 'Population']) # noqa
    >>> aux_table_1
       Country  Population
    0  Germany    84000000
    1   France    68000000
    2    Italy    59000000

    >>> aux_table_2 = pd.DataFrame([['France', 2937], ['Italy', 2099], ['Germany', 4223]], columns=['Country name', 'GDP (billion)']) # noqa
    >>> aux_table_2
        Country name  GDP (billion)
    0       France           2937
    1        Italy           2099
    2      Germany           4223

    >>> aux_table_3 = pd.DataFrame([['France', 'Paris'], ['Italy', 'Rome'], ['Germany', 'Berlin']], columns=['Countries', 'Capital']) # noqa
    >>> aux_table_3
      Countries Capital
    0    France   Paris
    1     Italy    Rome
    2   Germany  Berlin

    >>> aux_dict = {"Country": aux_table_1, "Country name": aux_table_2, "Countries": aux_table_3} # noqa

    >>> fa = FeatureAugmenter(tables=aux_dict, main_key='Country')

    >>> augmented_table = fa.fit_transform(main_table)
    >>> augmented_table
        Country Country_aux  Population Country name  GDP (billion) Countries Capital
    0   France      France    68000000       France           2937    France   Paris
    1  Germany     Germany    84000000      Germany           4223   Germany  Berlin
    2    Italy       Italy    59000000        Italy           2099     Italy    Rome
    """

    def __init__(
        self,
        tables: dict,
        main_key: str,
    ):
        self.tables = tables
        self.main_key = main_key

    def fit(self, main_table) -> "FeatureAugmenter":
        """Fit the Feature Augmenter to the main table.

        In practice, just checks if the key columns in X,
        the main table, and in the auxilliary tables exist.

        Parameters
        ----------
        main_table : DataFrame, shape [n_samples, n_features]
            The main table, to be joined to the
            auxilliary ones.

        Returns
        -------
        FeatureAugmenter
            Fitted FeatureAugmenter instance.
        """

        if self.main_key not in main_table.columns:
            raise ValueError(
                f"Got main_key={self.main_key!r}, but column missing in the main table."
            )

        for key in self.tables:
            if key not in self.tables[key].columns:
                raise ValueError(
                    f"Got column key {key!r}, "
                    "but column missing in the auxilliary table."
                )
        return self

    def transform(self, main_table) -> pd.DataFrame:
        """Transform X using the specified encoding scheme.

        Parameters
        ----------
        main_table : DataFrame, shape [n_samples, n_features]
            The main table, to be joined to the
            auxilliary ones.

        Returns
        -------
        DataFrame
            The final joined table.
        """

        for key in self.tables:
            # TODO: Add an option to fuzzy_join on multiple columns at once
            # (will be if len(inter_col)!=0)

            aux_table = self.tables[key]
            main_table = fuzzy_join(
                main_table,
                aux_table,
                left_on=self.main_key,
                right_on=key,
                suffixes=("", "_aux"),
            )

        return main_table

    def fit_transform(self, main_table) -> pd.DataFrame:
        """
        Fit the FeatureAugmenter to X, then transforms it.

        Parameters
        ----------
        main_table : DataFrame, shape [n_samples, n_features]
            The main table, to be joined to the
            auxilliary ones.

        Returns
        -------
        DataFrame
            The final joined table.
        """

        return self.fit(main_table).transform(main_table)
