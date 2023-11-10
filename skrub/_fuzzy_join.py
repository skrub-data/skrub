"""
Implements fuzzy_join, a function to perform fuzzy joining between two tables.
"""


import pandas as pd

from skrub import _join_utils
from skrub._joiner import DEFAULT_MATCHING, DEFAULT_STRING_ENCODER, Joiner


def fuzzy_join(
    left,
    right,
    left_on=None,
    right_on=None,
    on=None,
    suffix="",
    insert_match_info=False,
    drop_unmatched=False,
    string_encoder=DEFAULT_STRING_ENCODER,
    matching=DEFAULT_MATCHING,
) -> pd.DataFrame:
    """Join two tables based on approximate matching using the appropriate similarity \
    metric.

    The principle is as follows:

    1. We embed and transform the key string, numerical or datetime columns.
    2. For each category, we use the nearest neighbor method to find its
       closest neighbor and establish a match.
    3. We match the tables using the previous information.

    For string columns, categories from the two tables that share many sub-strings
    (n-grams) have greater probability of being matched together. The join is based on
    morphological similarities between strings.

    Simultaneous joins on multiple columns (e.g. longitude, latitude) is supported.

    Joining on numerical columns is also possible based on
    the Euclidean distance.

    Joining on datetime columns is based on the time difference.

    Parameters
    ----------
    left : :obj:`~pandas.DataFrame`
        A table to merge.
    right : :obj:`~pandas.DataFrame`
        A table used to merge with.
    left_on : str or list of str, optional
        Name of left table column(s) to join.
    right_on : str or list of str, optional
        Name of right table key column(s) to join
        with left table key column(s).
    on : str or list of str or int, optional
        Name of common left and right table join key columns.
        Must be found in both DataFrames. Use only if `left_on`
        and `right_on` parameters are not specified.
    string_encoder : vectorizer instance, optional
        Encoder parameter for the Vectorizer.
        By default, uses a HashingVectorizer.
        It is possible to pass a vectorizer instance inheriting
        _VectorizerMixin to tweak the parameters of the encoder.
    drop_unmatched : bool, default=False
        Remove categories for which a match was not found in the two tables.
    suffix: str, default=""

    Returns
    -------
    df_joined : :obj:`~pandas.DataFrame`
        The joined table.

    See Also
    --------
    Joiner
        fuzzy_join implemented as a scikit-learn transformer.

    Notes
    -----
    For regular joins, the output of fuzzy_join is identical
    to pandas.merge, except that both key columns are returned.

    Joining on indexes and multiple columns is not supported.

    When `return_score=True`, the returned :obj:`~pandas.DataFrame` gives
    the distances between the closest matches in a [0, 1] interval.
    0 corresponds to no matching n-grams, while 1 is a
    perfect match.

    When we use `match_score=0`, the function will be forced to impute the
    nearest match (of the left table category) across all possible matching
    options in the right table column.

    When the neighbors are distant, we may use the `match_score` parameter
    with a value bigger than 0 to define the minimal level of matching
    score tolerated. If it is not reached, matches will be
    considered as not found and NaN values will be imputed.

    Examples
    --------
    >>> df1 = pd.DataFrame({'a': ['ana', 'lala', 'nana'], 'b': [1, 2, 3]})
    >>> df2 = pd.DataFrame({'a': ['anna', 'lala', 'ana', 'nnana'], 'c': [5, 6, 7, 8]})

    >>> df1
          a  b
    0   ana  1
    1  lala  2
    2  nana  3

    >>> df2
           a  c
    0   anna  5
    1   lala  6
    2    ana  7
    3  nnana  8

    To do a simple join based on the nearest match:

    >>> fuzzy_join(df1, df2, on='a')
        a_x  b    a_y  c
    0   ana  1    ana  7
    1  lala  2   lala  6
    2  nana  3  nnana  8

    When we want to accept only a certain match precision,
    we can use the `match_score` argument:

    >>> fuzzy_join(df1, df2, on='a', match_score=1, return_score=True)
        a_x  b   a_y     c  matching_score
    0   ana  1   ana     7             1.0
    1  lala  2  lala     6             1.0
    2  nana  3  <NA>  <NA>             0.0

    As expected, the category "nana" has no exact match (`match_score=1`).
    """
    # duplicate the key checks performed by the Joiner so we can get better
    # names in error messages
    left_key, right_key = _join_utils.check_key(
        left_on,
        right_on,
        on,
        key_names={"main_key": "left_on", "aux_key": "right_on", "key": "on"},
    )
    _join_utils.check_missing_columns(left, left_key, "'left' (the left table)")
    _join_utils.check_missing_columns(right, right_key, "'right' (the right table)")

    join = Joiner(
        aux_table=right,
        main_key=left_on,
        aux_key=right_on,
        key=on,
        suffix=suffix,
        matching=matching,
        string_encoder=string_encoder,
        insert_match_info=True,
    ).fit_transform(left)
    if drop_unmatched:
        join = join[join["skrub.Joiner.matching.match_accepted"]]
    if not insert_match_info:
        join = join.drop(Joiner.match_info_columns, axis=1)
    return join
