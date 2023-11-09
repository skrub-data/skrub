"""
Implements fuzzy_join, a function to perform fuzzy joining between two tables.
"""


import pandas as pd

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
    how : {'left', 'right'}, default='left'
        Type of merge to be performed. Note that unlike pandas.merge,
        only "left" and "right" are supported so far, as the fuzzy-join comes
        with its own mechanism to resolve lack of correspondence between
        left and right tables.
    left_on : str or list of str, optional
        Name of left table column(s) to join.
    right_on : str or list of str, optional
        Name of right table key column(s) to join
        with left table key column(s).
    on : str or list of str or int, optional
        Name of common left and right table join key columns.
        Must be found in both DataFrames. Use only if `left_on`
        and `right_on` parameters are not specified.
    encoder : vectorizer instance, optional
        Encoder parameter for the Vectorizer.
        By default, uses a HashingVectorizer.
        It is possible to pass a vectorizer instance inheriting
        _VectorizerMixin to tweak the parameters of the encoder.
    analyzer : {'word', 'char', 'char_wb'}, default='char_wb'
        Analyzer parameter for the HashingVectorizer
        passed to the encoder and used for the string similarities.
        Describes whether the matrix `V` to factorize should be made of
        word counts or character n-gram counts.
        Option `char_wb` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    ngram_range : 2-tuple of int, default=(2, 4)
        The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.
    return_score : bool, default=True
        Whether to return matching score based on the distance between
        the nearest matched categories.
    match_score : float, default=0.0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. 1 means that only a perfect match will be
        accepted, and zero means that the closest match will be accepted,
        no matter how distant.
        For numerical joins, this defines the maximum Euclidean distance
        between the matches.
    drop_unmatched : bool, default=False
        Remove categories for which a match was not found in the two tables.
    sort : bool, default=False
        Sort the join keys lexicographically in the resulting :obj:`~pandas.DataFrame`.
        If False, the order of the join keys depends on the join type
        (`how` keyword).
    suffixes : 2-tuple of str, default=('_x', '_y')
        A list of strings indicating the suffix to add when overlaping
        column names.

    Returns
    -------
    df_joined : :obj:`~pandas.DataFrame`
        The joined table returned as a :obj:`~pandas.DataFrame`.
        If `return_score=True`, another column will be added
        to the DataFrame containing the matching scores.

    See Also
    --------
    Joiner
        Transformer to enrich a given table via one or more fuzzy joins to
        external resources.

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
    return Joiner(
        aux_table=right,
        main_key=left_on,
        aux_key=right_on,
        key=on,
        suffix=suffix,
        matching=matching,
        string_encoder=string_encoder,
    ).fit_transform(left)
