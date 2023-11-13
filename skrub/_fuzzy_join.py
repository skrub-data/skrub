"""
Implements fuzzy_join, a function to perform fuzzy joining between two tables.
"""
import pandas as pd

from skrub import _join_utils
from skrub._joiner import DEFAULT_REF_DIST, DEFAULT_STRING_ENCODER, Joiner


def fuzzy_join(
    left,
    right,
    left_on=None,
    right_on=None,
    on=None,
    suffix="",
    max_dist=1.0,
    insert_match_info=False,
    drop_unmatched=False,
    string_encoder=DEFAULT_STRING_ENCODER,
    ref_dist=DEFAULT_REF_DIST,
) -> pd.DataFrame:
    """Join two tables based on approximate matching using the appropriate metric.

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

    Joining on indexes is not supported.

    When `insert_match_info=True`, the returned :obj:`~pandas.DataFrame` contains
    additional columns which provide information about the match.

    When we use `max_dist=np.inf`, the function will be forced to impute the
    nearest match (of the left table category) across all possible matching
    options in the right table column.

    When the neighbors are distant, we may use the `max_dist` parameter
    define the maximal (rescaled) distance between 2 rows for them to match.
    If it is not reached, matches will be considered as not found and NaN values
    will be imputed.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> main_table = pd.DataFrame({"Country": ["France", "Italia", "Spain"]})
    >>> aux_table = pd.DataFrame( {"Country": ["Germany", "France", "Italy"],
    ...                            "Capital": ["Berlin", "Paris", "Rome"]} )
    >>> main_table
      Country
    0  France
    1  Italia
    2   Spain
    >>> aux_table
       Country Capital
    0  Germany  Berlin
    1   France   Paris
    2    Italy    Rome
    >>> fuzzy_join(
    ...     main_table,
    ...     aux_table,
    ...     on="Country",
    ...     suffix="_capitals",
    ...     max_dist=0.9,
    ... )
      Country Country_capitals Capital_capitals
    0  France           France            Paris
    1  Italia            Italy             Rome
    2   Spain              NaN              NaN
    >>> fuzzy_join(
    ...     main_table,
    ...     aux_table,
    ...     on="Country",
    ...     suffix="_capitals",
    ...     drop_unmatched=True,
    ...     max_dist=0.9,
    ... )
      Country Country_capitals Capital_capitals
    0  France           France            Paris
    1  Italia            Italy             Rome
    >>> fuzzy_join(
    ...     main_table,
    ...     aux_table,
    ...     on="Country",
    ...     suffix="_capitals",
    ...     max_dist=1.0,
    ... )
      Country Country_capitals Capital_capitals
    0  France           France            Paris
    1  Italia            Italy             Rome
    2   Spain          Germany           Berlin
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
        max_dist=max_dist,
        ref_dist=ref_dist,
        string_encoder=string_encoder,
        insert_match_info=True,
    ).fit_transform(left)
    if drop_unmatched:
        join = join[join["skrub.Joiner.match_accepted"]]
    if not insert_match_info:
        join = join.drop(Joiner.match_info_columns, axis=1)
    return join
