"""
Implements fuzzy_join, a function to perform fuzzy joining between two tables.
"""
import numpy as np

from skrub import _join_utils
from skrub._joiner import DEFAULT_REF_DIST, DEFAULT_STRING_ENCODER, Joiner


def fuzzy_join(
    left,
    right,
    left_on=None,
    right_on=None,
    on=None,
    suffix="",
    max_dist=np.inf,
    ref_dist=DEFAULT_REF_DIST,
    string_encoder=DEFAULT_STRING_ENCODER,
    add_match_info=False,
    drop_unmatched=False,
):
    """Fuzzy (approximate) join.

    Rows in the left table are joined to their closest match from the right
    table. The resulting table has the same rows (in the same order) as the
    left table, unless ``drop_unmatched`` is ``True``, in which case rows that
    are too far from their closest match will not appear in the result. Each
    row from the left table appears at most once in the result; if there are
    several equally good matching rows in the right table one of them will be
    used; which one is unspecified.

    To identify the best match for each row, values from the matching columns
    (``left_key`` and ``right_key``) are vectorized, ie represented by vectors of
    continuous values. Then, the Euclidean distances between these vectors are
    computed to find, for each left table row, its nearest neighbor within the
    right table.

    Optionally, a maximum distance threshold, ``max_dist``, can be set. Matches
    between vectors that are separated by a distance (strictly) greater than
    ``max_dist`` will be rejected. We will consider that left table rows that
    are farther than ``max_dist`` from their nearest neighbor do not have a
    matching row in the right table, and the output will contain nulls for
    the entries that would normally have come from the right table (as in a
    traditional left join).

    To make it easier to set a ``max_dist`` threshold, the distances are
    rescaled by dividing them by a reference distance, which can be chosen with
    ``ref_dist``. The default is ``'random_pairs'``. The possible choices are:

    'random_pairs'
        Pairs of rows are sampled randomly from the right table and their
        distance is computed. The reference distance is the first quartile of
        those distances.

    'second_neighbor'
        The reference distance is the distance to the *second* nearest neighbor
        in the right table.

    'self_join_neighbor'
        Once the match candidate (ie the nearest neigbor from the right
        table) has been found, we find its nearest neighbor in the right
        table (excluding itself). The reference distance is the distance that
        separates those 2 right rows.

    'no_rescaling'
        The reference distance is 1.0, ie no rescaling of the distances is
        applied.

    Parameters
    ----------
    left : :obj:`~pandas.DataFrame`
        Left operand of the join.
    right : :obj:`~pandas.DataFrame`
        Right operand of the join.
    left_on : str or list of str, default=None
        The column names in the left table on which the join will be performed.
        Can be a string if joining on a single column.
        If ``None``, `right_on` must also be ``None`` and `on` must be provided.
    right_on : str or list of str, default=None
        The column names in the right table on which the join will
        be performed. Can be a string if joining on a single column.
        If ``None``, `left_on` must also be ``None`` and `on` must be provided.
    on : str or list of str, default=None
        The column names to use for both ``left_on`` and ``right_on`` when they
        are the same. Provide either ``on`` or both ``left_on`` and ``right_on``.
    suffix : str, default=""
        Suffix to append to the ``right`` table's column names. You can use it
        to avoid duplicate column names in the join.
    max_dist : float, default=np.inf
        Maximum acceptable (rescaled) distance between a row in the
        ``left`` table and its nearest neighbor in the ``right`` table. Rows that
        are farther apart are not considered to match. By default, the distance
        is rescaled so that a value between 0 and 1 is typically a good choice,
        although rescaled distances can be greater than 1 for some choices of
        ``ref_dist``. ``None``, ``"inf"``, ``float("inf")`` or ``numpy.inf``
        mean that no matches are rejected.
    ref_dist : reference distance for rescaling, default = 'random_pairs'
        Options are {"random_pairs", "second_neighbor", "self_join_neighbor",
        "no_rescaling"}. See above for a description of each option. To
        facilitate the choice of ``max_dist``, distances between rows in
        ``left`` table and their nearest neighbor in ``right`` table will be
        rescaled by this reference distance.
    string_encoder : scikit-learn transformer used to vectorize text columns
        By default a ``HashingVectorizer`` combined with a ``TfidfTransformer``
        is used. Here we use raw TF-IDF features rather than transforming them
        for example with ``GapEncoder`` or ``MinHashEncoder`` because it is
        faster, these features are only used to find nearest neighbors and not
        used by downstream estimators, and distances between TF-IDF vectors
        have a somewhat simpler interpretation.
    add_match_info : bool, default=False
        Insert columns whose names start with `skrub_Joiner` containing
        the distance, rescaled distance and whether the rescaled distance is
        above the threshold. Those values can be helpful for an estimator that
        uses the joined features, or to inspect the result of the join and set
        a ``max_dist`` threshold.
    drop_unmatched : bool, default=False
        Remove rows for which a match was not found in the right table (ie for
        which the nearest neighbor is further than `max_dist`).

    Returns
    -------
    :obj:`~pandas.DataFrame`
        The joined tables.

    See Also
    --------
    Joiner :
        Same as fuzzy_join but as a scikit-learn transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> left_table = pd.DataFrame({"Country": ["France", "Italia", "Spain"]})
    >>> right_table = pd.DataFrame( {"Country": ["Germany", "France", "Italy"],
    ...                            "Capital": ["Berlin", "Paris", "Rome"]} )
    >>> left_table
      Country
    0  France
    1  Italia
    2   Spain
    >>> right_table
       Country Capital
    0  Germany  Berlin
    1   France   Paris
    2    Italy    Rome
    >>> fuzzy_join(
    ...     left_table,
    ...     right_table,
    ...     on="Country",
    ...     suffix="_capitals",
    ...     max_dist=1.0,
    ...     add_match_info=False,
    ... )
      Country Country_capitals Capital_capitals
    0  France           France            Paris
    1  Italia            Italy             Rome
    2   Spain              NaN              NaN
    >>> fuzzy_join(
    ...     left_table,
    ...     right_table,
    ...     on="Country",
    ...     suffix="_capitals",
    ...     drop_unmatched=True,
    ...     max_dist=1.0,
    ...     add_match_info=False,
    ... )
      Country Country_capitals Capital_capitals
    0  France           France            Paris
    1  Italia            Italy             Rome
    >>> fuzzy_join(
    ...     left_table,
    ...     right_table,
    ...     on="Country",
    ...     suffix="_capitals",
    ...     max_dist=float("inf"),
    ...     add_match_info=False,
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
        main_key_name="left_on",
        aux_key_name="right_on",
        key_name="on",
    )
    _join_utils.check_missing_columns(left, left_key, "'left' (the left table)")
    _join_utils.check_missing_columns(right, right_key, "'right' (the right table)")
    _join_utils.check_column_name_duplicates(
        left, right, suffix, main_table_name="left", aux_table_name="right"
    )

    join = Joiner(
        aux_table=right,
        main_key=left_on,
        aux_key=right_on,
        key=on,
        suffix=suffix,
        max_dist=max_dist,
        ref_dist=ref_dist,
        string_encoder=string_encoder,
        add_match_info=True,
    ).fit_transform(left)
    if drop_unmatched:
        join = join[join["skrub_Joiner_match_accepted"]]
    if not add_match_info:
        join = join.drop(Joiner.match_info_columns, axis=1)
    return join
