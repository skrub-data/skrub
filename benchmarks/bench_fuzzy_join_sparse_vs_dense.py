"""
This benchmark compares the performance of using sparse vs dense arrays
for a neareast neighbor search in the fuzzy_join.

The results seem to indicate that using dense arrays, with the current state,
is faster to search through than sparse arrays.

Date: June 2023
"""

import math
from pathlib import Path
from utils import default_parser, find_result, monitor
from utils.join import evaluate, fetch_big_data
from argparse import ArgumentParser
import numbers
import warnings
from collections.abc import Iterable
from time import perf_counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack, vstack
from sklearn import random_projection
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    _VectorizerMixin,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _numeric_encoding(
    main: pd.DataFrame,
    main_cols: list | str,
    aux: pd.DataFrame,
    aux_cols: list | str,
) -> tuple:
    """Encoding numerical columns.

    Parameters
    ----------
    main : :class:`~pandas.DataFrame`
        A table with numerical columns.
    main_cols : str or list
        The columns of the main table.
    aux : :class:`~pandas.DataFrame`
        Another table with numerical columns.
    aux_cols : str or list
        The columns of the aux table.

    Returns
    -------
    array-like
        An array of the encoded columns of the main table.
    array-like
        An array of the encoded columns of the aux table.
    """
    aux_array = aux[aux_cols].to_numpy()
    main_array = main[main_cols].to_numpy()
    # Re-weighting to avoid measure specificity
    scaler = StandardScaler()
    scaler.fit(np.vstack((aux_array, main_array)))
    aux_array = scaler.transform(aux_array)
    main_array = scaler.transform(main_array)
    return main_array, aux_array


def _string_encoding(
    main: pd.DataFrame,
    main_cols: list | str,
    aux: pd.DataFrame,
    aux_cols: list | str,
    analyzer: Literal["word", "char", "char_wb"],
    ngram_range: int | int,
    encoder: _VectorizerMixin = None,
    sparse: bool = True,
) -> tuple:
    """Encoding string columns.

    Parameters
    ----------
    main : :class:`~pandas.DataFrame`
        A table with string columns.
    main_cols : str or list
        The columns of the main table.
    aux : :class:`~pandas.DataFrame`
        Another table with string columns.
    aux_cols : str or list
        The columns of the aux table.
    analyzer : {"word", "char", "char_wb"}
        Analyzer parameter for the HashingVectorizer passed to
        the encoder and used for the string similarities.
        See fuzzy_join's docstring for more information.
    ngram_range : int 2-tuple, optional, default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams used in the string similarity.
        See fuzzy_join's docstring for more information.
    encoder: vectorizer instance, optional
        Encoder parameter for the Vectorizer.
        See fuzzy_join's docstring for more information.

    Returns
    -------
    array-like
        An array of the encoded columns of the main table.
    array-like
        An array of the encoded columns of the aux table.
    """
    # Make sure that the column types are string and categorical:
    main_cols_clean = main[main_cols].astype(str)
    aux_cols_clean = aux[aux_cols].astype(str)

    if isinstance(main_cols, list) and isinstance(aux_cols, list):
        first_col, other_cols = main_cols[0], main_cols[1:]
        main_cols_clean = main_cols_clean[first_col].str.cat(
            main_cols_clean[other_cols], sep="  "
        )
        first_col_aux, other_cols_aux = aux_cols[0], aux_cols[1:]
        aux_cols_clean = aux_cols_clean[first_col_aux].str.cat(
            aux_cols_clean[other_cols_aux], sep="  "
        )
    all_cats = pd.concat([main_cols_clean, aux_cols_clean], axis=0).unique()

    if encoder is None:
        encoder = HashingVectorizer(analyzer=analyzer, ngram_range=ngram_range)

    encoder = encoder.fit(all_cats)
    main_enc = encoder.transform(main_cols_clean)
    aux_enc = encoder.transform(aux_cols_clean)

    all_enc = vstack((main_enc, aux_enc))

    tfidf = TfidfTransformer().fit(all_enc)
    main_enc = tfidf.transform(main_enc)
    aux_enc = tfidf.transform(aux_enc)
    if sparse is True:
        return main_enc, aux_enc
    else:
        transformer = random_projection.GaussianRandomProjection(eps=0.4)
        main_enc_d = transformer.fit_transform(main_enc)
        aux_enc_d = transformer.fit_transform(main_enc)
        return main_enc_d, aux_enc_d


def _nearest_matches(main_array, aux_array, sparse=True) -> np.ndarray | np.ndarray:
    """Find the closest matches using the nearest neighbors method.

    Parameters
    ----------
    main_array : array-like
        An array of the encoded columns of the main table.
    aux_array : array-like
        An array of the encoded columns of the aux table.

    Returns
    -------
    np.ndarray
        Index of the closest matches of the main table in the aux table.
    np.ndarray
        Distance between the closest matches, on a scale between 0 and 1.
    """
    # Find nearest neighbor using KNN :
    neigh = NearestNeighbors(n_neighbors=1)
    if sparse is True:
        neigh.fit(aux_array)
        distance, neighbors = neigh.kneighbors(main_array, return_distance=True)
    else:
        neigh.fit(np.asarray(aux_array))
        distance, neighbors = neigh.kneighbors(
            np.asarray(main_array), return_distance=True
        )
    idx_closest = np.ravel(neighbors)
    distance = distance / np.max(distance)
    # Normalizing distance between 0 and 1:
    matching_score = 1 - (distance / 2)
    return idx_closest, matching_score


def fuzzy_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: Literal["left", "right"] = "left",
    left_on: str | list[str] | list[int] | None = None,
    right_on: str | list[str] | list[int] | None = None,
    on: str | list[str] | list[int] | None = None,
    numerical_match: Literal["string", "number"] = "number",
    encoder: _VectorizerMixin = None,
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    ngram_range: tuple[int, int] = (2, 4),
    return_score: bool = False,
    match_score: float = 0,
    drop_unmatched: bool = False,
    sort: bool = False,
    suffixes: tuple[str, str] = ("_x", "_y"),
    sparse: bool = True,
) -> pd.DataFrame:
    """
    Join two tables categorical string columns based on approximate
    matching and using morphological similarity.

    Parameters
    ----------
    left : :class:`~pandas.DataFrame`
        A table to merge.
    right : :class:`~pandas.DataFrame`
        A table used to merge with.
    how: typing.Literal["left", "right"], default=`left`
        Type of merge to be performed. Note that unlike :func:`~pandas.merge`,
        only "left" and "right" are supported so far, as the fuzzy-join comes
        with its own mechanism to resolve lack of correspondence between
        left and right tables.
    left_on : str or list of str, optional
        Name of left table column(s) to join.
    right_on : str or list of str, optional
        Name of right table key column(s) to join
        with left table key column(s).
    on : str or list of str or int, optional, default=None
        Name of common left and right table join key columns.
        Must be found in both DataFrames. Use only if `left_on`
        and `right_on` parameters are not specified.
    numerical_match: {`string`, `number`}, optional, default='string'
        For numerical columns, match using the Euclidean distance
        ("number"). If "string", uses the default n-gram string
        similarity on the string representation.
    encoder: _VectorizerMixin, default=None
        Encoder parameter for the Vectorizer.
        By default, uses a HashingVectorizer.
        It is possible to pass a vectorizer instance inheriting
        _VectorizerMixin to tweak the parameters of the encoder.
    analyzer : {"word", "char", "char_wb"}, optional, default=`char_wb`
        Analyzer parameter for the HashingVectorizer passed to
        the encoder and used for the string similarities.
        Options: {`word`, `char`, `char_wb`}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option `char_wb` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    ngram_range : int 2-tuple, optional, default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams used in the string similarity. All values of n such
        that min_n <= n <= max_n will be used.
    return_score : boolean, default=True
        Whether to return matching score based on the distance between
        the nearest matched categories.
    match_score : float, default=0.0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. 1 means that only a perfect match will be
        accepted, and zero means that the closest match will be accepted,
        no matter how distant.
        For numerical joins, this defines the maximum Euclidean distance
        between the matches.
    drop_unmatched : boolean, default=False
        Remove categories for which a match was not found in the two tables.
    sort : boolean, default=False
        Sort the join keys lexicographically in the result DataFrame.
        If False, the order of the join keys depends on the join type
        (`how` keyword).
    suffixes : str 2-tuple, default=('_x', '_y')
        A list of strings indicating the suffix to add when overlaping
        column names.
    sparse : boolean, default=True
        Use sparse or dense arrays for nearest neighbor search.

    Returns
    -------
    df_joined : :class:`~pandas.DataFrame`
        The joined table returned as a DataFrame. If `return_score` is True,
        another column will be added to the DataFrame containing the
        matching scores.

    See Also
    --------
    :class:`~skrub.Joiner` :
        Transformer to enrich a given table via one or more fuzzy joins to
        external resources.

    Notes
    -----
    For regular joins, the output of fuzzy_join is identical
    to :func:`~pandas.merge`, except that both key columns are returned.

    Joining on indexes and multiple columns is not supported.

    When `return_score=True`, the returned :class:`~pandas.DataFrame` gives
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
        a    c
    0  anna  5
    1  lala  6
    2  ana   7
    3  nnana 8

    To do a simple join based on the nearest match:

    >>> fuzzy_join(df1, df2, on='a')
        a_x  b   a_y    c
    0   ana  1   ana    7
    1  lala  2  lala    6
    2  nana  3  nnana   8

    When we want to accept only a certain match precision,
    we can use the `match_score` argument:

    >>> fuzzy_join(df1, df2, on='a', match_score=1, return_score=True)
        a_x  b   a_y    c  matching_score
    0   ana  1   ana  7.0  1.000000
    1  lala  2  lala  6.0  1.000000
    2  nana  3   NaN  NaN  0.532717

    As expected, the category "nana" has no exact match (`match_score=1`).
    """
    if analyzer not in ["char", "word", "char_wb"]:
        raise ValueError(
            f"analyzer should be either 'char', 'word' or 'char_wb', got {analyzer!r}",
        )

    if encoder is not None:
        if not issubclass(encoder.__class__, _VectorizerMixin):
            raise ValueError(
                "Parameter 'encoder' should be a vectorizer instance or "
                f"'hashing', got {encoder!r}. "
            )

    if how not in ["left", "right"]:
        raise ValueError(
            f"Parameter 'how' should be either 'left' or 'right', got {how!r}. "
        )

    if numerical_match not in ["string", "number"]:
        raise ValueError(
            "Parameter 'numerical_match' should be either 'string' or 'number', "
            f"got {numerical_match!r}. ",
        )

    for param in [on, left_on, right_on]:
        if param is not None and not isinstance(param, Iterable):
            raise TypeError(
                "Parameter 'left_on', 'right_on' or 'on' has invalid type,"
                "expected string or list of column names. "
            )

    if not isinstance(match_score, numbers.Number):
        raise TypeError(
            "Parameter 'match_score' has invalid type, expected int or float. "
        )

    if isinstance(on, str):
        left_col, right_col = [on], [on]
    elif isinstance(left_on, str) and isinstance(right_on, str):
        left_col, right_col = [left_on], [right_on]
    elif isinstance(on, Iterable):
        left_col = list(on)
        right_col = list(on)
    elif isinstance(left_on, Iterable) and isinstance(right_on, Iterable):
        left_col = list(left_on)
        right_col = list(right_on)
    else:
        raise KeyError(
            "Required parameter missing: either parameter "
            "'on' or 'left_on' & 'right_on' should be specified."
        )

    if how == "left":
        main_table = left.reset_index(drop=True)
        aux_table = right.reset_index(drop=True)
        main_cols = left_col
        aux_cols = right_col
    elif how == "right":
        main_table = right.reset_index(drop=True)
        aux_table = left.reset_index(drop=True)
        main_cols = right_col
        aux_cols = left_col

    # Warn if presence of missing values
    if main_table[main_cols].isna().any().any():
        warnings.warn(
            "You are merging on missing values. "
            "The output correspondence will be random or missing. "
            "To avoid unexpected errors you can drop them. ",
            UserWarning,
            stacklevel=2,
        )

    main_num_cols = main_table[main_cols].select_dtypes(include="number").columns
    aux_num_cols = aux_table[aux_cols].select_dtypes(include="number").columns

    # Check if included columns are numeric:
    any_numeric = len(main_num_cols) != 0
    # Check if included columns have mixed types:
    mixed_types = len(main_cols) > len(main_num_cols)

    if len(main_cols) == 1 and len(aux_cols) == 1 and any_numeric is False:
        main_cols = main_cols[0]
        aux_cols = aux_cols[0]

    if numerical_match in ["number"] and any_numeric and not mixed_types:
        main_enc, aux_enc = _numeric_encoding(
            main_table, main_num_cols, aux_table, aux_num_cols
        )
        idx_closest, matching_score = _nearest_matches(main_enc, aux_enc, sparse=sparse)
    elif numerical_match in ["number"] and any_numeric and mixed_types:
        main_num_enc, aux_num_enc = _numeric_encoding(
            main_table, main_num_cols, aux_table, aux_num_cols
        )
        main_str_cols = list(set(main_cols) - set(main_num_cols))
        aux_str_cols = list(set(aux_cols) - set(aux_num_cols))
        main_str_enc, aux_str_enc = _string_encoding(
            main_table,
            main_str_cols,
            aux_table,
            aux_str_cols,
            encoder=encoder,
            analyzer=analyzer,
            ngram_range=ngram_range,
            sparse=sparse,
        )
        if sparse is True:
            main_enc = hstack((main_num_enc, main_str_enc), format="csr")
            aux_enc = hstack((aux_num_enc, aux_str_enc), format="csr")
        else:
            main_enc = np.hstack((main_num_enc, main_str_enc))
            aux_enc = np.hstack((aux_num_enc, aux_str_enc))
        idx_closest, matching_score = _nearest_matches(main_enc, aux_enc, sparse=sparse)
    else:
        main_enc, aux_enc = _string_encoding(
            main_table,
            main_cols,
            aux_table,
            aux_cols,
            encoder=encoder,
            analyzer=analyzer,
            ngram_range=ngram_range,
            sparse=sparse,
        )
        idx_closest, matching_score = _nearest_matches(main_enc, aux_enc, sparse=sparse)

    main_table["fj_idx"] = idx_closest
    aux_table["fj_idx"] = aux_table.index

    if drop_unmatched:
        main_table = main_table[match_score <= matching_score]
        matching_score = matching_score[match_score <= matching_score]
    else:
        main_table.loc[np.ravel(match_score > matching_score), "fj_nan"] = 1

    if sort:
        main_table.sort_values(by=[main_cols], inplace=True)

    # To keep order of columns as in pandas.merge (always left table first)
    if how == "left":
        df_joined = pd.merge(
            main_table, aux_table, on="fj_idx", suffixes=suffixes, how=how
        )
    elif how == "right":
        df_joined = pd.merge(
            aux_table, main_table, on="fj_idx", suffixes=suffixes, how=how
        )

    if drop_unmatched:
        df_joined.drop(columns=["fj_idx"], inplace=True)
    else:
        idx = df_joined.index[df_joined["fj_nan"] == 1]
        if len(idx) != 0:
            df_joined.iloc[idx, df_joined.columns.get_loc("fj_idx") :] = np.NaN
        df_joined.drop(columns=["fj_idx", "fj_nan"], inplace=True)

    if return_score:
        df_joined["matching_score"] = matching_score

    return df_joined


#########################################################
# Benchmarking accuracy and speed on actual datasets
#########################################################

benchmark_name = "bench_fuzzy_join_sparse_vs_dense"


@monitor(
    parametrize={
        "sparse": [True, False],
        "dataset_name": [
            "iTunes-Amazon",
            "DBLP-ACM",
            "DBLP-GoogleScholar",
            "Walmart-Amazon",
        ],
        "analyzer": ["char_wb", "char", "word"],
        "ngram_range": [(2, 4), (3, 4)],
    },
    memory=True,
    time=True,
    repeat=5,
    save_as=benchmark_name,
)
def benchmark(
    sparse: bool,
    dataset_name: str,
    analyzer: Literal["char_wb", "char", "word"],
    ngram_range: tuple,
    data_home: Path | str | None = None,
    data_directory: str | None = "benchmarks_data",
):
    left_table, right_table, gt = fetch_big_data(
        dataset_name=dataset_name, data_home=data_home, data_directory=data_directory
    )

    start_time = perf_counter()
    joined_fj = fuzzy_join(
        left_table,
        right_table,
        how="left",
        left_on=left_table.columns[0],
        right_on=right_table.columns[0],
        analyzer=analyzer,
        ngram_range=ngram_range,
        sparse=sparse,
    )
    end_time = perf_counter()

    n_obs = len(joined_fj.index)

    avg_word = np.round(
        np.mean(
            [
                joined_fj["title_x"].str.split().map(len).mean(),
                joined_fj["title_y"].str.split().map(len).mean(),
            ]
        )
    )

    pr, re, f1 = evaluate(
        list(zip(joined_fj["title_x"], joined_fj["title_y"])),
        list(zip(gt["title_l"], gt["title_r"])),
    )

    res_dic = {
        "n_obs": n_obs,
        "precision": pr,
        "recall": re,
        "f1": f1,
        "time_fj": end_time - start_time,
        "avg_count": avg_word,
    }

    return res_dic


def plot(df: pd.DataFrame):
    sns.set_theme(style="ticks", palette="pastel")

    n_datasets = len(np.unique(df["dataset_name"]))
    n_rows = min(n_datasets, 3)
    f, axes = plt.subplots(
        n_rows,
        math.ceil(n_datasets / n_rows),
        squeeze=False,
        figsize=(20, 5),
    )
    plt.tight_layout()
    # Create the subplots but indexed by 1 value
    for i, dataset_name in enumerate(np.unique(df["dataset_name"])):
        current_df = df[df["dataset_name"] == dataset_name].reset_index()
        sns.scatterplot(
            x="time_fj",
            y="f1",
            hue="sparse",
            style="ngram_range",
            size="analyzer",
            alpha=0.8,
            data=current_df,
            ax=axes[i % n_rows, i // n_rows],
        )
        obs = current_df["n_obs"][0]
        avg_word = current_df["avg_count"][0]
        axes[i % n_rows, i // n_rows].set_title(
            f"{dataset_name} Obs={obs} Avg_word_length={avg_word} "
        )
        # remove legend
        axes[i % n_rows, i // n_rows].get_legend().remove()
        # Put a legend to the right side if last row
        if i == n_datasets - 1:
            axes[i % n_rows, i // n_rows].legend(loc="center right")
    plt.show()


if __name__ == "__main__":
    _args = ArgumentParser(
        description=(
            "Benchmark for the evaluating the fuzzy_join performance of sparse vs dense"
            " arrays."
        ),
        parents=[default_parser],
    ).parse_args()

    if _args.run:
        df = benchmark()
    else:
        result_file = find_result(benchmark_name)
        df = pd.read_parquet(result_file)

    if _args.plot:
        plot(df)
