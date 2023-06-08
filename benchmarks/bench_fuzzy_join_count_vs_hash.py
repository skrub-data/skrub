"""
This benchmark compares the performance of using the HashingVectorizer
or the CountVectorizer for the fuzzy join.

The results seem to indicate that the HashingVectorizer is almost always
faster than the CountVectorizer, without any significant loss in accuracy.
This leads to the conclusion that the HashingVectorizer should be used
by default for the fuzzy join, with the option to use the CountVectorizer if
results are unexpected (e.g hash collisions).

Date: December 2022
"""

import math
from argparse import ArgumentParser
from time import perf_counter
from typing import Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autofj.datasets import load_data
from scipy.sparse import vstack
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
)
from sklearn.neighbors import NearestNeighbors
from utils import default_parser, find_result, monitor
from utils.join import evaluate


# Function kept for reference
def fuzzy_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: Literal["left", "right"] = "left",
    left_on: Union[str, None] = None,
    right_on: Union[str, None] = None,
    on: Union[str, None] = None,
    encoder: Literal["count", "hash"] = "count",
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    ngram_range: Tuple[int, int] = (2, 4),
    return_score: bool = False,
    match_score: float = 0,
    drop_unmatched: bool = False,
    sort: bool = False,
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """
    Join two tables categorical string columns based on approximate
    matching and using morphological similarity.

    Parameters
    ----------
    left : pandas.DataFrame
        A table to merge.
    right : pandas.DataFrame
        A table used to merge with.
    how: typing.Literal["left", "right"], default=`left`
        Type of merge to be performed. Note that unlike pandas' merge,
        only "left" and "right" are supported so far, as the fuzzy-join comes
        with its own mechanism to resolve lack of correspondence between
        left and right tables.
    left_on : typing.Union[str, None]
        Name of left table column to join.
    right_on : typing.Union[str, None]
        Name of right table key column to join
        with left table key column.
    on : typing.Union[str, None]
        Name of common left and right table join key columns.
        Must be found in both DataFrames. Use only if `left_on`
        and `right_on` parameters are not specified.
    analyzer : typing.Literal["word", "char", "char_wb"], default=`char_wb`
        Analyzer parameter for the CountVectorizer used for the string
        similarities.
        Options: {`word`, `char`, `char_wb`}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option `char_wb` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    ngram_range : tuple (min_n, max_n), default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams used in the string similarity. All values of n such
        that min_n <= n <= max_n will be used.
    return_score : boolean, default=True
        Whether to return matching score based on the distance between the
        nearest matched categories.
    match_score : float, default=0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. Closer to 1 means the matches need to be very
        close to be accepted, and closer to 0 that a bigger matching distance
        is tolerated.
    drop_unmatched : boolean, default=False
        Remove categories for which a match was not found in the two tables.
    sort : boolean, default=False
        Sort the join keys lexicographically in the result DataFrame.
        If False, the order of the join keys depends on the join type
        (`how` keyword).
    suffixes : typing.Tuple[str, str], default=('_x', '_y')
        A list of strings indicating the suffix to add when overlaping
        column names.

    Returns
    -------
    df_joined: pandas.DataFrame
        The joined table returned as a DataFrame. If `return_score` is True,
        another column will be added to the DataFrame containing the
        matching scores.

    Notes
    -----
    For regular joins, the output of fuzzy_join is identical
    to pandas.merge, except that both key columns are returned.

    Joining on indexes and multiple columns is not
    supported.

    When return_score=True, the returned DataFrame gives
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

    if how not in ["left", "right"]:
        raise ValueError(
            f"how should be either 'left' or 'right', got {how!r}",
        )

    for param in [on, left_on, right_on]:
        if param is not None and not isinstance(param, str):
            raise KeyError(
                "Parameter 'left_on', 'right_on' or 'on' has invalid type, expected"
                " string"
            )

    # TODO: enable joining on multiple keys as in pandas.merge
    if on is not None:
        left_col = on
        right_col = on
    elif left_on is not None and right_on is not None:
        left_col = left_on
        right_col = right_on
    else:
        raise KeyError(
            "Required parameter missing: either parameter"
            "'on' or the pair 'left_on', 'right_on' should be specified."
        )

    if how == "left":
        main_table = left.reset_index(drop=True).copy()
        aux_table = right.reset_index(drop=True).copy()
        main_col = left_col
        aux_col = right_col
    else:
        main_table = right.reset_index(drop=True).copy()
        aux_table = left.reset_index(drop=True).copy()
        main_col = right_col
        aux_col = left_col

    # Drop missing values in key columns
    main_table.dropna(subset=[main_col], inplace=True)
    aux_table.dropna(subset=[aux_col], inplace=True)

    # Make sure that the column types are string and categorical:
    main_col_clean = main_table[main_col].astype(str)
    aux_col_clean = aux_table[aux_col].astype(str)

    if encoder == "count":
        enc = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    elif encoder == "hash":
        enc = HashingVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    else:
        raise ValueError(
            f"encoder should be either 'count' or 'hash', got {encoder!r}",
        )

    all_cats = pd.concat([main_col_clean, aux_col_clean], axis=0).unique()

    enc_cv = enc.fit(all_cats)
    main_enc = enc_cv.transform(main_col_clean)
    aux_enc = enc_cv.transform(aux_col_clean)

    all_enc = vstack((main_enc, aux_enc))

    tfidf = TfidfTransformer().fit(all_enc)
    main_enc = tfidf.transform(main_enc)
    aux_enc = tfidf.transform(aux_enc)

    # Find nearest neighbor using KNN :
    neigh = NearestNeighbors(n_neighbors=1)

    neigh.fit(aux_enc)
    distance, neighbors = neigh.kneighbors(main_enc, return_distance=True)
    idx_closest = np.ravel(neighbors)

    main_table["fj_idx"] = idx_closest
    aux_table["fj_idx"] = aux_table.index

    norm_distance = 1 - (distance / 2)
    if drop_unmatched:
        main_table = main_table[match_score <= norm_distance]
        norm_distance = norm_distance[match_score <= norm_distance]
    else:
        main_table.loc[np.ravel(match_score > norm_distance), "fj_nan"] = 1

    if sort:
        main_table.sort_values(by=[main_col], inplace=True)

    # To keep order of columns as in pandas.merge (always left table first)
    if how == "left":
        df_joined = pd.merge(
            main_table, aux_table, on="fj_idx", suffixes=suffixes, how=how
        )
    else:
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
        df_joined = pd.concat(
            [df_joined, pd.DataFrame(norm_distance, columns=["matching_score"])], axis=1
        )

    return df_joined


#########################################################
# Benchmarking accuracy and speed on actual datasets
#########################################################

benchmark_name = "bench_fuzzy_join_count_vs_hash"


@monitor(
    memory=True,
    time=True,
    parametrize={
        "encoder": ["hash", "count"],
        "dataset_name": [
            "Country",
            "BasketballTeam",
            "Drug",
            "Device",
            "ArtificialSatellite",
            "Amphibian",
            "Song",
            "HistoricBuilding",
            "Wrestler",
            "EthnicGroup",
        ],
        "analyzer": ["char_wb", "char", "word"],
        "ngram_range": [(2, 4), (2, 3), (2, 2)],
    },
    save_as=benchmark_name,
    repeat=10,
)
def benchmark(
    encoder: Literal["hash", "count"],
    dataset_name: str,
    analyzer: Literal["char_wb", "char", "word"],
    ngram_range: tuple,
):
    left_table, right_table, gt = load_data(dataset_name)

    start_time = perf_counter()
    joined_fj = fuzzy_join(
        left_table,
        right_table,
        how="left",
        left_on="title",
        right_on="title",
        encoder=encoder,
        analyzer=analyzer,
        ngram_range=ngram_range,
    )
    end_time = perf_counter()

    pr, re, f1 = evaluate(
        list(zip(joined_fj["title_x"], joined_fj["title_y"])),
        list(zip(gt["title_l"], gt["title_r"])),
    )

    res_dic = {
        "precision": pr,
        "recall": re,
        "f1": f1,
        "time_fj": end_time - start_time,
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
        sns.scatterplot(
            x="time_fj",
            y="f1",
            hue="encoder",
            style="ngram_range",
            size="analyzer",
            alpha=0.8,
            data=df[df["dataset_name"] == dataset_name],
            ax=axes[i % n_rows, i // n_rows],
        )
        axes[i % n_rows, i // n_rows].set_title(dataset_name)
        # remove legend
        axes[i % n_rows, i // n_rows].get_legend().remove()
        # Put a legend to the right side if last row
        if i == n_datasets - 1:
            axes[i % n_rows, i // n_rows].legend(loc="center right")
    plt.show()


if __name__ == "__main__":
    _args = ArgumentParser(
        description="Benchmark for the batch feature of the MinHashEncoder.",
        parents=[default_parser],
    ).parse_args()

    if _args.run:
        df = benchmark()
    else:
        result_file = find_result(benchmark_name)
        df = pd.read_parquet(result_file)

    if _args.plot:
        plot(df)
