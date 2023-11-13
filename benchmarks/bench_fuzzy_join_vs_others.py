"""
This benchmark compares the performance of dirty-cat's fuzzy_join compared
to other fuzzy joining functions available on small toy datasets.

skrub's fuzzy_join outperforms all other methods.

Date: September 2022
"""

import math
from argparse import ArgumentParser
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autofj import AutoFJ
from autofj.datasets import load_data
from thefuzz import process
from thefuzz.fuzz import partial_ratio
from utils import default_parser, find_result, monitor
from utils.join import evaluate

from skrub import fuzzy_join


def thefuzz_merge(
    df_1, df_2, left_on, right_on, threshold=0, limit=1, scorer=partial_ratio
):
    """
    Merging using thefuzz

    Parameters:
        df_1: the left table to join
        df_2: the right table to join
        left_on: key column of the left table
        right_on: key column of the right table
        threshold: how close the matches should be to return a match,
                   based on Levenshtein distance
        limit: the amount of matches that will get returned, these are sorted
               high to low

    Return:
        Dataframe with boths keys and matches.
    """
    s = df_2[right_on].tolist()
    m = df_1[left_on].apply(lambda x: process.extract(x, s, limit=limit, scorer=scorer))
    df_1["matches"] = m

    m2 = df_1["matches"].apply(
        lambda x: ", ".join([i[0] for i in x if i[1] >= threshold])
    )
    df_1["matches"] = m2
    thefuzz_join = df_1[df_1["matches"] != ""]
    return thefuzz_join


def autofj_merge(left, right, target=0.9):
    """Merging using AutomaticFuzzyJoin"""
    autofj = AutoFJ(precision_target=target, verbose=True)
    autofj_joins = autofj.join(left, right, id_column="id")
    return autofj_joins


#########################################################
# Benchmarking accuracy and speed on actual datasets
#########################################################

benchmark_name = "bench_fuzzy_join_vs_others"


@monitor(
    memory=True,
    time=True,
    parametrize={
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
        "join": [
            "fuzzy_join",
            "autofj",
            "thefuzz",
        ],
    },
    save_as=benchmark_name,
    repeat=5,
)
def benchmark(
    dataset_name: str,
    join: str,
):
    left_table, right_table, gt = load_data(dataset_name)

    if join == "fuzzy_join":
        start_time = perf_counter()
        joined_fj = fuzzy_join(
            left_table,
            right_table,
            on="title",
            suffix="_r",
        )
        end_time = perf_counter()
        pr, re, f1 = evaluate(
            list(zip(joined_fj["title"], joined_fj["title_r"])),
            list(zip(gt["title_l"], gt["title_r"])),
        )
    elif join == "autofj":
        start_time = perf_counter()
        joined_fj = autofj_merge(
            left_table,
            right_table,
        )
        end_time = perf_counter()
        pr, re, f1 = evaluate(
            list(zip(joined_fj["title_l"], joined_fj["title_r"])),
            list(zip(gt["title_l"], gt["title_r"])),
        )
    elif join == "thefuzz":
        left_table.rename(columns={"title": "title_l"}, inplace=True)
        right_table.rename(columns={"title": "title_r"}, inplace=True)
        start_time = perf_counter()
        joined_fj = thefuzz_merge(
            df_1=left_table,
            df_2=right_table,
            left_on="title_l",
            right_on="title_r",
        )
        end_time = perf_counter()
        pr, re, f1 = evaluate(
            list(zip(joined_fj["title_l"], joined_fj["matches"])),
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
            hue="join",
            # style="ngram_range",
            # size="analyzer",
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
