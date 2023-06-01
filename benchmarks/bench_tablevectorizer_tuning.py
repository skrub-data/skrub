"""
Performs a grid search to find the best parameters for the TableVectorizer
among a selection.
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


from dirty_cat import TableVectorizer, MinHashEncoder
from dirty_cat.datasets import (
    fetch_open_payments,
    fetch_drug_directory,
    fetch_road_safety,
    fetch_midwest_survey,
    fetch_medical_charge,
    fetch_employee_salaries,
    fetch_traffic_violations,
)

from typing import List, Tuple
from argparse import ArgumentParser
from utils import default_parser, find_result, monitor


def get_classification_datasets() -> List[Tuple[dict, str]]:
    return [
        (fetch_open_payments(), "open_payments"),
        (fetch_drug_directory(), 'drug_directory'),
        (fetch_road_safety(), "road_safety"),
        (fetch_midwest_survey(), "midwest_survey"),
        (fetch_traffic_violations(), "traffic_violations"),
    ]


def get_regression_datasets() -> List[Tuple[dict, str]]:
    return [
        (fetch_medical_charge(), "medical_charge"),
        (fetch_employee_salaries(), "employee_salaries"),
    ]


def get_dataset(info) -> Tuple[pd.DataFrame, pd.Series]:
    y = info.y
    X = info.X
    return X, y


###############################################
# Benchmarking TableVectorizer parameters
###############################################

benchmark_name = "bench_tablevectorizer_tuning"


@monitor(
    memory=True,
    time=True,
    parametrize={
        "tv_cardinality_threshold": [20, 30, 40, 50],
        "minhash_n_components": [10, 30, 50],
    },
    save_as=benchmark_name,
    repeat=5,
)
def benchmark(
    tv_cardinality_threshold: int,
    minhash_n_components: int,
):
    regression_pipeline = Pipeline(
        [
            ("tv", TableVectorizer(cardinality_threshold=tv_cardinality_threshold, high_card_cat_transformer=MinHashEncoder(n_components=minhash_n_components))),
            ("estimator", HistGradientBoostingRegressor()),
        ]
    )
    classification_pipeline = Pipeline(
        [
            ("tv", TableVectorizer(cardinality_threshold=tv_cardinality_threshold, high_card_cat_transformer=MinHashEncoder(n_components=minhash_n_components))),
            ("estimator", HistGradientBoostingClassifier()),
        ]
    )
    for pipeline, datasets in zip(
        [
            regression_pipeline,
            classification_pipeline,
        ],
        [
            get_regression_datasets(),
            get_classification_datasets(),
        ],
    ):
        for info, name in datasets:
            X, y = get_dataset(info)
            pipeline.fit(X, y)
            scores = cross_val_score(pipeline, X, y)
            score = np.mean(scores)
            dataset_name = name

    res_dic = {
            "grid_search_results": score,
            "dataset_name": dataset_name
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
    # Create the subplots but indexed by 1 value
    for i, dataset_name in enumerate(np.unique(df["dataset_name"])):
        sns.scatterplot(
            x="time_fj",
            y="grid_search_results",
            hue="tv_cardinality_threshold",
            size="minhash_n_components",
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
        description="Benchmark for the best parameters of the TableVectorizer.",
        parents=[default_parser],
    ).parse_args()

    if _args.run:
        benchmark()
    else:
        result_file = find_result(benchmark_name)
        df = pd.read_csv(result_file)

    if _args.plot:
        plot(df)
