"""
Performs a grid search to find the best parameters for the TableVectorizer
among a selection.

Date: September 2021
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


from skrub import TableVectorizer, MinHashEncoder

from argparse import ArgumentParser
from utils import (
    default_parser,
    find_result,
    monitor,
    get_dataset,
    get_classification_datasets,
    get_regression_datasets,
)


###############################################
# Benchmarking TableVectorizer parameters
###############################################

benchmark_name = "bench_tablevectorizer_tuning"


@monitor(
    memory=True,
    time=True,
    parametrize={
        "tv_cardinality_threshold": [20, 40, 60],
        "minhash_n_components": [10, 30, 50],
        "dataset_name": [
            "medical_charge",
            "open_payments",
            "midwest_survey",
            "medical_charge",
            "employee_salaries",
        ],
    },
    save_as=benchmark_name,
    repeat=3,
)
def benchmark(
    tv_cardinality_threshold: int,
    minhash_n_components: int,
    dataset_name: str,
):
    tv = TableVectorizer(
        cardinality_threshold=tv_cardinality_threshold,
        high_card_cat_transformer=MinHashEncoder(n_components=minhash_n_components),
    )
    regression_pipeline = Pipeline(
        [
            ("tv", tv),
            ("estimator", HistGradientBoostingRegressor()),
        ]
    )
    classification_pipeline = Pipeline(
        [
            ("tv", tv),
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
            if name == dataset_name:
                X, y = get_dataset(info)
                pipeline.fit(X, y)
                scores = cross_val_score(pipeline, X, y, cv=3)
                score = np.mean(scores)
    res_dic = {"grid_search_results": score, "dataset_name": dataset_name}
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
            x="time",
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
            axes[i % n_rows, i // n_rows].legend(loc="lower right")
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
        df = pd.read_parquet(result_file)

    if _args.plot:
        plot(df)
