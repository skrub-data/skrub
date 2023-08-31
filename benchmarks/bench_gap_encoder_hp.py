"""
Benchmark hyperparameters of GapEncoder on traffic_violations dataset
"""

from utils import default_parser, find_result, monitor
from time import perf_counter
import numpy as np
import pandas as pd
from skrub.datasets import fetch_traffic_violations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from skrub import GapEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

#######################################################
# Benchmarking accuracy and speed on traffic_violations
#######################################################

benchmark_name = "gap_encoder_benchmark_hp"


@monitor(
    memory=True,
    time=True,
    parametrize={
        "high_card_feature": [
            "seqid",
            "description",
            "location",
            "search_reason_for_stop",
            "state",
            "charge",
            "driver_city",
            "driver_state",
            "dl_state",
        ],
        "batch_size": [128, 512, 1024],
        "max_iter_e_step": [1, 3, 5, 10],
        "max_rows": [5_000, 20_000, 100_000],
        "max_no_improvement": [5, 10, 20],
        "random_state": [1, 2, 3],
    },
    save_as=benchmark_name,
    repeat=1,
)
def benchmark(
    high_card_feature: str,
    batch_size: int,
    max_iter_e_step: int,
    max_rows: int,
    max_no_improvement: int,
    random_state: int,
):
    X = np.array(ds.X[high_card_feature]).reshape(-1, 1).astype(str)
    y = ds.y
    # only keep the first max_rows rows
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X[:max_rows], y[:max_rows], test_size=0.2, random_state=random_state
    )

    gap = GapEncoder(
        batch_size=batch_size,
        max_iter_e_step=max_iter_e_step,
        max_no_improvement=max_no_improvement,
        random_state=random_state,
    )

    start_time = perf_counter()
    gap.fit(X_train)
    end_time = perf_counter()
    score_train = gap.score(X_train)
    score_test = gap.score(X_test)

    # evaluate the accuracy using the encoding
    X_train_encoded = gap.transform(X_train)
    X_test_encoded = gap.transform(X_test)

    clf = HistGradientBoostingClassifier()
    clf.fit(X_train_encoded, y_train)
    roc_auc_hgb_train = roc_auc_score(
        y_train, clf.predict_proba(X_train_encoded), multi_class="ovr"
    )
    roc_auc_hgb_test = roc_auc_score(
        y_test, clf.predict_proba(X_test_encoded), multi_class="ovr"
    )
    balanced_accuracy_hgb_train = balanced_accuracy_score(
        y_train, clf.predict(X_train_encoded)
    )
    balanced_accuracy_hgb_test = balanced_accuracy_score(
        y_test, clf.predict(X_test_encoded)
    )

    res_dic = {
        "time_fit": end_time - start_time,
        "score_train": score_train,
        "score_test": score_test,
        "roc_auc_hgb_train": roc_auc_hgb_train,
        "roc_auc_hgb_test": roc_auc_hgb_test,
        "balanced_accuracy_hgb_train": balanced_accuracy_hgb_train,
        "balanced_accuracy_hgb_test": balanced_accuracy_hgb_test,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
    }

    return res_dic


def plot(df: pd.DataFrame):
    base_values = {"batch_size": 1024, "max_iter_e_step": 1, "max_no_improvement": 5}
    for variable in base_values.keys():
        df_to_plot = df
        for other_variable in base_values.keys():
            if other_variable != variable:
                df_to_plot = df_to_plot[
                    df_to_plot[other_variable] == base_values[other_variable]
                ]
        # 2 subplots with a shared legend
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.lineplot(
            x="max_rows",
            y="score_train",
            hue="high_card_feature",
            style=variable,
            data=df_to_plot,
            ax=ax1,
            alpha=0.5,
        )
        ax1.get_legend().remove()
        sns.lineplot(
            x="max_rows",
            y="time_fit",
            hue="high_card_feature",
            style=variable,
            data=df_to_plot,
            ax=ax2,
            alpha=0.5,
        )
        # log scale
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        fig.suptitle(f"Effect of {variable} on score_train and time_fit")
        # legend outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # tight layout
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    _args = ArgumentParser(
        description="Benchmark for the GapEncoder's hp",
        parents=[default_parser],
    ).parse_args()

    ds = fetch_traffic_violations()

    if _args.run:
        logger.info("Running benchmark")
        df = benchmark()
    else:
        result_file = find_result(benchmark_name)
        df = pd.read_parquet(result_file)

    if _args.plot:
        plot(df)
