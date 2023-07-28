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


#########################################################
# Benchmarking accuracy and speed on actual traffic_violations dataset
#########################################################

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
        "max_iter_e_step": [10, 20, 30],
        "max_rows": [5_000, 20_000, 100_000],
    },
    save_as=benchmark_name,
    repeat=1,
)
def benchmark(
    high_card_feature: str,
    batch_size: int,
    max_iter_e_step: int,
    max_rows: int,
):
    print(f"Running benchmark")
    ds = fetch_traffic_violations()
    X = np.array(ds.X[high_card_feature]).reshape(-1, 1).astype(str)
    y = ds.y
    # only keep the first max_rows rows
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X[:max_rows], y[:max_rows], test_size=0.2, random_state=42
    )

    gap = GapEncoder(
        batch_size=batch_size,
        max_iter_e_step=max_iter_e_step,
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

    print("Done")
    print(res_dic)

    return res_dic


def plot(df: pd.DataFrame):
    # TODO
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser

    _args = ArgumentParser(
        description="Benchmark for the batch feature of the MinHashEncoder.",
        parents=[default_parser],
    ).parse_args()

    if _args.run:
        print("Running benchmark")
        df = benchmark()
    else:
        result_file = find_result(benchmark_name)
        df = pd.read_parquet(result_file)

    if _args.plot:
        plot(df)
