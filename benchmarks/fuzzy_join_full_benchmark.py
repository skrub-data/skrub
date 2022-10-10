import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from thefuzz import process
from autofj import AutoFJ
from thefuzz.fuzz import partial_ratio, WRatio, ratio
from fuzzy_join_benchmark import (
    fetch_data,
    fuzzy_join_precision_recall,
    thefuzz_precision_recall,
    autofj_precision_recall,
)
from dirty_cat._fuzzy_join import fuzzy_join


datasets = [
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
]
df = pd.DataFrame()
for dataset in datasets:
    left_1, right_1, gt_1 = fetch_data(dataset)
    for analyser, max_n_gram in product(["word", "char", "char_wb"], [3, 4]):
        if analyser == "word" and max_n_gram > 2:
            continue
        model_name = f"fuzzy_join_{analyser}_{max_n_gram}"
        precision, recall, f1 = fuzzy_join_precision_recall(
            right_1,
            left_1,
            gt_1,
            "title",
            "title",
            analyzer=analyser,
            ngram_range=(2, max_n_gram),
        )
        n_points = len(precision)
        df = df.append(
            pd.DataFrame.from_dict(
                {
                    "dataset": [dataset] * n_points,
                    "model": [model_name] * n_points,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
                orient="index",
            ).transpose(),
            ignore_index=True,
        )

    for scorer in partial_ratio, ratio, WRatio:
        model_name = f"thefuzz_{scorer.__name__}"
        precision_fw, recall_fw, f1_fw = thefuzz_precision_recall(
            left_1, right_1, gt_1, "title", "title", scorer=scorer
        )
        n_points = len(precision_fw)
        df = df.append(
            pd.DataFrame.from_dict(
                {
                    "dataset": [dataset] * n_points,
                    "model": [model_name] * n_points,
                    "precision": precision_fw,
                    "recall": recall_fw,
                    "f1": f1_fw,
                },
                orient="index",
            ).transpose(),
            ignore_index=True,
        )

    precision_fj, recall_fj, f1_fj = autofj_precision_recall(
        left_1, right_1, gt_1, n_points=10
    )
    model_name = "autofj_default"
    n_points = len(precision_fj)
    df = df.append(
        pd.DataFrame.from_dict(
            {
                "dataset": [dataset] * n_points,
                "model": [model_name] * n_points,
                "precision": precision_fj,
                "recall": recall_fj,
                "f1": f1_fj,
            },
            orient="index",
        ).transpose(),
        ignore_index=True,
    )

df.to_csv("full_benchmark_2.csv")
