"""
Gets the results output by `tablevectorizer_tuning.py`
and shows some graphs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


here = Path(__file__).parent

expected_end = "_results.csv"

param_name = "cardinality_threshold"
for file in here.iterdir():
    if file.name.endswith(expected_end):
        df: pd.DataFrame = pd.read_csv(file)
        name = file.name[: -len(expected_end)]

        card_unq = np.unique(df["param_sv__cardinality_threshold"])
        comp_unq = np.unique(df["param_sv__high_card_str_transformer__n_components"])

        card_amnt = len(card_unq)
        comp_amnt = len(comp_unq)

        scores = df["mean_test_score"].to_numpy().reshape(card_amnt, comp_amnt)

        fig, ax = plt.subplots()
        im = ax.imshow(scores)

        # We want to show all ticks...
        ax.set_xticks(np.arange(comp_amnt))
        ax.set_yticks(np.arange(card_amnt))
        # ... and label them with the respective list entries
        ax.set_xticklabels([f"n_components={v}" for v in comp_unq])
        ax.set_yticklabels([f"cardinality={v}" for v in card_unq])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(card_amnt):
            for j in range(comp_amnt):
                text = ax.text(
                    j, i, round(scores[i, j], 3), ha="center", va="center", color="w"
                )

        ax.set_title(name)
        fig.tight_layout()
        plt.show()
