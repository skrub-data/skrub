"""
To be launched with
```bash
py-spy record native -o py-spy-profile.svg -f speedscope -- python bench_minhash.py
```
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from utils import monitor
from dirty_cat import MinHashEncoder
from dirty_cat.tests.utils import generate_data


@monitor(
    memory=True,
    time=True,
    parametrize={
        "dataset_size": ["medium"],
        "batched": [True, False],
        "n_jobs": [1, 4, 8, 16, 32, 64],
    },
    save_benchmark_as="minhash_batch_comparison",
    repeat=10,
)
def benchmark(
    dataset_size: str,
    batched: bool,
    n_jobs: int,
):
    X = data[dataset_size]
    MinHashEncoder(batch=batched, n_jobs=n_jobs).fit(X).transform(X)


def plot(res: pd.DataFrame):
    sns.set_theme(style="darkgrid")
    sns.scatterplot(
        data=res,
        x="memory",
        y="time",
        style="call",
    )
    plt.show()


# Generate the data if not already on disk, and keep them in memory.
data = {}  # Will hold the datasets in memory.
_data_info = {
    "small": 10_000,
    "medium": 100_000,
    # "big": 1_000_000,
}
for name, size in _data_info.items():
    data_file = Path(f"data_{name}.pkl")
    if data_file.is_file():
        with data_file.open("rb") as fl:
            data.update({name: pickle.load(fl)})
    else:
        with data_file.open("wb") as fl:
            _gen = generate_data(size).reshape(-1, 1)
            pickle.dump(_gen, fl)
            data.update({name: _gen})


# results = benchmark()
results = pd.read_csv(
    "results/minhash_batch_comparison-20221119-0181acf6fe4933f17ea34ccbc85dca3975c8e152.csv"
)
for value in ["memory", "time"]:
    results[f"{value}_mean"] = results[value].apply(lambda val: np.mean(val))
    results[f"{value}_std"] = results[value].apply(lambda val: np.std(val))
print(results)
# plot(results)
