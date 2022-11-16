"""
To be launched with
```bash
py-spy record native -o py-spy-profile.svg -f speedscope -- python bench_minhash.py
```
"""

import pickle
from dirty_cat import MinHashEncoder
from dirty_cat.tests.utils import generate_data


def gen_data():
    """
    Creates three data files:
    - Small: 2K
    - Medium: 100K
    - Big: 2M
    """
    data = {
        "small": 10_000,
        "medium": 100_000,
        "big": 1_000_000,
    }
    for name, size in data.items():
        with open(f"data_{name}.pkl", "wb") as fl:
            pickle.dump(
                generate_data(size),
                fl,
            )


if __name__ == "__main__":
    mh = MinHashEncoder(n_jobs=1, batch=False)

    # gen_data(); exit()

    with open("../data_big.pkl", "rb") as fl:
        X = pickle.load(fl).reshape(-1, 1)

    mh.fit(X).transform(X)
