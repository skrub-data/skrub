# Inspired from :
# https://github.com/scikit-learn/scikit-learn/blob/main/benchmarks/bench_text_vectorizers.py
# plot run time and memory usage of the different encoders using the TableVectorizer method
# test if auto_cast=True plays an important role

"""

To run this benchmark, you will need,

 * dirty_cat
 * pandas
 * memory_profiler

"""
import timeit
import itertools

import numpy as np
import pandas as pd
from memory_profiler import memory_usage

from dirty_cat.datasets import fetch_traffic_violations

from dirty_cat import SimilarityEncoder, GapEncoder, MinHashEncoder
from dirty_cat import TableVectorizer

n_repeat = 3


def run_encoders(Encoder, X, **params):
    def f():
        enc = Encoder(**params)
        enc.fit_transform(X)

    return f


data = fetch_traffic_violations()

X = data[1][:5000]
y = data[2][:5000]

print("=" * 80 + "\n#" + "    Dirty_cat encoders benchmark" + "\n" + "=" * 80 + "\n")
print(f"Using a subset of the traffic violations dataset ({len(X)} observations).")
print("This benchmarks runs in ~1 min ...")

res = []

for Encoder, (auto_cast, high_card_cat_transformer) in itertools.product(
    [TableVectorizer],
    [
        (True, SimilarityEncoder()),
        (True, GapEncoder()),
        (True, MinHashEncoder()),
        (False, SimilarityEncoder()),
        (False, GapEncoder()),
        (False, MinHashEncoder()),
    ],
):
    params = {
        "auto_cast": auto_cast,
        "high_card_cat_transformer": high_card_cat_transformer,
    }
    bench = {"encoder": Encoder}
    bench.update(params)
    dt = timeit.repeat(run_encoders(Encoder, X, **params), number=1, repeat=n_repeat)
    bench["time"] = f"{np.mean(dt):.3f} (+-{np.std(dt):.3f})"

    mem_usage = memory_usage(run_encoders(Encoder, X, **params))

    bench["memory"] = f"{np.max(mem_usage):.1f}"

    res.append(bench)

df = pd.DataFrame(res).set_index(["auto_cast", "high_card_cat_transformer", "encoder"])

print("\n========== Run time performance (sec) ===========\n")
print(
    "Computing the mean and the standard deviation "
    f"of the run time over {n_repeat} runs...\n"
)
print(df["time"].unstack(level=-1))

print("\n=============== Memory usage (MB) ===============\n")
print(df["memory"].unstack(level=-1))
