from typing import List

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from skrub._deduplicate import (
    _create_spelling_correction,
    _guess_clusters,
    compute_ngram_distance,
    deduplicate,
)
from skrub.datasets import make_deduplication_data


@pytest.mark.parametrize(
    ["entries_per_category", "prob_mistake_per_letter"],
    [[[500, 100, 1500], 0.05], [[100, 100], 0.02], [[200, 50, 30, 200, 800], 0.01]],
)
def test_deduplicate(
    entries_per_category: List[int],
    prob_mistake_per_letter: float,
    seed: int = 123,
) -> None:
    rng = np.random.RandomState(seed)

    # hard coded to fix ground truth string similarities
    clean_categories = [
        "Example Category",
        "Generic",
        "Random Word",
        "Pretty similar category",
        "Final cluster",
    ]
    n_clusters = len(entries_per_category)
    clean_categories = clean_categories[:n_clusters]
    data = make_deduplication_data(
        clean_categories, entries_per_category, prob_mistake_per_letter, rng
    )
    deduplicated_data = np.array(deduplicate(data, n_clusters=None))
    assert deduplicated_data.shape[0] == len(data)
    recovered_categories = np.unique(deduplicated_data)
    assert recovered_categories.shape[0] == n_clusters
    assert np.isin(clean_categories, recovered_categories).all()
    deduplicated_data = deduplicate(data, n_clusters=n_clusters)
    translation_table = pd.Series(deduplicated_data, index=data)
    translation_table = translation_table[
        ~translation_table.index.duplicated(keep="first")
    ]
    assert np.isin(np.unique(deduplicated_data), recovered_categories).all()
    assert np.alltrue(translation_table[data] == np.array(deduplicated_data))
    deduplicated_other_analyzer = np.array(
        deduplicate(data, n_clusters=n_clusters, analyzer="char")
    )
    unique_other_analyzer = np.unique(deduplicated_other_analyzer)
    assert np.isin(unique_other_analyzer, recovered_categories).all()


def test_compute_ngram_distance() -> None:
    words = np.array(["aac", "aaa", "aaab", "aaa", "aaab", "aaa", "aaab", "aaa"])
    distance = compute_ngram_distance(words)
    distance = squareform(distance)
    assert distance.shape[0] == words.shape[0]
    assert np.allclose(np.diag(distance), 0)
    for un_word in np.unique(words):
        assert np.allclose(distance[words == un_word][:, words == un_word], 0)


def test__guess_clusters() -> None:
    words = np.array(["aac", "aaa", "aaab", "aaa", "aaab", "aaa", "aaab", "aaa"])
    distance = compute_ngram_distance(words)
    Z = linkage(distance, method="average")
    n_clusters = _guess_clusters(Z, distance)
    assert n_clusters == len(np.unique(words))


def test__create_spelling_correction(seed: int = 123) -> None:
    rng = np.random.RandomState(seed)
    n_clusters = 3
    samples_per_cluster = 10
    counts = np.concatenate(
        [rng.randint(0, 100, samples_per_cluster) for _ in range(n_clusters)]
    )
    clusters = (
        np.repeat(np.arange(n_clusters), samples_per_cluster).astype("int").tolist()
    )
    spelling_correction = _create_spelling_correction(
        counts.astype("str"),
        counts,
        clusters,
    )
    # Check that the most common sample per cluster is chosen as the 'correct' spelling
    for n in np.arange(n_clusters):
        assert (
            spelling_correction.values[clusters == n].astype("int")
            == counts[clusters == n].max()
        ).all()
