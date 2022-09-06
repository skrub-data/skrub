"""
Implements deduplication based on clustering string similarity matrices.
This works best if there is a number of underlying categories that
sometimes appear in the data with small variations and/or misspellings.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import silhouette_score


def deduplicate(
    data: Sequence[str],
    n_clusters: Optional[int] = None,
    ngram_range: Tuple[int, int] = (2, 4),
    analyzer: str = "char_wb",
    method: str = "average",
) -> List[str]:
    """Deduplicates data by computing the n-gram similarity between unique
    categories in data, performing hierarchical clustering on this similarity
    matrix, and choosing the most frequent element in each cluster as the
    'correct' spelling. This method works best if the true number of
    categories is significantly smaller than the number of observed spellings.

    Parameters
    ----------
    data : Sequence[str]
        The data to be deduplicated.
    n_clusters : Optional[int], optional
        number of clusters to use for hierarchical clustering, if None use the
        number of clusters that lead to the lowest silhouette score,
        by default None
    ngram_range : Tuple[int, int], optional
        range to use for computing n-gram similarity, by default (2, 4)
    analyzer : str, optional
        `CountVectorizer` analyzer for computing n-grams, by default "char_wb"
    method : str, optional
        linkage method to use for merging clusters, by default "average"

    Returns
    -------
    List[str]
       The deduplicated data.
    """
    unique_examples, counts = np.unique(data, return_counts=True)
    ex_series = pd.Series(counts, index=unique_examples)
    similarity_mat = compute_ngram_similarity(
        unique_examples, ngram_range=ngram_range, analyzer=analyzer
    )

    dense_distance = similarity_mat
    Z = linkage(dense_distance, method=method, optimal_ordering=True)
    if n_clusters is None:
        n_clusters = guess_clusters(Z, similarity_mat)
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    pd_spell_correct = create_spelling_correction(ex_series, clusters)
    unrolled_corrections = pd_spell_correct[data]
    return unrolled_corrections


def guess_clusters(Z: NDArray, similarity_mat: NDArray) -> int:
    """Finds the number of clusters that maximize the silhouette score
    when clustering `similarity_mat`.

    Parameters
    ----------
    Z : NDArray
        hierarchical linkage matrix, specifies which clusters to merge.
    similarity_mat : NDArray
        similarity matrix either in square or condensed form.

    Returns
    -------
    int
        number of clusters that maximize the silhouette score.
    """
    max_clusters = similarity_mat.shape[0]
    n_clusters = np.arange(2, max_clusters)
    # silhouette score needs a redundant distance matrix
    redundant_dist = squareform(similarity_mat)
    silhouette_scores = []
    for n_clust in n_clusters:
        labels = fcluster(Z, n_clust, criterion="maxclust")
        silhouette_avg = silhouette_score(redundant_dist, labels,
                                          metric="precomputed")
        silhouette_scores.append(silhouette_avg)
    return n_clusters[np.argmax(silhouette_scores)]


def create_spelling_correction(
    count_series: pd.Series, clusters: Sequence[int]
) -> pd.Series:
    """Creates a pandas Series that map each cluster member to the most
    frequent cluster member. The assumption is that the most common spelling
    is the correct one.

    Parameters
    ----------
    count_series : pd.Series
        A series with unique words (in the original data) as indices and number
        of occurrences of each word in the original data as values.
    clusters : Sequence[int]
        A sequence of ints, indicating cluster membership of each unique word
        in `count_series`.

    Returns
    -------
    pd.Series
        Series with unique (original) words as indices and (estimated)
        corrected spelling of each word as values.
    """
    original_spelling: List[str] = []
    corrected_spelling: List[str] = []
    for cluster in np.unique(clusters):
        sorted_spellings = (
            count_series.loc[clusters == cluster]
            .sort_values(ascending=False)
            .index.values
        )
        original_spelling.extend(sorted_spellings.tolist())
        # assumes spelling that occurs most frequently in cluster is correct
        corrected_spelling.extend(
            np.repeat(sorted_spellings[0], sorted_spellings.shape[0])
        )
    pd_spell_correct = pd.Series(corrected_spelling, index=original_spelling)
    return pd_spell_correct


def compute_ngram_similarity(
    unique_words: Sequence[str],
    ngram_range: Tuple[int, int] = (2, 4),
    analyzer: str = "char_wb",
) -> NDArray:
    """Computes the condensed n-gram similarity matrix between words in
    `unique_words`, using `CountVectorizer` and `TfidfTransformer`.

    Parameters
    ----------
    unique_words : Sequence[str]
        Sequence of unique words from the original data.
    ngram_range : Tuple[int, int], optional
        The n-gram range to compute the similarity in, by default (2, 4)
    analyzer : str, optional
        Analyzer to extract n-grams, by default "char_wb"

    Returns
    -------
    NDArray
        An n-by-(n-1)/2 matrix of n-gram similarities between `unique_words`.
    """
    enc = CountVectorizer(ngram_range=ngram_range, analyzer=analyzer)
    encoded = TfidfTransformer().fit_transform(enc.fit_transform(unique_words))

    similarity_mat = pdist(-encoded.todense(), metric="euclidean")
    return similarity_mat
