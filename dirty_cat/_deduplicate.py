"""
Implements deduplication based on clustering string distance matrices.
This works best if there is a number of underlying categories that
sometimes appear in the data with small variations and/or misspellings.
"""
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def compute_ngram_distance(
    unique_words: Union[Sequence[str], np.ndarray],
    ngram_range: Tuple[int, int] = (2, 4),
    analyzer: str = "char_wb",
) -> np.ndarray:
    """Compute the condensed pair-wise n-gram distance between `unique_words`.

    Parameters
    ----------
    unique_words : Union[Sequence[str], np.ndarray]
        Sequence or array of unique words from the original data.
    ngram_range : Tuple[int, int], optional, default=(2,4)
        The n-gram range to compute the distance in.
    analyzer : str, optional, default=`char_wb`
        Analyzer to extract n-grams.

    Returns
    -------
    np.ndarray
        An n-times-(n-1)/2 array of n-gram tf-idf distances between `unique_words`.

    Notes
    -----
    Extracts n-grams of all elements in `unique_words`, calculates the
    term frequency-inverse document frequency (tf-idf) for each n-gram, then
    computes the pair-wise euclidean distance between elements based on their n-gram
    tf-idf representation.
    """
    encoded = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer).fit_transform(
        unique_words
    )

    distance_mat = pdist(encoded.todense(), metric="euclidean")
    return distance_mat


def _guess_clusters(Z: np.ndarray, distance_mat: np.ndarray) -> int:
    """Finds the number of clusters that maximize the silhouette score
    when clustering `distance_mat`.

    Parameters
    ----------
    Z : np.ndarray
        hierarchical linkage matrix, specifies which clusters to merge.
    distance_mat : np.ndarray
        distance matrix either in square or condensed form.

    Returns
    -------
    int
        number of clusters that maximize the silhouette score.
    """
    max_clusters = distance_mat.shape[0]
    n_clusters = np.arange(2, max_clusters)
    # silhouette score needs a redundant distance matrix
    redundant_dist = squareform(distance_mat)
    silhouette_scores = []
    for n_clust in n_clusters:
        labels = fcluster(Z, n_clust, criterion="maxclust")
        silhouette_avg = silhouette_score(redundant_dist, labels, metric="precomputed")
        silhouette_scores.append(silhouette_avg)
    return n_clusters[np.argmax(silhouette_scores)]


def _create_spelling_correction(
    unique_words: Union[Sequence[str], np.ndarray],
    counts: Union[Sequence[int], np.ndarray],
    clusters: Sequence[int],
) -> pd.Series:
    """Creates a pandas Series that map each cluster member to the most
    frequent cluster member. The assumption is that the most common spelling
    is the correct one.

    Parameters
    ----------
    unique_words : Union[Sequence[str], np.ndarray]
        A sequence or array of unique words in the original data.
    counts : Union[Sequence[int], np.ndarray]
        A sequence or array of counts of how often each unique word appears in
        the original data.
    clusters : Sequence[int]
        A sequence of ints, indicating cluster membership of each unique word
        in `count_series`.

    Returns
    -------
    pd.Series
        Series with unique (original) words as indices and (estimated)
        corrected spelling of each word as values.
    """
    count_series = pd.Series(counts, index=unique_words)
    original_spelling: List[str] = []
    corrected_spelling: List[str] = []
    for cluster in np.unique(clusters):
        sorted_spellings = (
            count_series.loc[clusters == cluster]
            .sort_values(ascending=False)
            .index.tolist()
        )
        original_spelling.extend(sorted_spellings)
        # assumes spelling that occurs most frequently in cluster is correct
        corrected_spelling.extend(sorted_spellings[:1] * len(sorted_spellings))
    pd_spell_correct = pd.Series(corrected_spelling, index=original_spelling)
    return pd_spell_correct


def deduplicate(
    data: Sequence[str],
    n_clusters: Optional[int] = None,
    ngram_range: Tuple[int, int] = (2, 4),
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    method: Literal[
        "single", "complete", "average", "centroid", "median", "ward"
    ] = "average",
) -> List[str]:
    """Deduplicate data by hierarchically clustering similar strings.

    Parameters
    ----------
    data : Sequence[str]
        The data to be deduplicated.
    n_clusters : Optional[int], optional, default=None
        Number of clusters to use for hierarchical clustering, if None use the
        number of clusters that lead to the lowest silhouette score.
    ngram_range : Tuple[int, int], optional, default=(2, 4)
        Range to use for computing n-gram distance.
    analyzer : typing.Literal["word", "char", "char_wb"], optional, default=`char_wb`
        Analyzer parameter for the CountVectorizer used for the string
        similarities.
        Options: {`word`, `char`, `char_wb`}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option `char_wb` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    method : str, optional, default=`average`
        Linkage method parameter to use for merging clusters via scipy's
        `linkage` method.
        Options: {`single`, `complete`, `average`, `centroid`, `median`, `ward`},
        describing different methods to calculate the distance between two clusters.
        Option `average` calculates the distance between two clusters as the average
        distance between data points in the first and second cluster.

    Returns
    -------
    List[str]
       The deduplicated data.

    See Also
    --------
    :class:`~dirty_cat.GapEncoder` :
        Encodes dirty categories (strings) by constructing latent topics with continuous encoding.
    :class:`~dirty_cat.MinHashEncoder` :
        Encode string columns as a numeric array with the minhash method.
    :class:`~dirty_cat.SimilarityEncoder` :
        Encode string columns as a numeric array with n-gram string similarity.

    Notes
    -----
    Deduplication is done by first computing the n-gram distance between unique
    categories in data, then performing hierarchical clustering on this distance
    matrix, and choosing the most frequent element in each cluster as the
    'correct' spelling. This method works best if the true number of
    categories is significantly smaller than the number of observed spellings.

    Examples
    --------
    >>> from dirty_cat.datasets import make_deduplication_data
    >>> duplicated = make_deduplication_data(examples=['black', 'white'],
                                             entries_per_example=[5, 5],
                                             prob_mistake_per_letter=0.3,
                                             random_state=42)

    >>> duplicated
    ['blacn', 'black', 'black', 'black', 'black',
     'hvite', 'white', 'white', 'white', 'white']

    To deduplicate the data, we can build a correspondance matrix:

    >>> deduplicate_correspondence = deduplicate(duplicated)
    >>> deduplicate_correspondence
    blacn    black
    black    black
    black    black
    black    black
    black    black
    hvite    white
    white    white
    white    white
    white    white
    white    white
    dtype: object

    The translation table above is actually a series, giving the deduplicated values,
    and indexed by the original values.
    A deduplicated version of the initial list can easily be created:

    >>> deduplicated = list(deduplicate_correspondence)
    >>> deduplicated
    ['black', 'black', 'black', 'black', 'black',
    'white', 'white', 'white', 'white', 'white']

    We have our dirty categories deduplicated.
    """
    unique_words, counts = np.unique(data, return_counts=True)
    distance_mat = compute_ngram_distance(
        unique_words, ngram_range=ngram_range, analyzer=analyzer
    )

    Z = linkage(distance_mat, method=method, optimal_ordering=True)
    if n_clusters is None:
        n_clusters = _guess_clusters(Z, distance_mat)
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    translation_table = _create_spelling_correction(unique_words, counts, clusters)
    unrolled_corrections = translation_table[data]
    return unrolled_corrections
