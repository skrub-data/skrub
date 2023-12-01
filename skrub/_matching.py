import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


class Matching(BaseEstimator):
    """Base class for fuzzy-join matching & distance rescaling.

    This class is a helper for the ``Joiner`` and ``fuzzy_join``. It is
    provided with the main and auxiliary tables to join. It computes for each
    row in the main table its nearest neighbor in the auxiliary table. It also
    computes a reference distance, by which distances between a row and its
    nearest match are rescaled. This base class does not apply any rescaling:
    the reference distance is 1.0. Subclasses can override
    ``_get_reference_distances`` to modify the rescaling behavior.
    """

    def fit(self, aux):
        self.aux_ = aux
        self.neighbors_ = NearestNeighbors(n_neighbors=1).fit(aux)
        return self

    def match(self, main, max_dist):
        distances, indices = self.neighbors_.kneighbors(main, return_distance=True)
        distances, indices = distances.ravel(), indices.ravel()
        reference_distances = self._get_reference_distances(main, indices)
        rescaled_distances = self._rescale_distances(distances, reference_distances)
        return {
            "index": indices,
            "distance": distances,
            "rescaled_distance": rescaled_distances,
            "match_accepted": rescaled_distances <= max_dist,
        }

    def _get_reference_distances(self, main, indices):
        # Here we do no rescaling so the parameters are there because
        # subclasses overriding this function might need them but we do not use
        # them.
        del main, indices
        return 1.0

    def _rescale_distances(self, distances, reference_distances):
        reference_distances = np.asarray(reference_distances)
        ref_is_zero = reference_distances == 0.0
        rescaled_distances = np.zeros_like(distances)
        rescaled_distances[~ref_is_zero] = (
            distances[~ref_is_zero] / reference_distances[~ref_is_zero]
        )
        rescaled_distances[ref_is_zero] = np.inf
        rescaled_distances[distances == 0] = 0.0
        return rescaled_distances


def _sample_pairs(n_items, n_pairs, random_state):
    assert n_items > 1
    assert n_pairs > 0
    parts = []
    n_found = 0
    while n_found < n_pairs:
        new_part = random_state.randint(n_items, size=(n_pairs, 2))
        new_part = new_part[new_part[:, 0] != new_part[:, 1]]
        parts.append(new_part)
        n_found += new_part.shape[0]
    return np.concatenate(parts, axis=0)[:n_pairs]


class RandomPairs(Matching):
    """Fuzzy joining with rescaling by 1st quartile distance of random auxiliary rows.

    Pairs of (different) rows are sampled randomly from the auxiliary table,
    and the distance that separates each pair is computed. The reference
    distance is a percentile (by default 25%) of the resulting distances.
    """

    def __init__(self, percentile=25.0, n_sampled_pairs=500, random_state=0):
        self.percentile = percentile
        self.n_sampled_pairs = n_sampled_pairs
        self.random_state = random_state

    def fit(self, aux):
        super().fit(aux)
        self._check_inputs()
        n_rows = self.aux_.shape[0]
        pairs = _sample_pairs(n_rows, self.n_sampled_pairs, self.random_state_)
        diff = self.aux_[pairs[:, 0]] - self.aux_[pairs[:, 1]]
        if sparse.issparse(self.aux_):
            distances = sparse.linalg.norm(diff, axis=1)
        else:
            distances = np.linalg.norm(diff, axis=1)
        self.ref_dist_ = np.percentile(distances, self.percentile)
        return self

    def _get_reference_distances(self, main, indices):
        # Only the self.aux_ table is needed for this distance; the parameters
        # are there for compatibility and unused.
        del main, indices
        return self.ref_dist_

    def _check_inputs(self):
        if self.n_sampled_pairs <= 0:
            raise ValueError("n_sampled_pairs must be a positive integer")
        n_rows = self.aux_.shape[0]
        if n_rows <= 1:
            raise ValueError(
                "Cannot estimate the distribution of distances between rows  of a"
                f" table with only {n_rows} rows"
            )
        self.random_state_ = check_random_state(self.random_state)


class SelfJoinNeighbor(Matching):
    """Fuzzy joining rescaling by nearest neighbor to the aux row within the aux table.

    For each match, we take the auxiliary table row (the target of the match)
    and compute its nearest neighbor (excluding itself) within its own table
    (the auxiliary table). This is the reference distance (note that unlike
    some other rescaling strategies the reference distance differs for each
    pair of matching rows).

    Instead of the nearest neighbor, another (more distant neighbor) can be
    chosen by setting ``reference_neighbor``.
    """

    def __init__(self, reference_neighbor=1):
        self.reference_neighbor = reference_neighbor

    def _get_reference_distances(self, main, indices):
        # these parameters are there for compatibility with other distances but
        # are unused here
        del main
        reference_distances, _ = self.neighbors_.kneighbors(
            self.aux_[indices],
            return_distance=True,
            n_neighbors=self.reference_neighbor + 1,
        )
        reference_distances = reference_distances[:, -1]
        return reference_distances


class OtherNeighbor(Matching):
    """Fuzzy joining rescaling by nearest neighbor to the main row within the aux table.

    For each match, we rescale by the distance of the main (query) row to its
    _second_ nearest neighbor in the auxiliary table (the first being the
    actual match candidate). Note that unlike some other rescaling strategies
    the reference distance differs for each pair of matching rows

    Instead of the nearest neighbor, another (more distant neighbor) can be
    chosen by setting ``reference_neighbor``.
    """

    def __init__(self, reference_neighbor=1):
        self.reference_neighbor = reference_neighbor

    def _get_reference_distances(self, main, indices):
        # these parameters are there for compatibility with other distances but
        # are unused here
        del indices
        reference_distances, _ = self.neighbors_.kneighbors(
            main, return_distance=True, n_neighbors=self.reference_neighbor + 1
        )
        reference_distances = reference_distances[:, -1]
        return reference_distances
