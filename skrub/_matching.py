import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors


class Threshold(BaseEstimator):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, aux, main):
        del main
        self._neighbors = NearestNeighbors(n_neighbors=1).fit(aux)
        return self

    def match(self, main):
        distances, indices = self._neighbors.kneighbors(main, return_distance=True)
        distances, indices = distances.ravel(), indices.ravel()
        return {
            "nearest_neighbor_index": indices,
            "nearest_neighbor_distance": distances,
            "match_accepted": distances <= self.threshold,
        }

    def fit_match(self, aux, main):
        return self.fit(aux, main).match(main)


class TargetNeighborhood(BaseEstimator):
    def __init__(self, radius=1.0, tolerated_neighbors=0):
        self.radius = radius
        self.tolerated_neighbors = tolerated_neighbors

    def fit(self, aux, main):
        del main
        self._aux = aux
        self._neighbors = NearestNeighbors(
            n_neighbors=self.tolerated_neighbors + 2
        ).fit(self._aux)
        return self

    def match(self, main):
        distances, indices = self._neighbors.kneighbors(
            main, return_distance=True, n_neighbors=1
        )
        distances, indices = distances.ravel(), indices.ravel()
        competing_distances, _ = self._neighbors.kneighbors(
            self._aux[indices], return_distance=True
        )
        competing_distances = competing_distances[:, -1]
        accept = distances * self.radius < competing_distances.ravel()
        return {
            "nearest_neighbor_index": indices,
            "nearest_neighbor_distance": distances,
            "match_accepted": accept,
        }

    def fit_match(self, aux, main):
        return self.fit(aux, main).match(main)


class QueryNeighborhood(BaseEstimator):
    def __init__(self, radius=1.0, tolerated_neighbors=0):
        self.radius = radius
        self.tolerated_neighbors = tolerated_neighbors

    def fit(self, aux, main):
        del main
        self._neighbors = NearestNeighbors(
            n_neighbors=self.tolerated_neighbors + 2
        ).fit(aux)
        return self

    def match(self, main):
        distances, indices = self._neighbors.kneighbors(main, return_distance=True)
        competing_distances = distances[:, -1]
        distances, indices = distances[:, 0], indices[:, 0]
        accept = distances * self.radius < competing_distances
        return {
            "nearest_neighbor_index": indices,
            "nearest_neighbor_distance": distances,
            "match_accepted": accept,
        }

    def fit_match(self, aux, main):
        return self.fit(aux, main).match(main)


class MaxDistRescale(BaseEstimator):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, aux, main):
        self.fit_match(aux, main)
        return self

    def fit_match(self, aux, main):
        self._neighbors = NearestNeighbors(n_neighbors=1).fit(aux)
        distances, indices = self._neighbors.kneighbors(main, return_distance=True)
        distances, indices = distances.ravel(), indices.ravel()
        self.max_dist_ = distances.max()
        if self.max_dist_ != 0:
            distances /= self.max_dist_
        return {
            "nearest_neighbor_index": indices,
            "nearest_neighbor_distance": distances,
            "match_accepted": distances <= self.threshold,
        }

    def match(self, main):
        distances, indices = self._neighbors.kneighbors(main, return_distance=True)
        distances, indices = distances.ravel(), indices.ravel()
        if self.max_dist_ != 0:
            distances /= self.max_dist_
        else:
            distances[distances != 0] = np.inf
        return {
            "nearest_neighbor_index": indices,
            "nearest_neighbor_distance": distances,
            "match_accepted": distances <= self.threshold,
        }
