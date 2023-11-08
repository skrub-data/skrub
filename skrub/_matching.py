import numpy as np
from sklearn.neighbors import NearestNeighbors


class Threshold:
    def __init__(self, aux, threshold):
        self.aux = aux
        self.threshold = threshold

    def fit(self, main):
        del main
        self.neighbors_ = NearestNeighbors(n_neighbors=1).fit(self.aux)
        return self

    def match(self, main):
        distances, indices = self.neighbors_.kneighbors(main, return_distance=True)
        distances, indices = distances.ravel(), indices.ravel()
        return {
            "indices": indices,
            "distances": distances,
            "accept": distances <= self.threshold,
        }

    def fit_match(self, main):
        return self.fit(main).match(main)


class TargetNeighborhood:
    def __init__(self, aux, radius, tolerated_neighbors):
        self.aux = aux
        self.radius = radius
        self.tolerated_neighbors = tolerated_neighbors

    def fit(self, main):
        del main
        self.neighbors_ = NearestNeighbors(
            n_neighbors=self.tolerated_neighbors + 2
        ).fit(self.aux)
        return self

    def match(self, main):
        distances, indices = self.neighbors_.kneighbors(
            main, return_distance=True, n_neighbors=1
        )
        distances, indices = distances.ravel(), indices.ravel()
        competing_distances, _ = self.neighbors_.kneighbors(self.aux[indices])[:, -1]
        accept = distances * self.radius < competing_distances.ravel()
        return {"indices": indices, "distances": distances, "accept": accept}

    def fit_match(self, main):
        return self.fit(main).match(main)


class QueryNeighborhood:
    def __init__(self, aux, radius, tolerated_neighbors):
        self.aux = aux
        self.radius = radius
        self.tolerated_neighbors = tolerated_neighbors

    def fit(self, main):
        del main
        self.neighbors_ = NearestNeighbors(
            n_neighbors=self.tolerated_neighbors + 2
        ).fit(self.aux)
        return self

    def match(self, main):
        distances, indices = self.neighbors_.kneighbors(main, return_distance=True)
        competing_distances = distances[:, -1]
        distances, indices = distances[:, 0], indices[:, 0]
        accept = distances * self.radius < competing_distances
        return {"indices": indices, "distances": distances, "accept": accept}

    def fit_match(self, main):
        return self.fit(main).match(main)


class MaxDistRescale:
    def __init__(self, aux, threshold):
        self.aux = aux
        self.threshold = threshold

    def fit(self, main):
        self.fit_match(main)
        return self

    def fit_match(self, main):
        self.neighbors_ = NearestNeighbors(n_neighbors=1).fit(self.aux)
        distances, indices = self.neighbors_.kneighbors(main, return_distance=True)
        distances, indices = distances.ravel(), indices.ravel()
        self.max_dist_ = distances.max()
        if self.max_dist_ != 0:
            distances /= self.max_dist_
        return {
            "indices": indices,
            "distances": distances,
            "accept": distances <= self.threshold,
        }

    def match(self, main):
        distances, indices = self.neighbors_.kneighbors(main, return_distance=True)
        distances, indices = distances.ravel(), indices.ravel()
        if self.max_dist_ != 0:
            distances /= self.max_dist_
        else:
            distances[distances != 0] = np.inf
        return {
            "indices": indices,
            "distances": distances,
            "accept": distances <= self.threshold,
        }
