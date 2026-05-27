import numpy as np
from scipy.spatial.distance import cdist


def kmeans_cost(X, centers, weights=None):
    squared_distances = cdist(X, centers, "sqeuclidean")
    nearest_distances = squared_distances.min(axis=1)
    if weights is not None:
        weighted = weights * nearest_distances
        return float(np.sum(weighted))
    return float(np.sum(nearest_distances))
