import numpy as np


def kmeans_plus_plus_init(X_2d, k, random_state=None):
    """
    Perform k-means++ initialization on 2D data.

    Parameters
    ----------
    X_2d : np.ndarray of shape (n_samples, 2)
        2D data points.
    k : int
        Number of centers to choose.
    random_state : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    centers : np.ndarray of shape (k, 2)
        The chosen centers.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = X_2d.shape[0]

    # Choose first center uniformly at random
    first_idx = np.random.randint(0, n)
    centers = np.array([X_2d[first_idx]])  # shape (1, 2)

    # Initial distances from all points to the first center
    min_distances = np.linalg.norm(X_2d - centers[0], axis=1)  # shape (n,)

    # k-means++ style sampling of remaining centers
    for _ in range(2, k + 1):
        # Compute probabilities proportional to squared distance to closest center
        dist_sq = min_distances ** 2
        total_distance_sq = np.sum(dist_sq)
        probabilities = dist_sq / total_distance_sq

        # Sample a new center index according to these probabilities
        new_center_index = np.random.choice(n, p=probabilities)
        new_center = X_2d[new_center_index]

        # Add the new center to centers (keep shape (num_centers, 2))
        centers = np.vstack((centers, new_center))

        # Update min_distances: compare existing min distance with distance to new center
        distances_to_new_center = np.linalg.norm(X_2d - new_center, axis=1)
        min_distances = np.minimum(min_distances, distances_to_new_center)

    return centers

