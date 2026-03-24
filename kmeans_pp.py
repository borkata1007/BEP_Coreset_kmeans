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


def kmeans_plus_plus_init_weighted(X_2d, weights, k, random_state=None):
    """
    Perform k-means++ initialization on weighted 2D data (e.g., coreset points).

    Parameters
    ----------
    X_2d : np.ndarray of shape (n_samples, 2)
        2D weighted data points (e.g., coreset).
    weights : np.ndarray of shape (n_samples,)
        Per-point weights.
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
    weights = np.asarray(weights)

    # Choose first center uniformly at random (weighted)
    first_idx = np.random.choice(n, p=weights / weights.sum())
    centers = np.array([X_2d[first_idx]])  # shape (1, 2)

    # Initial distances from all points to the first center
    min_distances = np.linalg.norm(X_2d - centers[0], axis=1)  # shape (n,)

    # k-means++ style sampling of remaining centers
    for _ in range(2, k + 1):
        # Compute weighted probabilities proportional to squared distance to closest center
        dist_sq = min_distances ** 2
        weighted_dist_sq = weights * dist_sq
        total_weighted_dist_sq = np.sum(weighted_dist_sq)
        
        if total_weighted_dist_sq == 0:
            # Fallback: use uniform weighted sampling
            probabilities = weights / weights.sum()
        else:
            probabilities = weighted_dist_sq / total_weighted_dist_sq

        # Sample a new center index according to these probabilities
        new_center_index = np.random.choice(n, p=probabilities)
        new_center = X_2d[new_center_index]

        # Add the new center to centers
        centers = np.vstack((centers, new_center))

        # Update min_distances: compare existing min distance with distance to new center
        distances_to_new_center = np.linalg.norm(X_2d - new_center, axis=1)
        min_distances = np.minimum(min_distances, distances_to_new_center)

    return centers


def compute_kmeans_cost(X_2d, centers, weights=None):
    """
    Compute the k-means cost function.

    - If `weights` is None: sum of squared distances from each point to its closest center.
    - If `weights` is provided: weighted sum of squared distances (for coreset).

    Parameters
    ----------
    X_2d : np.ndarray of shape (n_samples, 2)
        Data points.
    centers : np.ndarray of shape (k, 2)
        The centers.
    weights : None or np.ndarray of shape (n_samples,)
        Optional per-point weights.

    Returns
    -------
    cost : float
        The total (weighted) cost.
    """
    if centers.size == 0 or X_2d.size == 0:
        return 0.0

    # Compute distances from each point to each center
    distances = np.linalg.norm(X_2d[:, np.newaxis] - centers[np.newaxis, :], axis=2)  # shape (n, k)
    min_distances_sq = np.min(distances, axis=1) ** 2

    if weights is None:
        return np.sum(min_distances_sq)

    weights = np.asarray(weights)
    if weights.shape[0] != X_2d.shape[0]:
        raise ValueError("weights must have the same number of rows as X_2d")

    return np.sum(min_distances_sq * weights)


def _kmeans_plus_plus_local_search(X_2d, centers, weights, n_steps=150, random_state=None):
    """
    General local search for k-means++ replacement procedure.

    X_2d : np.ndarray (n,2)
    centers: np.ndarray (k,2)
    weights: None or np.ndarray (n,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_2d = np.asarray(X_2d)
    n = X_2d.shape[0]

    if n == 0:
        return np.empty((0, 2)), 0.0

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape[0] != n:
            raise ValueError("weights must have same length as X_2d")

    current_centers = centers.copy()
    current_cost = compute_kmeans_cost(X_2d, current_centers, weights=weights)

    for step in range(n_steps):
        # Compute squared distances to nearest center
        dist_sq = np.min(np.linalg.norm(X_2d[:, np.newaxis] - current_centers[np.newaxis, :], axis=2) ** 2, axis=1)

        if weights is None:
            weighted_dist = dist_sq
        else:
            weighted_dist = weights * dist_sq

        total_weighted_dist = weighted_dist.sum()
        if total_weighted_dist <= 0:
            break

        probabilities = weighted_dist / total_weighted_dist
        q_idx = np.random.choice(n, p=probabilities)
        q_point = X_2d[q_idx]

        best_centers = current_centers
        best_cost = current_cost

        for j in range(current_centers.shape[0]):
            candidate_centers = current_centers.copy()
            candidate_centers[j] = q_point
            candidate_cost = compute_kmeans_cost(X_2d, candidate_centers, weights=weights)

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_centers = candidate_centers

        if best_cost < current_cost:
            current_centers = best_centers
            current_cost = best_cost

    return current_centers, current_cost


def kmeans_plus_plus_local_search_full(X_2d, k, n_steps=150, random_state=None):
    """kmeans++ + local search for unweighted full point set."""
    centers = kmeans_plus_plus_init(X_2d, k, random_state=random_state)
    return _kmeans_plus_plus_local_search(X_2d, centers, weights=None, n_steps=n_steps, random_state=random_state)


def kmeans_plus_plus_local_search_weighted(X_2d, weights, k, n_steps=150, random_state=None):
    """kmeans++ + local search for weighted point set."""
    centers = kmeans_plus_plus_init_weighted(X_2d, weights, k, random_state=random_state)
    return _kmeans_plus_plus_local_search(X_2d, centers, weights=weights, n_steps=n_steps, random_state=random_state)



