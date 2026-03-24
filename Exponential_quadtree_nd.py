import numpy as np
from kmeans_pp_nd import compute_kmeans_cost


def exponential_quadtree_coreset(X, centers, eps, random_state=None):
    """
    Build an exponential quadtree-style coreset for arbitrary dimensions.

    For depth i (root has depth 0), a hypercube is split into 2^d sub-hypercubes
    if it contains at least 2**i points. Otherwise it becomes a leaf and we:
      - choose one point in the hypercube uniformly at random
      - assign it a weight equal to the number of points in the hypercube

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, d)
        Data points in d dimensions.
    centers : np.ndarray of shape (k, d)
        The centers to use for cost calculation.
    eps : float
        Approximation parameter.
    random_state : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    reps : np.ndarray of shape (n_leaves, d)
        Representative points for each leaf hypercube.
    weights : np.ndarray of shape (n_leaves,)
        Weights = number of original points in that leaf hypercube.
    hypercubes : list of tuples
        Each tuple is (min_dim1, max_dim1, min_dim2, max_dim2, ..., min_dimd, max_dimd) for a leaf hypercube.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.asarray(X)
    n, d = X.shape
    if n == 0:
        return np.empty((0, d)), np.empty((0,)), []

    # Compute bounding box
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    # Make the bounding box a hypercube by extending each dimension to match the max range
    ranges = maxs - mins
    max_range = np.max(ranges)
    extended_maxs = mins + max_range
    maxs = np.where(ranges < max_range, extended_maxs, maxs)

    n = X.shape[0]
    k = centers.shape[0]

    cost = compute_kmeans_cost(X, centers)

    reps = []
    weights = []
    hypercubes = []

    def recurse(indices, bounds_min, bounds_max, depth):
        m = indices.size
        # Even if there are no points, we still want this hypercube
        # to appear in the visual grid (though visualization might be tricky in high D)
        if m == 0:
            hypercube = []
            for i in range(d):
                hypercube.extend([bounds_min[i], bounds_max[i]])
            hypercubes.append(tuple(hypercube))
            return

        # Threshold calculation: generalize from 2D formula
        # Original: (cost / (side_length/(2**depth))) * (eps ** (2 * d + 1) / (4 * k * np.log(n) ** 2))
        # side_length is now max_range / (2**depth) for the current hypercube size
        current_side = max_range / (2 ** depth)
        threshold = (cost / current_side) * (eps ** (2 * d + 1) / (4 * k * np.log(n) ** 2))

        # If not enough points to justify splitting at this depth, make a leaf
        if m < threshold:
            chosen_idx = np.random.choice(indices)
            reps.append(X[chosen_idx])
            weights.append(m)
            hypercube = []
            for i in range(d):
                hypercube.extend([bounds_min[i], bounds_max[i]])
            hypercubes.append(tuple(hypercube))
            return

        # Split current hypercube into 2^d sub-hypercubes
        midpoints = 0.5 * (bounds_min + bounds_max)

        # Generate all combinations of splits
        # For each dimension, we can go left (0) or right (1) of midpoint
        num_subcubes = 2 ** d
        for subcube_idx in range(num_subcubes):
            sub_bounds_min = bounds_min.copy()
            sub_bounds_max = bounds_max.copy()
            sub_indices = indices.copy()

            for dim in range(d):
                if (subcube_idx // (2 ** dim)) % 2 == 0:
                    # Left half in this dimension
                    sub_bounds_max[dim] = midpoints[dim]
                    pts_dim = X[sub_indices, dim]
                    mask = pts_dim <= midpoints[dim]
                else:
                    # Right half in this dimension
                    sub_bounds_min[dim] = midpoints[dim]
                    pts_dim = X[sub_indices, dim]
                    mask = pts_dim > midpoints[dim]
                sub_indices = sub_indices[mask]

            next_depth = depth + 1
            recurse(sub_indices, sub_bounds_min, sub_bounds_max, next_depth)

    all_indices = np.arange(n, dtype=int)
    recurse(all_indices, mins, maxs, depth=0)

    return np.vstack(reps), np.asarray(weights, dtype=float), hypercubes