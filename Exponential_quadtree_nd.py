import numpy as np
from kmeans_pp_nd import compute_kmeans_cost


def exponential_quadtree_coreset(X, centers, eps, random_state=None, compression_ratio=None, tolerance=0.1, max_iter=8):
    """Build an exponential quadtree-style coreset for arbitrary dimensions.

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
    compression_ratio : float or None
        Target fraction of coresets points / original points; if provided, the
        internal threshold constant is adjusted by binary search.
    tolerance : float
        Relative tolerance around compression_ratio (default 0.1 = 10%).
    max_iter : int
        Maximum binary search iterations for the constant.

    Returns
    -------
    reps : np.ndarray of shape (n_leaves, d)
    weights : np.ndarray of shape (n_leaves,)
    hypercubes : list of tuples
        Each tuple is (min_dim1, max_dim1, ..., min_dimd, max_dimd).
    """

    if random_state is not None:
        np.random.seed(random_state)

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    print(f"! EQT start: n={n}, d={d}, eps={eps}, compression_ratio={compression_ratio}")
    if n == 0:
        return np.empty((0, d)), np.empty((0,)), []

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    ranges = maxs - mins
    max_range = np.max(ranges)
    extended_maxs = mins + max_range
    maxs = np.where(ranges < max_range, extended_maxs, maxs)

    k = centers.shape[0]
    cost = compute_kmeans_cost(X, centers)

    def evaluate(constant):
        reps = []
        weights = []
        hypercubes = []

        def recurse(indices, bounds_min, bounds_max, depth):
            m = indices.size
            if m == 0:
                hypercube = []
                for i in range(d):
                    hypercube.extend([bounds_min[i], bounds_max[i]])
                hypercubes.append(tuple(hypercube))
                return

            current_side = max_range / (2 ** depth)
            const_val = constant
            threshold = (
                cost / (current_side ** 2)
            ) * (
                eps ** d / (const_val * k * (np.log(n) + 1.0))
            )

            if m < threshold:
                chosen_idx = np.random.choice(indices)
                reps.append(X[chosen_idx])
                weights.append(float(m))
                hypercube = []
                for i in range(d):
                    hypercube.extend([bounds_min[i], bounds_max[i]])
                hypercubes.append(tuple(hypercube))
                return

            midpoints = 0.5 * (bounds_min + bounds_max)
            num_subcubes = 2 ** d

            for subcube_idx in range(num_subcubes):
                sub_bounds_min = bounds_min.copy()
                sub_bounds_max = bounds_max.copy()
                sub_indices = indices

                for dim in range(d):
                    if (subcube_idx // (2 ** dim)) % 2 == 0:
                        sub_bounds_max[dim] = midpoints[dim]
                        mask = X[sub_indices, dim] <= midpoints[dim]
                    else:
                        sub_bounds_min[dim] = midpoints[dim]
                        mask = X[sub_indices, dim] > midpoints[dim]
                    sub_indices = sub_indices[mask]

                recurse(sub_indices, sub_bounds_min, sub_bounds_max, depth + 1)

        recurse(np.arange(n, dtype=int), mins, maxs, 0)

        reps_arr = np.vstack(reps) if reps else np.empty((0, d))
        print(f"! EQT evaluate: constant={constant:.6g}, reps={reps_arr.shape[0]}, ratio={reps_arr.shape[0] / n:.6f}")
        return reps_arr, np.asarray(weights, dtype=float), hypercubes

    if compression_ratio is None:
        result = evaluate(4.0)
        print("! EQT done: used default constant=4")
        return result

    if not (0 < compression_ratio <= 1.0):
        raise ValueError("compression_ratio must be in (0, 1]")

    target = compression_ratio
    lo, hi = 1e-6, 1e6
    best = None

    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        reps_arr, weights_arr, hypercubes = evaluate(mid)
        ratio = reps_arr.shape[0] / n
        print(f"! EQT tune iter={it} constant={mid:.6g} ratio={ratio:.6f} target={target:.6f}")

        if best is None or abs(ratio - target) < abs(best[0].shape[0] / n - target):
            best = (reps_arr, weights_arr, hypercubes)

        if abs(ratio - target) <= target * tolerance:
            break

        if ratio > target:
            hi = mid
        else:
            lo = mid

    print(f"! EQT done: best reps={best[0].shape[0]}, ratio={best[0].shape[0] / n:.6f}, target={target:.6f}")
    return best
