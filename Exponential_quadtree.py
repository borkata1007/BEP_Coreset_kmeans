import numpy as np
from kmeans_pp import compute_kmeans_cost


def exponential_quadtree_coreset(X_2d, centers, eps, random_state=None):
    """
    Build an exponential quadtree-style coreset.

    For depth i (root has depth 0), a square is split into 4 if it contains
    at least 2**i points. Otherwise it becomes a leaf and we:
      - choose one point in the square uniformly at random
      - assign it a weight equal to the number of points in the square

    Parameters
    ----------
    X_2d : np.ndarray of shape (n_samples, 2)
        2D data points.
    centers : np.ndarray of shape (k, 2)
        The centers to use for cost calculation.
    eps : float
        Approximation parameter.
    random_state : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    reps : np.ndarray of shape (n_leaves, 2)
        Representative points for each leaf square.
    weights : np.ndarray of shape (n_leaves,)
        Weights = number of original points in that leaf square.
    squares : list of tuples
        Each tuple is (x0, x1, y0, y1) for a leaf square.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_2d = np.asarray(X_2d)
    n = X_2d.shape[0]
    if n == 0:
        return np.empty((0, 2)), np.empty((0,)), []

    x_min, x_max = np.min(X_2d[:, 0]), np.max(X_2d[:, 0])
    y_min, y_max = np.min(X_2d[:, 1]), np.max(X_2d[:, 1])

    # Make the bounding box a square
    x_range = x_max - x_min
    y_range = y_max - y_min
    side_length = max(x_range, y_range)
    if x_range < side_length:
        x_max = x_min + side_length
    else:
        y_max = y_min + side_length

    n = X_2d.shape[0]
    k = centers.shape[0]
    d = 2  # dimension

    cost = compute_kmeans_cost(X_2d, centers)

    reps = []
    weights = []
    squares = []

    def recurse(indices, x0, x1, y0, y1, depth):
        m = indices.size
        # Even if there are no points, we still want this square
        # to appear in the visual grid.
        if m == 0:
            squares.append((x0, x1, y0, y1))
            return

        #print(cost, depth, m, (eps ** (2 * d + 1) / (4 * k * np.log(n) ** 2)))
        threshold = (cost / (side_length/(2**depth))) * (eps ** (2 * d + 1) / (4 * k * np.log(n) ** 2))
        #print(f"Depth {depth}: {m} points, threshold = {threshold:.4f}")
        # If not enough points to justify splitting at this depth, make a leaf
        if m < threshold:
            chosen_idx = np.random.choice(indices)
            reps.append(X_2d[chosen_idx])
            weights.append(m)
            squares.append((x0, x1, y0, y1))
            return

        # Split current square into 4 sub-squares
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)

        pts = X_2d[indices]
        xs = pts[:, 0]
        ys = pts[:, 1]

        left = xs <= xm
        right = ~left
        bottom = ys <= ym
        top = ~bottom

        # Quadrants: (left, bottom), (left, top), (right, bottom), (right, top)
        lb_idx = indices[left & bottom]
        lt_idx = indices[left & top]
        rb_idx = indices[right & bottom]
        rt_idx = indices[right & top]

        next_depth = depth + 1

        recurse(lb_idx, x0, xm, y0, ym, next_depth)
        recurse(lt_idx, x0, xm, ym, y1, next_depth)
        recurse(rb_idx, xm, x1, y0, ym, next_depth)
        recurse(rt_idx, xm, x1, ym, y1, next_depth)

    all_indices = np.arange(n, dtype=int)
    recurse(all_indices, x_min, x_max, y_min, y_max, depth=0)

    return np.vstack(reps), np.asarray(weights, dtype=float), squares

