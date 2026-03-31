from dataclasses import dataclass

import numpy as np

from kmeans_pp_nd import compute_kmeans_cost


@dataclass
class QuadNode:
    indices: np.ndarray
    count: int
    depth: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    side_length: float
    children: list
    is_empty: bool
    rep_idx: int | None
    midpoints: np.ndarray | None = None


def _bounds_to_hypercube(bounds_min, bounds_max):
    values = []
    for i in range(bounds_min.size):
        values.extend([float(bounds_min[i]), float(bounds_max[i])])
    return tuple(values)


def _make_bounding_cube(X):
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    side = float(np.max(maxs - mins))
    maxs = mins + side
    return mins, maxs, side


def _build_node(
    X,
    indices,
    bounds_min,
    bounds_max,
    depth,
    max_depth,
    rng,
    progress=None,
    progress_interval=20000,
    keep_empty_cells=False,
):
    if progress is not None:
        progress["nodes"] += 1
        if depth > progress["max_depth"]:
            progress["max_depth"] = depth
        if progress["verbose"] and progress["nodes"] % progress_interval == 0:
            print(
                f"! EQT build progress: nodes={progress['nodes']}, current_depth={depth}, max_depth={progress['max_depth']}",
                flush=True,
            )
    count = int(indices.size)
    side = float(bounds_max[0] - bounds_min[0])
    is_empty = count == 0
    rep_idx = None if is_empty else int(rng.choice(indices))
    node = QuadNode(
        indices=indices,
        count=count,
        depth=depth,
        bounds_min=bounds_min.copy(),
        bounds_max=bounds_max.copy(),
        side_length=side,
        children=[],
        is_empty=is_empty,
        rep_idx=rep_idx,
    )

    # Build-only stopping conditions must not depend on beta.
    if is_empty or count <= 1 or side <= 0.0 or (max_depth is not None and depth >= max_depth):
        return node

    d = X.shape[1]
    mid = 0.5 * (bounds_min + bounds_max)
    node.midpoints = mid

    num_subcubes = 2 ** d
    child_nodes = []
    non_empty_count = 0
    max_child_size = 0

    for subcube_idx in range(num_subcubes):
        sub_min = bounds_min.copy()
        sub_max = bounds_max.copy()
        sub_indices = indices

        for dim in range(d):
            if (subcube_idx >> dim) & 1 == 0:
                sub_max[dim] = mid[dim]
                mask = X[sub_indices, dim] <= mid[dim]
            else:
                sub_min[dim] = mid[dim]
                mask = X[sub_indices, dim] > mid[dim]
            sub_indices = sub_indices[mask]

        if keep_empty_cells or sub_indices.size > 0:
            child_nodes.append(
                _build_node(
                    X,
                    sub_indices,
                    sub_min,
                    sub_max,
                    depth + 1,
                    max_depth,
                    rng,
                    progress=progress,
                    progress_interval=progress_interval,
                    keep_empty_cells=keep_empty_cells,
                )
            )
        if sub_indices.size > 0:
            non_empty_count += 1
            if sub_indices.size > max_child_size:
                max_child_size = int(sub_indices.size)

    # Degenerate split guard: if all points fall into a single child, recursion
    # would keep shrinking geometry without improving partition membership.
    if non_empty_count <= 1 or max_child_size == count:
        return node

    if len(child_nodes) >= 2:
        node.children = child_nodes

    return node


def build_exponential_quadtree(
    X,
    max_depth=None,
    random_state=None,
    verbose=False,
    progress_interval=20000,
    keep_empty_cells=False,
):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n == 0:
        return None, {"n": 0, "d": d, "mins": None, "maxs": None, "side": 0.0}

    if max_depth is None:
        max_depth = 64

    rng = np.random.default_rng(random_state)
    mins, maxs, side = _make_bounding_cube(X)
    progress = {
        "nodes": 0,
        "max_depth": 0,
        "verbose": bool(verbose),
    }
    root = _build_node(
        X,
        np.arange(n, dtype=int),
        mins,
        maxs,
        depth=0,
        max_depth=max_depth,
        rng=rng,
        progress=progress,
        progress_interval=progress_interval,
        keep_empty_cells=keep_empty_cells,
    )
    return root, {"n": n, "d": d, "mins": mins, "maxs": maxs, "side": side}


def _tree_stats(root):
    if root is None:
        return {
            "nodes": 0,
            "leaves": 0,
            "empty_leaves": 0,
            "max_depth": 0,
        }

    nodes = 0
    leaves = 0
    empty_leaves = 0
    max_depth = 0
    stack = [root]

    while stack:
        node = stack.pop()
        nodes += 1
        if node.depth > max_depth:
            max_depth = node.depth
        if not node.children:
            leaves += 1
            if node.is_empty:
                empty_leaves += 1
        else:
            stack.extend(node.children)

    return {
        "nodes": nodes,
        "leaves": leaves,
        "empty_leaves": empty_leaves,
        "max_depth": max_depth,
    }


def _threshold_for_node(node, beta, eps, cost, k, n, d):
    if node.side_length <= 0.0:
        return np.inf
    return beta * (cost / (node.side_length ** 2)) * (eps ** d / (k * (np.log(n) + 1.0)))


def _count_coreset_size_from_tree(node, beta, eps, cost, k, n, d):
    if node is None or node.is_empty:
        return 0

    threshold = _threshold_for_node(node, beta, eps, cost, k, n, d)
    if node.count < threshold or not node.children:
        return 1

    total = 0
    for child in node.children:
        total += _count_coreset_size_from_tree(child, beta, eps, cost, k, n, d)
    return total


def _extract_coreset_from_tree(node, beta, eps, cost, k, n, d, X, reps, weights, hypercubes, keep_empty_cells=False):
    if node is None:
        return

    if node.is_empty:
        if keep_empty_cells:
            hypercubes.append(_bounds_to_hypercube(node.bounds_min, node.bounds_max))
        return

    threshold = _threshold_for_node(node, beta, eps, cost, k, n, d)
    if node.count < threshold or not node.children:
        reps.append(X[node.rep_idx])
        weights.append(float(node.count))
        hypercubes.append(_bounds_to_hypercube(node.bounds_min, node.bounds_max))
        return

    for child in node.children:
        _extract_coreset_from_tree(
            child,
            beta,
            eps,
            cost,
            k,
            n,
            d,
            X,
            reps,
            weights,
            hypercubes,
            keep_empty_cells=keep_empty_cells,
        )


def count_coreset_size(root, beta, eps, cost, k, n, d):
    return _count_coreset_size_from_tree(root, beta, eps, cost, k, n, d)


def extract_coreset(root, beta, eps, cost, k, n, d, X, keep_empty_cells=False):
    reps = []
    weights = []
    hypercubes = []
    _extract_coreset_from_tree(
        root,
        beta,
        eps,
        cost,
        k,
        n,
        d,
        X,
        reps,
        weights,
        hypercubes,
        keep_empty_cells=keep_empty_cells,
    )

    if reps:
        reps_arr = np.vstack(reps)
        weights_arr = np.asarray(weights, dtype=float)
    else:
        reps_arr = np.empty((0, d))
        weights_arr = np.empty((0,), dtype=float)

    return reps_arr, weights_arr, hypercubes


def _collect_critical_betas(root, eps, cost, k, n, d):
    values = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or node.is_empty or node.side_length <= 0.0:
            continue
        base = (cost / (node.side_length ** 2)) * (eps ** d / (k * (np.log(n) + 1.0)))
        if base > 0:
            values.append(node.count / base)
        stack.extend(node.children)
    if not values:
        return np.asarray([], dtype=float)
    return np.unique(np.asarray(values, dtype=float))


def _direct_coreset_with_beta(X, eps, cost, k, beta, random_state=None, max_depth=None, keep_empty_cells=False):
    """Direct threshold-based recursion (beta-aware during construction)."""
    n, d = X.shape
    mins, maxs, _ = _make_bounding_cube(X)
    rng = np.random.default_rng(random_state)

    reps = []
    weights = []
    hypercubes = []

    if max_depth is None:
        max_depth = 64

    def recurse(indices, bounds_min, bounds_max, depth):
        m = int(indices.size)
        side_length = float(bounds_max[0] - bounds_min[0])

        if m == 0:
            if keep_empty_cells:
                hypercubes.append(_bounds_to_hypercube(bounds_min, bounds_max))
            return

        if side_length <= 0.0 or depth >= max_depth:
            rep_idx = int(rng.choice(indices))
            reps.append(X[rep_idx])
            weights.append(float(m))
            hypercubes.append(_bounds_to_hypercube(bounds_min, bounds_max))
            return

        threshold = beta * (cost / (side_length ** 2)) * (eps ** d / (k * (np.log(n) + 1.0)))
        if m < threshold:
            rep_idx = int(rng.choice(indices))
            reps.append(X[rep_idx])
            weights.append(float(m))
            hypercubes.append(_bounds_to_hypercube(bounds_min, bounds_max))
            return

        mid = 0.5 * (bounds_min + bounds_max)
        num_subcubes = 2 ** d
        subcells = []
        non_empty_count = 0
        max_child_size = 0

        for subcube_idx in range(num_subcubes):
            sub_min = bounds_min.copy()
            sub_max = bounds_max.copy()
            sub_indices = indices

            for dim in range(d):
                if (subcube_idx >> dim) & 1 == 0:
                    sub_max[dim] = mid[dim]
                    mask = X[sub_indices, dim] <= mid[dim]
                else:
                    sub_min[dim] = mid[dim]
                    mask = X[sub_indices, dim] > mid[dim]
                sub_indices = sub_indices[mask]

            subcells.append((sub_indices, sub_min, sub_max))
            if sub_indices.size > 0:
                non_empty_count += 1
                if sub_indices.size > max_child_size:
                    max_child_size = int(sub_indices.size)

        # Degenerate split guard.
        if non_empty_count <= 1 or max_child_size == m:
            rep_idx = int(rng.choice(indices))
            reps.append(X[rep_idx])
            weights.append(float(m))
            hypercubes.append(_bounds_to_hypercube(bounds_min, bounds_max))
            return

        for sub_indices, sub_min, sub_max in subcells:
            if keep_empty_cells or sub_indices.size > 0:
                recurse(sub_indices, sub_min, sub_max, depth + 1)

    recurse(np.arange(n, dtype=int), mins, maxs, 0)

    if reps:
        reps_arr = np.vstack(reps)
        weights_arr = np.asarray(weights, dtype=float)
    else:
        reps_arr = np.empty((0, d))
        weights_arr = np.empty((0,), dtype=float)

    return reps_arr, weights_arr, hypercubes


def _tune_beta(root, target_ratio, eps, cost, k, n, d, tolerance, max_iter, use_critical_search=False, verbose=False):
    if use_critical_search:
        critical = _collect_critical_betas(root, eps, cost, k, n, d)
        if critical.size > 0:
            lo_i, hi_i = 0, critical.size - 1
            best_beta = float(critical[0])
            best_gap = float("inf")
            for it in range(max_iter):
                mid_i = (lo_i + hi_i) // 2
                beta = float(critical[mid_i])
                ratio = count_coreset_size(root, beta, eps, cost, k, n, d) / n
                gap = abs(ratio - target_ratio)
                if gap < best_gap:
                    best_gap = gap
                    best_beta = beta
                    if verbose:
                        print(
                            f"! EQT tune update(best): iter={it} beta={best_beta:.6g} ratio={ratio:.6f} gap={best_gap:.6f}",
                            flush=True,
                        )
                if gap <= target_ratio * tolerance:
                    if verbose:
                        print(
                            f"! EQT tune done: iter={it} beta={best_beta:.6g} ratio={ratio:.6f}",
                            flush=True,
                        )
                    return best_beta
                if ratio > target_ratio:
                    lo_i = mid_i + 1
                else:
                    hi_i = mid_i - 1
                if verbose:
                    print(
                        f"! EQT tune step: iter={it} beta={beta:.6g} ratio={ratio:.6f} index_range=[{lo_i},{hi_i}]",
                        flush=True,
                    )
                if lo_i > hi_i:
                    break
            return best_beta

    # Continuous binary search on monotone size(beta).
    lo, hi = 1e-12, 1.0
    ratio_lo = count_coreset_size(root, lo, eps, cost, k, n, d) / n
    ratio_hi = count_coreset_size(root, hi, eps, cost, k, n, d) / n

    for _ in range(30):
        if ratio_hi <= target_ratio:
            break
        hi *= 2.0
        ratio_hi = count_coreset_size(root, hi, eps, cost, k, n, d) / n

    for _ in range(30):
        if ratio_lo >= target_ratio:
            break
        lo *= 0.5
        ratio_lo = count_coreset_size(root, lo, eps, cost, k, n, d) / n

    if verbose:
        print(
            f"! EQT tune init-range: lo={lo:.6g} (ratio={ratio_lo:.6f}), hi={hi:.6g} (ratio={ratio_hi:.6f}), target={target_ratio:.6f}",
            flush=True,
        )

    best_beta = lo
    best_gap = abs(ratio_lo - target_ratio)
    if abs(ratio_hi - target_ratio) < best_gap:
        best_beta = hi
        best_gap = abs(ratio_hi - target_ratio)

    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        ratio_mid = count_coreset_size(root, mid, eps, cost, k, n, d) / n
        gap = abs(ratio_mid - target_ratio)

        if gap < best_gap:
            best_gap = gap
            best_beta = mid
            if verbose:
                print(
                    f"! EQT tune update(best): iter={it} beta={best_beta:.6g} ratio={ratio_mid:.6f} gap={best_gap:.6f}",
                    flush=True,
                )

        if gap <= target_ratio * tolerance:
            if verbose:
                print(
                    f"! EQT tune done: iter={it} beta={mid:.6g} ratio={ratio_mid:.6f}",
                    flush=True,
                )
            return mid

        if ratio_mid > target_ratio:
            lo = mid
        else:
            hi = mid

        if verbose:
            print(
                f"! EQT tune step: iter={it} mid={mid:.6g} ratio={ratio_mid:.6f} range=[{lo:.6g}, {hi:.6g}]",
                flush=True,
            )

    return best_beta


def exponential_quadtree_coreset(
    X,
    centers,
    eps,
    random_state=None,
    beta=None,
    compression_ratio=None,
    tolerance=0.1,
    max_iter=8,
    verbose=False,
    max_depth=None,
    use_critical_search=False,
    return_info=False,
    keep_empty_cells=False,
):
    """Build an exponential quadtree coreset using one-time tree construction.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, d)
        Input points.
    centers : np.ndarray of shape (k, d)
        Reference centers used only to compute fixed reference cost.
    eps : float
        Approximation parameter (typically 0.05, 0.1, or 0.2).
    random_state : int or None
        Seed for deterministic representative choices.
    beta : float or None
        If provided, use this fixed beta directly and skip compression-ratio tuning.
    compression_ratio : float or None
        Desired coreset size fraction in (0, 1]. If None, uses default beta=4.0.
    tolerance : float
        Relative tolerance used when targeting compression_ratio.
    max_iter : int
        Iterations for beta search.
    verbose : bool
        If True, prints compact diagnostics.
    max_depth : int or None
        Optional geometric depth cap for tree construction.
    use_critical_search : bool
        If True, searches over cached node critical beta values.
    return_info : bool
        If True, also returns a metadata dict with tuned beta and sizes.
    keep_empty_cells : bool
        If True, keep empty leaf cells for visualization; slower and higher memory.

    Returns
    -------
    reps : np.ndarray
    weights : np.ndarray
    hypercubes : list[tuple]
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    n, d = X.shape
    if n == 0:
        return np.empty((0, d)), np.empty((0,), dtype=float), []

    k = int(centers.shape[0])
    if k <= 0:
        raise ValueError("centers must contain at least one point")

    if verbose:
        print(f"! EQT build start: n={n}, d={d}, eps={eps}, beta={beta}, target_ratio={compression_ratio}", flush=True)

    cost = float(compute_kmeans_cost(X, centers))

    # Fixed-beta mode: use direct threshold recursion so beta impacts construction immediately.
    if beta is not None:
        beta = float(beta)
        if beta <= 0:
            raise ValueError("beta must be > 0")
        reps, weights, hypercubes = _direct_coreset_with_beta(
            X,
            eps,
            cost,
            k,
            beta,
            random_state=random_state,
            max_depth=max_depth,
            keep_empty_cells=keep_empty_cells,
        )
        achieved_ratio = 0.0 if n == 0 else reps.shape[0] / n
        info = {
            "beta": float(beta),
            "n_initial": int(n),
            "n_coreset": int(reps.shape[0]),
            "compression_ratio_achieved": float(achieved_ratio),
            "compression_ratio_target": None,
            "cost_ref": float(cost),
            "k": int(k),
            "d": int(d),
            "eps": float(eps),
        }

        if verbose:
            print(
                "! EQT direct-beta summary:",
                f"n={n}",
                f"d={d}",
                f"k={k}",
                f"eps={eps}",
                f"beta={beta:.6g}",
                f"cost_ref={cost:.6f}",
                f"coreset_size={reps.shape[0]}",
                f"ratio={achieved_ratio:.6f}",
                flush=True,
            )

        if return_info:
            return reps, weights, hypercubes, info
        return reps, weights, hypercubes

    root, meta = build_exponential_quadtree(
        X,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose,
        keep_empty_cells=keep_empty_cells,
    )
    stats = _tree_stats(root)

    if verbose:
        print(
            "! EQT build done:",
            f"nodes={stats['nodes']}",
            f"leaves={stats['leaves']}",
            f"empty_leaves={stats['empty_leaves']}",
            f"tree_depth={stats['max_depth']}",
            f"root_side={meta['side']:.6g}",
            flush=True,
        )

    if compression_ratio is None:
        beta = 4.0
    else:
        if not (0 < compression_ratio <= 1.0):
            raise ValueError("compression_ratio must be in (0, 1]")
        beta = _tune_beta(
            root,
            compression_ratio,
            eps,
            cost,
            k,
            n,
            d,
            tolerance,
            max_iter,
            use_critical_search=use_critical_search,
            verbose=verbose,
        )

    reps, weights, hypercubes = extract_coreset(root, beta, eps, cost, k, n, d, X, keep_empty_cells=keep_empty_cells)
    achieved_ratio = 0.0 if n == 0 else reps.shape[0] / n

    info = {
        "beta": float(beta),
        "n_initial": int(n),
        "n_coreset": int(reps.shape[0]),
        "compression_ratio_achieved": float(achieved_ratio),
        "compression_ratio_target": None if compression_ratio is None else float(compression_ratio),
        "cost_ref": float(cost),
        "k": int(k),
        "d": int(d),
        "eps": float(eps),
    }

    if verbose:
        print(
            "! EQT summary:",
            f"n={meta['n']}",
            f"d={meta['d']}",
            f"k={k}",
            f"eps={eps}",
            f"target={compression_ratio}",
            f"beta={beta:.6g}",
            f"cost_ref={cost:.6f}",
            f"coreset_size={reps.shape[0]}",
            f"ratio={achieved_ratio:.6f}",
        )

    if return_info:
        return reps, weights, hypercubes, info

    return reps, weights, hypercubes
