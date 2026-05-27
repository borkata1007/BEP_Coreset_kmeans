"""Standalone EGQ coreset construction for the srcjean experiment format.

This file does not import from the older implementation in the repo. It keeps
the same algorithmic ingredients here so Jean's runner can still use the usual
AlgoClass(m).generate(data) interface.
"""

import heapq

import numpy as np


class EGQCoreset:
    """Exponential-grid/quadtree coreset with the original threshold rule.

    The mathematical rule is not "split until exactly m points". A cell with
    side length L and count |P_cell| becomes one weighted representative when

        |P_cell| < beta * (cost / L^2) * eps^d / (k * (log(n) + 1)).

    Jean's runner supplies only a target m, so this class tunes beta until the
    formula-driven construction gets as close as possible to m. If exact m is
    impossible because a split jumps from, say, 1 cell to 4 cells, the algorithm
    returns the nearest formula-valid size instead of merging cells by hand.
    """

    def __init__(
        self,
        m,
        eps=0.1,
        k=None,
        beta=None,
        max_depth=64,
        beta_search_steps=32,
        lloyd_iterations=8,
        sizing_workflow="ranked",
        verify_ranked=False,
        trim_overshoot=True,
    ):
        self.m = m
        self.eps = eps
        self.k = k
        self.beta = beta
        self.max_depth = max_depth
        self.beta_search_steps = beta_search_steps
        self.lloyd_iterations = lloyd_iterations
        self.sizing_workflow = sizing_workflow
        self.verify_ranked = verify_ranked
        self.trim_overshoot = trim_overshoot

        self.weights = None
        self.indices = None
        self.sensitivities = None
        self.cells = None
        self.raw_size = None
        self.trimmed_to_target = False
        self.trimmed_weight_rescaled = False
        self.beta_used = None
        self.eps_used = None
        self.z_used = None
        self.eps_min = None
        self.eps_report = None
        self.rank_interval = None
        self.rank_verification = None
        self.reference_centers = None
        self.reference_cost = None

    def generate(self, data):
        data = np.asarray(data, dtype=float)
        n, d = data.shape

        if n == 0:
            self.weights = np.empty(0, dtype=float)
            self.indices = np.empty(0, dtype=int)
            self.sensitivities = np.empty(0, dtype=float)
            self.cells = []
            self.beta_used = self.beta
            return data

        target_size = max(1, min(int(self.m), n))
        k = _resolve_k(self.k, target_size, n)

        # The threshold uses a reference k-means cost. In the original workflow
        # this came from centers computed outside the coreset builder; in the
        # srcjean API, the coreset only receives data, so we derive them here.
        self.reference_centers = _reference_centers(
            data,
            k,
            n_iterations=self.lloyd_iterations,
        )
        self.reference_cost = _kmeans_cost(data, self.reference_centers)

        if self.beta is None and self.sizing_workflow == "ranked":
            leaves, z_used, rank_interval, verification = _build_cells_by_ranked_splits(
                data,
                target_size,
                self.reference_cost,
                k,
                self.max_depth,
                verify=self.verify_ranked,
            )
            eps_min = z_used ** (1.0 / d) if z_used >= 0.0 else np.nan
            eps_report = _round_up_to_step(eps_min, 0.1)
            beta_used = z_used / (eps_report ** d) if eps_report > 0.0 else np.nan
            beta = beta_used
            self.eps_used = eps_report
            self.z_used = z_used
            self.eps_min = eps_min
            self.eps_report = eps_report
            self.rank_interval = rank_interval
            if self.verify_ranked and np.isfinite(beta_used) and np.isfinite(eps_report):
                verified = _build_cells_for_beta(
                    data,
                    beta_used,
                    eps_report,
                    self.reference_cost,
                    k,
                    self.max_depth,
                )
                verification = {
                    "matched_size": len(verified) == len(leaves),
                    "ranked_size": len(leaves),
                    "formula_size": len(verified),
                }
            self.rank_verification = verification
        elif self.beta is None:
            beta, leaves = _tune_beta_to_target_size(
                data,
                target_size,
                self.eps,
                self.reference_cost,
                k,
                self.max_depth,
                self.beta_search_steps,
            )
            self.eps_used = self.eps
            self.z_used = beta * (self.eps ** d)
        else:
            beta = float(self.beta)
            leaves = _build_cells_for_beta(
                data,
                beta,
                self.eps,
                self.reference_cost,
                k,
                self.max_depth,
            )
            self.eps_used = self.eps
            self.z_used = beta * (self.eps ** d)

        reps, weights, indices, cells = _representatives_from_cells(data, leaves)
        self.raw_size = int(reps.shape[0])

        if (
            self.trim_overshoot
            and self.sizing_workflow == "ranked"
            and reps.shape[0] > target_size
        ):
            keep = _top_weight_indices(weights, target_size)
            reps = reps[keep]
            weights = weights[keep]
            indices = indices[keep]
            cells = [cells[idx] for idx in keep]
            weight_sum = float(np.sum(weights))
            if weight_sum > 0.0:
                weights = weights * (n / weight_sum)
                self.trimmed_weight_rescaled = True
            self.trimmed_to_target = True

        self.weights = weights
        self.indices = indices
        self.sensitivities = np.full(n, 1.0 / n)
        self.cells = cells
        self.beta_used = beta

        return reps.reshape((-1, d))


def _resolve_k(k, target_size, n):
    if k is not None:
        return max(1, min(int(k), n))

    # Jean's unchanged make_coreset API does not pass the downstream k, even
    # though k is part of the EGQ formula. Its coreset_size helper always has
    # m >= k + 1, so m - 1 is a conservative derived value that keeps the
    # returned coreset large enough for the later weighted KMeans call.
    inferred = max(1, target_size - 1)
    return min(inferred, n)


def _reference_centers(data, k, n_iterations):
    centers = _kmeans_plus_plus_init(data, k)

    for _ in range(n_iterations):
        labels = _nearest_center_indices(data, centers)
        updated = centers.copy()

        for center_idx in range(k):
            members = data[labels == center_idx]
            if members.size == 0:
                updated[center_idx] = data[np.random.randint(data.shape[0])]
            else:
                updated[center_idx] = np.mean(members, axis=0)

        if np.allclose(updated, centers):
            break
        centers = updated

    return centers


def _kmeans_plus_plus_init(data, k):
    n = data.shape[0]
    centers = np.empty((k, data.shape[1]), dtype=float)
    first_idx = int(np.random.randint(n))
    centers[0] = data[first_idx]

    closest_sq = _squared_distances_to_point(data, centers[0])
    for center_idx in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 0.0:
            next_idx = int(np.random.randint(n))
        else:
            probs = closest_sq / total
            next_idx = int(np.random.choice(n, p=probs))

        centers[center_idx] = data[next_idx]
        new_sq = _squared_distances_to_point(data, centers[center_idx])
        closest_sq = np.minimum(closest_sq, new_sq)

    return centers


def _squared_distances_to_point(data, point):
    diff = data - point
    return np.sum(diff * diff, axis=1)


def _nearest_center_indices(data, centers, chunk_size=250000):
    labels = np.empty(data.shape[0], dtype=int)

    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start:start + chunk_size]
        distances = _squared_distances_to_centers(chunk, centers)
        labels[start:start + chunk.shape[0]] = np.argmin(distances, axis=1)

    return labels


def _kmeans_cost(data, centers, chunk_size=250000):
    cost = 0.0

    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start:start + chunk_size]
        distances = _squared_distances_to_centers(chunk, centers)
        cost += float(np.sum(np.min(distances, axis=1)))

    return cost


def _squared_distances_to_centers(chunk, centers):
    diff = chunk[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)


def _tune_beta_to_target_size(data, target_size, eps, cost, k, max_depth, steps):
    min_size = min(k, data.shape[0])
    best_beta = None
    best_leaves = None
    best_gap = np.inf

    beta = 1.0
    leaves = _build_cells_for_beta(data, beta, eps, cost, k, max_depth)
    fallback_beta = beta
    fallback_leaves = leaves
    fallback_gap = abs(len(leaves) - target_size)

    best_beta, best_leaves, best_gap = _maybe_update_best(
        best_beta, best_leaves, best_gap, beta, leaves, target_size, min_size
    )

    # Larger beta means a larger stopping threshold, hence fewer cells.
    if len(leaves) > target_size:
        lo = beta
        lo_leaves = leaves
        hi = beta
        hi_leaves = leaves

        while len(hi_leaves) > target_size and hi < 1e300:
            hi *= 2.0
            hi_leaves = _build_cells_for_beta(data, hi, eps, cost, k, max_depth)
            fallback_beta, fallback_leaves, fallback_gap = _maybe_update_fallback(
                fallback_beta, fallback_leaves, fallback_gap, hi, hi_leaves, target_size
            )
            best_beta, best_leaves, best_gap = _maybe_update_best(
                best_beta, best_leaves, best_gap, hi, hi_leaves, target_size, min_size
            )
    elif len(leaves) < target_size:
        hi = beta
        hi_leaves = leaves
        lo = beta
        lo_leaves = leaves

        while len(lo_leaves) < target_size and lo > 1e-300:
            lo *= 0.5
            lo_leaves = _build_cells_for_beta(data, lo, eps, cost, k, max_depth)
            fallback_beta, fallback_leaves, fallback_gap = _maybe_update_fallback(
                fallback_beta, fallback_leaves, fallback_gap, lo, lo_leaves, target_size
            )
            best_beta, best_leaves, best_gap = _maybe_update_best(
                best_beta, best_leaves, best_gap, lo, lo_leaves, target_size, min_size
            )
    else:
        return beta, leaves

    if len(lo_leaves) < target_size or len(hi_leaves) > target_size:
        if best_leaves is not None:
            return best_beta, best_leaves
        return fallback_beta, fallback_leaves

    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        mid_leaves = _build_cells_for_beta(data, mid, eps, cost, k, max_depth)
        mid_size = len(mid_leaves)
        gap = abs(mid_size - target_size)

        fallback_beta, fallback_leaves, fallback_gap = _maybe_update_fallback(
            fallback_beta,
            fallback_leaves,
            fallback_gap,
            mid,
            mid_leaves,
            target_size,
        )
        best_beta, best_leaves, best_gap = _maybe_update_best(
            best_beta,
            best_leaves,
            best_gap,
            mid,
            mid_leaves,
            target_size,
            min_size,
        )
        if best_gap == 0:
            break

        if mid_size > target_size:
            lo = mid
        else:
            hi = mid

    if best_leaves is not None:
        return best_beta, best_leaves
    return fallback_beta, fallback_leaves


def _maybe_update_best(best_beta, best_leaves, best_gap, beta, leaves, target_size, min_size):
    if len(leaves) < min_size:
        return best_beta, best_leaves, best_gap

    gap = abs(len(leaves) - target_size)
    if gap < best_gap:
        return beta, leaves, gap

    return best_beta, best_leaves, best_gap


def _maybe_update_fallback(best_beta, best_leaves, best_gap, beta, leaves, target_size):
    gap = abs(len(leaves) - target_size)
    if gap < best_gap:
        return beta, leaves, gap

    return best_beta, best_leaves, best_gap


def _build_cells_by_ranked_splits(data, target_size, cost, k, max_depth, verify=False):
    """Build a target-sized tree by splitting cells in formula-priority order.

    For a fixed dataset, k, and cost, the split condition can be written with
    z = beta * eps^d as

        z <= count * side_length^2 * k * (log(n) + 1) / cost.

    The right-hand side is a cell-specific z cutoff. Changing z moves only the
    global cutoff, so the descending priority order is unchanged.
    """
    n, d = data.shape
    root = _make_root_cell(data)
    leaves = [root]
    active_leaf_ids = {id(root)}
    active_leaf_count = 1

    if target_size <= 1 or cost <= 0.0:
        z_used, interval = _z_from_priority_interval(np.inf, 0.0, k, n)
        return leaves, z_used, interval, None

    heap = []
    serial = 0
    first_blocked_priority = None
    last_split_priority = None
    _push_ranked_cell(heap, root, cost, serial)
    serial += 1

    while active_leaf_count < target_size and heap:
        priority, cell = _pop_ranked_cell(heap, active_leaf_ids)
        if cell is None:
            break
        children = _split_cell(data, cell, max_depth)
        if len(children) <= 1:
            cell["splittable"] = False
            first_blocked_priority = priority
            continue

        active_leaf_ids.discard(id(cell))
        leaves.extend(children)
        active_leaf_count += len(children) - 1
        last_split_priority = priority

        for child in children:
            active_leaf_ids.add(id(child))
            _push_ranked_cell(heap, child, cost, serial)
            serial += 1

    if last_split_priority is not None:
        _split_equal_priority_boundary(
            data,
            leaves,
            active_leaf_ids,
            heap,
            cost,
            max_depth,
            last_split_priority,
            serial,
        )

    next_priority, _ = _pop_ranked_cell(heap, active_leaf_ids)

    if next_priority is None:
        next_priority = first_blocked_priority

    z_used, interval = _z_from_priority_interval(
        last_split_priority,
        next_priority,
        k,
        n,
    )

    verification = None
    leaves = [leaf for leaf in leaves if id(leaf) in active_leaf_ids]
    if verify and np.isfinite(z_used):
        verified = _build_cells_for_beta(data, z_used, 1.0, cost, k, max_depth)
        verification = {
            "matched_size": len(verified) == len(leaves),
            "ranked_size": len(leaves),
            "formula_size": len(verified),
        }

    return leaves, z_used, interval, verification


def _push_ranked_cell(heap, cell, cost, serial):
    if not cell["splittable"]:
        return

    priority = _split_priority(cell, cost)
    heapq.heappush(heap, (-priority, serial, cell))


def _pop_ranked_cell(heap, active_leaf_ids):
    while heap:
        neg_priority, _, cell = heapq.heappop(heap)
        if cell["splittable"] and id(cell) in active_leaf_ids:
            return -neg_priority, cell

    return None, None


def _split_equal_priority_boundary(
    data,
    leaves,
    active_leaf_ids,
    heap,
    cost,
    max_depth,
    boundary_priority,
    serial,
):
    pending = []

    while True:
        priority, cell = _pop_ranked_cell(heap, active_leaf_ids)
        if cell is None:
            break

        if not np.isclose(priority, boundary_priority, rtol=1e-12, atol=1e-15):
            heapq.heappush(heap, (-priority, serial, cell))
            return

        children = _split_cell(data, cell, max_depth)
        if len(children) <= 1:
            cell["splittable"] = False
            continue

        active_leaf_ids.discard(id(cell))
        leaves.extend(children)
        pending.extend(children)

        for child in pending:
            active_leaf_ids.add(id(child))
            _push_ranked_cell(heap, child, cost, serial)
            serial += 1
        pending.clear()


def _split_priority(cell, cost):
    if cost <= 0.0:
        return 0.0

    return cell["count"] * (cell["side_length"] ** 2) / cost


def _z_from_priority_interval(last_split_priority, next_priority, k, n):
    log_term = np.log(n) + 1.0

    lower_alpha = 0.0 if next_priority is None else next_priority
    upper_alpha = np.inf if last_split_priority is None else last_split_priority

    if lower_alpha is None:
        lower_alpha = 0.0
    if upper_alpha is None:
        upper_alpha = np.inf

    if not np.isfinite(upper_alpha):
        alpha = max(lower_alpha, 1.0)
    elif lower_alpha <= 0.0:
        alpha = 0.5 * upper_alpha
    elif lower_alpha < upper_alpha:
        alpha = np.sqrt(lower_alpha * upper_alpha)
    else:
        alpha = upper_alpha

    z = alpha * k * log_term
    return z, (lower_alpha, upper_alpha)


def _round_up_to_step(value, step):
    if not np.isfinite(value) or value <= 0.0:
        return value

    return np.ceil(value / step) * step


def _build_cells_for_beta(data, beta, eps, cost, k, max_depth):
    n, d = data.shape
    leaves = [_make_root_cell(data)]

    while True:
        split_idx = _most_violating_cell_index(leaves, beta, eps, cost, k, n, d)
        if split_idx is None:
            break

        cell = leaves.pop(split_idx)
        children = _split_cell(data, cell, max_depth)
        if len(children) <= 1:
            cell["splittable"] = False
            leaves.append(cell)
            continue

        leaves.extend(children)

    return leaves


def _most_violating_cell_index(cells, beta, eps, cost, k, n, d):
    best_idx = None
    best_score = 1.0

    for idx, cell in enumerate(cells):
        if not cell["splittable"]:
            continue

        threshold = _threshold(cell, beta, eps, cost, k, n, d)
        if cell["count"] < threshold:
            continue

        if threshold <= 0.0:
            score = np.inf
        else:
            score = cell["count"] / threshold

        if score > best_score:
            best_idx = idx
            best_score = score

    return best_idx


def _threshold(cell, beta, eps, cost, k, n, d):
    side_length = cell["side_length"]
    if side_length <= 0.0:
        return np.inf

    log_term = np.log(n) + 1.0
    return beta * (cost / (side_length ** 2)) * (eps ** d / (k * log_term))


def _make_root_cell(data):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    side_length = float(np.max(maxs - mins))
    maxs = mins + side_length

    return {
        "indices": np.arange(data.shape[0], dtype=int),
        "count": int(data.shape[0]),
        "depth": 0,
        "bounds_min": mins,
        "bounds_max": maxs,
        "side_length": side_length,
        "splittable": side_length > 0.0 and data.shape[0] > 1,
    }


def _split_cell(data, cell, max_depth):
    if cell["depth"] >= max_depth or cell["side_length"] <= 0.0 or cell["count"] <= 1:
        return []

    d = data.shape[1]
    mid = 0.5 * (cell["bounds_min"] + cell["bounds_max"])
    children = []

    for subcube_idx in range(2 ** d):
        sub_min = cell["bounds_min"].copy()
        sub_max = cell["bounds_max"].copy()
        sub_indices = cell["indices"]

        for dim in range(d):
            if (subcube_idx >> dim) & 1 == 0:
                sub_max[dim] = mid[dim]
                mask = data[sub_indices, dim] <= mid[dim]
            else:
                sub_min[dim] = mid[dim]
                mask = data[sub_indices, dim] > mid[dim]
            sub_indices = sub_indices[mask]

        if sub_indices.size == 0:
            continue

        side_length = float(sub_max[0] - sub_min[0])
        children.append(
            {
                "indices": sub_indices,
                "count": int(sub_indices.size),
                "depth": cell["depth"] + 1,
                "bounds_min": sub_min,
                "bounds_max": sub_max,
                "side_length": side_length,
                "splittable": side_length > 0.0 and sub_indices.size > 1,
            }
        )

    if len(children) <= 1 or max(child["count"] for child in children) == cell["count"]:
        return []

    return children


def _representatives_from_cells(data, cells):
    reps = []
    weights = []
    indices = []
    bounds = []

    for cell in cells:
        rep_idx = int(np.random.choice(cell["indices"]))
        reps.append(data[rep_idx])
        weights.append(float(cell["count"]))
        indices.append(rep_idx)
        bounds.append(_bounds_to_tuple(cell["bounds_min"], cell["bounds_max"]))

    return (
        np.vstack(reps),
        np.asarray(weights, dtype=float),
        np.asarray(indices, dtype=int),
        bounds,
    )


def _top_weight_indices(weights, target_size):
    order = np.lexsort((np.arange(weights.size), -weights))
    return np.sort(order[:target_size])


def _bounds_to_tuple(bounds_min, bounds_max):
    values = []
    for dim in range(bounds_min.size):
        values.extend([float(bounds_min[dim]), float(bounds_max[dim])])
    return tuple(values)


ExponentialQuadtreeCoreset = EGQCoreset
