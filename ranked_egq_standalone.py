"""Standalone ranked EGQ experiments for donuts and image palettes.

This file is intentionally self-contained so it can be copied into a new repo.

Dependencies:
    pip install numpy pandas matplotlib pillow scikit-learn

Examples:
    python ranked_egq_standalone.py donuts --csv final_datasets/donuts.csv
    python ranked_egq_standalone.py image --image pictures/cvete.jpg
"""

from __future__ import annotations

import argparse
import heapq
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KDTree


class RankedEGQCoreset:
    """Ranked exponential-grid/quadtree coreset.

    The construction ranks cells by the z cutoff induced by

        threshold = z * cost / (side_length^2 * k * (log(n) + 1)),
        z = beta * eps^d.

    It tunes z to reach the closest attainable quadtree size, then reports a
    conservative epsilon by eps_report = ceil(z^(1/d) / 0.1) * 0.1 and chooses
    beta so beta * eps_report^d = z.
    """

    def __init__(
        self,
        target_size,
        reference_k,
        max_depth=64,
        lloyd_iterations=4,
        verify=True,
        trim_overshoot=True,
        random_state=0,
    ):
        self.target_size = int(target_size)
        self.reference_k = int(reference_k)
        self.max_depth = int(max_depth)
        self.lloyd_iterations = int(lloyd_iterations)
        self.verify = bool(verify)
        self.trim_overshoot = bool(trim_overshoot)
        self.random_state = random_state

        self.weights = None
        self.indices = None
        self.cells = None
        self.raw_size = None
        self.trimmed_to_target = False
        self.trimmed_weight_rescaled = False
        self.reference_centers = None
        self.reference_cost = None
        self.z_used = None
        self.eps_min = None
        self.eps_report = None
        self.beta_used = None
        self.rank_interval = None
        self.verification = None

    def generate(self, data):
        np.random.seed(self.random_state)
        data = np.asarray(data, dtype=float)
        n, d = data.shape

        if n == 0:
            self.weights = np.empty(0, dtype=float)
            self.indices = np.empty(0, dtype=int)
            self.cells = []
            return data

        target_size = max(1, min(self.target_size, n))
        k = max(1, min(self.reference_k, n))

        self.reference_centers = _reference_centers(data, k, self.lloyd_iterations)
        self.reference_cost = _kmeans_cost(data, self.reference_centers)

        leaves, z_used, interval = _build_cells_by_ranked_splits(
            data,
            target_size,
            self.reference_cost,
            k,
            self.max_depth,
        )

        eps_min = z_used ** (1.0 / d) if z_used >= 0.0 else np.nan
        eps_report = _round_up_to_step(eps_min, 0.1)
        beta_used = z_used / (eps_report ** d) if eps_report > 0.0 else np.nan

        self.z_used = z_used
        self.eps_min = eps_min
        self.eps_report = eps_report
        self.beta_used = beta_used
        self.rank_interval = interval

        if self.verify and np.isfinite(beta_used) and np.isfinite(eps_report):
            verified = _build_cells_for_beta(
                data,
                beta_used,
                eps_report,
                self.reference_cost,
                k,
                self.max_depth,
            )
            self.verification = {
                "matched_size": len(verified) == len(leaves),
                "ranked_size": len(leaves),
                "formula_size": len(verified),
            }

        reps, weights, indices, cells = _representatives_from_cells(data, leaves)
        self.raw_size = int(reps.shape[0])

        if self.trim_overshoot and reps.shape[0] > target_size:
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
        self.cells = cells
        return reps.reshape((-1, d))


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
    centers[0] = data[np.random.randint(n)]
    closest_sq = _squared_distances_to_point(data, centers[0])
    for center_idx in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 0.0:
            next_idx = int(np.random.randint(n))
        else:
            next_idx = int(np.random.choice(n, p=closest_sq / total))
        centers[center_idx] = data[next_idx]
        closest_sq = np.minimum(closest_sq, _squared_distances_to_point(data, centers[center_idx]))
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


def _build_cells_by_ranked_splits(data, target_size, cost, k, max_depth):
    n, _ = data.shape
    root = _make_root_cell(data)
    leaves = [root]
    active_leaf_ids = {id(root)}
    active_leaf_count = 1

    if target_size <= 1 or cost <= 0.0:
        return leaves, _z_from_priority_interval(np.inf, 0.0, k, n)[0], (0.0, np.inf)

    heap = []
    serial = 0
    last_split_priority = None
    first_blocked_priority = None
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

    z_used, interval = _z_from_priority_interval(last_split_priority, next_priority, k, n)
    leaves = [leaf for leaf in leaves if id(leaf) in active_leaf_ids]
    return leaves, z_used, interval


def _push_ranked_cell(heap, cell, cost, serial):
    if cell["splittable"]:
        heapq.heappush(heap, (-_split_priority(cell, cost), serial, cell))


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
        for child in children:
            active_leaf_ids.add(id(child))
            _push_ranked_cell(heap, child, cost, serial)
            serial += 1


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
    return alpha * k * log_term, (lower_alpha, upper_alpha)


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
        else:
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
        score = np.inf if threshold <= 0.0 else cell["count"] / threshold
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def _threshold(cell, beta, eps, cost, k, n, d):
    side_length = cell["side_length"]
    if side_length <= 0.0:
        return np.inf
    return beta * (cost / (side_length ** 2)) * (eps ** d / (k * (np.log(n) + 1.0)))


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
    return np.vstack(reps), np.asarray(weights), np.asarray(indices), bounds


def _top_weight_indices(weights, target_size):
    order = np.lexsort((np.arange(weights.size), -weights))
    return np.sort(order[:target_size])


def _bounds_to_tuple(bounds_min, bounds_max):
    values = []
    for dim in range(bounds_min.size):
        values.extend([float(bounds_min[dim]), float(bounds_max[dim])])
    return tuple(values)


def run_donuts(
    csv_path="final_datasets/donuts.csv",
    reference_k=3,
    sizes=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    output_dir="results/ranked_egq_size",
):
    df = pd.read_csv(csv_path)
    X_raw = df[["x", "y"]].to_numpy(dtype=float)
    X = X_raw
    title = "donuts"

    rows = []
    for size in sizes:
        out = Path(output_dir) / "points" / title / f"reference_k_{reference_k}_palette_{size}"
        out.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        egq = RankedEGQCoreset(size, reference_k=reference_k, random_state=0)
        reps = egq.generate(X)
        tree = KDTree(reps)
        _, nearest = tree.query(X, k=1)
        assignments = nearest[:, 0]
        reconstructed = reps[assignments]
        reconstruction_cost = float(np.sum((X - reconstructed) ** 2))
        elapsed = time.perf_counter() - start

        summary = _summary_dict("donuts", X, reference_k, size, egq, reconstruction_cost, elapsed)
        pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
        pd.Series(summary).to_json(out / "summary.json", indent=2)
        pd.DataFrame({"point_index": np.arange(X.shape[0]), "coreset_index": assignments}).to_csv(
            out / "assignments.csv", index=False
        )
        coreset_df = pd.DataFrame(reps, columns=["x", "y"])
        coreset_df["weight"] = egq.weights
        coreset_df["source_index"] = egq.indices
        coreset_df.to_csv(out / "palette_coreset.csv", index=False)

        _save_point_assignment_plot(X, reps, assignments, out / "assignment_coloring.png")
        _save_point_coreset_plot(X, reps, egq.weights, out / "coreset.png")
        _save_point_quadtree_boxes_plot(X, reps, egq.cells, out / "quadtree_boxes.png")
        rows.append(summary)
        print(f"donuts size={size} actual={reps.shape[0]} cost={reconstruction_cost:.3f} time={elapsed:.3f}s")

    sweep_out = Path(output_dir) / "points" / title / f"reference_k_{reference_k}_palette_sweep"
    sweep_out.mkdir(parents=True, exist_ok=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(sweep_out / "sweep_summary.csv", index=False)
    _save_cost_plots(sweep, sweep_out, "donuts", reference_k, "EGQ coreset size")


def run_cvete(
    image_path="pictures/cvete.jpg",
    reference_k=8,
    sizes=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    output_dir="results/ranked_egq_size",
):
    image_path = Path(image_path)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    X = image.reshape(-1, 3).astype(float)
    image_name = image_path.stem.lower().replace(" ", "_")

    rows = []
    for size in sizes:
        out = Path(output_dir) / "images" / image_name / f"reference_k_{reference_k}_palette_{size}"
        out.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        egq = RankedEGQCoreset(size, reference_k=reference_k, random_state=0)
        palette = np.clip(egq.generate(X), 0, 255)
        tree = KDTree(palette)
        _, nearest = tree.query(X, k=1)
        assignments = nearest[:, 0]
        compressed_flat = palette[assignments]
        compressed = compressed_flat.astype(np.uint8).reshape(height, width, 3)
        reconstruction_cost = float(np.sum((X - compressed_flat) ** 2))
        elapsed = time.perf_counter() - start

        summary = _summary_dict(image_name, X, reference_k, size, egq, reconstruction_cost, elapsed)
        summary.update({"height": height, "width": width})
        pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
        pd.Series(summary).to_json(out / "summary.json", indent=2)
        _save_image(compressed, out / "compressed.png")
        _save_image(image, out / "original.png")
        _save_palette_image(palette, out / "palette.png")
        _save_image_comparison(image, compressed, out / "comparison.png", size)
        palette_df = pd.DataFrame(palette, columns=["r", "g", "b"])
        palette_df["weight"] = egq.weights
        palette_df["source_index"] = egq.indices
        palette_df.to_csv(out / "palette_coreset.csv", index=False)
        rows.append(summary)
        print(f"cvete size={size} actual={palette.shape[0]} cost={reconstruction_cost:.3f} time={elapsed:.3f}s")

    sweep_out = Path(output_dir) / "images" / image_name / f"reference_k_{reference_k}_palette_sweep"
    sweep_out.mkdir(parents=True, exist_ok=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(sweep_out / "sweep_summary.csv", index=False)
    _save_cost_plots(sweep, sweep_out, image_name, reference_k, "EGQ palette size")


def _summary_dict(name, X, reference_k, size, egq, reconstruction_cost, elapsed):
    return {
        "dataset": name,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "reference_k": int(reference_k),
        "target_palette_size": int(size),
        "raw_palette_size": int(egq.raw_size),
        "actual_palette_size": int(egq.weights.shape[0]),
        "trimmed_to_target": bool(egq.trimmed_to_target),
        "trimmed_weight_rescaled": bool(egq.trimmed_weight_rescaled),
        "weights_sum": float(egq.weights.sum()),
        "z_used": float(egq.z_used),
        "beta_used": float(egq.beta_used),
        "eps_min": float(egq.eps_min),
        "eps_report": float(egq.eps_report),
        "rank_interval_low": float(egq.rank_interval[0]),
        "rank_interval_high": float(egq.rank_interval[1]),
        "reference_cost": float(egq.reference_cost),
        "palette_reconstruction_cost": float(reconstruction_cost),
        "cost_over_initial": float(reconstruction_cost / egq.reference_cost),
        "runtime_seconds": float(elapsed),
        "verification_matched_size": None
        if egq.verification is None
        else bool(egq.verification["matched_size"]),
        "verification_ranked_size": None if egq.verification is None else int(egq.verification["ranked_size"]),
        "verification_formula_size": None if egq.verification is None else int(egq.verification["formula_size"]),
    }


def _save_cost_plots(df, out, name, reference_k, x_label):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["target_palette_size"], df["cost_over_initial"], marker="o", linewidth=1.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("cost / initial reference cost")
    ax.set_title(f"{name}: EGQ cost vs reference k={reference_k} cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cost_ratio_line.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["target_palette_size"], df["palette_reconstruction_cost"], marker="o", linewidth=1.8)
    ax.axhline(df["reference_cost"].iloc[0], color="tab:red", linestyle="--", linewidth=1.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("squared reconstruction cost")
    ax.set_title(f"{name}: reconstruction cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cost_line.png", dpi=200)
    plt.close(fig)


def _save_point_assignment_plot(X, reps, assignments, path):
    sample_idx = _sample_indices(X.shape[0], 120000)
    cmap = plt.get_cmap("turbo", max(reps.shape[0], 2))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[sample_idx, 0], X[sample_idx, 1], c=assignments[sample_idx], cmap=cmap, s=2, alpha=0.45)
    ax.scatter(reps[:, 0], reps[:, 1], c=np.arange(reps.shape[0]), cmap=cmap, s=24, edgecolors="black")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_point_coreset_plot(X, reps, weights, path):
    sample_idx = _sample_indices(X.shape[0], 120000)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[sample_idx, 0], X[sample_idx, 1], c="lightgray", s=2, alpha=0.35)
    sizes = 10 + 90 * (weights / weights.max())
    scatter = ax.scatter(reps[:, 0], reps[:, 1], c=weights, cmap="viridis", s=sizes, edgecolors="black")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(scatter, ax=ax).set_label("weight")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_point_quadtree_boxes_plot(X, reps, cells, path):
    sample_idx = _sample_indices(X.shape[0], 120000)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[sample_idx, 0], X[sample_idx, 1], c="lightgray", s=2, alpha=0.25)
    for bounds in cells:
        x0, x1, y0, y1 = bounds[0], bounds[1], bounds[2], bounds[3]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="black", linewidth=0.35, alpha=0.35)
    ax.scatter(reps[:, 0], reps[:, 1], c="tab:blue", s=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sample_indices(n, max_size):
    rng = np.random.default_rng(0)
    size = min(max_size, n)
    return rng.choice(n, size=size, replace=False)


def _save_image(array, path):
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _save_palette_image(palette, path):
    colors = np.clip(palette, 0, 255).astype(np.uint8)
    swatch_width = 40
    swatch_height = 60
    image = np.zeros((swatch_height, swatch_width * colors.shape[0], 3), dtype=np.uint8)
    for idx, color in enumerate(colors):
        image[:, idx * swatch_width:(idx + 1) * swatch_width] = color
    _save_image(image, path)


def _save_image_comparison(original, compressed, path, size):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(compressed)
    axes[1].set_title(f"{size} EGQ colors")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_metrics_biased_grid(
    data_root="data_jean_final_25_05/05",
    output_dir="results_data_jean_25_05",
    clusters=(2, 4, 8, 16, 32, 64),
    budgets=(64, 128, 256, 512, 1024, 4096),
    iterations=range(5),
    ref_lloyd_iterations=0,
    weighted_steps=20,
    resume=True,
    dataset_filter=None,
):
    """Run the metrics_biased.csv grid with ranked EGQ.

    Output:
        metrics_ranked_egq.csv has exactly Dataset,Clusters,Budget,Iteration,Cost,Time.
        extended_results.csv adds raw size, epsilon, beta, and detailed timings.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics_ranked_egq.csv"
    extended_path = out / "extended_results.csv"

    existing = _load_existing_metrics(extended_path) if resume else set()
    datasets = _load_metrics_datasets(data_root)
    if dataset_filter is not None:
        datasets = {name: datasets[name] for name in dataset_filter if name in datasets}

    metrics_rows = []
    extended_rows = []
    if resume and metrics_path.exists():
        metrics_rows.extend(pd.read_csv(metrics_path).to_dict("records"))
    if resume and extended_path.exists():
        extended_rows.extend(pd.read_csv(extended_path).to_dict("records"))

    for dataset_name, X in datasets.items():
        for k in clusters:
            for budget in budgets:
                for iteration in iterations:
                    key = (dataset_name, int(k), int(budget), int(iteration))
                    if key in existing:
                        continue

                    print(f"running {dataset_name} k={k} budget={budget} iter={iteration}", flush=True)
                    row, ext = _run_one_metrics_job(
                        dataset_name,
                        X,
                        k=int(k),
                        budget=int(budget),
                        iteration=int(iteration),
                        ref_lloyd_iterations=ref_lloyd_iterations,
                        weighted_steps=weighted_steps,
                        output_dir=out,
                    )
                    metrics_rows.append(row)
                    extended_rows.append(ext)
                    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
                    pd.DataFrame(extended_rows).to_csv(extended_path, index=False)

    _save_metrics_visualizations(out, datasets)
    print(f"Saved metrics CSV to: {metrics_path}")
    print(f"Saved extended CSV to: {extended_path}")


def _load_existing_metrics(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["Dataset"], df["Clusters"], df["Budget"], df["Iteration"]))


def _load_metrics_datasets(data_root):
    root = Path(data_root)
    data = {}

    image_map = {
        "birb": root / "final_image" / "birb.png",
        "balloons": root / "final_image" / "image.png",
    }
    for name, path in image_map.items():
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        data[name] = image.reshape(-1, 3).astype(float) / 255.0

    tabular_map = {
        "donuts": (root / "final_synthetic" / "donuts.csv", ["x", "y"]),
        "spotify": (
            root / "final_real" / "spotify_9d.csv",
            [
                "danceability",
                "energy",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "valence",
            ],
        ),
        "uber": (root / "final_real" / "uber.csv", None),
    }
    for name, (path, columns) in tabular_map.items():
        df = pd.read_csv(path)
        if columns is None:
            numeric = df.select_dtypes(include=[np.number])
            if "id" in numeric.columns:
                numeric = numeric.drop(columns=["id"])
        else:
            numeric = df[columns]
        X = numeric.to_numpy(dtype=float)
        X = X[~np.isnan(X).any(axis=1)]
        data[name] = X

    return data


def _run_one_metrics_job(
    dataset_name,
    X,
    k,
    budget,
    iteration,
    ref_lloyd_iterations,
    weighted_steps,
    output_dir,
):
    coreset_start = time.perf_counter()
    egq = RankedEGQCoreset(
        budget,
        reference_k=k,
        lloyd_iterations=ref_lloyd_iterations,
        random_state=iteration,
    )
    Q = egq.generate(X)
    coreset_time = time.perf_counter() - coreset_start

    final_start = time.perf_counter()
    centers = _weighted_kmeans_plus_plus_local_search(
        Q,
        egq.weights,
        k,
        n_steps=weighted_steps,
        random_state=iteration,
    )
    final_kmeans_time = time.perf_counter() - final_start

    cost_start = time.perf_counter()
    cost = _kmeans_cost(X, centers)
    cost_time = time.perf_counter() - cost_start

    run_dir = output_dir / "runs" / dataset_name / f"k_{k}" / f"budget_{budget}" / f"iter_{iteration}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_dict(dataset_name, X, k, budget, egq, cost, coreset_time)
    summary.update(
        {
            "Iteration": int(iteration),
            "final_kmeans_time": float(final_kmeans_time),
            "cost_time": float(cost_time),
            "weighted_steps": int(weighted_steps),
            "ref_lloyd_iterations": int(ref_lloyd_iterations),
        }
    )
    pd.Series(summary).to_json(run_dir / "summary.json", indent=2)
    coreset_df = pd.DataFrame(Q, columns=[f"x{i}" for i in range(Q.shape[1])])
    coreset_df["weight"] = egq.weights
    coreset_df["source_index"] = egq.indices
    coreset_df.to_csv(run_dir / "coreset.csv", index=False)

    row = {
        "Dataset": dataset_name,
        "Clusters": int(k),
        "Budget": int(budget),
        "Iteration": int(iteration),
        "Cost": float(cost),
        "Time": float(coreset_time),
    }
    ext = {
        **row,
        "RawSize": int(egq.raw_size),
        "ActualSize": int(Q.shape[0]),
        "Trimmed": bool(egq.trimmed_to_target),
        "Z": float(egq.z_used),
        "EpsReport": float(egq.eps_report),
        "Beta": float(egq.beta_used),
        "ReferenceCost": float(egq.reference_cost),
        "FinalKMeansTime": float(final_kmeans_time),
        "CostTime": float(cost_time),
        "WeightedSteps": int(weighted_steps),
        "RefLloydIterations": int(ref_lloyd_iterations),
    }
    return row, ext


def _weighted_kmeans_plus_plus_local_search(X, weights, k, n_steps=20, random_state=0):
    rng = np.random.default_rng(random_state)
    centers = _weighted_kmeans_plus_plus_init(X, weights, k, rng)
    for _ in range(n_steps):
        labels = _nearest_center_indices(X, centers)
        updated = centers.copy()
        for center_idx in range(k):
            mask = labels == center_idx
            if not np.any(mask):
                updated[center_idx] = X[rng.integers(X.shape[0])]
            else:
                w = weights[mask]
                updated[center_idx] = np.average(X[mask], axis=0, weights=w)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def _weighted_kmeans_plus_plus_init(X, weights, k, rng):
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    probs = weights / np.sum(weights)
    first_idx = int(rng.choice(n, p=probs))
    centers[0] = X[first_idx]
    closest_sq = _squared_distances_to_point(X, centers[0])
    for center_idx in range(1, k):
        weighted = weights * closest_sq
        total = float(np.sum(weighted))
        if total <= 0.0:
            next_idx = int(rng.choice(n, p=probs))
        else:
            next_idx = int(rng.choice(n, p=weighted / total))
        centers[center_idx] = X[next_idx]
        closest_sq = np.minimum(closest_sq, _squared_distances_to_point(X, centers[center_idx]))
    return centers


def _save_metrics_visualizations(output_dir, datasets):
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    # Keep this focused: one illustrative run per visual dataset.
    specs = [
        ("donuts", 3, 512, 0),
        ("birb", 8, 512, 0),
        ("balloons", 8, 512, 0),
    ]
    for dataset_name, k, budget, iteration in specs:
        if dataset_name not in datasets:
            continue
        X = datasets[dataset_name]
        egq = RankedEGQCoreset(budget, reference_k=k, lloyd_iterations=0, random_state=iteration)
        Q = egq.generate(X)
        if X.shape[1] == 2:
            tree = KDTree(Q)
            _, nearest = tree.query(X, k=1)
            _save_point_assignment_plot(X, Q, nearest[:, 0], vis_dir / f"{dataset_name}_assignment_coloring.png")
            _save_point_coreset_plot(X, Q, egq.weights, vis_dir / f"{dataset_name}_coreset.png")
            _save_point_quadtree_boxes_plot(X, Q, egq.cells, vis_dir / f"{dataset_name}_quadtree_boxes.png")
        elif dataset_name in ("birb", "balloons"):
            tree = KDTree(np.clip(Q, 0, 255))
            _, nearest = tree.query(X, k=1)
            compressed = np.clip(Q[nearest[:, 0]], 0, 255).astype(np.uint8)
            # Shape lookup from pixel count is not reliable here, so save palette only.
            _save_palette_image(Q, vis_dir / f"{dataset_name}_palette.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["donuts", "image", "both", "metrics"])
    parser.add_argument("--csv", default="final_datasets/donuts.csv")
    parser.add_argument("--image", default="pictures/cvete.jpg")
    parser.add_argument("--output-dir", default="results_data_jean_25_05_raw")
    parser.add_argument("--donuts-k", type=int, default=3)
    parser.add_argument("--image-k", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="*", default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--data-root", default="data_jean_final_25_05/05")
    parser.add_argument("--weighted-steps", type=int, default=20)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    if args.mode in ("donuts", "both"):
        run_donuts(args.csv, reference_k=args.donuts_k, sizes=tuple(args.sizes), output_dir=args.output_dir)
    if args.mode in ("image", "both"):
        run_cvete(args.image, reference_k=args.image_k, sizes=tuple(args.sizes), output_dir=args.output_dir)
    if args.mode == "metrics":
        run_metrics_biased_grid(
            data_root=args.data_root,
            output_dir=args.output_dir,
            weighted_steps=args.weighted_steps,
            dataset_filter=args.datasets,
        )


if __name__ == "__main__":
    main()
