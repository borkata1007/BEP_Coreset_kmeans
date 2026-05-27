import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from kmeans_pp_nd import kmeans_plus_plus_local_search_full, kmeans_plus_plus_local_search_weighted, compute_kmeans_cost
from Exponential_quadtree_nd import exponential_quadtree_coreset
from image_processors import (
    _assign_nearest_centers_chunked,
    _compute_kmeans_cost_chunked,
    compress_image_with_coreset,
    save_compressed_image,
)
from srcjean.coresets.egq_coreset import EGQCoreset


def load_dataset_uber(csv_path="uber-raw-data-jul14.csv"):
    """Parse Uber CSV into standardized [Lat, Lon, hour, day] features."""
    df = pd.read_csv(csv_path)
    dt = pd.to_datetime(df["Date/Time"])
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day

    X = df[["Lat", "Lon", "hour", "day"]].values.astype(float)
    X = StandardScaler().fit_transform(X)

    k = 8
    title = "Uber Pickups NYC"
    return X, k, title


def load_dataset_donuts(csv_path="final_datasets/donuts.csv"):
    """Parse the 2D donuts CSV into standardized [x, y] features."""
    df = pd.read_csv(csv_path)
    X = df[["x", "y"]].values.astype(float)
    X = StandardScaler().fit_transform(X)

    k = 9
    title = "Donuts"
    return X, k, title


def workflow_ranked_egq_size(
    X,
    k,
    title,
    target_size,
    lloyd_iterations=4,
    verify_ranked=True,
    seed=0,
    output_dir="results/ranked_egq_size",
):
    """Run Jean's ranked EGQ size-targeting workflow."""
    import time

    output_path = Path(output_dir) / title.lower().replace(" ", "_") / f"target_{target_size}"
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"RANKED EGQ SIZE WORKFLOW: {title}")
    print("=" * 60)
    print(f"target coreset size: {target_size}")
    print(f"k: {k}")

    np.random.seed(seed)
    start = time.perf_counter()
    egq = EGQCoreset(
        target_size,
        k=k,
        lloyd_iterations=lloyd_iterations,
        verify_ranked=verify_ranked,
        sizing_workflow="ranked",
    )
    Q = egq.generate(X)
    elapsed = time.perf_counter() - start

    print("Number of original points:", X.shape[0])
    print("Raw formula coreset points:", egq.raw_size)
    print("Number of coreset points:", Q.shape[0])
    print("Trimmed to target:", egq.trimmed_to_target)
    print("Trimmed weights rescaled:", egq.trimmed_weight_rescaled)
    print("Weights sum:", egq.weights.sum())
    print("Z used:", egq.z_used)
    print("Beta used:", egq.beta_used)
    print("Minimum epsilon implied by z:", egq.eps_min)
    print("Reported epsilon:", egq.eps_report)
    print("Rank interval:", egq.rank_interval)
    print("Verification:", egq.rank_verification)
    print("Reference cost:", egq.reference_cost)
    print(f"Runtime seconds: {elapsed:.3f}")
    print("Workflow completed.\n")

    summary = {
        "dataset": title,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "k": int(k),
        "target_coreset_size": int(target_size),
        "raw_coreset_size": int(egq.raw_size),
        "actual_coreset_size": int(Q.shape[0]),
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
        "runtime_seconds": float(elapsed),
        "verification_matched_size": None
        if egq.rank_verification is None
        else bool(egq.rank_verification["matched_size"]),
        "verification_ranked_size": None
        if egq.rank_verification is None
        else int(egq.rank_verification["ranked_size"]),
        "verification_formula_size": None
        if egq.rank_verification is None
        else int(egq.rank_verification["formula_size"]),
    }

    pd.DataFrame([summary]).to_csv(output_path / "summary.csv", index=False)
    pd.Series(summary).to_json(output_path / "summary.json", indent=2)

    coreset_df = pd.DataFrame(Q, columns=[f"x{i}" for i in range(Q.shape[1])])
    coreset_df["weight"] = egq.weights
    coreset_df["source_index"] = egq.indices
    coreset_df.to_csv(output_path / "coreset.csv", index=False)

    if X.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        rng = np.random.default_rng(seed)
        sample_size = min(20000, X.shape[0])
        sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        ax.scatter(
            X[sample_idx, 0],
            X[sample_idx, 1],
            c="lightgray",
            s=2,
            alpha=0.35,
            edgecolors="none",
            label="data sample",
        )
        sizes = 10 + 90 * (egq.weights / egq.weights.max())
        scatter = ax.scatter(
            Q[:, 0],
            Q[:, 1],
            c=egq.weights,
            cmap="viridis",
            s=sizes,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
            label="coreset",
        )
        ax.set_title(f"{title}: ranked EGQ target {target_size}, actual {Q.shape[0]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("weight")
        fig.tight_layout()
        fig.savefig(output_path / "coreset.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            X[sample_idx, 0],
            X[sample_idx, 1],
            c="lightgray",
            s=2,
            alpha=0.25,
            edgecolors="none",
            label="data sample",
        )
        for bounds in egq.cells:
            x0, x1 = bounds[0], bounds[1]
            y0, y1 = bounds[2], bounds[3]
            xs = [x0, x1, x1, x0, x0]
            ys = [y0, y0, y1, y1, y0]
            ax.plot(xs, ys, color="black", linewidth=0.35, alpha=0.35)
        ax.scatter(
            Q[:, 0],
            Q[:, 1],
            c="tab:blue",
            s=8,
            alpha=0.85,
            edgecolors="none",
            label="coreset reps",
        )
        ax.set_title(f"{title}: ranked EGQ quadtree cells")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path / "quadtree_boxes.png", dpi=200)
        plt.close(fig)

    print(f"Saved results to: {output_path}")
    return Q, egq


def workflow_ranked_egq_image(
    image_path,
    n_colors,
    target_coreset_size,
    lloyd_iterations=4,
    local_search_steps=20,
    verify_ranked=True,
    seed=0,
    output_dir="results/ranked_egq_size",
):
    """Compress an image with size-targeted ranked EGQ on RGB points."""
    import time

    image_path = Path(image_path)
    output_path = (
        Path(output_dir)
        / "images"
        / image_path.stem.lower().replace(" ", "_")
        / f"colors_{n_colors}_target_{target_coreset_size}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"RANKED EGQ IMAGE WORKFLOW: {image_path.name}")
    print("=" * 60)
    print(f"colors: {n_colors}")
    print(f"target coreset size: {target_coreset_size}")

    img = plt.imread(image_path)
    if img.dtype.kind == "f":
        rgb_image = np.clip(img[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_image = img[:, :, :3].astype(np.uint8)

    height, width = rgb_image.shape[:2]
    rgb_points = rgb_image.reshape(-1, 3).astype(float)

    np.random.seed(seed)
    start = time.perf_counter()
    egq = EGQCoreset(
        target_coreset_size,
        k=n_colors,
        lloyd_iterations=lloyd_iterations,
        verify_ranked=verify_ranked,
        sizing_workflow="ranked",
    )
    coreset_points = egq.generate(rgb_points)

    final_centers, coreset_final_cost = kmeans_plus_plus_local_search_weighted(
        coreset_points,
        egq.weights,
        n_colors,
        n_steps=local_search_steps,
        random_state=seed,
        verbose=False,
    )
    chunk_size = _image_chunk_size(n_colors)
    final_cost = _compute_kmeans_cost_chunked(
        rgb_points,
        final_centers,
        chunk_size=chunk_size,
    )
    nearest_center_indices = _assign_nearest_centers_chunked(
        rgb_points,
        final_centers,
        chunk_size=chunk_size,
    )
    compressed_rgb = np.clip(final_centers[nearest_center_indices], 0, 255).astype(np.uint8)
    compressed_img = compressed_rgb.reshape((height, width, 3))
    elapsed = time.perf_counter() - start

    compressed_path = output_path / "compressed.png"
    original_path = output_path / "original.png"
    palette_path = output_path / "palette.png"

    save_compressed_image(compressed_img, compressed_path)
    save_compressed_image(rgb_image, original_path)
    _save_palette_image(final_centers, palette_path)

    summary = {
        "image": image_path.name,
        "height": int(height),
        "width": int(width),
        "full_size": int(rgb_points.shape[0]),
        "colors": int(n_colors),
        "target_coreset_size": int(target_coreset_size),
        "raw_coreset_size": int(egq.raw_size),
        "actual_coreset_size": int(coreset_points.shape[0]),
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
        "full_final_cost": float(final_cost),
        "coreset_final_cost": float(coreset_final_cost),
        "runtime_seconds": float(elapsed),
        "chunk_size": int(chunk_size),
        "verification_matched_size": None
        if egq.rank_verification is None
        else bool(egq.rank_verification["matched_size"]),
        "verification_ranked_size": None
        if egq.rank_verification is None
        else int(egq.rank_verification["ranked_size"]),
        "verification_formula_size": None
        if egq.rank_verification is None
        else int(egq.rank_verification["formula_size"]),
        "compressed_path": str(compressed_path),
        "palette_path": str(palette_path),
    }
    pd.DataFrame([summary]).to_csv(output_path / "summary.csv", index=False)
    pd.Series(summary).to_json(output_path / "summary.json", indent=2)

    coreset_df = pd.DataFrame(coreset_points, columns=["r", "g", "b"])
    coreset_df["weight"] = egq.weights
    coreset_df["source_index"] = egq.indices
    coreset_df.to_csv(output_path / "coreset.csv", index=False)

    palette_df = pd.DataFrame(np.clip(final_centers, 0, 255), columns=["r", "g", "b"])
    palette_df.to_csv(output_path / "palette.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(compressed_img)
    axes[1].set_title(f"{n_colors} colors")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path / "comparison.png", dpi=200)
    plt.close(fig)

    print("Raw formula coreset points:", egq.raw_size)
    print("Final coreset points:", coreset_points.shape[0])
    print("Trimmed to target:", egq.trimmed_to_target)
    print("Z used:", egq.z_used)
    print("Reported epsilon:", egq.eps_report)
    print("Beta used:", egq.beta_used)
    print("Full final cost:", final_cost)
    print(f"Runtime seconds: {elapsed:.3f}")
    print(f"Saved image results to: {output_path}")
    print("Workflow completed.\n")

    return compressed_img, egq, final_centers


def workflow_ranked_egq_image_color_sweep(
    image_path,
    color_powers=range(1, 12),
    target_coreset_size=2500,
    lloyd_iterations=4,
    local_search_steps=20,
    verify_ranked=True,
    seed=0,
    output_dir="results/ranked_egq_size",
):
    rows = []
    for power in color_powers:
        n_colors = 2 ** power
        _, egq, centers = workflow_ranked_egq_image(
            image_path,
            n_colors=n_colors,
            target_coreset_size=target_coreset_size,
            lloyd_iterations=lloyd_iterations,
            local_search_steps=local_search_steps,
            verify_ranked=verify_ranked,
            seed=seed,
            output_dir=output_dir,
        )
        rows.append(
            {
                "power": int(power),
                "colors": int(n_colors),
                "target_coreset_size": int(target_coreset_size),
                "raw_coreset_size": int(egq.raw_size),
                "actual_coreset_size": int(egq.weights.shape[0]),
                "trimmed_to_target": bool(egq.trimmed_to_target),
                "z_used": float(egq.z_used),
                "eps_report": float(egq.eps_report),
                "beta_used": float(egq.beta_used),
                "n_centers": int(centers.shape[0]),
            }
        )

    image_name = Path(image_path).stem.lower().replace(" ", "_")
    sweep_path = Path(output_dir) / "images" / image_name / f"sweep_target_{target_coreset_size}"
    sweep_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(sweep_path / "sweep_summary.csv", index=False)
    print(f"Saved sweep summary to: {sweep_path / 'sweep_summary.csv'}")
    return df


def workflow_ranked_egq_image_coreset_sweep(
    image_path,
    n_colors=8,
    target_sizes=(512, 1024, 2048),
    lloyd_iterations=4,
    local_search_steps=20,
    verify_ranked=True,
    seed=0,
    output_dir="results/ranked_egq_size",
):
    rows = []
    for target_size in target_sizes:
        _, egq, centers = workflow_ranked_egq_image(
            image_path,
            n_colors=n_colors,
            target_coreset_size=target_size,
            lloyd_iterations=lloyd_iterations,
            local_search_steps=local_search_steps,
            verify_ranked=verify_ranked,
            seed=seed,
            output_dir=output_dir,
        )
        rows.append(
            {
                "colors": int(n_colors),
                "target_coreset_size": int(target_size),
                "raw_coreset_size": int(egq.raw_size),
                "actual_coreset_size": int(egq.weights.shape[0]),
                "trimmed_to_target": bool(egq.trimmed_to_target),
                "z_used": float(egq.z_used),
                "eps_report": float(egq.eps_report),
                "beta_used": float(egq.beta_used),
                "n_centers": int(centers.shape[0]),
            }
        )

    image_name = Path(image_path).stem.lower().replace(" ", "_")
    sweep_path = Path(output_dir) / "images" / image_name / f"colors_{n_colors}_coreset_sweep"
    sweep_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(sweep_path / "sweep_summary.csv", index=False)
    print(f"Saved coreset-size sweep summary to: {sweep_path / 'sweep_summary.csv'}")
    return df


def workflow_ranked_egq_image_palette(
    image_path,
    reference_k=8,
    target_palette_size=512,
    lloyd_iterations=4,
    verify_ranked=True,
    seed=0,
    output_dir="results/ranked_egq_size",
):
    """Compress an image by using EGQ representatives directly as palette colors."""
    import time
    from sklearn.neighbors import KDTree

    image_path = Path(image_path)
    output_path = (
        Path(output_dir)
        / "images"
        / image_path.stem.lower().replace(" ", "_")
        / f"reference_k_{reference_k}_palette_{target_palette_size}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"RANKED EGQ IMAGE PALETTE WORKFLOW: {image_path.name}")
    print("=" * 60)
    print(f"reference k for EGQ: {reference_k}")
    print(f"target palette colors: {target_palette_size}")

    img = plt.imread(image_path)
    if img.dtype.kind == "f":
        rgb_image = np.clip(img[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_image = img[:, :, :3].astype(np.uint8)

    height, width = rgb_image.shape[:2]
    rgb_points = rgb_image.reshape(-1, 3).astype(float)

    np.random.seed(seed)
    start = time.perf_counter()
    egq = EGQCoreset(
        target_palette_size,
        k=reference_k,
        lloyd_iterations=lloyd_iterations,
        verify_ranked=verify_ranked,
        sizing_workflow="ranked",
    )
    palette = np.clip(egq.generate(rgb_points), 0, 255)

    tree = KDTree(palette)
    _, nearest = tree.query(rgb_points, k=1)
    compressed_rgb = palette[nearest[:, 0]].astype(np.uint8)
    compressed_img = compressed_rgb.reshape((height, width, 3))
    final_cost = float(np.sum((rgb_points - palette[nearest[:, 0]]) ** 2))
    elapsed = time.perf_counter() - start

    compressed_path = output_path / "compressed.png"
    original_path = output_path / "original.png"
    palette_path = output_path / "palette.png"
    save_compressed_image(compressed_img, compressed_path)
    save_compressed_image(rgb_image, original_path)
    _save_palette_image(palette, palette_path)

    summary = {
        "image": image_path.name,
        "height": int(height),
        "width": int(width),
        "full_size": int(rgb_points.shape[0]),
        "reference_k": int(reference_k),
        "target_palette_size": int(target_palette_size),
        "raw_palette_size": int(egq.raw_size),
        "actual_palette_size": int(palette.shape[0]),
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
        "palette_reconstruction_cost": float(final_cost),
        "runtime_seconds": float(elapsed),
        "verification_matched_size": None
        if egq.rank_verification is None
        else bool(egq.rank_verification["matched_size"]),
        "verification_ranked_size": None
        if egq.rank_verification is None
        else int(egq.rank_verification["ranked_size"]),
        "verification_formula_size": None
        if egq.rank_verification is None
        else int(egq.rank_verification["formula_size"]),
        "compressed_path": str(compressed_path),
        "palette_path": str(palette_path),
    }
    pd.DataFrame([summary]).to_csv(output_path / "summary.csv", index=False)
    pd.Series(summary).to_json(output_path / "summary.json", indent=2)

    coreset_df = pd.DataFrame(palette, columns=["r", "g", "b"])
    coreset_df["weight"] = egq.weights
    coreset_df["source_index"] = egq.indices
    coreset_df.to_csv(output_path / "palette_coreset.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(compressed_img)
    axes[1].set_title(f"{palette.shape[0]} EGQ colors")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path / "comparison.png", dpi=200)
    plt.close(fig)

    print("Raw formula palette colors:", egq.raw_size)
    print("Final palette colors:", palette.shape[0])
    print("Trimmed to target:", egq.trimmed_to_target)
    print("Reference k:", reference_k)
    print("Z used:", egq.z_used)
    print("Reported epsilon:", egq.eps_report)
    print("Palette reconstruction cost:", final_cost)
    print(f"Runtime seconds: {elapsed:.3f}")
    print(f"Saved palette image results to: {output_path}")
    print("Workflow completed.\n")
    return compressed_img, egq, palette, summary


def workflow_ranked_egq_image_palette_sweep(
    image_path,
    reference_k=8,
    palette_sizes=(512, 1024, 2048),
    lloyd_iterations=4,
    verify_ranked=True,
    seed=0,
    output_dir="results/ranked_egq_size",
):
    rows = []
    for palette_size in palette_sizes:
        _, egq, palette, summary = workflow_ranked_egq_image_palette(
            image_path,
            reference_k=reference_k,
            target_palette_size=palette_size,
            lloyd_iterations=lloyd_iterations,
            verify_ranked=verify_ranked,
            seed=seed,
            output_dir=output_dir,
        )
        rows.append(
            {
                "reference_k": int(reference_k),
                "target_palette_size": int(palette_size),
                "raw_palette_size": int(egq.raw_size),
                "actual_palette_size": int(palette.shape[0]),
                "trimmed_to_target": bool(egq.trimmed_to_target),
                "z_used": float(egq.z_used),
                "eps_report": float(egq.eps_report),
                "beta_used": float(egq.beta_used),
                "initial_reference_cost": float(summary["reference_cost"]),
                "palette_reconstruction_cost": float(summary["palette_reconstruction_cost"]),
                "cost_over_initial": float(
                    summary["palette_reconstruction_cost"] / summary["reference_cost"]
                ),
                "runtime_seconds": float(summary["runtime_seconds"]),
            }
        )

    image_name = Path(image_path).stem.lower().replace(" ", "_")
    sweep_path = Path(output_dir) / "images" / image_name / f"reference_k_{reference_k}_palette_sweep"
    sweep_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(sweep_path / "sweep_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        df["target_palette_size"],
        df["cost_over_initial"],
        marker="o",
        linewidth=1.8,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EGQ palette size")
    ax.set_ylabel("cost / initial reference cost")
    ax.set_title(f"{image_name}: EGQ palette cost vs reference k={reference_k} cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(sweep_path / "cost_ratio_line.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        df["target_palette_size"],
        df["palette_reconstruction_cost"],
        marker="o",
        linewidth=1.8,
        label="EGQ palette reconstruction",
    )
    ax.axhline(
        df["initial_reference_cost"].iloc[0],
        color="tab:red",
        linestyle="--",
        linewidth=1.4,
        label=f"initial reference k={reference_k} cost",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EGQ palette size")
    ax.set_ylabel("full-image k-means cost")
    ax.set_title(f"{image_name}: reconstruction cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(sweep_path / "cost_line.png", dpi=200)
    plt.close(fig)

    print(f"Saved palette-size sweep summary to: {sweep_path / 'sweep_summary.csv'}")
    print(f"Saved cost ratio plot to: {sweep_path / 'cost_ratio_line.png'}")
    return df


def workflow_ranked_egq_points_palette(
    X,
    title,
    reference_k=3,
    target_palette_size=512,
    lloyd_iterations=4,
    verify_ranked=True,
    seed=0,
    output_dir="results/ranked_egq_size",
):
    """Use EGQ representatives directly as a 2D point palette/summary."""
    import time
    from sklearn.neighbors import KDTree

    output_path = (
        Path(output_dir)
        / "points"
        / title.lower().replace(" ", "_")
        / f"reference_k_{reference_k}_palette_{target_palette_size}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"RANKED EGQ POINT PALETTE WORKFLOW: {title}")
    print("=" * 60)
    print(f"reference k for EGQ: {reference_k}")
    print(f"target coreset points: {target_palette_size}")

    np.random.seed(seed)
    start = time.perf_counter()
    egq = EGQCoreset(
        target_palette_size,
        k=reference_k,
        lloyd_iterations=lloyd_iterations,
        verify_ranked=verify_ranked,
        sizing_workflow="ranked",
    )
    coreset_points = egq.generate(X)

    tree = KDTree(coreset_points)
    _, nearest = tree.query(X, k=1)
    assignments = nearest[:, 0]
    assigned_points = coreset_points[assignments]
    reconstruction_cost = float(np.sum((X - assigned_points) ** 2))
    elapsed = time.perf_counter() - start

    summary = {
        "dataset": title,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "reference_k": int(reference_k),
        "target_palette_size": int(target_palette_size),
        "raw_palette_size": int(egq.raw_size),
        "actual_palette_size": int(coreset_points.shape[0]),
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
        if egq.rank_verification is None
        else bool(egq.rank_verification["matched_size"]),
        "verification_ranked_size": None
        if egq.rank_verification is None
        else int(egq.rank_verification["ranked_size"]),
        "verification_formula_size": None
        if egq.rank_verification is None
        else int(egq.rank_verification["formula_size"]),
    }
    pd.DataFrame([summary]).to_csv(output_path / "summary.csv", index=False)
    pd.Series(summary).to_json(output_path / "summary.json", indent=2)

    coreset_df = pd.DataFrame(coreset_points, columns=["x", "y"])
    coreset_df["weight"] = egq.weights
    coreset_df["source_index"] = egq.indices
    coreset_df.to_csv(output_path / "palette_coreset.csv", index=False)

    assignments_df = pd.DataFrame(
        {
            "point_index": np.arange(X.shape[0], dtype=int),
            "coreset_index": assignments.astype(int),
        }
    )
    assignments_df.to_csv(output_path / "assignments.csv", index=False)

    _save_point_assignment_plot(
        X,
        coreset_points,
        assignments,
        output_path / "assignment_coloring.png",
        title=f"{title}: {coreset_points.shape[0]} EGQ summaries",
        seed=seed,
    )
    _save_point_coreset_plot(
        X,
        coreset_points,
        egq.weights,
        output_path / "coreset.png",
        title=f"{title}: ranked EGQ target {target_palette_size}, actual {coreset_points.shape[0]}",
        seed=seed,
    )
    _save_point_quadtree_boxes_plot(
        X,
        coreset_points,
        egq.cells,
        output_path / "quadtree_boxes.png",
        title=f"{title}: ranked EGQ quadtree cells",
        seed=seed,
    )

    print("Raw formula coreset points:", egq.raw_size)
    print("Final coreset points:", coreset_points.shape[0])
    print("Trimmed to target:", egq.trimmed_to_target)
    print("Reference k:", reference_k)
    print("Z used:", egq.z_used)
    print("Reported epsilon:", egq.eps_report)
    print("Palette reconstruction cost:", reconstruction_cost)
    print("Cost over initial reference cost:", reconstruction_cost / egq.reference_cost)
    print(f"Runtime seconds: {elapsed:.3f}")
    print(f"Saved point palette results to: {output_path}")
    print("Workflow completed.\n")
    return egq, coreset_points, summary


def workflow_ranked_egq_points_palette_sweep(
    X,
    title,
    reference_k=3,
    palette_sizes=tuple(2 ** power for power in range(1, 12)),
    lloyd_iterations=4,
    verify_ranked=True,
    seed=0,
    output_dir="results/ranked_egq_size",
):
    rows = []
    for palette_size in palette_sizes:
        egq, coreset_points, summary = workflow_ranked_egq_points_palette(
            X,
            title,
            reference_k=reference_k,
            target_palette_size=palette_size,
            lloyd_iterations=lloyd_iterations,
            verify_ranked=verify_ranked,
            seed=seed,
            output_dir=output_dir,
        )
        rows.append(
            {
                "reference_k": int(reference_k),
                "target_palette_size": int(palette_size),
                "raw_palette_size": int(egq.raw_size),
                "actual_palette_size": int(coreset_points.shape[0]),
                "trimmed_to_target": bool(egq.trimmed_to_target),
                "z_used": float(egq.z_used),
                "eps_report": float(egq.eps_report),
                "beta_used": float(egq.beta_used),
                "initial_reference_cost": float(summary["reference_cost"]),
                "palette_reconstruction_cost": float(summary["palette_reconstruction_cost"]),
                "cost_over_initial": float(summary["cost_over_initial"]),
                "runtime_seconds": float(summary["runtime_seconds"]),
            }
        )

    sweep_path = (
        Path(output_dir)
        / "points"
        / title.lower().replace(" ", "_")
        / f"reference_k_{reference_k}_palette_sweep"
    )
    sweep_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(sweep_path / "sweep_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["target_palette_size"], df["cost_over_initial"], marker="o", linewidth=1.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EGQ coreset size")
    ax.set_ylabel("cost / initial reference cost")
    ax.set_title(f"{title}: EGQ summary cost vs reference k={reference_k} cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(sweep_path / "cost_ratio_line.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        df["target_palette_size"],
        df["palette_reconstruction_cost"],
        marker="o",
        linewidth=1.8,
        label="EGQ representative reconstruction",
    )
    ax.axhline(
        df["initial_reference_cost"].iloc[0],
        color="tab:red",
        linestyle="--",
        linewidth=1.4,
        label=f"initial reference k={reference_k} cost",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EGQ coreset size")
    ax.set_ylabel("full-data squared reconstruction cost")
    ax.set_title(f"{title}: reconstruction cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(sweep_path / "cost_line.png", dpi=200)
    plt.close(fig)

    print(f"Saved points sweep summary to: {sweep_path / 'sweep_summary.csv'}")
    print(f"Saved points cost ratio plot to: {sweep_path / 'cost_ratio_line.png'}")
    return df


def _save_point_assignment_plot(X, coreset_points, assignments, output_path, title, seed=0):
    sample_size = min(120000, X.shape[0])
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)
    n_colors = coreset_points.shape[0]
    cmap = plt.get_cmap("turbo", max(n_colors, 2))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        X[sample_idx, 0],
        X[sample_idx, 1],
        c=assignments[sample_idx],
        cmap=cmap,
        s=2,
        alpha=0.45,
        edgecolors="none",
        vmin=0,
        vmax=max(n_colors - 1, 1),
    )
    ax.scatter(
        coreset_points[:, 0],
        coreset_points[:, 1],
        c=np.arange(n_colors),
        cmap=cmap,
        s=24,
        edgecolors="black",
        linewidths=0.35,
        vmin=0,
        vmax=max(n_colors - 1, 1),
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_point_coreset_plot(X, coreset_points, weights, output_path, title, seed=0):
    sample_size = min(120000, X.shape[0])
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        X[sample_idx, 0],
        X[sample_idx, 1],
        c="lightgray",
        s=2,
        alpha=0.35,
        edgecolors="none",
        label="data sample",
    )
    sizes = 10 + 90 * (weights / weights.max())
    scatter = ax.scatter(
        coreset_points[:, 0],
        coreset_points[:, 1],
        c=weights,
        cmap="viridis",
        s=sizes,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
        label="coreset",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_point_quadtree_boxes_plot(X, coreset_points, cells, output_path, title, seed=0):
    sample_size = min(120000, X.shape[0])
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        X[sample_idx, 0],
        X[sample_idx, 1],
        c="lightgray",
        s=2,
        alpha=0.25,
        edgecolors="none",
        label="data sample",
    )
    for bounds in cells:
        x0, x1 = bounds[0], bounds[1]
        y0, y1 = bounds[2], bounds[3]
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        ax.plot(xs, ys, color="black", linewidth=0.35, alpha=0.35)
    ax.scatter(
        coreset_points[:, 0],
        coreset_points[:, 1],
        c="tab:blue",
        s=8,
        alpha=0.85,
        edgecolors="none",
        label="coreset reps",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _image_chunk_size(n_centers, max_distance_entries=5_000_000):
    return max(512, min(250_000, max_distance_entries // max(1, int(n_centers))))


def _save_palette_image(centers, output_path):
    colors = np.clip(centers, 0, 255).astype(np.uint8)
    swatch_width = 40
    swatch_height = 60
    palette = np.zeros((swatch_height, swatch_width * colors.shape[0], 3), dtype=np.uint8)
    for idx, color in enumerate(colors):
        start = idx * swatch_width
        palette[:, start:start + swatch_width] = color
    save_compressed_image(palette, output_path)


def collect_coreset_metrics(
    X,
    k,
    eps,
    local_search_steps=67,
    compression_ratio=None,
    beta=None,
    beta_search_precision=None,
    verbose=True,
):
    """Run one experiment and return all metrics/results needed by workflows."""
    print("begins kmeans++ local search (full data)...")
    c, _ = kmeans_plus_plus_local_search_full(
        X,
        k,
        n_steps=local_search_steps,
        random_state=0,
        verbose=verbose,
    )
    print("finished kmeans++ local search")

    print("begins coreset building...")
    coreset_points, coreset_weights, cells, info = exponential_quadtree_coreset(
        X,
        c,
        eps,
        random_state=0,
        beta=beta,
        compression_ratio=compression_ratio,
        beta_search_precision=beta_search_precision,
        verbose=verbose,
        return_info=True,
    )
    print("finished coreset building")

    full_cost = compute_kmeans_cost(X, c)
    coreset_cost = compute_kmeans_cost(coreset_points, c, weights=coreset_weights)

    print("begins weighted kmeans++ local search (on coreset)...")
    c_prime, c_prime_cost_on_coreset = kmeans_plus_plus_local_search_weighted(
        coreset_points,
        coreset_weights,
        k,
        n_steps=local_search_steps,
        random_state=0,
        verbose=verbose,
    )
    print("finished weighted kmeans++ local search")

    full_cost_with_c_prime = compute_kmeans_cost(X, c_prime)
    p_over_q = info["n_initial"] / info["n_coreset"]
    qc_over_pc = coreset_cost / full_cost
    qcp_over_pcp = c_prime_cost_on_coreset / full_cost_with_c_prime
    pcp_over_pc = full_cost_with_c_prime / full_cost

    return {
        "c": c,
        "c_prime": c_prime,
        "coreset_points": coreset_points,
        "coreset_weights": coreset_weights,
        "cells": cells,
        "info": info,
        "full_cost": full_cost,
        "coreset_cost": coreset_cost,
        "c_prime_cost_on_coreset": c_prime_cost_on_coreset,
        "full_cost_with_c_prime": full_cost_with_c_prime,
        "p_over_q": p_over_q,
        "qc_over_pc": qc_over_pc,
        "qcp_over_pcp": qcp_over_pcp,
        "pcp_over_pc": pcp_over_pc,
    }


def run_coreset_workflow(
    X,
    k,
    title,
    eps,
    local_search_steps=67,
    compression_ratio=None,
    beta=None,
    beta_search_precision=None,
    verbose=True,
    plot_dims=(0, 1),
    plot_labels=None,
    equal_aspect=False,
):
    """Generic coreset workflow for any numeric dataset in shape (n, d)."""
    print("=" * 60)
    print(f"DATASET WORKFLOW: {title}")
    print("=" * 60)

    results = collect_coreset_metrics(
        X,
        k,
        eps,
        local_search_steps=local_search_steps,
        compression_ratio=compression_ratio,
        beta=beta,
        beta_search_precision=beta_search_precision,
        verbose=verbose,
    )
    c = results["c"]
    c_prime = results["c_prime"]
    coreset_points = results["coreset_points"]
    coreset_weights = results["coreset_weights"]
    cells = results["cells"]
    info = results["info"]
    full_cost = results["full_cost"]
    coreset_cost = results["coreset_cost"]
    c_prime_cost_on_coreset = results["c_prime_cost_on_coreset"]
    full_cost_with_c_prime = results["full_cost_with_c_prime"]
    p_over_q = results["p_over_q"]
    qc_over_pc = results["qc_over_pc"]
    qcp_over_pcp = results["qcp_over_pcp"]
    pcp_over_pc = results["pcp_over_pc"]

    print("Number of original points:", info["n_initial"])
    print("Number of coreset points:", info["n_coreset"])
    print("Beta used:", info["beta"])
    requested_p_over_q = None if compression_ratio is None else 1.0 / compression_ratio

    print("\n--- Cost Summary ---")
    print("Full data k-means cost:", full_cost)
    print("Coreset weighted cost:", coreset_cost)
    print("Requested compression ratio:", compression_ratio)
    print("Achieved compression ratio:", info["compression_ratio_achieved"])
    if requested_p_over_q is not None:
        print("Requested inverse compression ratio (|P|/|Q| target):", requested_p_over_q)
    print("Achieved inverse compression ratio (|P|/|Q|):", info["n_initial"] / info["n_coreset"])

    print("\n--- Comparison of centers ---")
    print("Full data cost with centers c (full local):", full_cost)
    print("Full data cost with centers c' (coreset local):", full_cost_with_c_prime)
    print("Cost ratio (c' vs c):", pcp_over_pc)
    print("Coreset internal cost c' (weighted):", c_prime_cost_on_coreset)

    print("\n--- Table Metrics ---")
    print(f"eps: {eps:.8f}")
    if requested_p_over_q is not None:
        print(f"|P|/|Q| target: {requested_p_over_q:.8f}")
    print(f"|P|/|Q| achieved: {p_over_q:.8f}")
    print(f"k: {k}")
    print(f"beta: {info['beta']:.8f}")
    print(f"Cost(Q,C)/Cost(P,C): {qc_over_pc:.8f}")
    print(f"Cost(Q,C')/Cost(P,C'): {qcp_over_pcp:.8f}")
    print(f"Cost(P,C')/Cost(P,C): {pcp_over_pc:.8f}")

    x_dim, y_dim = plot_dims
    if plot_labels is None:
        x_label = f"Feature {x_dim}"
        y_label = f"Feature {y_dim}"
    else:
        x_label, y_label = plot_labels

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, x_dim], X[:, y_dim], c="lightgray", alpha=0.5, edgecolor="none", label="Data points")
    plt.scatter(c[:, x_dim], c[:, y_dim], c="red", s=150, marker="X", label="Centers c (full local)", linewidths=2)
    plt.scatter(c_prime[:, x_dim], c_prime[:, y_dim], c="blue", s=150, marker="+", label="Centers c' (coreset local)", linewidths=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}: centers comparison")
    if equal_aspect:
        plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, x_dim], X[:, y_dim], c="lightgray", alpha=0.5, edgecolor="none", label="Original points")
    for cube in cells:
        x0, x1 = cube[2 * x_dim], cube[2 * x_dim + 1]
        y0, y1 = cube[2 * y_dim], cube[2 * y_dim + 1]
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        plt.plot(xs, ys, color="gray", linewidth=0.6, alpha=0.6)

    sizes = 15 * (coreset_weights / coreset_weights.max())
    scatter = plt.scatter(
        coreset_points[:, x_dim],
        coreset_points[:, y_dim],
        c=coreset_weights,
        cmap="Greens",
        s=40 + sizes,
        edgecolor="k",
        vmin=coreset_weights.min(),
        vmax=coreset_weights.max(),
        label="Coreset reps",
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Weight")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}: EQT coreset projection")
    if equal_aspect:
        plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Workflow completed.\n")


def workflow_fixed_beta_eps_sweep_to_csv(
    X,
    k,
    title,
    beta,
    eps_values=None,
    local_search_steps=67,
    verbose=True,
    output_csv="Plots/fixed_beta_eps_sweep.csv",
):
    """Sweep epsilon values for a fixed beta and save metrics to CSV."""
    if eps_values is None:
        eps_values = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FIXED-BETA EPS SWEEP: {title}")
    print("=" * 60)
    print(f"beta: {beta}")
    print(f"eps values: {list(eps_values)}")

    rows = []
    for eps in eps_values:
        print("\n" + "-" * 60)
        print(f"running eps={eps}")
        print("-" * 60)

        results = collect_coreset_metrics(
            X,
            k,
            eps,
            local_search_steps=local_search_steps,
            compression_ratio=None,
            beta=beta,
            beta_search_precision=None,
            verbose=verbose,
        )
        info = results["info"]

        row = {
            "dataset": title,
            "eps": float(eps),
            "k": int(k),
            "beta": float(info["beta"]),
            "n_initial": int(info["n_initial"]),
            "n_coreset": int(info["n_coreset"]),
            "compression_ratio_achieved": float(info["compression_ratio_achieved"]),
            "p_over_q_achieved": float(results["p_over_q"]),
            "cost_p_c": float(results["full_cost"]),
            "cost_q_c": float(results["coreset_cost"]),
            "cost_p_c_prime": float(results["full_cost_with_c_prime"]),
            "cost_q_c_prime": float(results["c_prime_cost_on_coreset"]),
            "cost_qc_over_pc": float(results["qc_over_pc"]),
            "cost_qcprime_over_pcprime": float(results["qcp_over_pcp"]),
            "cost_pcprime_over_pc": float(results["pcp_over_pc"]),
        }
        rows.append(row)

        print(
            "saved row:",
            f"eps={row['eps']:.8f}",
            f"|P|/|Q|={row['p_over_q_achieved']:.8f}",
            f"beta={row['beta']:.8f}",
            f"Cost(Q,C)/Cost(P,C)={row['cost_qc_over_pc']:.8f}",
            f"Cost(Q,C')/Cost(P,C')={row['cost_qcprime_over_pcprime']:.8f}",
            f"Cost(P,C')/Cost(P,C)={row['cost_pcprime_over_pc']:.8f}",
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved epsilon sweep CSV to: {output_path}")
    return df


def workflow_eps(X, k, title, eps=0.1, local_search_steps=67, verbose=True, plot_dims=(0, 1)):
    """Generic workflow with eps only (default beta=4 behavior)."""
    return run_coreset_workflow(
        X,
        k,
        title,
        eps,
        local_search_steps=local_search_steps,
        compression_ratio=None,
        beta=None,
        verbose=verbose,
        plot_dims=plot_dims,
    )


def workflow_eps_compression_ratio(
    X,
    k,
    title,
    eps=0.1,
    compression_ratio=0.05,
    local_search_steps=67,
    beta_search_precision=None,
    verbose=True,
    plot_dims=(0, 1),
):
    """Generic workflow with eps + compression-ratio targeting."""
    return run_coreset_workflow(
        X,
        k,
        title,
        eps,
        local_search_steps=local_search_steps,
        compression_ratio=compression_ratio,
        beta=None,
        beta_search_precision=beta_search_precision,
        verbose=verbose,
        plot_dims=plot_dims,
    )


def workflow_uber_cost_ratios(
    eps=0.1,
    compression_ratio=0.05,
    local_search_steps=67,
    beta_search_precision=None,
    verbose=True,
    csv_path="uber-raw-data-jul14.csv",
):
    """Uber workflow that reports coreset/full-dataset cost comparisons."""
    X, k, title = load_dataset_uber(csv_path)

    print("=" * 60)
    print(f"UBER COST-RATIO WORKFLOW: {title}")
    print("=" * 60)
    print(f"eps: {eps}")
    print(f"requested compression ratio: {compression_ratio}")

    print("begins kmeans++ local search (full data)...")
    original_centers, points_to_original_centers = kmeans_plus_plus_local_search_full(
        X,
        k,
        n_steps=local_search_steps,
        random_state=0,
        verbose=verbose,
    )
    print("finished kmeans++ local search")

    print("begins coreset building...")
    coreset_points, coreset_weights, _, info = exponential_quadtree_coreset(
        X,
        original_centers,
        eps,
        random_state=0,
        compression_ratio=compression_ratio,
        beta_search_precision=beta_search_precision,
        verbose=verbose,
        return_info=True,
    )
    print("finished coreset building")
    print("Number of original points:", info["n_initial"])
    print("Number of coreset points:", info["n_coreset"])

    coreset_to_original_centers = compute_kmeans_cost(
        coreset_points,
        original_centers,
        weights=coreset_weights,
    )

    print("begins weighted kmeans++ local search (on coreset)...")
    coreset_centers, coreset_to_coreset_centers = kmeans_plus_plus_local_search_weighted(
        coreset_points,
        coreset_weights,
        k,
        n_steps=local_search_steps,
        random_state=0,
        verbose=verbose,
    )
    print("finished weighted kmeans++ local search")

    points_to_coreset_centers = compute_kmeans_cost(X, coreset_centers)

    print("\n--- Cost Values ---")
    print("Coreset to original centers:", coreset_to_original_centers)
    print("Pointset to original centers:", points_to_original_centers)
    print("Coreset to coreset centers:", coreset_to_coreset_centers)
    print("Pointset to coreset centers:", points_to_coreset_centers)

    print("\n--- Cost Ratios ---")
    print(
        "(coreset to original centers) / (pointset to original centers):",
        coreset_to_original_centers / points_to_original_centers,
    )
    print(
        "(coreset to coreset centers) / (pointset to coreset centers):",
        coreset_to_coreset_centers / points_to_coreset_centers,
    )
    print(
        "(pointset to coreset centers) / (pointset to initial centers):",
        points_to_coreset_centers / points_to_original_centers,
    )
    print("Beta used:", info["beta"])
    print("Achieved compression ratio:", info["compression_ratio_achieved"])
    print("Inverse compression ratio (n / coreset size):", info["n_initial"] / info["n_coreset"])
    print("Workflow completed.\n")


def workflow_uber_cost_ratios_beta(
    eps=0.1,
    beta=4.0,
    local_search_steps=67,
    verbose=True,
    csv_path="uber-raw-data-jul14.csv",
):
    """Uber workflow that reports coreset/full-dataset cost comparisons for a fixed beta."""
    X, k, title = load_dataset_uber(csv_path)

    print("=" * 60)
    print(f"UBER FIXED-BETA WORKFLOW: {title}")
    print("=" * 60)
    print(f"eps: {eps}")
    print(f"beta: {beta}")

    print("begins kmeans++ local search (full data)...")
    original_centers, points_to_original_centers = kmeans_plus_plus_local_search_full(
        X,
        k,
        n_steps=local_search_steps,
        random_state=0,
        verbose=verbose,
    )
    print("finished kmeans++ local search")

    print("begins coreset building...")
    coreset_points, coreset_weights, _, info = exponential_quadtree_coreset(
        X,
        original_centers,
        eps,
        random_state=0,
        beta=beta,
        verbose=verbose,
        return_info=True,
    )
    print("finished coreset building")
    print("Number of original points:", info["n_initial"])
    print("Number of coreset points:", info["n_coreset"])

    coreset_to_original_centers = compute_kmeans_cost(
        coreset_points,
        original_centers,
        weights=coreset_weights,
    )

    print("begins weighted kmeans++ local search (on coreset)...")
    coreset_centers, coreset_to_coreset_centers = kmeans_plus_plus_local_search_weighted(
        coreset_points,
        coreset_weights,
        k,
        n_steps=local_search_steps,
        random_state=0,
        verbose=verbose,
    )
    print("finished weighted kmeans++ local search")

    points_to_coreset_centers = compute_kmeans_cost(X, coreset_centers)

    print("\n--- Cost Values ---")
    print("Coreset to original centers:", coreset_to_original_centers)
    print("Pointset to original centers:", points_to_original_centers)
    print("Coreset to coreset centers:", coreset_to_coreset_centers)
    print("Pointset to coreset centers:", points_to_coreset_centers)

    print("\n--- Cost Ratios ---")
    print(
        "(coreset to original centers) / (pointset to original centers):",
        coreset_to_original_centers / points_to_original_centers,
    )
    print(
        "(coreset to coreset centers) / (pointset to coreset centers):",
        coreset_to_coreset_centers / points_to_coreset_centers,
    )
    print(
        "(pointset to coreset centers) / (pointset to initial centers):",
        points_to_coreset_centers / points_to_original_centers,
    )
    print("Beta used:", info["beta"])
    print("Achieved compression ratio:", info["compression_ratio_achieved"])
    print("Inverse compression ratio (n / coreset size):", info["n_initial"] / info["n_coreset"])
    print("Workflow completed.\n")


def workflow_eps_beta(
    X,
    k,
    title,
    eps=0.1,
    beta=4.0,
    local_search_steps=67,
    verbose=True,
    plot_dims=(0, 1),
):
    """Generic workflow with eps + explicit fixed beta."""
    return run_coreset_workflow(
        X,
        k,
        title,
        eps,
        local_search_steps=local_search_steps,
        compression_ratio=None,
        beta=beta,
        beta_search_precision=None,
        verbose=verbose,
        plot_dims=plot_dims,
    )


def workflow_image(
    image_path,
    t,
    eps=0.1,
    compression_ratio=0.02,
    local_search_steps=67,
    beta_search_precision=None,
    verbose=True,
):
    """Image compression workflow remains separate file-processing path."""
    print("=" * 60)
    print("IMAGE WORKFLOW: Image Compression with Coreset k-means")
    print("=" * 60)

    print(f"Loading image from: {image_path}")
    print("begins image compression...")
    compressed_img, original_shape, stats = compress_image_with_coreset(
        image_path,
        t,
        eps=eps,
        random_state=0,
        n_steps=local_search_steps,
        compression_ratio=compression_ratio,
        beta_search_precision=beta_search_precision,
        verbose=verbose,
    )
    print("finished image compression")

    output_path = image_path.replace(".png", f"_compressed_t{t}.png").replace(".jpg", f"_compressed_t{t}.jpg")
    save_compressed_image(compressed_img, output_path)
    print(f"Compressed image saved to: {output_path}")
    print(f"Image dimensions: {original_shape[0]} x {original_shape[1]} with {t} colors")
    print("\n--- Image Compression Statistics ---")
    print(f"Full dataset size (pixels): {stats['full_size']}")
    print(f"Coreset size: {stats['coreset_size']}")
    print(f"Requested compression ratio: {compression_ratio}")
    print(f"Achieved compression ratio: {stats['compression_ratio_achieved']:.4f}")
    print(f"Cost with initial centers: {stats['initial_cost']:.2f}")
    print(f"Cost with final centers: {stats['final_cost']:.2f}")
    print(f"Cost improvement ratio (final / initial): {stats['final_cost'] / stats['initial_cost']:.4f}")
    print("Image workflow completed.\n")


def main():
    """Choose loader + generic workflow mode."""
    reference_k = 3
    coreset_sizes = tuple(2 ** power for power in range(1, 12))
    lloyd_iterations = 4

    X, _, title = load_dataset_donuts("final_datasets/donuts.csv")
    workflow_ranked_egq_points_palette_sweep(
        X,
        title,
        reference_k=reference_k,
        palette_sizes=coreset_sizes,
        lloyd_iterations=lloyd_iterations,
        verify_ranked=True,
    )
    return

    # workflow_uber_cost_ratios_beta(
    #     eps=eps,
    #     beta=beta,
    #     local_search_steps=local_search_steps,
    #     verbose=verbose,
    #     csv_path="uber-raw-data-jul14.csv",
    # )

    # workflow_uber_cost_ratios(
    #     eps=eps,
    #     compression_ratio=compression_ratio,
    #     local_search_steps=local_search_steps,
    #     beta_search_precision=beta_search_precision,
    #     verbose=verbose,
    #     csv_path="uber-raw-data-jul14.csv",
    # )

    #Choose one generic mode:
    workflow_fixed_beta_eps_sweep_to_csv(
        X,
        k,
        title,
        beta=beta,
        eps_values=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        local_search_steps=local_search_steps,
        verbose=verbose,
        output_csv="Plots/fixed_beta_eps_sweep.csv",
    )

    # workflow_eps(
    #     X,
    #     k,
    #     title,
    #     eps=eps,
    #     local_search_steps=local_search_steps,
    #     verbose=verbose,
    #     plot_dims=(0, 1),
    # )

    # workflow_eps_beta(
    #     X,
    #     k,
    #     title,
    #     eps=eps,
    #     beta=beta,
    #     local_search_steps=local_search_steps,
    #     verbose=verbose,
    #     plot_dims=(0, 1),
    # )

    # workflow_image(
    #     "pictures/cvete.jpg",
    #     t=8,
    #     eps=0.1,
    #     compression_ratio=0.02,
    #     local_search_steps=67,
    #     beta_search_precision=beta_search_precision,
    #     verbose=True,
    # )


if __name__ == "__main__":
    main()
