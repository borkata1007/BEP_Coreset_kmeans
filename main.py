import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from kmeans_pp_nd import kmeans_plus_plus_local_search_full, kmeans_plus_plus_local_search_weighted, compute_kmeans_cost
from Exponential_quadtree_nd import exponential_quadtree_coreset
from image_processors import compress_image_with_coreset, save_compressed_image


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
    eps = 0.05
    compression_ratio = 1 / 5
    beta = 640
    local_search_steps = 4
    beta_search_precision = 0.5
    verbose = True

    X, k, title = load_dataset_uber("uber-raw-data-jul14.csv")

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
