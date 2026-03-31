import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


def run_coreset_workflow(
    X,
    k,
    title,
    eps,
    local_search_steps=67,
    compression_ratio=None,
    beta=None,
    verbose=True,
    plot_dims=(0, 1),
):
    """Generic coreset workflow for any numeric dataset in shape (n, d)."""
    print("=" * 60)
    print(f"DATASET WORKFLOW: {title}")
    print("=" * 60)

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
        verbose=verbose,
        return_info=True,
    )
    print("finished coreset building")
    print("Number of original points:", info["n_initial"])
    print("Number of coreset points:", info["n_coreset"])
    print("Beta used:", info["beta"])

    full_cost = compute_kmeans_cost(X, c)
    coreset_cost = compute_kmeans_cost(coreset_points, c, weights=coreset_weights)

    print("\n--- Cost Summary ---")
    print("Full data k-means cost:", full_cost)
    print("Coreset weighted cost:", coreset_cost)
    print("Requested compression ratio:", compression_ratio)
    print("Achieved compression ratio:", info["compression_ratio_achieved"])

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
    print("\n--- Comparison of centers ---")
    print("Full data cost with centers c (full local):", full_cost)
    print("Full data cost with centers c' (coreset local):", full_cost_with_c_prime)
    print("Cost ratio (c' vs c):", full_cost_with_c_prime / full_cost)
    print("Coreset internal cost c' (weighted):", c_prime_cost_on_coreset)

    x_dim, y_dim = plot_dims
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, x_dim], X[:, y_dim], c="lightgray", alpha=0.5, edgecolor="none", label="Data points")
    plt.scatter(c[:, x_dim], c[:, y_dim], c="red", s=150, marker="X", label="Centers c (full local)", linewidths=2)
    plt.scatter(c_prime[:, x_dim], c_prime[:, y_dim], c="blue", s=150, marker="+", label="Centers c' (coreset local)", linewidths=2)
    plt.xlabel(f"Feature {x_dim}")
    plt.ylabel(f"Feature {y_dim}")
    plt.title(f"{title}: centers comparison")
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
    plt.xlabel(f"Feature {x_dim}")
    plt.ylabel(f"Feature {y_dim}")
    plt.title(f"{title}: EQT coreset projection")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Workflow completed.\n")

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
        verbose=verbose,
        plot_dims=plot_dims,
    )


def workflow_uber_cost_ratios(
    eps=0.1,
    compression_ratio=0.05,
    local_search_steps=67,
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
        verbose=verbose,
        plot_dims=plot_dims,
    )


def workflow_image(image_path, t, eps=0.1, compression_ratio=0.02, local_search_steps=67, verbose=True):
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
    eps = 0.07
    compression_ratio = 1 / 100
    beta = 21
    local_search_steps = 4
    verbose = True

    X, k, title = load_dataset_uber("uber-raw-data-jul14.csv")

    workflow_uber_cost_ratios_beta(
        eps=eps,
        beta=beta,
        local_search_steps=local_search_steps,
        verbose=verbose,
        csv_path="uber-raw-data-jul14.csv",
    )

    # workflow_uber_cost_ratios(
    #     eps=eps,
    #     compression_ratio=compression_ratio,
    #     local_search_steps=local_search_steps,
    #     verbose=verbose,
    #     csv_path="uber-raw-data-jul14.csv",
    # )

    #Choose one generic mode:
    # workflow_eps_compression_ratio(
    #     X,
    #     k,
    #     title,
    #     eps=eps,
    #     compression_ratio=compression_ratio,
    #     local_search_steps=local_search_steps,
    #     verbose=verbose,
    #     plot_dims=(0, 1),
    # )

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

    # workflow_image("pictures/image.png", t=8, eps=0.1, compression_ratio=0.02, local_search_steps=67, verbose=True)


if __name__ == "__main__":
    main()
