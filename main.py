import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from kmeans_pp import kmeans_plus_plus_init, kmeans_plus_plus_local_search_full, kmeans_plus_plus_local_search_weighted, compute_kmeans_cost
from Exponential_quadtree import exponential_quadtree_coreset
from image_processors import compress_image_with_coreset, save_compressed_image


def workflow_dataset(eps=10):
    """
    Dataset workflow: Load MNIST, build coreset, compare original centers vs coreset-derived centers.
    
    Parameters
    ----------
    eps : float
        Approximation parameter for coreset.
    """
    print("=" * 60)
    print("DATASET WORKFLOW: Coreset on MNIST with Center Comparison")
    print("=" * 60)
    
    # Load a medium/large dataset (~70k samples): MNIST from OpenML
    print("begins MNIST loading...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(int)

    # Set k equal to the true number of classes
    k = len(np.unique(y))

    # Reduce to 2 dimensions using PCA
    print("begins PCA...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Compute initial centers via kmeans++ + local search on full data
    print("begins kmeans++ local search (full data)...")
    c, full_local_cost = kmeans_plus_plus_local_search_full(
        X_2d,
        k,
        n_steps=100,
        random_state=0,
    )
    print("finished kmeans++ local search")

    # Visualize full-data local search centers (optional initial check)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", alpha=0.6, edgecolor="k", label="Data points")
    plt.scatter(c[:, 0], c[:, 1], c="red", s=120, marker="X", label="Centers c (full local)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"MNIST with local-search kmeans++ centers c (k = {k})")
    plt.legend()
    plt.tight_layout()

    # Build exponential quadtree coreset
    print("begins coreset building...")
    coreset_points, coreset_weights, squares = exponential_quadtree_coreset(X_2d, c, eps, random_state=0)
    print("finished coreset building")
    print("Number of original points:", X_2d.shape[0])
    print("Number of coreset points:", coreset_points.shape[0])

    # Print full-data k-means cost (unweighted)
    print("begins cost computation...")
    full_cost = compute_kmeans_cost(X_2d, c)
    print("Full data k-means cost:", full_cost)

    # Print coreset weighted cost
    coreset_cost = compute_kmeans_cost(coreset_points, c, weights=coreset_weights)
    print("finished cost computation")
    print("Coreset weighted cost:", coreset_cost)

    # Print logging of cost ratio for verification
    print("Boundary range", full_cost * (1-eps), full_cost * (1+eps))
    print("Cost ratio (coreset / full):", coreset_cost / full_cost)

    # Run local-search weighted kmeans++ on coreset to get improved centers c'
    print("begins weighted kmeans++ local search (on coreset)...")
    c_prime, c_prime_cost_on_coreset = kmeans_plus_plus_local_search_weighted(
        coreset_points,
        coreset_weights,
        k,
        n_steps=100,
        random_state=0,
    )
    print("finished weighted kmeans++ local search")
    full_cost_with_c_prime = compute_kmeans_cost(X_2d, c_prime)

    print("\n--- Comparison of centers ---")
    print("Full data cost with centers c (full local):", full_cost)
    print("Full data cost with centers c' (coreset local):", full_cost_with_c_prime)
    print("Cost ratio (c' vs c):", full_cost_with_c_prime / full_cost)
    print("Coreset internal cost c' (weighted):", c_prime_cost_on_coreset)
    
    # Plot comparison of centers c and c'
    print("begins plotting...")
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="lightgray", alpha=0.5, edgecolor="none", label="Data points")
    plt.scatter(c[:, 0], c[:, 1], c="red", s=150, marker="X", label="Centers c (full local)", linewidths=2)
    plt.scatter(c_prime[:, 0], c_prime[:, 1], c="blue", s=150, marker="+", label="Centers c' (coreset local)", linewidths=2)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Comparison: Original centers c vs. Coreset-derived centers c'")
    plt.legend()
    plt.tight_layout()
    
    # Plot coreset and quadtree squares on a separate figure with weight color-coded points
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="lightgray", alpha=0.5, edgecolor="none", label="Original points")
    # Draw leaf squares
    for (x0, x1, y0, y1) in squares:
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        plt.plot(xs, ys, color="gray", linewidth=0.8, alpha=0.7)

    # Color code the coreset reps by weight (green sequential colormap)
    sizes = 15 * (coreset_weights / coreset_weights.max())
    scatter = plt.scatter(
        coreset_points[:, 0],
        coreset_points[:, 1],
        c=coreset_weights,
        cmap="Greens",  # light green to dark green
        s=40 + sizes,
        edgecolor="k",
        vmin=coreset_weights.min(),
        vmax=coreset_weights.max(),
        label="Coreset reps",
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Weight")

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Exponential quadtree coreset (weight-coded)")
    plt.legend()
    plt.tight_layout()
    print("finished plotting")

    plt.show()
    print("Dataset workflow completed.\n")


def workflow_image(image_path, t, eps=0.1):
    """
    Image compression workflow: Compress image using coreset-based k-means on RGB data.
    
    Parameters
    ----------
    image_path : str
        Path to the input image file.
    t : int
        Number of colors (clusters) to use.
    eps : float
        Approximation parameter for coreset.
    """
    print("=" * 60)
    print("IMAGE WORKFLOW: Image Compression with Coreset k-means")
    print("=" * 60)
    
    print(f"Loading image from: {image_path}")
    print("begins image compression...")
    compressed_img, original_shape, stats = compress_image_with_coreset(image_path, t, eps=eps, random_state=0)
    print("finished image compression")
    
    output_path = image_path.replace(".png", f"_compressed_t{t}.png").replace(".jpg", f"_compressed_t{t}.jpg")
    save_compressed_image(compressed_img, output_path)
    print(f"Compressed image saved to: {output_path}")
    print(f"Image dimensions: {original_shape[0]} x {original_shape[1]} with {t} colors")
    print(f"\n--- Image Compression Statistics ---")
    print(f"Full dataset size (pixels): {stats['full_size']}")
    print(f"Coreset size: {stats['coreset_size']}")
    print(f"Compression ratio: {stats['coreset_size'] / stats['full_size']:.4f}")
    print(f"Cost with initial centers: {stats['initial_cost']:.2f}")
    print(f"Cost with final centers: {stats['final_cost']:.2f}")
    print(f"Cost improvement ratio (final / initial): {stats['final_cost'] / stats['initial_cost']:.4f}")
    print("Image workflow completed.\n")


def main():
    """Main function to choose which workflow(s) to run."""
    # Constants
    eps_dataset = 1
    t = 16
    # number of colors for image compression
    eps_image = 2 # approximation parameter for image compression
    
    # Choose which workflow(s) to run:
    # Uncomment the workflows you want to execute
    
    # Workflow 1: Dataset with center comparison
    workflow_dataset(eps=eps_dataset)
    
    # Workflow 2: Image compression
    # Uncomment to run image compression workflow
    # workflow_image("pictures/cune.png", t=t, eps=eps_image)


if __name__ == "__main__":
    main()