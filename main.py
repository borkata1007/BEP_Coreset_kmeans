import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from kmeans_pp import kmeans_plus_plus_init
from Exponential_quadtree import exponential_quadtree_coreset


def main():
    # Load a medium/large dataset (~70k samples): MNIST from OpenML
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(int)

    # Set k equal to the true number of classes
    k = len(np.unique(y))

    # Reduce to 2 dimensions using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    # Get k centers using k-means++ initialization
    c = kmeans_plus_plus_init(X_2d, k)

    # Visualize data points and chosen centers
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", alpha=0.6, edgecolor="k", label="Data points")
    plt.scatter(c[:, 0], c[:, 1], c="red", s=120, marker="X", label="Chosen centers")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"MNIST with k-means++ centers (k = {k})")
    plt.legend()
    plt.tight_layout()

    # Build exponential quadtree coreset
    coreset_points, coreset_weights, squares = exponential_quadtree_coreset(X_2d, random_state=0)
    print("Number of original points:", X_2d.shape[0])
    print("Number of coreset points:", coreset_points.shape[0])
    print("Coreset points with weights (x, y, w):")
    for p, w in zip(coreset_points, coreset_weights):
        print(p[0], p[1], w)

    # Plot coreset and quadtree squares on a separate figure
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="lightgray", alpha=0.5, edgecolor="none", label="Original points")
    # Draw leaf squares
    for (x0, x1, y0, y1) in squares:
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        plt.plot(xs, ys, color="gray", linewidth=0.8, alpha=0.7)

    # Scale marker size by weight for visualization
    sizes = 30 * (coreset_weights / coreset_weights.max())
    plt.scatter(coreset_points[:, 0], coreset_points[:, 1], s=80 + sizes, c="blue", edgecolor="k", label="Coreset reps")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Exponential quadtree coreset")
    plt.legend()
    plt.tight_layout()

    # Third visualization: only weighted points, colored by weight (blue=1, red=max)
    plt.figure(figsize=(6, 5))
    w_min, w_max = 1, coreset_weights.max()
    scatter = plt.scatter(
        coreset_points[:, 0],
        coreset_points[:, 1],
        c=coreset_weights,
        cmap="coolwarm",  # blue (low) to red (high)
        s=5,             # smaller dots
        edgecolor="none",
        vmin=w_min,
        vmax=w_max,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Coreset weight (blue = 1, red = max)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Coreset points colored by weight")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()