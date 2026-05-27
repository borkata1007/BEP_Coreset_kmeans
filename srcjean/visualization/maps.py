import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from ..experiments.experiment import make_coreset, fit_centers, fit_full_kmeans, coreset_size

MAP_K = [4, 16, 64]
MAP_FRACS = [0.01, 0.10, 0.50]
PLOT_N = 100000

# MAP_K = [2, 4, 8, 16, 32, 64]
# MAP_FRACS = [0.0025, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50]


def _assign(X, centers):
    distances = cdist(X, centers, "sqeuclidean")
    return distances.argmin(axis=1)


def _subsample(P, P_raw, n_target, seed=42):
    n = len(P)
    plot_n = min(n_target, n)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n, plot_n, replace=False)
    return P[indices], P_raw[indices]


def _draw_panel(ax, points, labels, centers, dx, dy, xlabel, ylabel, title):
    ax.scatter(points[:, dx], points[:, dy], c=labels, cmap=plt.cm.tab10, s=0.5, alpha=0.3)
    ax.scatter(centers[:, dx], centers[:, dy], c="red", marker="*", s=200, edgecolors="white", linewidths=1, zorder=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_maps(P, P_raw, scaler, info, algorithms, out_dir, k_values=MAP_K, q_fractions=MAP_FRACS):
    os.makedirs(out_dir, exist_ok=True)
    n, _ = P.shape
    dx, dy = info["map_dims"]
    xlabel, ylabel = info["map_labels"]

    P_sub, P_plot = _subsample(P, P_raw, PLOT_N)

    for k in k_values:
        C = fit_full_kmeans(P, k, seed=0)
        labels_full = _assign(P_sub, C)
        C_orig = scaler.inverse_transform(C)

        for frac in q_fractions:
            m = coreset_size(n, k, frac)
            panel_count = len(algorithms) + 1

            with plt.style.context("dark_background"):
                fig, axes = plt.subplots(1, panel_count, figsize=(8 * panel_count, 8))

                full_title = f"Full k-means (k={k}, n={n:,})"
                _draw_panel(axes[0], P_plot, labels_full, C_orig, dx, dy, xlabel, ylabel, full_title)

                for idx, (algo_name, AlgoClass) in enumerate(algorithms.items(), start=1):
                    Q, w = make_coreset(P, AlgoClass, m, seed=0)
                    Cp = fit_centers(Q, w, k, seed=0)
                    labels_algo = _assign(P_sub, Cp)
                    Cp_orig = scaler.inverse_transform(Cp)
                    compression = n / m
                    algo_title = f"{algo_name} (|Q|={m}, {compression:.1f}x compression)"
                    _draw_panel(axes[idx], P_plot, labels_algo, Cp_orig, dx, dy, xlabel, ylabel, algo_title)

                fig.suptitle(f"k={k}, |Q|={m} ({frac:.2%} of n={n:,})", fontsize=14)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"k={k}_Q={frac * 100:.2f}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
