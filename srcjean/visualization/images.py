import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from ..experiments.experiment import make_coreset, fit_centers, fit_full_kmeans, coreset_size

IMG_K = [4, 16, 64]
IMG_FRACS = [0.01, 0.10, 0.50]

# IMG_K = [2, 4, 8, 16, 32, 64]
# IMG_FRACS = [0.0025, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50]


def _quantize(P, scaler, centers, shape):
    distances = cdist(P, centers, "sqeuclidean")
    labels = distances.argmin(axis=1)
    centers_rgb = scaler.inverse_transform(centers)
    centers_rgb = np.clip(centers_rgb, 0, 255)
    pixels = centers_rgb[labels]
    H, W = shape
    return pixels.reshape(H, W, 3).astype(np.uint8)


def _compute_full_quantizations(P, scaler, shape, k_values):
    quantizations = {}
    for k in k_values:
        C = fit_full_kmeans(P, k, seed=0)
        quantizations[k] = _quantize(P, scaler, C, shape)
    return quantizations


def _compute_algo_quantizations(P, scaler, shape, algorithms, k_values, q_fractions):
    n = len(P)
    quantizations = {name: {} for name in algorithms}
    for k in k_values:
        for frac in q_fractions:
            m = coreset_size(n, k, frac)
            for algo_name, AlgoClass in algorithms.items():
                Q, w = make_coreset(P, AlgoClass, m, seed=0)
                Cp = fit_centers(Q, w, k, seed=0)
                quantizations[algo_name][(k, frac)] = _quantize(P, scaler, Cp, shape)
    return quantizations


def _save_compare_figure(original, full_quant, algo_quants, k, frac, m, n, out_path):
    panel_count = len(algo_quants) + 2
    fig, axes = plt.subplots(1, panel_count, figsize=(5 * panel_count, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(full_quant)
    axes[1].set_title(f"Full (k={k})")

    for idx, (algo_name, image) in enumerate(algo_quants.items(), start=2):
        axes[idx].imshow(image)
        axes[idx].set_title(f"{algo_name} (|Q|={m}, {n / m:.1f}x)")

    for ax in axes:
        ax.set_axis_off()

    fig.suptitle(f"k={k}, |Q|={m} ({frac:.2%} of n={n:,})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_per_algo_grid(algo_name, algo_images, k_values, q_fractions, n, out_path):
    rows = len(k_values)
    cols = len(q_fractions)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for r, k in enumerate(k_values):
        for c, frac in enumerate(q_fractions):
            m = coreset_size(n, k, frac)
            image = algo_images[(k, frac)]
            ax = axes[r][c]
            ax.imshow(image)
            ax.set_title(f"k={k}, |Q|={m} ({frac:.2%})", fontsize=10)
            ax.set_axis_off()

    fig.suptitle(f"{algo_name} — quantization across (k, |Q|), n={n:,}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_images(P, P_raw, scaler, info, algorithms, out_dir, k_values=IMG_K, q_fractions=IMG_FRACS):
    per_algo_dir = os.path.join(out_dir, "per_algo")
    compare_dir = os.path.join(out_dir, "compare")
    os.makedirs(per_algo_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    n = len(P)
    H, W = info["shape"]
    shape = (H, W)
    original = P_raw.reshape(H, W, 3).astype(np.uint8)

    full_quants = _compute_full_quantizations(P, scaler, shape, k_values)
    algo_quants = _compute_algo_quantizations(P, scaler, shape, algorithms, k_values, q_fractions)

    for k in k_values:
        for frac in q_fractions:
            m = coreset_size(n, k, frac)
            current_algo_quants = {}
            for algo_name in algorithms:
                current_algo_quants[algo_name] = algo_quants[algo_name][(k, frac)]
            file_name = f"k={k}_Q={frac * 100:.2f}.png"
            out_path = os.path.join(compare_dir, file_name)
            _save_compare_figure(original, full_quants[k], current_algo_quants, k, frac, m, n, out_path)

    for algo_name in algorithms:
        out_path = os.path.join(per_algo_dir, f"{algo_name}.png")
        _save_per_algo_grid(algo_name, algo_quants[algo_name], k_values, q_fractions, n, out_path)
