import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .lines import RATIO_LABELS


def _build_matrix(means, ks, fracs, ratio):
    matrix = np.zeros((len(ks), len(fracs)))
    for i, k in enumerate(ks):
        for j, frac in enumerate(fracs):
            cell = means[(means["k"] == k) & (means["frac"] == frac)]
            matrix[i, j] = cell[ratio].iloc[0]
    return matrix


def _color_norm(matrix):
    vmin = float(matrix.min())
    vmax = float(matrix.max())
    if vmin < 1.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    return None


def _annotate_cells(ax, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)


def plot_heatmap(df, dataset, algo, ratio, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df_algo = df[df["algo"] == algo]
    means = df_algo.groupby(["k", "frac"], as_index=False).mean(numeric_only=True)

    ks = sorted(means["k"].unique())
    fracs = sorted(means["frac"].unique())
    matrix = _build_matrix(means, ks, fracs, ratio)
    norm = _color_norm(matrix)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", norm=norm)

    x_tick_labels = [f"{frac * 100:.2f}%" for frac in fracs]
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks)
    ax.set_xlabel("|Q| (fraction of n)")
    ax.set_ylabel("k")

    _annotate_cells(ax, matrix)

    ax.set_title(f"{dataset} — {RATIO_LABELS[ratio]} ({algo})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{algo}_{ratio}.png"), dpi=120)
    plt.close(fig)
