import os
import matplotlib.pyplot as plt

RATIO_LABELS = {
    "r1": "Cost(Q,C) / Cost(P,C)",
    "r2": "Cost(Q,C') / Cost(P,C')",
    "r3": "Cost(P,C') / Cost(P,C)",
}


def _q_tick_labels(sub):
    labels = []
    for m, frac in zip(sub["m"], sub["frac"]):
        label = f"|Q|={int(m)}\n{frac * 100:.2f}%"
        labels.append(label)
    return labels


def _mean_over_seeds(df, group_cols):
    grouped = df.groupby(group_cols, as_index=False)
    return grouped.mean(numeric_only=True)


def plot_per_algo(df, dataset, algo, n, d, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df_algo = df[df["algo"] == algo]
    means = _mean_over_seeds(df_algo, ["k", "frac", "m"])

    ks = sorted(means["k"].unique())
    fracs = sorted(means["frac"].unique())

    first_k = ks[0]
    first_k_rows = means[means["k"] == first_k].sort_values("frac")
    x_labels = _q_tick_labels(first_k_rows)
    x_positions = list(range(len(x_labels)))

    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ratio_idx, ratio in enumerate(["r1", "r2", "r3"]):
        ax = axes[ratio_idx]
        for k_idx, k in enumerate(ks):
            k_rows = means[means["k"] == k].sort_values("frac")
            y = k_rows[ratio].values
            color = cmap(k_idx / max(1, len(ks) - 1))
            ax.plot(x_positions, y, "o-", color=color, label=f"k={k}", linewidth=2, markersize=7)
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel("coreset size |Q|")
        ax.set_ylabel(RATIO_LABELS[ratio])
        ax.set_title(RATIO_LABELS[ratio])
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(f"{dataset} — {algo} across (k, |Q|), d={d}, |P|={n:,}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{algo}.png"), dpi=120)
    plt.close(fig)


def plot_algo_compare(df, dataset, k, ratio, n, d, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df_k = df[df["k"] == k]
    means = _mean_over_seeds(df_k, ["algo", "frac", "m"])

    algos = sorted(means["algo"].unique())
    first_algo_rows = means[means["algo"] == algos[0]].sort_values("frac")
    x_labels = _q_tick_labels(first_algo_rows)
    x_positions = list(range(len(x_labels)))

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 5))

    for algo_idx, algo in enumerate(algos):
        algo_rows = means[means["algo"] == algo].sort_values("frac")
        y = algo_rows[ratio].values
        ax.plot(x_positions, y, "o-", color=cmap(algo_idx), label=algo, linewidth=2, markersize=8)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("coreset size |Q|")
    ax.set_ylabel(RATIO_LABELS[ratio])
    ax.set_title(f"{dataset} — {RATIO_LABELS[ratio]} (k={k}, d={d}, |P|={n:,})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{ratio}_k={k}.png"), dpi=120)
    plt.close(fig)
