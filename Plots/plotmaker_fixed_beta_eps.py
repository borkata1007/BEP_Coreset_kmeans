from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(__file__).with_name("fixed_beta_eps_sweep.csv")
OUTPUT_DIR = Path(__file__).resolve().parent


def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df = df.sort_values("eps").reset_index(drop=True)
    return df


def style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.figsize": (8, 5),
        }
    )


def savefig(name):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"saved: {path}")


def eps_tick_labels(df):
    labels = []
    for index, (eps, nq) in enumerate(zip(df["eps"], df["n_coreset"])):
        prefix = "\n\n" if index % 2 else ""
        labels.append(f"{prefix}{eps:g}\n{nq}")
    return labels


def add_total_points_note(ax, df):
    n_total = int(df["n_initial"].iloc[0])
    beta = float(df["beta"].iloc[0])
    ax.text(
        0.01,
        1.02,
        f"|P|={n_total:,}    beta={beta:g}",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )


def plot_size_and_compression(df):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(df["eps"], df["n_coreset"], color="#1f77b4", marker="o", linewidth=2)
    ax1.set_xlabel("epsilon")
    ax1.set_ylabel("coreset size |Q|", color="#1f77b4")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["eps"], df["p_over_q_achieved"], color="#d62728", marker="s", linewidth=2)
    ax2.set_ylabel("inverse compression |P| / |Q|", color="#d62728")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Fixed-beta sweep: coreset size collapse as epsilon grows")
    savefig("fixed_beta_eps_size_and_inverse_ratio.png")
    plt.show()


def plot_cost_ratios_vs_eps(df):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ratio_specs = [
        ("cost_qc_over_pc", "Cost(Q,C) / Cost(P,C)", "#1f77b4", "o"),
        ("cost_qcprime_over_pcprime", "Cost(Q,C') / Cost(P,C')", "#ff7f0e", "s"),
        ("cost_pcprime_over_pc", "Cost(P,C') / Cost(P,C)", "#2ca02c", "^"),
    ]

    for col, label, color, marker in ratio_specs:
        ax.plot(df["eps"], df[col], label=label, color=color, marker=marker, linewidth=2)

    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_xticks(df["eps"])
    ax.set_xticklabels(eps_tick_labels(df))
    ax.set_xlabel("tick labels: epsilon on top, coreset size |Q| below")
    ax.set_ylabel("ratio value")
    ax.set_title("Cost ratios across epsilon for fixed beta")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    add_total_points_note(ax, df)
    savefig("fixed_beta_eps_cost_ratios.png")
    plt.show()


def plot_actual_costs_vs_eps(df):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cost_specs = [
        ("cost_p_c", "Cost(P,C)", "#4c78a8", "o"),
        ("cost_q_c", "Cost(Q,C)", "#f58518", "s"),
        ("cost_p_c_prime", "Cost(P,C')", "#54a24b", "^"),
        ("cost_q_c_prime", "Cost(Q,C')", "#e45756", "D"),
    ]

    for col, label, color, marker in cost_specs:
        ax.plot(df["eps"], df[col], label=label, color=color, marker=marker, linewidth=2)

    ax.set_xticks(df["eps"])
    ax.set_xticklabels(eps_tick_labels(df))
    ax.set_xlabel("epsilon and coreset size")
    ax.set_ylabel("actual cost value")
    ax.set_title("Actual cost values across epsilon for fixed beta")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    add_total_points_note(ax, df)
    savefig("fixed_beta_eps_actual_costs.png")
    plt.show()


def plot_ratio_deviation_heatmap(df):
    ratio_cols = [
        "cost_qc_over_pc",
        "cost_qcprime_over_pcprime",
        "cost_pcprime_over_pc",
    ]
    ratio_labels = [
        "Cost(Q,C)/Cost(P,C)",
        "Cost(Q,C')/Cost(P,C')",
        "Cost(P,C')/Cost(P,C)",
    ]

    values = np.abs(df[ratio_cols].to_numpy().T - 1.0)

    plt.figure(figsize=(10, 3.8))
    im = plt.imshow(values, aspect="auto", cmap="magma")
    plt.colorbar(im, label="absolute deviation from 1")
    plt.xticks(range(len(df)), [f"{eps:g}" for eps in df["eps"]], rotation=45)
    plt.yticks(range(len(ratio_labels)), ratio_labels)
    plt.xlabel("epsilon")
    plt.title("Where fixed beta starts drifting from ideal cost preservation")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] > values.max() * 0.45 else "black"
            plt.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    savefig("fixed_beta_eps_ratio_deviation_heatmap.png")
    plt.show()


def plot_quality_vs_compression(df):
    fig, ax = plt.subplots(figsize=(9, 5))

    scatter = ax.scatter(
        df["p_over_q_achieved"],
        df["cost_pcprime_over_pc"],
        c=df["eps"],
        cmap="viridis",
        s=90,
        edgecolor="black",
    )
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("inverse compression |P| / |Q|")
    ax.set_ylabel("Cost(P,C') / Cost(P,C)")
    ax.set_title("Tradeoff: stronger compression versus final center quality")
    ax.grid(True, alpha=0.3)

    for _, row in df.iterrows():
        ax.annotate(f"{row['eps']:g}", (row["p_over_q_achieved"], row["cost_pcprime_over_pc"]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    cbar = plt.colorbar(scatter)
    cbar.set_label("epsilon")
    savefig("fixed_beta_eps_quality_vs_inverse_ratio.png")
    plt.show()


def main():
    style()
    df = load_data()

    print("Loaded rows:", len(df))
    print(df[["eps", "beta", "n_coreset", "p_over_q_achieved"]])

    plot_size_and_compression(df)
    plot_cost_ratios_vs_eps(df)
    plot_actual_costs_vs_eps(df)
    plot_ratio_deviation_heatmap(df)
    plot_quality_vs_compression(df)


if __name__ == "__main__":
    main()
