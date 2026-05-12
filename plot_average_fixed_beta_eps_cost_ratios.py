from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "aggregate"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

RATIO_COLUMNS = [
    "cost_qc_over_pc",
    "cost_qcprime_over_pcprime",
    "cost_pcprime_over_pc",
]


def load_fixed_beta_tables() -> pd.DataFrame:
    csv_paths = sorted(RESULTS_DIR.glob("*/tables/*fixed_beta_eps_sweep.csv"))
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if not set(RATIO_COLUMNS + ["eps"]).issubset(df.columns):
            continue
        df = df.copy()
        df["source_file"] = str(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No fixed-beta epsilon sweep CSVs found under results/*/tables")

    return pd.concat(frames, ignore_index=True)


def average_by_epsilon(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("eps", as_index=False)
    avg = grouped.agg(
        n_runs=("source_file", "nunique"),
        cost_qc_over_pc=("cost_qc_over_pc", "mean"),
        cost_qcprime_over_pcprime=("cost_qcprime_over_pcprime", "mean"),
        cost_pcprime_over_pc=("cost_pcprime_over_pc", "mean"),
        avg_n_coreset=("n_coreset", "mean"),
        avg_p_over_q=("p_over_q_achieved", "mean"),
    )
    return avg.sort_values("eps").reset_index(drop=True)


def plot_average_cost_ratios(df: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    ratio_specs = [
        ("cost_qc_over_pc", "Avg Cost(Q,C) / Cost(P,C)", "#1f77b4", "o"),
        ("cost_qcprime_over_pcprime", "Avg Cost(Q,C') / Cost(P,C')", "#ff7f0e", "s"),
        ("cost_pcprime_over_pc", "Avg Cost(P,C') / Cost(P,C)", "#2ca02c", "^"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col, label, color, marker in ratio_specs:
        ax.plot(df["eps"], df[col], label=label, color=color, marker=marker, linewidth=2)

    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_xticks(df["eps"])
    ax.set_xticklabels([f"{eps:g}" for eps in df["eps"]])
    ax.set_xlabel("epsilon")
    ax.set_ylabel("average ratio value")
    ax.set_title("Average Fixed-Beta Cost Ratios Across Dataset Runs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.text(
        0.01,
        1.02,
        f"averaged over {int(df['n_runs'].max())} runs",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )

    output_path = PLOT_DIR / "average_fixed_beta_eps_cost_ratios.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_fixed_beta_tables()
    avg = average_by_epsilon(df)

    table_path = TABLE_DIR / "average_fixed_beta_eps_cost_ratios.csv"
    avg.to_csv(table_path, index=False)
    print(f"saved: {table_path}")
    print(avg)

    plot_average_cost_ratios(avg)


if __name__ == "__main__":
    main()
