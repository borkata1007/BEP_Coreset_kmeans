from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kmeans_pp_nd import kmeans_plus_plus_local_search_full


INPUT_FILE = Path("final_datasets") / "donuts.csv"
OUTPUT_DIR = Path("results") / "donuts" / "plots"


def plot_centers(df, k: int) -> None:
    local_search_steps = 100
    random_state = 0
    output_file = OUTPUT_DIR / f"donuts_k{k}_steps{local_search_steps}_centers.png"

    points = df[["x", "y"]].to_numpy(dtype=float)
    centers, cost = kmeans_plus_plus_local_search_full(
        points,
        k,
        n_steps=local_search_steps,
        random_state=random_state,
        verbose=True,
    )

    sample_size = min(200_000, len(df))
    plot_df = df.sample(n=sample_size, random_state=random_state) if len(df) > sample_size else df

    plt.figure(figsize=(9, 9))
    plt.scatter(
        plot_df["x"],
        plot_df["y"],
        s=1.5,
        alpha=0.12,
        linewidths=0,
        c="#8f99a6",
        label="Donuts points",
    )
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        s=230,
        marker="X",
        c="#d62728",
        edgecolors="black",
        linewidths=1.2,
        label=f"k={k} centers",
        zorder=3,
    )

    for index, (x_value, y_value) in enumerate(centers, start=1):
        plt.annotate(
            str(index),
            (x_value, y_value),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color="#111111",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Donuts Dataset with k={k} Cluster Centers")
    plt.grid(alpha=0.2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=240)
    plt.close()

    print(f"Wrote {output_file} with {len(df)} points, plotted {sample_size} points")
    print(f"k-means cost: {cost:.6f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, usecols=["x", "y"]).dropna()

    for k in [8, 9, 10]:
        plot_centers(df, k)


if __name__ == "__main__":
    main()
