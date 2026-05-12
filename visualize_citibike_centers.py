from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kmeans_pp_nd import kmeans_plus_plus_local_search_full


INPUT_FILE = Path("boris_citibike_2d.csv")
OUTPUT_DIR = Path("Plots") / "citibike_center_checks"
OUTPUT_TEMPLATE = "citibike_k{k}_centers.png"


def plot_centers(df, scaled_coordinates, scaler, k: int) -> None:
    local_search_steps = 4
    random_state = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / OUTPUT_TEMPLATE.format(k=k)

    centers_scaled, cost = kmeans_plus_plus_local_search_full(
        scaled_coordinates,
        k,
        n_steps=local_search_steps,
        random_state=random_state,
        verbose=True,
    )
    centers = scaler.inverse_transform(centers_scaled)

    sample_size = min(200_000, len(df))
    plot_df = df.sample(n=sample_size, random_state=random_state) if len(df) > sample_size else df

    plt.figure(figsize=(9, 9))
    plt.scatter(
        plot_df["longitude"],
        plot_df["latitude"],
        s=2.0,
        alpha=0.14,
        linewidths=0,
        c="#9aa3ad",
        label="Citibike start locations",
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

    for index, (longitude, latitude) in enumerate(centers, start=1):
        plt.annotate(
            str(index),
            (longitude, latitude),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color="#111111",
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Citibike Start Locations with k={k} Cluster Centers")
    plt.grid(alpha=0.2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=240)
    plt.close()

    print(f"Wrote {output_file} with {len(df)} points, plotted {sample_size} points")
    print(f"k-means cost on standardized coordinates: {cost:.6f}")
    print("Centers:")
    for index, (longitude, latitude) in enumerate(centers, start=1):
        print(f"{index}: longitude={longitude:.8f}, latitude={latitude:.8f}")


def main() -> None:
    k_values = [10, 11, 12, 13]

    df = pd.read_csv(INPUT_FILE, usecols=["longitude", "latitude"])
    df = df.dropna()

    coordinates = df[["longitude", "latitude"]].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled_coordinates = scaler.fit_transform(coordinates)

    for k in k_values:
        plot_centers(df, scaled_coordinates, scaler, k)


if __name__ == "__main__":
    main()
