import os

from ..coresets import ALGORITHMS
from ..data import DATASETS, load, ROOT
from .experiment import run_grid, centers_to_dataframe, K_VALUES
from ..visualization.maps import plot_maps
from ..visualization.images import plot_images
from ..visualization.lines import plot_per_algo, plot_algo_compare
from ..visualization.heatmap import plot_heatmap

OUT_ROOT = os.path.join(ROOT, "output")
RATIOS = ["r1", "r2", "r3"]


def save_results(df, out_dir):
    out_path = os.path.join(out_dir, "results.csv")
    df.to_csv(out_path, index=False)


def save_centers(centers, scaler, feature_names, out_dir):
    centers_df = centers_to_dataframe(centers, scaler, feature_names)
    out_path = os.path.join(out_dir, "centers.csv")
    centers_df.to_csv(out_path, index=False)


def plot_cost_ratio_per_algo(df, name, n, d, out_dir):
    per_algo_dir = os.path.join(out_dir, "cost_ratio", "per_algo")
    for algo in ALGORITHMS:
        plot_per_algo(df, name, algo, n, d, per_algo_dir)


def plot_cost_ratio_compare(df, name, n, d, out_dir):
    compare_dir = os.path.join(out_dir, "cost_ratio", "compare")
    for k in K_VALUES:
        for ratio in RATIOS:
            plot_algo_compare(df, name, k, ratio, n, d, compare_dir)


def plot_all_heatmaps(df, name, out_dir):
    heatmaps_dir = os.path.join(out_dir, "heatmaps")
    for algo in ALGORITHMS:
        for ratio in RATIOS:
            plot_heatmap(df, name, algo, ratio, heatmaps_dir)


def plot_dataset_specific(P, P_raw, scaler, info, out_dir):
    if info.get("type") == "image":
        plot_images(P, P_raw, scaler, info, ALGORITHMS, os.path.join(out_dir, "image"))
    elif info.get("map"):
        plot_maps(P, P_raw, scaler, info, ALGORITHMS, os.path.join(out_dir, "maps"))


def run_dataset(name):
    P, P_raw, scaler, info = load(name)
    n, d = P.shape
    print(f"[{name}] n={n:,} d={d}", flush=True)

    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    df, centers = run_grid(P, ALGORITHMS)

    save_results(df, out_dir)
    save_centers(centers, scaler, info["features"], out_dir)

    plot_cost_ratio_per_algo(df, name, n, d, out_dir)
    plot_cost_ratio_compare(df, name, n, d, out_dir)
    plot_all_heatmaps(df, name, out_dir)
    plot_dataset_specific(P, P_raw, scaler, info, out_dir)


def main():
    for name in DATASETS:
        run_dataset(name)


if __name__ == "__main__":
    main()
