import csv
from pathlib import Path

import numpy as np
from PIL import Image

from Exponential_quadtree_nd import _direct_coreset_with_beta
from image_processors import _assign_nearest_centers_chunked, _compute_kmeans_cost_chunked
from kmeans_pp_nd import kmeans_plus_plus_init


INPUT_DIR = Path("final_datasets")
OUTPUT_ROOT = Path("results") / "pictures_fixed_coreset"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

TARGET_CORESET_SIZES = [2, 4, 8, 32, 128]
REFERENCE_K = 8
EPS = 0.1
RANDOM_STATE = 0
MAX_SEARCH_STEPS = 28


def load_rgb_points(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)
    height, width, _ = image_array.shape
    rgb_points = image_array.reshape((-1, 3)).astype(float)
    return rgb_points, (height, width, 3)


def output_folder_for(image_path: Path) -> Path:
    return OUTPUT_ROOT / image_path.stem.replace(" ", "_")


def coreset_for_beta(rgb_points, cost, beta, target_size):
    reps, weights, _ = _direct_coreset_with_beta(
        rgb_points,
        EPS,
        cost,
        REFERENCE_K,
        beta,
        random_state=RANDOM_STATE,
    )
    size = reps.shape[0]
    gap = abs(size - target_size)
    return reps, weights, size, gap


def trim_to_target(reps, weights, target_size):
    if reps.shape[0] <= target_size:
        return reps, weights, False

    # Keep the representatives that cover the most original pixels.
    keep = np.argsort(weights)[::-1][:target_size]
    return reps[keep], weights[keep], True


def find_fixed_size_coreset(rgb_points, target_size, reference_cost):
    cache = {}

    def evaluate(beta):
        key = round(float(beta), 14)
        if key not in cache:
            cache[key] = coreset_for_beta(rgb_points, reference_cost, float(beta), target_size)
        return cache[key]

    lo = 1e-12
    hi = 1.0
    lo_result = evaluate(lo)
    hi_result = evaluate(hi)

    while lo_result[2] < target_size and lo > 1e-300:
        lo *= 0.5
        lo_result = evaluate(lo)

    while hi_result[2] > target_size:
        hi *= 2.0
        hi_result = evaluate(hi)

    best_any = lo_result
    best_any_beta = lo
    best_over = lo_result if lo_result[2] >= target_size else None
    best_over_beta = lo if lo_result[2] >= target_size else None

    for beta, result in [(hi, hi_result)]:
        if result[3] < best_any[3]:
            best_any = result
            best_any_beta = beta
        if result[2] >= target_size and (best_over is None or result[2] < best_over[2]):
            best_over = result
            best_over_beta = beta

    exact_result = None
    exact_beta = None

    for _ in range(MAX_SEARCH_STEPS):
        mid = 0.5 * (lo + hi)
        result = evaluate(mid)
        size = result[2]

        if result[3] < best_any[3]:
            best_any = result
            best_any_beta = mid
        if size >= target_size and (best_over is None or size < best_over[2]):
            best_over = result
            best_over_beta = mid

        if size == target_size:
            exact_result = result
            exact_beta = mid
            break

        # Larger beta gives coarser/smaller coresets.
        if size > target_size:
            lo = mid
        else:
            hi = mid

    if exact_result is not None:
        reps, weights, raw_size, _ = exact_result
        return {
            "reps": reps,
            "weights": weights,
            "beta": exact_beta,
            "raw_size": raw_size,
            "final_size": reps.shape[0],
            "trimmed": False,
            "reference_cost": reference_cost,
        }

    if best_over is not None:
        reps, weights, raw_size, _ = best_over
        beta = best_over_beta
    else:
        reps, weights, raw_size, _ = best_any
        beta = best_any_beta

    reps, weights, trimmed = trim_to_target(reps, weights, target_size)
    return {
        "reps": reps,
        "weights": weights,
        "beta": beta,
        "raw_size": raw_size,
        "final_size": reps.shape[0],
        "trimmed": trimmed,
        "reference_cost": reference_cost,
    }


def render_from_representatives(rgb_points, image_shape, representatives):
    nearest = _assign_nearest_centers_chunked(rgb_points, representatives)
    compressed_rgb = representatives[nearest].astype(np.uint8)
    return compressed_rgb.reshape(image_shape)


def compress_one(image_path: Path, rgb_points, image_shape, reference_cost, target_size: int):
    output_dir = output_folder_for(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image={image_path.name} target_coreset_size={target_size}")
    result = find_fixed_size_coreset(rgb_points, target_size, reference_cost)
    compressed_img = render_from_representatives(rgb_points, image_shape, result["reps"])

    output_path = output_dir / f"{image_path.stem}_coreset_{target_size}{image_path.suffix.lower()}"
    Image.fromarray(compressed_img).save(output_path)

    final_cost = _compute_kmeans_cost_chunked(rgb_points, result["reps"])
    row = {
        "image": image_path.name,
        "target_coreset_size": target_size,
        "raw_coreset_size": result["raw_size"],
        "final_coreset_size": result["final_size"],
        "trimmed_to_exact": result["trimmed"],
        "beta": result["beta"],
        "eps": EPS,
        "reference_k": REFERENCE_K,
        "full_size": rgb_points.shape[0],
        "height": image_shape[0],
        "width": image_shape[1],
        "reference_cost": result["reference_cost"],
        "render_cost": final_cost,
        "output_path": str(output_path),
    }

    stats_path = output_dir / "fixed_coreset_stats.csv"
    write_header = not stats_path.exists()
    with stats_path.open("a", newline="") as stats_file:
        writer = csv.DictWriter(stats_file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        "Wrote",
        output_path,
        f"raw_size={result['raw_size']}",
        f"final_size={result['final_size']}",
        f"trimmed={result['trimmed']}",
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        rgb_points, image_shape = load_rgb_points(image_path)
        reference_centers = kmeans_plus_plus_init(
            rgb_points,
            REFERENCE_K,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        reference_cost = _compute_kmeans_cost_chunked(rgb_points, reference_centers)
        for target_size in [2, 4, 8, 32]:
            compress_one(image_path, rgb_points, image_shape, reference_cost, target_size)

    for image_path in image_paths:
        rgb_points, image_shape = load_rgb_points(image_path)
        reference_centers = kmeans_plus_plus_init(
            rgb_points,
            REFERENCE_K,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        reference_cost = _compute_kmeans_cost_chunked(rgb_points, reference_centers)
        compress_one(image_path, rgb_points, image_shape, reference_cost, 128)


if __name__ == "__main__":
    main()
