import csv
from pathlib import Path

from image_processors import compress_image_with_coreset, save_compressed_image


INPUT_DIR = Path("final_datasets")
OUTPUT_ROOT = Path("results") / "pictures"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

EPS = 0.1
BETA = 640.0
LOCAL_SEARCH_STEPS = 8
RANDOM_STATE = 0


def output_folder_for(image_path: Path) -> Path:
    safe_name = image_path.stem.replace(" ", "_")
    return OUTPUT_ROOT / safe_name


def compress_one(image_path: Path, colors: int) -> dict:
    output_dir = output_folder_for(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_compressed_{colors}_colors{image_path.suffix.lower()}"
    compressed_img, original_shape, stats = compress_image_with_coreset(
        str(image_path),
        colors,
        eps=EPS,
        random_state=RANDOM_STATE,
        n_steps=LOCAL_SEARCH_STEPS,
        beta=BETA,
        verbose=True,
    )
    save_compressed_image(compressed_img, str(output_path))

    row = {
        "image": image_path.name,
        "colors": colors,
        "output_path": str(output_path),
        "height": original_shape[0],
        "width": original_shape[1],
        "full_size": stats["full_size"],
        "coreset_size": stats["coreset_size"],
        "compression_ratio_achieved": stats["compression_ratio_achieved"],
        "initial_cost": stats["initial_cost"],
        "final_cost": stats["final_cost"],
        "eps": EPS,
        "beta": BETA,
        "local_search_steps": LOCAL_SEARCH_STEPS,
    }

    stats_path = output_dir / "compression_stats.csv"
    write_header = not stats_path.exists()
    with stats_path.open("a", newline="") as stats_file:
        writer = csv.DictWriter(stats_file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Wrote {output_path}")
    return row


def run_pass(colors_to_run):
    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in image_paths:
        for colors in colors_to_run:
            compress_one(image_path, colors)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_pass([2, 4, 8, 32])
    run_pass([128])


if __name__ == "__main__":
    main()
