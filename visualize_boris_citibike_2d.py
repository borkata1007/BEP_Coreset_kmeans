import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_FILE = Path("boris_citibike_2d.csv")
OUTPUT_DIR = Path("Plots") / "citibike_overview"
OUTPUT_FILE = OUTPUT_DIR / "boris_citibike_2d_scatter.png"


def main() -> None:
    longitudes = []
    latitudes = []

    with INPUT_FILE.open(newline="") as source_file:
        reader = csv.DictReader(source_file)
        for row in reader:
            longitudes.append(float(row["longitude"]))
            latitudes.append(float(row["latitude"]))

    plt.figure(figsize=(9, 9))
    plt.scatter(longitudes, latitudes, s=3.0, alpha=0.18, linewidths=0)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Citibike Trips: Start Locations")
    plt.grid(alpha=0.2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=220)
    print(f"Wrote {OUTPUT_FILE} with {len(longitudes)} points")


if __name__ == "__main__":
    main()
