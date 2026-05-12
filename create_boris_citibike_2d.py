import csv
from pathlib import Path


INPUT_FILE = Path("boris_citibike_duration.csv")
OUTPUT_FILE = Path("boris_citibike_2d.csv")
KEEP_COLUMNS = ["id", "longitude", "latitude"]


def main() -> None:
    with INPUT_FILE.open(newline="") as source_file:
        reader = csv.DictReader(source_file)

        missing_columns = [column for column in KEEP_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing expected column(s): {missing}")

        with OUTPUT_FILE.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=KEEP_COLUMNS)
            writer.writeheader()

            for row in reader:
                writer.writerow({column: row[column] for column in KEEP_COLUMNS})

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
