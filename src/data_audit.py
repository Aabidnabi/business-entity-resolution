from pathlib import Path
import pandas as pd


FILES = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
]


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def audit_file(file_path: str) -> None:
    path = Path(file_path)

    if not path.exists():
        print(f"\nNOT FOUND: {path}")
        return

    row_count = count_rows(path)
    sample = pd.read_csv(path, sep="\t", nrows=10_000)

    print("\n" + "=" * 70)
    print(f"FILE: {path}")
    print(f"ROWS: {row_count:,}")
    print(f"COLUMNS: {sample.columns.tolist()}")

    print("\nMISSING VALUES:")
    print(sample.isna().sum().to_string())

    first_column = sample.columns[0]
    print(f"\nDUPLICATE {first_column} IN SAMPLE:")
    print(sample[first_column].duplicated().sum())

    print("\nSAMPLE:")
    print(sample.head(3).to_string(index=False))


def main() -> None:
    for file_path in FILES:
        audit_file(file_path)


if __name__ == "__main__":
    main()