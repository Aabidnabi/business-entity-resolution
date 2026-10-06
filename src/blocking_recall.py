from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "train"

SAMPLE_S1 = 2_000
SAMPLE_S2 = 20_000
SAMPLE_S3 = 20_000


def main():
    print("Loading small samples...")

    s1 = pd.read_csv(
        DATASET / "train_source1.tsv",
        sep="\t",
        nrows=SAMPLE_S1,
    )

    s2 = pd.read_csv(
        DATASET / "train_source2.tsv",
        sep="\t",
        nrows=SAMPLE_S2,
    )

    s3 = pd.read_csv(
        DATASET / "train_source3.tsv",
        sep="\t",
        nrows=SAMPLE_S3,
    )

    print(f"S1: {len(s1):,}")
    print(f"S2: {len(s2):,}")
    print(f"S3: {len(s3):,}")

    print("\nColumns:")
    print("S1:", list(s1.columns))

    print("\nSample loaded successfully.")
    print("No full-dataset normalization yet.")


if __name__ == "__main__":
    main()