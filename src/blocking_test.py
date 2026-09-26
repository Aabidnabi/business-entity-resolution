import pandas as pd

from preprocessing import add_normalized_columns
from blocking import (
    build_exact_name_index,
    build_inverted_index,
    generate_candidates_for_record,
    add_country_filter,
)


TRAIN_DIR = "dataset/train"

S1_FILE = f"{TRAIN_DIR}/train_source1.tsv"
S2_FILE = f"{TRAIN_DIR}/train_source2.tsv"
S3_FILE = f"{TRAIN_DIR}/train_source3.tsv"


def load_sample(path: str, nrows: int) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
    )


def main() -> None:
    print("Loading samples...")

    s1 = load_sample(S1_FILE, 10_000)
    s2 = load_sample(S2_FILE, 100_000)
    s3 = load_sample(S3_FILE, 100_000)

    print(f"S1 sample: {len(s1):,}")
    print(f"S2 sample: {len(s2):,}")
    print(f"S3 sample: {len(s3):,}")

    print("\nNormalizing...")

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    print("Building indexes...")

    exact2 = build_exact_name_index(s2)
    exact3 = build_exact_name_index(s3)

    token2 = build_inverted_index(s2, "name_tokens")
    token3 = build_inverted_index(s3, "name_tokens")

    print(f"S2 exact-name keys: {len(exact2):,}")
    print(f"S3 exact-name keys: {len(exact3):,}")
    print(f"S2 token keys: {len(token2):,}")
    print(f"S3 token keys: {len(token3):,}")

    total_candidates = 0
    zero_candidates = 0
    candidate_counts = []

    print("\nTesting candidate generation...")

    for _, row in s1.iterrows():
        candidates2 = generate_candidates_for_record(
            row["name_norm"],
            row["name_tokens"],
            exact2,
            token2,
        )

        candidates3 = generate_candidates_for_record(
            row["name_norm"],
            row["name_tokens"],
            exact3,
            token3,
        )

        candidates2 = add_country_filter(
            candidates2,
            s2,
            row["country_norm"],
        )

        candidates3 = add_country_filter(
            candidates3,
            s3,
            row["country_norm"],
        )

        count = len(candidates2) + len(candidates3)

        candidate_counts.append(count)
        total_candidates += count

        if count == 0:
            zero_candidates += 1

    counts = pd.Series(candidate_counts)

    print("\n" + "=" * 70)
    print("BLOCKING RESULTS")
    print("=" * 70)

    print(f"S1 records tested: {len(s1):,}")
    print(f"Total candidate pairs: {total_candidates:,}")
    print(f"Average candidates / S1: {counts.mean():.2f}")
    print(f"Median candidates / S1: {counts.median():.2f}")
    print(f"95th percentile: {counts.quantile(0.95):.2f}")
    print(f"99th percentile: {counts.quantile(0.99):.2f}")
    print(f"Max candidates: {counts.max():,}")
    print(f"S1 with zero candidates: {zero_candidates:,}")
    print(
        f"S1 zero-candidate rate: "
        f"{zero_candidates / len(s1):.2%}"
    )


if __name__ == "__main__":
    main()