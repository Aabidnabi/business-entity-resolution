from pathlib import Path
import sys
import ast
import pandas as pd

# Allow imports from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import add_normalized_columns
from src.blocking import (
    build_inverted_index,
    build_exact_name_index,
    generate_candidates_for_record,
    add_country_filter,
)


DATASET = ROOT / "dataset" / "train"


def parse_matches(value):
    """Convert ground-truth matched_entity_ids into a Python list."""
    if pd.isna(value) or str(value).strip() == "":
        return []

    text = str(value).strip()

    try:
        value = ast.literal_eval(text)
        if isinstance(value, list):
            return [str(x) for x in value]
    except (ValueError, SyntaxError):
        pass

    # Fallback for simple delimiter-separated values
    for sep in ["|", ",", ";"]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()]

    return [text]


def load_data():
    print("Loading train data...")

    s1 = pd.read_csv(
        DATASET / "train_source1.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    s2 = pd.read_csv(
        DATASET / "train_source2.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    s3 = pd.read_csv(
        DATASET / "train_source3.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
    )

    gt = pd.read_csv(
        DATASET / "train_ground_truth.tsv",
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    return s1, s2, s3, gt


def build_indexes(df):
    exact = build_exact_name_index(df)

    token = build_inverted_index(
        df,
        column="name_token_string",
        min_token_length=2,
        max_df_ratio=0.01,
    )

    return exact, token


def main():
    s1, s2, s3, gt = load_data()

    print(f"S1: {len(s1):,}")
    print(f"S2: {len(s2):,}")
    print(f"S3: {len(s3):,}")
    print(f"Ground truth: {len(gt):,}")

    print("\nNormalizing...")
    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    print("\nBuilding indexes...")
    s2_exact, s2_token = build_indexes(s2)
    s3_exact, s3_token = build_indexes(s3)

    print("Indexes ready.")

    # ---------------------------------------------------------
    # Use a manageable validation sample first.
    # We select S1 records that actually have ground-truth matches.
    # ---------------------------------------------------------
    gt["matched_list"] = gt["matched_entity_ids"].apply(parse_matches)

    matched_gt = gt[gt["matched_list"].map(len) > 0].copy()

    # Deterministic sample
    sample_gt = matched_gt.sample(
        n=min(10_000, len(matched_gt)),
        random_state=42,
    )

    s1_ids = set(sample_gt["source1_entity_id"].astype(str))

    s1_sample = s1[
        s1["entity_id"].astype(str).isin(s1_ids)
    ].copy()

    gt_map = dict(
        zip(
            sample_gt["source1_entity_id"].astype(str),
            sample_gt["matched_list"],
        )
    )

    print(f"\nMatched S1 records tested: {len(s1_sample):,}")

    # ---------------------------------------------------------
    # Candidate recall
    # ---------------------------------------------------------
    found_matches = 0
    total_true_matches = 0
    total_candidates = 0
    records_with_all_matches = 0

    print("\nTesting candidate recall...")

    for i, row in enumerate(s1_sample.itertuples(index=False), start=1):

        s1_id = str(row.entity_id)

        # Convert namedtuple to dictionary-like access
        record = row._asdict()

        candidates_s2 = generate_candidates_for_record(
            record,
            s2,
            s2_exact,
            s2_token,
        )

        candidates_s3 = generate_candidates_for_record(
            record,
            s3,
            s3_exact,
            s3_token,
        )

        candidates_s2 = add_country_filter(
            record,
            candidates_s2,
            s2,
        )

        candidates_s3 = add_country_filter(
            record,
            candidates_s3,
            s3,
        )

        candidate_ids = set()

        for idx in candidates_s2:
            candidate_ids.add(str(s2.iloc[idx]["entity_id"]))

        for idx in candidates_s3:
            candidate_ids.add(str(s3.iloc[idx]["entity_id"]))

        true_ids = set(gt_map.get(s1_id, []))

        total_true_matches += len(true_ids)
        found = len(true_ids.intersection(candidate_ids))

        found_matches += found
        total_candidates += len(candidate_ids)

        if found == len(true_ids):
            records_with_all_matches += 1

        if i % 1000 == 0:
            recall = (
                found_matches / total_true_matches
                if total_true_matches
                else 0
            )

            avg_candidates = total_candidates / i

            print(
                f"Processed {i:,} | "
                f"Recall: {recall:.4%} | "
                f"Avg candidates: {avg_candidates:.1f}"
            )

    candidate_recall = (
        found_matches / total_true_matches
        if total_true_matches
        else 0
    )

    entity_recall = (
        records_with_all_matches / len(s1_sample)
        if len(s1_sample)
        else 0
    )

    avg_candidates = (
        total_candidates / len(s1_sample)
        if len(s1_sample)
        else 0
    )

    print("\n" + "=" * 70)
    print("CANDIDATE RECALL RESULTS")
    print("=" * 70)

    print(f"S1 records tested:        {len(s1_sample):,}")
    print(f"True matches:             {total_true_matches:,}")
    print(f"True matches retrieved:   {found_matches:,}")
    print(f"Candidate recall:         {candidate_recall:.4%}")
    print(f"Entity full-match recall: {entity_recall:.4%}")
    print(f"Average candidates/S1:    {avg_candidates:.2f}")

    print("=" * 70)


if __name__ == "__main__":
    main()