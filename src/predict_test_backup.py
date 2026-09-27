import duckdb
import lightgbm as lgb
import pandas as pd
import numpy as np
import os

MODEL = "output/lightgbm_model.txt"
FEATURES_FILE = "output/test_features.parquet"
CANDIDATES_FILE = "output/candidate_pairs_test.parquet"

THRESHOLD = float(open("output/best_threshold.txt").read().strip())

FEATURES = [
    "name_exact",
    "name_token_exact",
    "name_edit_sim",
    "name_length_ratio",
    "address_exact",
    "address_edit_sim",
    "country_exact",
    "s1_name_missing",
    "s2_name_missing",
    "s1_address_missing",
    "s2_address_missing",
]

os.makedirs("output", exist_ok=True)

print(f"Loading model...")
model = lgb.Booster(model_file=MODEL)
print(f"Threshold: {THRESHOLD}")

con = duckdb.connect()
con.execute("PRAGMA threads=4")

print("Scoring test features in chunks...")

con.execute("""
CREATE OR REPLACE TABLE scored AS
SELECT
    source1_entity_id,
    matched_entity_id,
    CAST(NULL AS DOUBLE) AS score
FROM read_parquet(?)
WHERE FALSE
""", [FEATURES_FILE])

offset = 0
batch = 500000

while True:
    df = con.execute("""
        SELECT
            source1_entity_id,
            matched_entity_id,
            name_exact,
            name_token_exact,
            name_edit_sim,
            name_length_ratio,
            address_exact,
            address_edit_sim,
            country_exact,
            s1_name_missing,
            s2_name_missing,
            s1_address_missing,
            s2_address_missing
        FROM read_parquet(?)
        LIMIT ? OFFSET ?
    """, [FEATURES_FILE, batch, offset]).df()

    if df.empty:
        break

    X = df[FEATURES]
    df["score"] = model.predict(X)

    con.register("batch_df", df[
        ["source1_entity_id", "matched_entity_id", "score"]
    ])

    con.execute("""
        INSERT INTO scored
        SELECT * FROM batch_df
    """)

    offset += len(df)
    print(f"Scored {offset:,} rows")

print("Creating matching_results.tsv...")

# Keep only model-approved matches.
matches = con.execute("""
SELECT
    source1_entity_id,
    matched_entity_id
FROM scored
WHERE score >= ?
ORDER BY source1_entity_id, matched_entity_id
""", [THRESHOLD]).df()

print(f"Predicted matches: {len(matches):,}")

# Build one row for EVERY test S1.
s1 = con.execute("""
SELECT entity_id
FROM read_parquet('data/processed/test_source1.parquet')
ORDER BY entity_id
""").df()

grouped = (
    matches.groupby("source1_entity_id")["matched_entity_id"]
    .apply(lambda x: ",".join(map(str, x)))
    .to_dict()
)

result = pd.DataFrame({
    "source1_entity_id": s1["entity_id"],
    "matched_entity_ids": [
        grouped.get(str(x), grouped.get(x, ""))
        for x in s1["entity_id"]
    ],
})

result.to_csv(
    "output/matching_results.tsv",
    sep="\t",
    index=False
)

# Candidate file required by the competition.
cand = con.execute("""
SELECT
    source1_entity_id,
    matched_entity_id
FROM read_parquet(?)
ORDER BY source1_entity_id, matched_entity_id
""", [CANDIDATES_FILE]).df()

cand.to_csv(
    "output/candidate_pairs.tsv",
    sep="\t",
    index=False
)

print(f"Submission rows: {len(result):,}")
print(f"Candidate rows: {len(cand):,}")
print("DONE")
