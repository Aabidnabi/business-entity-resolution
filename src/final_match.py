import duckdb
import pandas as pd
import os

os.makedirs("output", exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA threads=4")

print("Building precision-focused matches...")

# Strong evidence:
# 1. exact normalized name
# 2. exact normalized name + address
# 3. exact normalized name + country (already country-blocked)
#
# For F0.5, avoid accepting every token-block candidate.

con.execute("""
CREATE OR REPLACE TEMP TABLE strong AS
SELECT
    c.source1_entity_id,
    c.matched_entity_id
FROM read_parquet('output/test_features.parquet') c
WHERE c.name_exact = 1
   OR (c.address_exact = 1 AND c.name_token_exact = 1)
""")

print("Strong matches created.")

# Keep only S1 entities where strong evidence exists.
matches = con.execute("""
SELECT source1_entity_id, matched_entity_id
FROM strong
ORDER BY source1_entity_id, matched_entity_id
""").df()

print(f"Strong predicted pairs: {len(matches):,}")

# Every test S1 must appear exactly once.
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
        grouped.get(x, grouped.get(str(x), ""))
        for x in s1["entity_id"]
    ]
})

result.to_csv(
    "output/matching_results.tsv",
    sep="\t",
    index=False
)

# Required candidate file.
cand = con.execute("""
SELECT source1_entity_id, matched_entity_id
FROM read_parquet('output/candidate_pairs_test.parquet')
ORDER BY source1_entity_id, matched_entity_id
""").df()

cand.to_csv(
    "output/candidate_pairs.tsv",
    sep="\t",
    index=False
)

print(f"Submission S1 rows: {len(result):,}")
print(f"Candidate pairs: {len(cand):,}")
print(f"Non-empty predictions: {(result.matched_entity_ids != '').sum():,}")
print("DONE")
