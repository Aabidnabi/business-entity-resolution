import duckdb
import os

os.makedirs("output", exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA threads=8")

print("Loading ground truth...")

con.execute("""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT
    source1_entity_id,
    matched_entity_ids
FROM read_csv_auto(
    'dataset/train/train_ground_truth.tsv',
    delim='\t',
    header=true
)
""")

print("Creating positive pairs...")

con.execute("""
CREATE OR REPLACE TEMP TABLE positives AS
SELECT
    source1_entity_id,
    trim(x) AS matched_entity_id,
    1 AS label
FROM gt,
UNNEST(string_split(
    COALESCE(matched_entity_ids, ''),
    ','
)) t(x)
WHERE trim(x) <> ''
""")

p = con.execute("SELECT COUNT(*) FROM positives").fetchone()[0]
print(f"Positive pairs: {p:,}")

print("Creating candidate negatives from token block...")

# Candidate pairs excluding known positives.
# LIMIT keeps training set practical.
con.execute("""
CREATE OR REPLACE TEMP TABLE negatives AS
SELECT
    a.entity_id AS source1_entity_id,
    b.entity_id AS matched_entity_id,
    0 AS label
FROM read_parquet('data/processed/train_source1.parquet') a
JOIN read_parquet('data/processed/train_source2.parquet') b
  ON a.country_norm = b.country_norm
 AND a.name_token_string = b.name_token_string
WHERE a.name_token_string <> ''
  AND NOT EXISTS (
      SELECT 1
      FROM positives p
      WHERE p.source1_entity_id = a.entity_id
        AND p.matched_entity_id = b.entity_id
  )
LIMIT 1000000
""")

n = con.execute("SELECT COUNT(*) FROM negatives").fetchone()[0]
print(f"Negative pairs: {n:,}")

print("Writing training pairs...")

con.execute("""
COPY (
    SELECT * FROM positives
    UNION ALL
    SELECT * FROM negatives
) TO 'output/training_pairs.parquet'
(FORMAT PARQUET)
""")

total = con.execute("""
SELECT COUNT(*)
FROM read_parquet('output/training_pairs.parquet')
""").fetchone()[0]

print(f"TOTAL TRAINING PAIRS: {total:,}")
print("DONE")
