from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
TRAIN = ROOT / "dataset" / "train"

SAMPLE_N = 10000

con = duckdb.connect()

print("Loading 10,000 matched S1 records...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT
    source1_entity_id::VARCHAR AS s1_id,
    matched_entity_ids::VARCHAR AS matched_ids
FROM read_csv(
    '{TRAIN / "train_ground_truth.tsv"}',
    delim='\t',
    header=true
)
WHERE matched_entity_ids IS NOT NULL
LIMIT {SAMPLE_N}
""")

print("Creating S1 sample...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT *
FROM read_parquet('{P / "train_source1.parquet"}')
WHERE CAST(entity_id AS VARCHAR) IN (
    SELECT s1_id FROM gt
)
""")

print("Exact normalized-name blocking: S1 -> S2...")

s2_exact = con.execute(f"""
SELECT
    s1.entity_id AS s1_id,
    s2.entity_id AS s2_id
FROM s1
JOIN read_parquet('{P / "train_source2.parquet"}') s2
    ON s1.name_norm = s2.name_norm
   AND s1.country_norm = s2.country_norm
WHERE s1.name_norm <> ''
""").fetchdf()

print(f"S2 exact candidates: {len(s2_exact):,}")

print("Exact normalized-name blocking: S1 -> S3...")

s3_exact = con.execute(f"""
SELECT
    s1.entity_id AS s1_id,
    s3.entity_id AS s3_id
FROM s1
JOIN read_parquet('{P / "train_source3.parquet"}') s3
    ON s1.name_norm = s3.name_norm
   AND s1.country_norm = s3.country_norm
WHERE s1.name_norm <> ''
""").fetchdf()

print(f"S3 exact candidates: {len(s3_exact):,}")

print("\n" + "=" * 70)
print("FAST EXACT-BLOCKING BASELINE")
print("=" * 70)

s1_count = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]

print(f"S1 tested:       {s1_count:,}")
print(f"S2 candidates:   {len(s2_exact):,}")
print(f"S3 candidates:   {len(s3_exact):,}")
print(f"Total candidates:{len(s2_exact) + len(s3_exact):,}")

print("=" * 70)
print("\nExact-name baseline completed successfully.")

print("\nCalculating ground-truth candidate recall...")

# Build candidate lookup
candidate_pairs = {}

for row in s2_exact.itertuples(index=False):
    candidate_pairs.setdefault(str(row.s1_id), set()).add(str(row.s2_id))

for row in s3_exact.itertuples(index=False):
    candidate_pairs.setdefault(str(row.s1_id), set()).add(str(row.s3_id))

gt_rows = con.execute("""
SELECT s1_id, matched_ids
FROM gt
""").fetchall()

total_true = 0
found_true = 0
entities_with_all = 0

for s1_id, matched_ids in gt_rows:
    s1_id = str(s1_id)

    if not matched_ids:
        continue

    # Ground truth format: parse common list formats safely
    text = str(matched_ids).strip()
    text = text.strip("[]")

    true_ids = {
        x.strip().strip("'").strip('"')
        for x in text.replace(";", ",").split(",")
        if x.strip()
    }

    if not true_ids:
        continue

    candidates = candidate_pairs.get(s1_id, set())

    total_true += len(true_ids)
    found = len(true_ids & candidates)
    found_true += found

    if found == len(true_ids):
        entities_with_all += 1

candidate_recall = found_true / total_true if total_true else 0
entity_recall = (
    entities_with_all / len(gt_rows)
    if gt_rows else 0
)

print("\n" + "=" * 70)
print("EXACT-NAME CANDIDATE RECALL")
print("=" * 70)
print(f"True matches:              {total_true:,}")
print(f"True matches retrieved:    {found_true:,}")
print(f"Candidate recall:          {candidate_recall:.4%}")
print(f"Entity full-match recall:  {entity_recall:.4%}")
print("=" * 70)

