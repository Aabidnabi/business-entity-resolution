from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
TRAIN = ROOT / "dataset" / "train"

N = 10000

con = duckdb.connect()
con.execute("PRAGMA threads=4")

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
LIMIT {N}
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT *
FROM read_parquet('{P / "train_source1.parquet"}')
WHERE CAST(entity_id AS VARCHAR) IN
      (SELECT s1_id FROM gt)
""")

print("S1 loaded.")

# ---------------------------------------------------------
# Helper: evaluate a candidate table against ground truth
# ---------------------------------------------------------
def evaluate(candidate_table, label):
    rows = con.execute(f"""
        SELECT s1_id, candidate_id
        FROM {candidate_table}
    """).fetchall()

    lookup = {}

    for s1_id, cid in rows:
        lookup.setdefault(str(s1_id), set()).add(str(cid))

    gt_rows = con.execute(
        "SELECT s1_id, matched_ids FROM gt"
    ).fetchall()

    total_true = 0
    found_true = 0
    full_entities = 0
    matched_entities = 0

    for s1_id, matched_ids in gt_rows:
        if not matched_ids:
            continue

        text = str(matched_ids).strip()
        text = text.strip("[](){}")

        true_ids = {
            x.strip().strip("'").strip('"')
            for x in text.replace(";", ",").split(",")
            if x.strip()
        }

        if not true_ids:
            continue

        matched_entities += 1
        candidates = lookup.get(str(s1_id), set())

        found = len(true_ids & candidates)

        total_true += len(true_ids)
        found_true += found

        if found == len(true_ids):
            full_entities += 1

    recall = found_true / total_true if total_true else 0
    entity_recall = (
        full_entities / matched_entities
        if matched_entities else 0
    )

    count = len(rows)

    print(
        f"{label:<28} "
        f"candidates={count:,}  "
        f"avg/S1={count / len(gt_rows):.2f}  "
        f"recall={recall:.4%}  "
        f"full={entity_recall:.4%}"
    )

    return lookup


# ---------------------------------------------------------
# 1. Exact normalized name + country
# ---------------------------------------------------------
print("\n[1/3] Exact-name blocking...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE exact AS
SELECT DISTINCT
    s1.entity_id::VARCHAR AS s1_id,
    s2.entity_id::VARCHAR AS candidate_id
FROM s1
JOIN read_parquet('{P / "train_source2.parquet"}') s2
  ON s1.name_norm = s2.name_norm
 AND s1.country_norm = s2.country_norm
WHERE s1.name_norm <> ''
""")

con.execute(f"""
INSERT INTO exact
SELECT DISTINCT
    s1.entity_id::VARCHAR,
    s3.entity_id::VARCHAR
FROM s1
JOIN read_parquet('{P / "train_source3.parquet"}') s3
  ON s1.name_norm = s3.name_norm
 AND s1.country_norm = s3.country_norm
WHERE s1.name_norm <> ''
""")

evaluate("exact", "Exact name")


# ---------------------------------------------------------
# 2. Rare name-token blocking
# ---------------------------------------------------------
print("\n[2/3] Rare-token blocking...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1_tokens AS
SELECT DISTINCT
    entity_id::VARCHAR AS s1_id,
    country_norm,
    token
FROM s1,
     UNNEST(name_tokens) AS t(token)
WHERE token IS NOT NULL
  AND len(token) >= 3
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s2_token_freq AS
SELECT
    country_norm,
    token,
    COUNT(*) AS freq
FROM read_parquet('{P / "train_source2.parquet"}'),
     UNNEST(name_tokens) AS t(token)
WHERE len(token) >= 3
GROUP BY country_norm, token
HAVING COUNT(*) <= 50000
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s3_token_freq AS
SELECT
    country_norm,
    token,
    COUNT(*) AS freq
FROM read_parquet('{P / "train_source3.parquet"}'),
     UNNEST(name_tokens) AS t(token)
WHERE len(token) >= 3
GROUP BY country_norm, token
HAVING COUNT(*) <= 50000
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE token_candidates AS
SELECT DISTINCT
    a.s1_id,
    b.entity_id::VARCHAR AS candidate_id
FROM s1_tokens a
JOIN s2_token_freq f
  ON a.country_norm = f.country_norm
 AND a.token = f.token
JOIN read_parquet('{P / "train_source2.parquet"}') b
  ON b.country_norm = f.country_norm
 AND list_contains(b.name_tokens, f.token)

UNION

SELECT DISTINCT
    a.s1_id,
    b.entity_id::VARCHAR AS candidate_id
FROM s1_tokens a
JOIN s3_token_freq f
  ON a.country_norm = f.country_norm
 AND a.token = f.token
JOIN read_parquet('{P / "train_source3.parquet"}') b
  ON b.country_norm = f.country_norm
 AND list_contains(b.name_tokens, f.token)
""")

evaluate("token_candidates", "Rare name token")


# ---------------------------------------------------------
# 3. UNION: exact + rare-token
# ---------------------------------------------------------
print("\n[3/3] Union blocking...")

con.execute("""
CREATE OR REPLACE TEMP TABLE all_candidates AS
SELECT s1_id, candidate_id FROM exact
UNION
SELECT s1_id, candidate_id FROM token_candidates
""")

evaluate("all_candidates", "EXACT + TOKEN UNION")

print("\n" + "=" * 80)
print("MULTI-BLOCK RECALL TEST COMPLETE")
print("=" * 80)
