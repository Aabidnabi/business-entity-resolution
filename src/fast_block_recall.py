from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
TRAIN = ROOT / "dataset" / "train"
N = 10000

con = duckdb.connect()
con.execute("PRAGMA threads=4")

print("Loading sample...")

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
WHERE CAST(entity_id AS VARCHAR) IN (SELECT s1_id FROM gt)
""")

def evaluate(table, label):
    rows = con.execute(
        f"SELECT s1_id, candidate_id FROM {table}"
    ).fetchall()

    candidates = {}
    for s1, cid in rows:
        candidates.setdefault(str(s1), set()).add(str(cid))

    total = found = full = entities = 0

    for s1, raw in con.execute(
        "SELECT s1_id, matched_ids FROM gt"
    ).fetchall():

        if not raw:
            continue

        true_ids = {
            x.strip().strip("'").strip('"')
            for x in str(raw).strip("[](){}").split(",")
            if x.strip()
        }

        if not true_ids:
            continue

        entities += 1
        got = candidates.get(str(s1), set())
        hit = len(true_ids & got)

        total += len(true_ids)
        found += hit

        if hit == len(true_ids):
            full += 1

    print(
        f"{label:<25} "
        f"candidates={len(rows):,}  "
        f"recall={found/total:.4%}  "
        f"full={full/entities:.4%}"
    )

# 1. Exact normalized name
print("\n[1] Exact name...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE name_candidates AS
SELECT DISTINCT
    s1.entity_id::VARCHAR AS s1_id,
    s2.entity_id::VARCHAR AS candidate_id
FROM s1
JOIN read_parquet('{P / "train_source2.parquet"}') s2
  ON s1.name_norm = s2.name_norm
 AND s1.country_norm = s2.country_norm
WHERE s1.name_norm <> ''

UNION

SELECT DISTINCT
    s1.entity_id::VARCHAR,
    s3.entity_id::VARCHAR
FROM s1
JOIN read_parquet('{P / "train_source3.parquet"}') s3
  ON s1.name_norm = s3.name_norm
 AND s1.country_norm = s3.country_norm
WHERE s1.name_norm <> ''
""")

evaluate("name_candidates", "Exact name")

# 2. Exact token-string (handles word order)
print("\n[2] Token-string...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE token_candidates AS
SELECT DISTINCT
    s1.entity_id::VARCHAR AS s1_id,
    s2.entity_id::VARCHAR AS candidate_id
FROM s1
JOIN read_parquet('{P / "train_source2.parquet"}') s2
  ON s1.name_token_string = s2.name_token_string
 AND s1.country_norm = s2.country_norm
WHERE s1.name_token_string <> ''

UNION

SELECT DISTINCT
    s1.entity_id::VARCHAR,
    s3.entity_id::VARCHAR
FROM s1
JOIN read_parquet('{P / "train_source3.parquet"}') s3
  ON s1.name_token_string = s3.name_token_string
 AND s1.country_norm = s3.country_norm
WHERE s1.name_token_string <> ''
""")

evaluate("token_candidates", "Token-string")

# 3. Address-number blocking

print("\nDONE")
