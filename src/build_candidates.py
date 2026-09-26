import duckdb
import os

os.makedirs("output", exist_ok=True)

con = duckdb.connect()

print("Building candidate pairs...")

# S1
con.execute("""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT
    entity_id,
    name_token_string,
    country_norm,
    left(regexp_replace(name_norm, '[^a-z0-9]', '', 'g'), 4) AS sig4
FROM read_parquet('data/processed/train_source1.parquet')
""")

# S2/S3 combined candidate generation.
# Main reliable block = normalized token-string.
print("Token block S2...")
con.execute("""
COPY (
    SELECT
        a.entity_id AS source1_entity_id,
        b.entity_id AS matched_entity_id
    FROM s1 a
    JOIN read_parquet('data/processed/train_source2.parquet') b
      ON a.country_norm = b.country_norm
     AND a.name_token_string = b.name_token_string
    WHERE a.name_token_string <> ''
) TO 'output/candidates_s2.parquet' (FORMAT PARQUET)
""")

print("Token block S3...")
con.execute("""
COPY (
    SELECT
        a.entity_id AS source1_entity_id,
        b.entity_id AS matched_entity_id
    FROM s1 a
    JOIN read_parquet('data/processed/train_source3.parquet') b
      ON a.country_norm = b.country_norm
     AND a.name_token_string = b.name_token_string
    WHERE a.name_token_string <> ''
) TO 'output/candidates_s3.parquet' (FORMAT PARQUET)
""")

# Final token-only candidate set
print("Combining candidates...")

con.execute("""
COPY (
    SELECT DISTINCT source1_entity_id, matched_entity_id
    FROM (
        SELECT * FROM read_parquet('output/candidates_s2.parquet')
        UNION ALL
        SELECT * FROM read_parquet('output/candidates_s3.parquet')
    )
) TO 'output/candidate_pairs_train.parquet' (FORMAT PARQUET)
""")

n = con.execute("""
SELECT COUNT(*) FROM read_parquet('output/candidate_pairs_train.parquet')
""").fetchone()[0]

print(f"FINAL TRAIN CANDIDATES: {n:,}")
print("DONE")
