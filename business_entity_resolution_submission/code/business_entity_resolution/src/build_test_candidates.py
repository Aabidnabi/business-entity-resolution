import duckdb
import os

os.makedirs("output", exist_ok=True)

con = duckdb.connect()

print("Building TEST candidate pairs...")

# Test S1
con.execute("""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT
    entity_id,
    name_token_string,
    country_norm
FROM read_parquet('data/processed/test_source1.parquet')
""")

# S2
print("Token block TEST S2...")

con.execute("""
COPY (
    SELECT
        a.entity_id AS source1_entity_id,
        b.entity_id AS matched_entity_id
    FROM s1 a
    JOIN read_parquet('data/processed/test_source2.parquet') b
      ON a.country_norm = b.country_norm
     AND a.name_token_string = b.name_token_string
    WHERE a.name_token_string <> ''
) TO 'output/test_candidates_s2.parquet' (FORMAT PARQUET)
""")

# S3
print("Token block TEST S3...")

con.execute("""
COPY (
    SELECT
        a.entity_id AS source1_entity_id,
        b.entity_id AS matched_entity_id
    FROM s1 a
    JOIN read_parquet('data/processed/test_source3.parquet') b
      ON a.country_norm = b.country_norm
     AND a.name_token_string = b.name_token_string
    WHERE a.name_token_string <> ''
) TO 'output/test_candidates_s3.parquet' (FORMAT PARQUET)
""")

print("Combining TEST candidates...")

con.execute("""
COPY (
    SELECT DISTINCT source1_entity_id, matched_entity_id
    FROM (
        SELECT * FROM read_parquet('output/test_candidates_s2.parquet')
        UNION ALL
        SELECT * FROM read_parquet('output/test_candidates_s3.parquet')
    )
) TO 'output/candidate_pairs_test.parquet' (FORMAT PARQUET)
""")

n = con.execute("""
SELECT COUNT(*)
FROM read_parquet('output/candidate_pairs_test.parquet')
""").fetchone()[0]

s1_total = con.execute("""
SELECT COUNT(*)
FROM read_parquet('data/processed/test_source1.parquet')
""").fetchone()[0]

s1_with_candidates = con.execute("""
SELECT COUNT(DISTINCT source1_entity_id)
FROM read_parquet('output/candidate_pairs_test.parquet')
""").fetchone()[0]

print(f"TEST S1 TOTAL: {s1_total:,}")
print(f"TEST S1 WITH CANDIDATES: {s1_with_candidates:,}")
print(f"TEST CANDIDATE PAIRS: {n:,}")
print("DONE")
