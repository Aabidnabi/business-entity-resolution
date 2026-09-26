import duckdb

con = duckdb.connect()

print("Loading sample...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT *
FROM read_parquet('data/processed/train_source1.parquet')
LIMIT 10000
""")

# Only S1 entities that actually have a GT match
con.execute("""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT DISTINCT source1_entity_id
FROM read_csv_auto(
    'dataset/train/train_ground_truth.tsv',
    delim='\t',
    header=true
)
WHERE matched_entity_ids IS NOT NULL
  AND matched_entity_ids <> ''
LIMIT 10000
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1m AS
SELECT s1.*
FROM s1
JOIN gt ON s1.entity_id = gt.source1_entity_id
""")

print("[1] Exact address + country -> S2")
r2 = con.execute("""
SELECT COUNT(*)
FROM s1m a
JOIN read_parquet('data/processed/train_source2.parquet') b
  ON a.address_norm = b.address_norm
 AND a.country_norm = b.country_norm
WHERE a.address_norm <> ''
""").fetchone()[0]

print("S2 candidates:", r2)

print("[2] Exact address + country -> S3")
r3 = con.execute("""
SELECT COUNT(*)
FROM s1m a
JOIN read_parquet('data/processed/train_source3.parquet') b
  ON a.address_norm = b.address_norm
 AND a.country_norm = b.country_norm
WHERE a.address_norm <> ''
""").fetchone()[0]

print("S3 candidates:", r3)
print("DONE")
