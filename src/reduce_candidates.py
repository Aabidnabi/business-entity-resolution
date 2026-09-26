import duckdb

con = duckdb.connect()

print("Counting candidate frequency...")

con.execute("""
CREATE OR REPLACE TEMP TABLE freq AS
SELECT
    name_token_string,
    country_norm,
    COUNT(*) AS freq
FROM read_parquet('data/processed/train_source2.parquet')
WHERE name_token_string <> ''
GROUP BY name_token_string, country_norm
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE freq3 AS
SELECT
    name_token_string,
    country_norm,
    COUNT(*) AS freq
FROM read_parquet('data/processed/train_source3.parquet')
WHERE name_token_string <> ''
GROUP BY name_token_string, country_norm
""")

print("Building reduced candidates...")

con.execute("""
COPY (
    SELECT DISTINCT
        a.entity_id AS source1_entity_id,
        b.entity_id AS matched_entity_id
    FROM read_parquet('data/processed/train_source1.parquet') a
    JOIN read_parquet('data/processed/train_source2.parquet') b
      ON a.country_norm = b.country_norm
     AND a.name_token_string = b.name_token_string
    JOIN freq f
      ON b.name_token_string = f.name_token_string
     AND b.country_norm = f.country_norm
    WHERE a.name_token_string <> ''
      AND f.freq <= 20

    UNION

    SELECT DISTINCT
        a.entity_id,
        b.entity_id
    FROM read_parquet('data/processed/train_source1.parquet') a
    JOIN read_parquet('data/processed/train_source3.parquet') b
      ON a.country_norm = b.country_norm
     AND a.name_token_string = b.name_token_string
    JOIN freq3 f
      ON b.name_token_string = f.name_token_string
     AND b.country_norm = f.country_norm
    WHERE a.name_token_string <> ''
      AND f.freq <= 20
) TO 'output/candidate_pairs_train.parquet'
(FORMAT PARQUET)
""")

n = con.execute("""
SELECT COUNT(*)
FROM read_parquet('output/candidate_pairs_train.parquet')
""").fetchone()[0]

print(f"REDUCED CANDIDATES: {n:,}")
print("DONE")
