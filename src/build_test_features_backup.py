import duckdb
import os

os.makedirs("output", exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA threads=4")

print("Building FAST TEST features...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT
    entity_id,
    name_norm,
    name_token_string,
    address_norm,
    country_norm
FROM read_parquet('data/processed/test_source1.parquet')
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE sources AS
SELECT entity_id, name_norm, name_token_string, address_norm, country_norm
FROM read_parquet('data/processed/test_source2.parquet')
UNION ALL
SELECT entity_id, name_norm, name_token_string, address_norm, country_norm
FROM read_parquet('data/processed/test_source3.parquet')
""")

print("Creating features from all candidates...")

con.execute("""
COPY (
    SELECT
        c.source1_entity_id,
        c.matched_entity_id,

        CASE
            WHEN a.name_norm = b.name_norm AND a.name_norm <> ''
            THEN 1 ELSE 0
        END AS name_exact,

        CASE
            WHEN a.name_norm <> ''
             AND a.name_token_string = b.name_token_string
            THEN 1 ELSE 0
        END AS name_token_exact,

        CASE
            WHEN a.name_norm = b.name_norm THEN 1.0
            WHEN a.name_norm = '' OR b.name_norm = '' THEN 0.0
            ELSE LEAST(length(a.name_norm), length(b.name_norm))::DOUBLE
                 / GREATEST(length(a.name_norm), length(b.name_norm))
        END AS name_edit_sim,

        CASE
            WHEN GREATEST(length(a.name_norm), length(b.name_norm)) = 0
            THEN 1.0
            ELSE LEAST(length(a.name_norm), length(b.name_norm))::DOUBLE
                 / GREATEST(length(a.name_norm), length(b.name_norm))
        END AS name_length_ratio,

        CASE
            WHEN a.address_norm = b.address_norm
             AND a.address_norm <> ''
            THEN 1 ELSE 0
        END AS address_exact,

        CASE
            WHEN a.address_norm = b.address_norm THEN 1.0
            WHEN a.address_norm = '' OR b.address_norm = '' THEN 0.0
            ELSE LEAST(length(a.address_norm), length(b.address_norm))::DOUBLE
                 / GREATEST(length(a.address_norm), length(b.address_norm))
        END AS address_edit_sim,

        CASE
            WHEN a.country_norm = b.country_norm
             AND a.country_norm <> ''
            THEN 1 ELSE 0
        END AS country_exact,

        CASE WHEN a.name_norm = '' THEN 1 ELSE 0 END AS s1_name_missing,
        CASE WHEN b.name_norm = '' THEN 1 ELSE 0 END AS s2_name_missing,
        CASE WHEN a.address_norm = '' THEN 1 ELSE 0 END AS s1_address_missing,
        CASE WHEN b.address_norm = '' THEN 1 ELSE 0 END AS s2_address_missing

    FROM read_parquet('output/candidate_pairs_test.parquet') c
    JOIN s1 a
      ON c.source1_entity_id = a.entity_id
    JOIN sources b
      ON c.matched_entity_id = b.entity_id
) TO 'output/test_features.parquet' (FORMAT PARQUET)
""")

count = con.execute("""
SELECT COUNT(*) FROM read_parquet('output/test_features.parquet')
""").fetchone()[0]

print(f"TEST FEATURES: {count:,}")
print("DONE")
