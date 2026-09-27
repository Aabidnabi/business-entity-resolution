import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8")

print("Building pairwise features...")

con.execute("""
CREATE OR REPLACE TEMP TABLE pairs AS
SELECT * FROM read_parquet('output/training_pairs.parquet')
""")

con.execute("""
COPY (
    SELECT
        p.source1_entity_id,
        p.matched_entity_id,
        p.label,

        -- NAME
        CASE WHEN a.name_norm = b.name_norm
             THEN 1 ELSE 0 END AS name_exact,

        CASE WHEN a.name_token_string = b.name_token_string
             THEN 1 ELSE 0 END AS name_token_exact,

        CASE
            WHEN a.name_norm = '' OR b.name_norm = '' THEN 0.0
            ELSE 1.0 - (
                levenshtein(a.name_norm, b.name_norm)::DOUBLE
                / GREATEST(length(a.name_norm), length(b.name_norm), 1)
            )
        END AS name_edit_sim,

        CASE
            WHEN a.name_norm = '' OR b.name_norm = '' THEN 0.0
            ELSE
                LEAST(length(a.name_norm), length(b.name_norm))::DOUBLE
                / GREATEST(length(a.name_norm), length(b.name_norm), 1)
        END AS name_length_ratio,

        -- ADDRESS
        CASE WHEN a.address_norm = b.address_norm
                  AND a.address_norm <> ''
             THEN 1 ELSE 0 END AS address_exact,

        CASE
            WHEN a.address_norm = '' OR b.address_norm = '' THEN 0.0
            ELSE 1.0 - (
                levenshtein(a.address_norm, b.address_norm)::DOUBLE
                / GREATEST(length(a.address_norm), length(b.address_norm), 1)
            )
        END AS address_edit_sim,

        -- COUNTRY
        CASE WHEN a.country_norm = b.country_norm
             THEN 1 ELSE 0 END AS country_exact,

        -- MISSINGNESS
        CASE WHEN a.name_norm = '' THEN 1 ELSE 0 END AS s1_name_missing,
        CASE WHEN b.name_norm = '' THEN 1 ELSE 0 END AS s2_name_missing,
        CASE WHEN a.address_norm = '' THEN 1 ELSE 0 END AS s1_address_missing,
        CASE WHEN b.address_norm = '' THEN 1 ELSE 0 END AS s2_address_missing

    FROM pairs p
    JOIN read_parquet('data/processed/train_source1.parquet') a
      ON p.source1_entity_id = a.entity_id
    JOIN (
        SELECT * FROM read_parquet('data/processed/train_source2.parquet')
        UNION ALL
        SELECT * FROM read_parquet('data/processed/train_source3.parquet')
    ) b
      ON p.matched_entity_id = b.entity_id
) TO 'output/training_features.parquet'
(FORMAT PARQUET)
""")

n = con.execute("""
SELECT COUNT(*)
FROM read_parquet('output/training_features.parquet')
""").fetchone()[0]

print(f"FEATURE ROWS: {n:,}")
print("DONE")
