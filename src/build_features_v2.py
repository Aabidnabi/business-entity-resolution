import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8")

print("Building V2 pairwise features...")

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

        -- NAME TOKEN JACCARD
        CASE
            WHEN a.name_token_string = '' OR b.name_token_string = '' THEN 0.0
            ELSE (
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT token
                    FROM unnest(string_split(a.name_token_string, ' ')) AS x(token)
                    WHERE token <> ''
                ) aa
                WHERE token IN (
                    SELECT DISTINCT token
                    FROM unnest(string_split(b.name_token_string, ' ')) AS y(token)
                    WHERE token <> ''
                )
            )::DOUBLE
            /
            NULLIF((
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT token
                    FROM unnest(
                        string_split(a.name_token_string || ' ' || b.name_token_string, ' ')
                    ) AS z(token)
                    WHERE token <> ''
                ) uu
            ), 0)
        END AS name_jaccard,

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
        CASE
            WHEN a.address_norm = b.address_norm
                 AND a.address_norm <> ''
            THEN 1 ELSE 0
        END AS address_exact,

        CASE
            WHEN a.address_norm = '' OR b.address_norm = '' THEN 0.0
            ELSE 1.0 - (
                levenshtein(a.address_norm, b.address_norm)::DOUBLE
                / GREATEST(length(a.address_norm), length(b.address_norm), 1)
            )
        END AS address_edit_sim,

        -- ADDRESS TOKEN OVERLAP
        CASE
            WHEN a.address_norm = '' OR b.address_norm = '' THEN 0.0
            ELSE (
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT token
                    FROM unnest(string_split(a.address_norm, ' ')) AS x(token)
                    WHERE length(token) >= 2
                ) aa
                WHERE token IN (
                    SELECT DISTINCT token
                    FROM unnest(string_split(b.address_norm, ' ')) AS y(token)
                    WHERE length(token) >= 2
                )
            )::DOUBLE
            /
            NULLIF((
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT token
                    FROM unnest(
                        string_split(a.address_norm || ' ' || b.address_norm, ' ')
                    ) AS z(token)
                    WHERE length(token) >= 2
                ) uu
            ), 0)
        END AS address_jaccard,

        -- ADDRESS NUMBER AGREEMENT
        CASE
            WHEN a.address_numbers = b.address_numbers
                 AND a.address_norm <> ''
                 AND b.address_norm <> ''
            THEN 1 ELSE 0
        END AS address_numbers_exact,

        CASE
            WHEN a.address_norm = '' OR b.address_norm = '' THEN 0
            WHEN regexp_extract(a.address_norm, '[0-9]{5,6}') <> ''
             AND regexp_extract(a.address_norm, '[0-9]{5,6}')
                 = regexp_extract(b.address_norm, '[0-9]{5,6}')
            THEN 1
            ELSE 0
        END AS postal_code_match,

        -- COUNTRY
        CASE WHEN a.country_norm = b.country_norm
             THEN 1 ELSE 0 END AS country_exact,

        -- COUNTRY-AWARE ADDRESS SIGNALS
        CASE
            WHEN a.country_norm = b.country_norm
                 AND a.country_norm <> ''
                 AND a.address_norm <> ''
                 AND b.address_norm <> ''
            THEN 1 ELSE 0
        END AS same_country_with_address,

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
) TO 'output/training_features_v2.parquet'
(FORMAT PARQUET)
""")

n = con.execute("""
SELECT COUNT(*)
FROM read_parquet('output/training_features_v2.parquet')
""").fetchone()[0]

print(f"V2 FEATURE ROWS: {n:,}")
print("DONE")
