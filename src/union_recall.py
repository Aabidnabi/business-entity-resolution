import duckdb

con = duckdb.connect()

print("Loading sample + ground truth...")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT *
FROM read_parquet('data/processed/train_source1.parquet')
LIMIT 10000
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT
    source1_entity_id,
    matched_entity_ids
FROM read_csv_auto(
    'dataset/train/train_ground_truth.tsv',
    delim='\t',
    header=true
)
WHERE matched_entity_ids IS NOT NULL
  AND matched_entity_ids <> ''
LIMIT 10000
""")

# Ground-truth matched pairs
con.execute("""
CREATE OR REPLACE TEMP TABLE truth AS
SELECT
    source1_entity_id,
    trim(x) AS matched_id
FROM gt,
UNNEST(string_split(matched_entity_ids, ',')) t(x)
WHERE trim(x) <> ''
""")

print("[1] Building token + character candidates...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidates AS

-- TOKEN STRING BLOCK
SELECT
    a.entity_id AS s1_id,
    b.entity_id AS candidate_id
FROM s1 a
JOIN read_parquet('data/processed/train_source2.parquet') b
  ON a.country_norm = b.country_norm
 AND a.name_token_string = b.name_token_string
WHERE a.name_token_string <> ''

UNION

SELECT
    a.entity_id,
    b.entity_id
FROM s1 a
JOIN read_parquet('data/processed/train_source3.parquet') b
  ON a.country_norm = b.country_norm
 AND a.name_token_string = b.name_token_string
WHERE a.name_token_string <> ''

UNION

-- CHARACTER SIGNATURE BLOCK
SELECT
    a.entity_id,
    b.entity_id
FROM (
    SELECT *,
           left(regexp_replace(name_norm, '[^a-z0-9]', '', 'g'), 4) AS sig4,
           left(regexp_replace(name_norm, '[^a-z0-9]', '', 'g'), 5) AS sig5
    FROM s1
) a
JOIN read_parquet('data/processed/train_source2.parquet') b
  ON a.country_norm = b.country_norm
 AND (
      a.sig4 = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 4)
      OR a.sig5 = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 5)
 )
WHERE a.sig4 <> ''

UNION

SELECT
    a.entity_id,
    b.entity_id
FROM (
    SELECT *,
           left(regexp_replace(name_norm, '[^a-z0-9]', '', 'g'), 4) AS sig4,
           left(regexp_replace(name_norm, '[^a-z0-9]', '', 'g'), 5) AS sig5
    FROM s1
) a
JOIN read_parquet('data/processed/train_source3.parquet') b
  ON a.country_norm = b.country_norm
 AND (
      a.sig4 = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 4)
      OR a.sig5 = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 5)
 )
WHERE a.sig4 <> ''
""")

total = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
print(f"Total unique candidates: {total:,}")

print("[2] Measuring ground-truth recall...")

retrieved = con.execute("""
SELECT COUNT(*)
FROM truth t
JOIN candidates c
  ON t.source1_entity_id = c.s1_id
 AND t.matched_id = c.candidate_id
""").fetchone()[0]

truth_total = con.execute("SELECT COUNT(*) FROM truth").fetchone()[0]

entity_full = con.execute("""
SELECT COUNT(*)
FROM (
    SELECT t.source1_entity_id,
           COUNT(*) AS truth_n,
           COUNT(c.candidate_id) AS found_n
    FROM truth t
    LEFT JOIN candidates c
      ON t.source1_entity_id = c.s1_id
     AND t.matched_id = c.candidate_id
    GROUP BY t.source1_entity_id
)
WHERE truth_n = found_n
""").fetchone()[0]

entities = con.execute("""
SELECT COUNT(DISTINCT source1_entity_id)
FROM truth
""").fetchone()[0]

print()
print("========== UNION RESULT ==========")
print(f"True matches:              {truth_total:,}")
print(f"Retrieved true matches:    {retrieved:,}")
print(f"Candidate recall:          {retrieved / truth_total * 100:.4f}%")
print(f"Full-match entity recall:  {entity_full / entities * 100:.4f}%")
print("===================================")
