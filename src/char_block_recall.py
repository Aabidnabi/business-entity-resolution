import duckdb

con = duckdb.connect()

print("Loading 10k matched S1...")
con.execute("""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT *
FROM read_parquet('data/processed/train_source1.parquet')
LIMIT 10000
""")

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
SELECT *
FROM s1
WHERE entity_id IN (SELECT source1_entity_id FROM gt)
""")

# Character-prefix signatures.
# Multiple signatures = better recall without huge joins.
con.execute("""
CREATE OR REPLACE TEMP TABLE s1sig AS
SELECT *,
       left(regexp_replace(name_norm, '[^a-z0-9]', '', 'g'), 4) AS sig4,
       left(regexp_replace(name_norm, '[^a-z0-9]', '', 'g'), 5) AS sig5
FROM s1m
WHERE name_norm <> ''
""")

for source in ["train_source2", "train_source3"]:
    print(f"[{source}] character signatures...")

    q = f"""
    SELECT COUNT(*)
    FROM s1sig a
    JOIN read_parquet('data/processed/{source}.parquet') b
      ON a.country_norm = b.country_norm
     AND (
          a.sig4 = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 4)
          OR a.sig5 = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 5)
     )
    WHERE a.sig4 <> ''
    """

    n = con.execute(q).fetchone()[0]
    print(f"{source}: {n:,} candidates")

print("DONE")
