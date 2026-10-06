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

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE cand_{source} AS
    SELECT DISTINCT
        a.entity_id::VARCHAR AS s1_id,
        b.entity_id::VARCHAR AS candidate_id
    FROM s1sig a
    JOIN read_parquet('data/processed/{source}.parquet') b
      ON a.country_norm = b.country_norm
     AND (
          a.sig4 = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 4)
          OR a.sig5 = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 5)
     )
    WHERE a.sig4 <> ''
    """)

    n = con.execute(f"SELECT COUNT(*) FROM cand_{source}").fetchone()[0]

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE eval_{source} AS
    SELECT
        g.source1_entity_id::VARCHAR AS s1_id,
        g.matched_entity_ids::VARCHAR AS matched_ids
    FROM read_csv(
        'dataset/train/train_ground_truth.tsv',
        delim='\\t',
        header=true
    ) g
    WHERE g.source1_entity_id::VARCHAR IN
          (SELECT entity_id::VARCHAR FROM s1m)
      AND g.matched_entity_ids IS NOT NULL
      AND g.matched_entity_ids <> ''
    """)

    rows = con.execute(f"""
        SELECT s1_id, matched_ids
        FROM eval_{source}
    """).fetchall()

    cand = con.execute(f"""
        SELECT s1_id, candidate_id
        FROM cand_{source}
    """).fetchall()

    lookup = {}
    for sid, cid in cand:
        lookup.setdefault(str(sid), set()).add(str(cid))

    total = 0
    found = 0
    full = 0
    entities = 0

    for sid, ids in rows:
        text = str(ids).strip().strip("[](){}")
        true_ids = {
            x.strip().strip("'").strip('"')
            for x in text.replace(";", ",").split(",")
            if x.strip()
        }

        if not true_ids:
            continue

        entities += 1
        hit = len(true_ids & lookup.get(str(sid), set()))
        total += len(true_ids)
        found += hit

        if hit == len(true_ids):
            full += 1

    recall = found / total if total else 0
    full_recall = full / entities if entities else 0

    print(
        f"{source}: candidates={n:,} "
        f"recall={recall:.4%} "
        f"full={full_recall:.4%}"
    )

print("DONE")
