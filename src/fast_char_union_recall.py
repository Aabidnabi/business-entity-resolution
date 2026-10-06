
import duckdb

con = duckdb.connect()

print("Loading matched 10k S1...")

con.execute("""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT DISTINCT source1_entity_id::VARCHAR AS source1_entity_id,
       matched_entity_ids
FROM read_csv(
    'dataset/train/train_ground_truth.tsv',
    delim='\t',
    header=true
)
WHERE matched_entity_ids IS NOT NULL
  AND matched_entity_ids <> ''
LIMIT 10000
""")

con.execute("""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT *
FROM read_parquet('data/processed/train_source1.parquet')
WHERE entity_id::VARCHAR IN (SELECT source1_entity_id FROM gt)
""")

for source in ["train_source2", "train_source3"]:

    print(f"\n[{source}] building union...")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE cand AS
    SELECT DISTINCT
        a.entity_id::VARCHAR AS s1_id,
        b.entity_id::VARCHAR AS candidate_id
    FROM s1 a
    JOIN read_parquet('data/processed/{source}.parquet') b
      ON a.country_norm = b.country_norm
     AND (
          a.name_token_string = b.name_token_string
          OR left(regexp_replace(a.name_norm, '[^a-z0-9]', '', 'g'), 4)
             = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 4)
          OR left(regexp_replace(a.name_norm, '[^a-z0-9]', '', 'g'), 5)
             = left(regexp_replace(b.name_norm, '[^a-z0-9]', '', 'g'), 5)
     )
    WHERE a.name_norm <> ''
    """)

    n = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]

    print(f"candidates={n:,}")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE truth AS
    SELECT source1_entity_id, matched_entity_ids
    FROM gt
    """)

    rows = con.execute("""
        SELECT source1_entity_id, matched_entity_ids
        FROM truth
    """).fetchall()

    cand = con.execute("""
        SELECT s1_id, candidate_id
        FROM cand
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
        hits = true_ids & lookup.get(str(sid), set())

        total += len(true_ids)
        found += len(hits)

        if len(hits) == len(true_ids):
            full += 1

    print(f"pair recall={found / total:.4%}")
    print(f"full entity recall={full / entities:.4%}")

print("\nDONE")
