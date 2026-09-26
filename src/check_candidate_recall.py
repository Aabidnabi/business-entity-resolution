import duckdb

con = duckdb.connect()

print("Checking candidate recall on 10k S1...")

con.execute("""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT source1_entity_id, matched_entity_ids
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
CREATE OR REPLACE TEMP TABLE truth AS
SELECT
    source1_entity_id,
    trim(x) AS matched_id
FROM gt,
UNNEST(string_split(matched_entity_ids, ',')) t(x)
WHERE trim(x) <> ''
""")

total = con.execute("""
SELECT COUNT(*) FROM truth
""").fetchone()[0]

found = con.execute("""
SELECT COUNT(*)
FROM truth t
JOIN read_parquet('output/candidate_pairs_train.parquet') c
  ON t.source1_entity_id = c.source1_entity_id
 AND t.matched_id = c.matched_entity_id
""").fetchone()[0]

print(f"True matches:           {total:,}")
print(f"Retrieved:              {found:,}")
print(f"Candidate recall:       {found/total*100:.2f}%")
print("DONE")
