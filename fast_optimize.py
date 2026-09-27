import duckdb
import lightgbm as lgb
import pandas as pd

MODEL = "output/lightgbm_model.txt"
TRAIN = "output/training_features.parquet"

FEATURES = [
    "name_exact",
    "name_token_exact",
    "name_edit_sim",
    "name_length_ratio",
    "address_exact",
    "address_edit_sim",
    "country_exact",
    "s1_name_missing",
    "s2_name_missing",
    "s1_address_missing",
    "s2_address_missing",
]

print("=" * 70)
print("FAST TRAINING SCORE OPTIMIZATION")
print("=" * 70)

print("[1] Loading model...")
model = lgb.Booster(model_file=MODEL)

print("[2] Loading training features...")

con = duckdb.connect()
con.execute("PRAGMA threads=8")

df = con.execute("""
SELECT
    source1_entity_id,
    matched_entity_id,
    label,
    name_exact,
    name_token_exact,
    name_edit_sim,
    name_length_ratio,
    address_exact,
    address_edit_sim,
    country_exact,
    s1_name_missing,
    s2_name_missing,
    s1_address_missing,
    s2_address_missing
FROM read_parquet('output/training_features.parquet')
""").fetchdf()

print(f"Rows: {len(df):,}")

print("[3] Predicting scores ONCE...")

df["score"] = model.predict(df[FEATURES])

actual = int(df["label"].sum())

print(f"Actual positive pairs: {actual:,}")
print()
print("=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

best = None

for t in [
    0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85,
    0.88, 0.90, 0.92, 0.94,
    0.95, 0.96, 0.97, 0.98, 0.99
]:

    selected = df["score"] >= t

    predicted = int(selected.sum())
    tp = int(df.loc[selected, "label"].sum())

    precision = tp / predicted if predicted else 0
    recall = tp / actual if actual else 0

    f05 = (
        1.25 * precision * recall /
        (0.25 * precision + recall)
        if precision + recall > 0 else 0
    )

    print(
        f"{t:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F0.5={f05:.4f} | "
        f"Pairs={predicted:,}"
    )

    if best is None or f05 > best[0]:
        best = (f05, t, precision, recall, predicted)

print()
print("=" * 70)
print("BEST TRAINING THRESHOLD")
print("=" * 70)

print(f"F0.5:      {best[0]:.6f}")
print(f"Threshold: {best[1]:.2f}")
print(f"Precision: {best[2]:.6f}")
print(f"Recall:    {best[3]:.6f}")
print(f"Pairs:     {best[4]:,}")
print("=" * 70)