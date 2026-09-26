import duckdb
import lightgbm as lgb
import pandas as pd
import numpy as np

FEATURE_FILE = "output/training_features.parquet"
MODEL_FILE = "output/lightgbm_model.txt"

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

con = duckdb.connect()

print("Loading validation data...")

df = con.execute(f"""
    SELECT
        source1_entity_id,
        label,
        {", ".join(FEATURES)}
    FROM read_parquet('{FEATURE_FILE}')
""").df()

entity_hash = pd.util.hash_pandas_object(
    df["source1_entity_id"].astype(str),
    index=False
)

df = df.loc[(entity_hash % 5) == 0].copy()

print(f"Validation rows: {len(df):,}")
print(f"Validation entities: {df.source1_entity_id.nunique():,}")

model = lgb.Booster(model_file=MODEL_FILE)

print("Predicting...")

df["prob"] = model.predict(
    df[FEATURES],
    num_iteration=model.best_iteration
)

# Sort once. Threshold metrics can then be computed with cumulative counts.
df = df.sort_values("prob", ascending=False).reset_index(drop=True)

y = df["label"].to_numpy(dtype=np.int8)
p = df["prob"].to_numpy()

# Candidate thresholds.
thresholds = np.arange(0.10, 0.951, 0.01)

best_threshold = 0.5
best_score = -1.0

# Entity-level ground-truth positive counts.
gt = df.groupby("source1_entity_id", sort=False)["label"].sum()

# Number of validation entities.
n_entities = len(gt)

print("Searching thresholds...")

for threshold in thresholds:
    selected = p >= threshold

    # Predicted positives per entity.
    pred_counts = (
        df.loc[selected]
        .groupby("source1_entity_id", sort=False)
        .size()
    )

    # True positives per entity.
    tp_counts = (
        df.loc[selected & (y == 1)]
        .groupby("source1_entity_id", sort=False)
        .size()
    )

    # Align all entities; entities with no prediction get zero.
    pred = pred_counts.reindex(gt.index, fill_value=0).to_numpy()
    tp = tp_counts.reindex(gt.index, fill_value=0).to_numpy()
    actual = gt.to_numpy()

    fp = pred - tp
    fn = actual - tp

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) != 0
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fn) != 0
    )

    f05 = np.divide(
        1.25 * precision * recall,
        0.25 * precision + recall,
        out=np.zeros_like(precision),
        where=(0.25 * precision + recall) != 0
    )

    score = float(f05.mean())

    print(f"threshold={threshold:.2f}  macro_F0.5={score:.6f}")

    if score > best_score:
        best_score = score
        best_threshold = float(threshold)

print()
print("====================================")
print(f"BEST THRESHOLD: {best_threshold:.2f}")
print(f"BEST MACRO F0.5: {best_score:.6f}")
print("====================================")

with open("output/best_threshold.txt", "w") as f:
    f.write(f"{best_threshold:.2f}\n")
