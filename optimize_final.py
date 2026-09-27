import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb

MODEL = "output/lightgbm_model.txt"
TRAIN_FEATURES = "output/training_features.parquet"
TEST_SCORED = "output/test_scored.parquet"

print("=" * 70)
print("FAST F0.5 OPTIMIZATION")
print("=" * 70)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
print("[1/4] Loading LightGBM model...")
model = lgb.Booster(model_file=MODEL)

feature_cols = [
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

# ---------------------------------------------------------
# Score training data
# ---------------------------------------------------------
print("[2/4] Scoring training features...")

con = duckdb.connect()

train = con.execute("""
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

X = train[feature_cols]
train["score"] = model.predict(X)

print(f"Training rows: {len(train):,}")

# ---------------------------------------------------------
# F0.5 evaluation
# ---------------------------------------------------------
print("[3/4] Evaluating thresholds and TOP-K...")

def evaluate(df, threshold, k=None):

    d = df[df["score"] >= threshold].copy()

    if k is not None:
        d = (
            d.sort_values(
                ["source1_entity_id", "score"],
                ascending=[True, False]
            )
            .groupby("source1_entity_id", sort=False)
            .head(k)
        )

    # TP
    tp = int(d["label"].sum())

    # predicted positives
    predicted = len(d)

    # total true positives
    actual = int(df["label"].sum())

    precision = tp / predicted if predicted else 0
    recall = tp / actual if actual else 0

    beta2 = 0.25

    f05 = (
        (1 + beta2) * precision * recall /
        (beta2 * precision + recall)
        if precision + recall > 0
        else 0
    )

    return precision, recall, f05, predicted


results = []

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]

for t in thresholds:

    p, r, f, n = evaluate(train, t, None)

    results.append(
        ("THRESHOLD", t, None, p, r, f, n)
    )

    for k in [1, 2, 3]:

        p, r, f, n = evaluate(train, t, k)

        results.append(
            ("TOPK", t, k, p, r, f, n)
        )


res = pd.DataFrame(
    results,
    columns=[
        "method",
        "threshold",
        "k",
        "precision",
        "recall",
        "f05",
        "predicted"
    ]
)

res = res.sort_values("f05", ascending=False)

print()
print("=" * 70)
print("BEST CONFIGURATIONS")
print("=" * 70)

print(
    res.head(15).to_string(index=False)
)

best = res.iloc[0]

print()
print("=" * 70)
print("BEST")
print("=" * 70)

print(f"Method:     {best.method}")
print(f"Threshold:  {best.threshold}")
print(f"TOP-K:      {best.k}")
print(f"Precision:  {best.precision:.4f}")
print(f"Recall:     {best.recall:.4f}")
print(f"F0.5:       {best.f05:.4f}")
print(f"Predicted:  {int(best.predicted):,}")

res.to_csv("output/optimization_results.csv", index=False)

print()
print("Saved: output/optimization_results.csv")