import pandas as pd
import numpy as np
import lightgbm as lgb

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
    "address_jaccard",
]

df = pd.read_parquet("output/training_features_jaccard.parquet")
valid = pd.util.hash_pandas_object(df["source1_entity_id"].astype(str)) % 5 == 0
v = df.loc[valid].copy()

model = lgb.Booster(model_file="output/experiment_jaccard_model.txt")
v["score"] = model.predict(v[FEATURES])

n = v.groupby("source1_entity_id")["label"].transform("size").to_numpy()
score = v["score"].to_numpy()
y = v["label"].to_numpy()

best = None

for a in np.arange(0.70, 0.851, 0.025):
    for b in np.arange(a, 0.951, 0.025):
        for c in np.arange(b, 0.976, 0.025):
            threshold = np.where(n <= 5, a, np.where(n <= 100, b, c))
            pred = score >= threshold

            tp = np.sum(pred & (y == 1))
            fp = np.sum(pred & (y == 0))
            fn = np.sum((~pred) & (y == 1))

            p = tp / (tp + fp) if tp + fp else 0
            r = tp / (tp + fn) if tp + fn else 0
            f = 1.25 * p * r / (0.25 * p + r) if p + r else 0

            if best is None or f > best[0]:
                best = (f, p, r, tp, fp, fn, a, b, c)

print("\nADAPTIVE RESULT")
print(f"F0.5      : {best[0]:.9f}")
print(f"Precision : {best[1]:.9f}")
print(f"Recall    : {best[2]:.9f}")
print(f"TP/FP/FN  : {best[3]:,} / {best[4]:,} / {best[5]:,}")
print(f"Thresholds: <=5={best[6]:.3f}, 6-100={best[7]:.3f}, >100={best[8]:.3f}")
