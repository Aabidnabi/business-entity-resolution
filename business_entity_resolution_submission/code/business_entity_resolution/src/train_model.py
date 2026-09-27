import os
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
    "s1_name_missing", "s2_name_missing",
    "s1_address_missing", "s2_address_missing",

]

con = duckdb.connect()

print("Loading training data...")

df = con.execute(f"""
    SELECT
        source1_entity_id,
        label,
        {", ".join(FEATURES)}
    FROM read_parquet('{FEATURE_FILE}')
""").df()

print(f"Rows loaded: {len(df):,}")

# Entity-level split: same S1 entity never appears in both train and validation.
df["entity_hash"] = pd.util.hash_pandas_object(
    df["source1_entity_id"].astype(str),
    index=False
)

val_mask = (df["entity_hash"] % 5) == 0

train_df = df.loc[~val_mask].copy()
val_df = df.loc[val_mask].copy()

print(f"Train rows: {len(train_df):,}")
print(f"Validation rows: {len(val_df):,}")
print(f"Train entities: {train_df.source1_entity_id.nunique():,}")
print(f"Validation entities: {val_df.source1_entity_id.nunique():,}")

X_train = train_df[FEATURES]
y_train = train_df["label"]

X_val = val_df[FEATURES]
y_val = val_df["label"]

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

print("Training LightGBM...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(25),
    ],
)

os.makedirs("output", exist_ok=True)
model.booster_.save_model(MODEL_FILE)

print(f"MODEL SAVED: {MODEL_FILE}")
print(f"Best iteration: {model.best_iteration_}")
