import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost as xgb

# -----------------------------
# 1) Load + features
# -----------------------------
df = pd.read_excel("multibank_nr_link_adapt_dataset.xlsx")

feat_cols = [
    "cqi_bankA_FR1_sub6",
    "ri_bankA_FR1_sub6",
    "pmi_bankA_FR1_sub6",
]

optional = [
    "estRmsDs_bankA_FR1_sub6_ns",
    "freqSel_bankA_FR1_sub6_dB",
    "cohTime_bankA_FR1_sub6_ms",
]
for c in optional:
    if c in df.columns:
        feat_cols.append(c)

X = df[feat_cols].to_numpy()
X = np.nan_to_num(X, nan=0.0)
y = df["bestMCS_bankA_FR1_sub6"].to_numpy().astype(int)

print("Features used:", feat_cols)

# -----------------------------
# 2) Remap labels to 0..K-1
# -----------------------------
classes = np.sort(np.unique(y))
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for i, c in enumerate(classes)}
y_mapped = np.array([class_to_idx[v] for v in y], dtype=int)
num_classes = len(classes)

print("Original MCS classes present:", classes)
print("num_classes:", num_classes)

# -----------------------------
# 3) Train/val/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_mapped, test_size=0.2, random_state=0, stratify=y_mapped
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test, label=y_test)

# -----------------------------
# 4) Try a few configs with early stopping
# -----------------------------
configs = [
    ("XGB_A_shallow", dict(max_depth=4, eta=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, gamma=0)),
    ("XGB_B_regularized", dict(max_depth=6, eta=0.05, subsample=0.7, colsample_bytree=0.7, min_child_weight=5, gamma=0)),
    ("XGB_C_conservative", dict(max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=1)),
]

base_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "seed": 0,
}

for name, extra in configs:
    print(f"\n--- {name} ---")

    params = dict(base_params)
    params.update(extra)

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predict probs -> class indices
    proba = bst.predict(dtest)            # shape (N, num_classes)
    pred_mapped = np.argmax(proba, axis=1)

    # Map back to original MCS values for your metrics
    pred = np.array([idx_to_class[int(i)] for i in pred_mapped], dtype=int)
    y_true = np.array([idx_to_class[int(i)] for i in y_test], dtype=int)

    acc = accuracy_score(y_true, pred)
    within1 = np.mean(np.abs(pred - y_true) <= 1)
    over = np.mean(pred > y_true)
    under = np.mean(pred < y_true)

    print("best_iteration:", bst.best_iteration)
    print("accuracy:", acc)
    print("within ±1:", within1)
    print("over-predict rate:", over)
    print("under-predict rate:", under)
    print("confusion matrix shape:", confusion_matrix(y_true, pred).shape)