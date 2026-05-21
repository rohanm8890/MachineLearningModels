import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

model = ExtraTreesClassifier(
    n_estimators=1200,
    random_state=0,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
within1 = np.mean(np.abs(pred - y_test) <= 1)
over = np.mean(pred > y_test)
under = np.mean(pred < y_test)

print("CQI+RI+PMI ExtraTrees")
print("accuracy:", acc)
print("within ±1:", within1)
print("over-predict rate:", over)
print("under-predict rate:", under)
print("confusion matrix shape:", confusion_matrix(y_test, pred).shape)