import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# 1) Load + prepare data (same as your code)
# -----------------------------
df = pd.read_excel("Data/multibank_nr_link_adapt_dataset.xlsx")

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

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# (Optional but recommended) make a validation split for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
)

# -----------------------------
# 2) Scale features (DNN needs this)
# -----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# -----------------------------
# 3) Torch datasets/dataloaders
# -----------------------------
def to_tensor(x, y):
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return x_t, y_t

Xtr_t, ytr_t = to_tensor(X_train_s, y_train)
Xva_t, yva_t = to_tensor(X_val_s, y_val)
Xte_t, yte_t = to_tensor(X_test_s, y_test)

train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=512, shuffle=False)
test_loader  = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=512, shuffle=False)

# -----------------------------
# 4) Define a basic DNN (MLP)
# -----------------------------
torch.manual_seed(0)
np.random.seed(0)

input_dim = X_train_s.shape[1]
# Labels span 0..27 (even if only 23 values appear), so safest is 28 outputs:
num_classes = int(y.max()) + 1  # -> 28

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim, num_classes).to(device)

# Class weights (optional): helps imbalance a bit
counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
weights = counts.sum() / (counts + 1e-9)
weights = weights / weights.mean()
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 5) Train loop + simple early stopping
# -----------------------------
def eval_loss(loader):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    return total_loss / max(n, 1)

best_val = float("inf")
patience = 5
bad_epochs = 0

for epoch in range(1, 51):  # 50 epochs is plenty for a baseline
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    val_loss = eval_loss(val_loader)
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:02d} | val_loss {val_loss:.4f}")

    # early stopping
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        bad_epochs = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("early stopping")
            break

# restore best
model.load_state_dict(best_state)
model.to(device)

# -----------------------------
# 6) Evaluate on test set with your same metrics
# -----------------------------
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)

pred = np.concatenate(all_preds)

acc = (pred == y_test).mean()
within1 = np.mean(np.abs(pred - y_test) <= 1)
over = np.mean(pred > y_test)
under = np.mean(pred < y_test)

print("\nCQI+RI+PMI MLP (basic DNN)")
print("accuracy:", acc)
print("within ±1:", within1)
print("over-predict rate:", over)
print("under-predict rate:", under)
print("num_classes used:", num_classes, "(outputs)")
