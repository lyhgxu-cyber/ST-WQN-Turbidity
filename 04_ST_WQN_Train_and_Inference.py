import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
DATA_PATH = "LS_1day_clean.csv"
DEVICE = "cpu"
SEED = 42
BATCH_SIZE = 256
EPOCHS = 80
PATIENCE = 10
LR = 5e-4

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=[
    "Blue","Green","Red","NIR","SWIR1","SWIR2",
    "Turbidity","Longitude","Latitude","Date","Satellite_Flag"
])

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["DOY"] = df["Date"].dt.dayofyear

# =========================
# SPLIT (by station)
# =========================
stations = df["Station_ID_En"].unique()
np.random.shuffle(stations)
split_idx = int(len(stations)*0.8)

train_df = df[df["Station_ID_En"].isin(stations[:split_idx])].copy()
test_df  = df[df["Station_ID_En"].isin(stations[split_idx:])].copy()

# =========================
# TIME FEATURES
# =========================
for d in [train_df, test_df]:
    d["DOY_sin"] = np.sin(2*np.pi*d["DOY"]/365)
    d["DOY_cos"] = np.cos(2*np.pi*d["DOY"]/365)

# =========================
# PHYSICAL FEATURES
# =========================
def add_physical_features(df):
    eps = 1e-6
    df["NDTI"] = (df["Red"] - df["Green"]) / (df["Red"] + df["Green"] + eps)
    df["NIR_Red"] = df["NIR"] / (df["Red"] + eps)
    df["Red_Blue"] = df["Red"] / (df["Blue"] + eps)
    return df

train_df = add_physical_features(train_df)
test_df  = add_physical_features(test_df)

spec_cols = [
    "Blue","Green","Red","NIR","SWIR1","SWIR2",
    "NDTI","NIR_Red","Red_Blue"
]

# =========================
# STANDARDIZE
# =========================
scaler_spec = StandardScaler().fit(train_df[spec_cols])
train_df[spec_cols] = scaler_spec.transform(train_df[spec_cols])
test_df[spec_cols]  = scaler_spec.transform(test_df[spec_cols])

# =========================
# TARGET
# =========================
train_df["logT"] = np.log10(train_df["Turbidity"])
test_df["logT"]  = np.log10(test_df["Turbidity"])

# =========================
# GEO ENCODING
# =========================
def encode_geo(lon, lat):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    return np.stack([
        np.sin(lat), np.cos(lat),
        np.sin(lon), np.cos(lon)
    ], axis=1)

geo_cols = ["sin_lat","cos_lat","sin_lon","cos_lon"]

train_df[geo_cols] = encode_geo(train_df["Longitude"].values, train_df["Latitude"].values)
test_df[geo_cols]  = encode_geo(test_df["Longitude"].values, test_df["Latitude"].values)

# =========================
# LONG-TAIL WEIGHTS
# =========================
def compute_weights(logT, bins=30):
    hist, bin_edges = np.histogram(logT, bins=bins)
    bin_idx = np.digitize(logT, bin_edges[:-1], right=True)
    freq = hist[np.clip(bin_idx - 1, 0, len(hist)-1)]
    weights = 1.0 / (freq + 1e-6)
    weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return weights.astype(np.float32)

train_df["weight"] = compute_weights(train_df["logT"].values)
test_df["weight"] = 1.0

# =========================
# DATASET
# =========================
class WaterDataset(Dataset):
    def __init__(self, df):
        self.spec = df[spec_cols].values.astype(np.float32)
        self.time = df[["DOY_sin","DOY_cos"]].values.astype(np.float32)
        self.geo  = df[geo_cols].values.astype(np.float32)
        self.flag = df["Satellite_Flag"].values.astype(np.int64)
        self.y    = df["logT"].values.astype(np.float32)
        self.w    = df["weight"].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.spec[idx], self.time[idx], self.geo[idx], self.flag[idx], self.y[idx], self.w[idx]

train_loader = DataLoader(WaterDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(WaterDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL (ST-WQN + Spectral Attention)
# =========================
class ST_WQN(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Spectral Attention (SENet-like) ----
        self.attention = nn.Sequential(
            nn.Linear(9,16),
            nn.GELU(),
            nn.Linear(16,9),
            nn.Sigmoid()
        )

        # spectral branch
        self.spec_net = nn.Sequential(
            nn.Linear(9,64),
            nn.GELU(),
            nn.Linear(64,32)
        )

        # env branch
        self.sensor_emb = nn.Embedding(2,4)
        self.env_net = nn.Sequential(
            nn.Linear(2+4+4,32),
            nn.GELU(),
            nn.Linear(32,16)
        )

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(48,128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Linear(64,1)
        )

    def forward(self, spec, time, geo, flag):
        # attention weighting
        attn = self.attention(spec)
        spec = spec * attn

        spec_feat = self.spec_net(spec)
        sensor_feat = self.sensor_emb(flag)
        env_input = torch.cat([time, geo, sensor_feat], dim=1)
        env_feat = self.env_net(env_input)

        x = torch.cat([spec_feat, env_feat], dim=1)
        return self.fusion(x).squeeze()

# =========================
# LOSS (Asymmetric Weighted Huber)
# =========================
class AsymmetricWeightedHuber(nn.Module):
    def forward(self, p, t, w):
        e = torch.abs(p - t)
        huber = torch.where(e < 1, 0.5 * e**2, e - 0.5)

        penalty = torch.where((t > 1.5) & (p < t), 2.0, 1.0)

        return torch.mean(w * penalty * huber)

# =========================
# TRAIN
# =========================
def train_model(model, train_loader, val_loader):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = AsymmetricWeightedHuber()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    best_val = float("inf")
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        for spec,time,geo,flag,y,w in train_loader:
            spec,time,geo,flag,y,w = spec.to(DEVICE),time.to(DEVICE),geo.to(DEVICE),flag.to(DEVICE),y.to(DEVICE),w.to(DEVICE)

            opt.zero_grad()
            pred = model(spec,time,geo,flag)
            loss = loss_fn(pred,y,w)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        val_loss = 0
        preds_list, gts_list = [], []

        with torch.no_grad():
            for spec,time,geo,flag,y,w in val_loader:
                spec,time,geo,flag,y = spec.to(DEVICE),time.to(DEVICE),geo.to(DEVICE),flag.to(DEVICE),y.to(DEVICE)
                pred = model(spec,time,geo,flag)
                val_loss += nn.functional.huber_loss(pred,y,reduction='sum').item()

                preds_list.append(pred.cpu().numpy())
                gts_list.append(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "wqn.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, val_loss={val_loss:.5f}")

    return model

# =========================
# RUN
# =========================
model = ST_WQN()
model = train_model(model, train_loader, val_loader)

# =========================
# EVAL
# =========================
def evaluate_physical(y_true_log, y_pred_log):
    y_true = 10**y_true_log
    y_pred = 10 ** y_pred_log
    y_pred = np.clip(y_pred, 1e-6, 1e6)
    y_true = np.clip(y_true, 1e-6, 1e6)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/(y_true + 1e-6))) * 100
    r2 = r2_score(y_true_log, y_pred_log)

    return r2, rmse, mae, mape, y_true, y_pred

model.load_state_dict(torch.load("wqn.pth"))
model.eval()

y_pred_all, y_true_all = [], []

with torch.no_grad():
    for spec,time,geo,flag,y,w in val_loader:
        spec,time,geo,flag = spec.to(DEVICE),time.to(DEVICE),geo.to(DEVICE),flag.to(DEVICE)
        pred = model(spec,time,geo,flag)
        y_pred_all.append(pred.cpu().numpy())
        y_true_all.append(y.numpy())

y_pred_all = np.concatenate(y_pred_all)
y_true_all = np.concatenate(y_true_all)

r2, rmse, mae, mape, y_true_phys, y_pred_phys = evaluate_physical(y_true_all, y_pred_all)

print("=====================================")
print(f"Validation: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.2f}%")
print("=====================================")

mask = (
    np.isfinite(y_true_phys) &
    np.isfinite(y_pred_phys) &
    (y_true_phys > 0) &
    (y_pred_phys > 0)
)

y_true_phys = y_true_phys[mask]
y_pred_phys = y_pred_phys[mask]

# =========================
# FIGURE
# =========================
plt.figure(figsize=(6,6))
sns.kdeplot(
    x=y_true_phys,
    y=y_pred_phys,
    fill=True,
    cmap="viridis",
    thresh=0.05,
    levels=50
)

plt.xscale("log")
plt.yscale("log")

min_val = min(y_true_phys.min(), y_pred_phys.min())
max_val = max(y_true_phys.max(), y_pred_phys.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel("Observed Turbidity (NTU)")
plt.ylabel("Predicted Turbidity (NTU)")

plt.tight_layout()
plt.savefig("wqn.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# METRICS
# =========================
with open("wqn.txt", "w") as f:
    f.write(f"R2 = {r2:.3f}\n")
    f.write(f"RMSE = {rmse:.3f}\n")
    f.write(f"MAE = {mae:.3f}\n")
    f.write(f"MAPE = {mape:.2f}%\n")