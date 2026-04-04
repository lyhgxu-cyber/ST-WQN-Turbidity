import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

# 1. 修正：顶刊画图全局设置（你原代码的写法会报错，必须用 update）
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "pdf.fonttype": 42
})

OUT = "E:/nationwide_turbidity/SI"
os.makedirs(OUT, exist_ok=True)
log_records =[]

def log_step(step, rule, n0, n_rm, n_rt):
    log_records.append({
        "Step": step, "Rule": rule, "Initial N": n0, "Removed N": n_rm, "Retained N": n_rt
    })

# ==========================================
# 数据加载与预处理
# ==========================================
print("Loading data...")
df = pd.read_csv("E:/nationwide_turbidity/LS_1day.csv")
df = df.rename(columns={
    "Blue_mean": "Blue", "Green_mean": "Green", "Red_mean": "Red",
    "NIR_mean": "NIR", "SWIR1_mean": "SWIR1", "SWIR2_mean": "SWIR2"
})
bands =['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']


# ==========================================
# 四级清洗核心逻辑 (Four-Step Quality Control)
# ==========================================
n0 = len(df)

# Step 0: 填充 NaN 并截断异常值，保证列完整

for col in bands:
    # 填充中位数，防止 NaN 影响后续 CV、KMeans
    df[col] = df[col].fillna(df[col].median())
    # 截断到合理范围 [0,1]，保证物理意义
    df[col] = df[col].clip(0,1)

# Turbidity 处理，保证 logT 安全
df["Turbidity"] = df["Turbidity"].clip(lower=1e-3, upper=5000)

# Step 1: 物理+光谱逻辑约束
mask_valid = (df["Turbidity"] > 0) & df[bands].notna().all(axis=1) & \
             (df[bands] >= 0).all(axis=1) & (df[bands] <= 1).all(axis=1) & \
             (df["Red"] < 0.5) # 强行截断异常高亮/强耀斑像元
mask_spectral = (df["Turbidity"] >= 50) | ((df["Turbidity"] < 50) & (df["NIR"] < df["Red"]))
df1 = df[mask_valid & mask_spectral].copy()
log_step("Step1", "Physical (Red<0.5) + Spectral", n0, n0 - len(df1), len(df1))

# Step 2: 空间同质性约束 (CV <= 0.2)
cv_cols = []
for b in ["Blue", "Green", "Red", "NIR"]:
    df1[f"{b}_CV"] = (df1[f"{b}_stdDev"] / (df1[b].replace(0,1e-6))).clip(0, 1)
    cv_cols.append(f"{b}_CV")
mask_cv = (df1[cv_cols] <= 0.2).all(axis=1)
df2 = df1[mask_cv].copy()
log_step("Step2", "CV <= 0.2", len(df1), len(df1) - len(df2), len(df2))

# Step 3: 多维异常值剔除 (Isolation Forest)
df2["logT"] = np.log10(df2["Turbidity"].replace(0,1e-3))
X_iso = StandardScaler().fit_transform(df2[["Red", "NIR", "logT"]])
df2["iso_label"] = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1).fit_predict(X_iso)
df3 = df2[df2["iso_label"] == 1].copy()
log_step("Step3", "Isolation Forest (5%)", len(df2), len(df2) - len(df3), len(df3))

# Step 4: 光谱聚类 + Huber 稳健残差剔除
# 修正：将你写成一行的违规 for 循环拆解，保证逻辑安全
X_owt = StandardScaler().fit_transform(df3[bands])
df3["OWT"] = KMeans(n_clusters=4, random_state=42).fit_predict(X_owt)
df3["residual_z"] = np.nan

for c in range(4):
    sub = df3[df3["OWT"] == c].dropna(subset=["Red","NIR","logT"])
    if len(sub) < 50:
        continue
    Xr = sub[["Red", "NIR"]]
    yr = sub["logT"]
    huber = HuberRegressor().fit(Xr, yr)
    preds = huber.predict(Xr)
    df3.loc[sub.index, "residual_z"] = (yr - preds) / huber.scale_

df4 = df3[np.abs(df3["residual_z"]) <= 3].copy()
log_step("Step4", "Huber |z| <= 3", len(df3), len(df3) - len(df4), len(df4))
df4.to_csv("E:/nationwide_turbidity/LS_1day_clean.csv", index=False)
print(f"Final cleaned CSV saved. Retained {len(df4)} samples.")

# 导出质控表与最终清洗数据
pd.DataFrame(log_records).to_csv(f"{OUT}/Table_S1.csv", index=False)
df4.to_csv("E:/nationwide_turbidity/LS_1day_clean.csv", index=False)
print(f"Data cleaning finished. Retained {len(df4)} / {n0} samples.")

# ==========================================
# 出版级附图绘制 (Publication-Quality Plots)
# ==========================================
print("Generating figures...")

# ==========================================
# Fig S1: 时空分布图（修正时间差 + 空间分布用 df4）
# ==========================================
print("Generating Fig S1...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 时间差计算（Satellite_Date UTC -> BJ）
sat_utc = pd.to_datetime(df["Satellite_Date"], errors='coerce')
date_bj = pd.to_datetime(df["Date"], errors='coerce')
time_diff_hours = (sat_utc + pd.Timedelta(hours=8) - date_bj).dt.total_seconds() / 3600

# (a) Temporal Distribution
sns.histplot(time_diff_hours.dropna(), bins=24, ax=axes[0], color="steelblue", edgecolor="black")
axes[0].set_xlabel("Time Difference (Hours)")
axes[0].set_ylabel("Match-up Counts")
axes[0].set_title("(a) Temporal Distribution", loc='left', fontweight='bold')

# (b) Spatial Distribution - 使用最终保留数据 df4
if "Longitude" in df4.columns and "Latitude" in df4.columns:
    sc = axes[1].scatter(df4["Longitude"], df4["Latitude"],
                         c=np.log10(df4["Turbidity"]), cmap="viridis",
                         s=2, alpha=0.6, vmin=0, vmax=3)
    plt.colorbar(sc, ax=axes[1], label=r"$\log_{10}(\text{Turbidity})$")

axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("(b) Spatial Distribution", loc='left', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/Fig_S1.png")
plt.close()

# Fig S2: CV 清洗前后的 ECDF (累积分布函数) - 解决你之前曲线断崖的物理常识错误
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for i, (ax, b) in enumerate(zip(axes.flatten(), ["Blue", "Green", "Red", "NIR"])):
    # 顶刊绝不容忍概率图出错：用 ecdfplot 画累积分布，保证最终点均到达 1.0
    sns.ecdfplot(data=df1, x=f"{b}_CV", ax=ax, label="Before (Step 1)", color="gray", linestyle="--")
    sns.ecdfplot(data=df2, x=f"{b}_CV", ax=ax, label="After (Step 2)", color="darkred", linewidth=2)
    ax.set_xlim(0, 0.5)
    ax.axvline(0.2, color='black', linestyle=':', label='Threshold=0.2')
    ax.set_xlabel(f"{b} CV")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"({chr(97+i)})", loc='left', fontweight='bold')
    if i == 0:
        ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUT}/Fig_S2.png")
plt.close()

# Fig S3: 数据降噪过程的海量数据 Hexbin 图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Step 3: Isolation Forest Outliers
ax1 = axes[0]
inliers = df2[df2["iso_label"] == 1]
outliers = df2[df2["iso_label"] == -1]
# 用 Hexbin 绘制主体正常数据（bins='log' 表示颜色按对数映射，防止重叠显示不全）
hb1 = ax1.hexbin(inliers["Red"], inliers["logT"], gridsize=60, cmap="Blues", bins='log', mincnt=1)
# 散点标出被孤立森林剔除的异常点
ax1.scatter(outliers["Red"], outliers["logT"], facecolors='none', edgecolors='red', s=8, alpha=0.6, label="IF Outliers")
ax1.set_xlim(0, 0.5)
ax1.set_xlabel("Red Reflectance")
ax1.set_ylabel(r"$\log_{10}(\text{Turbidity})$ [NTU/FNU]")
ax1.legend(loc="upper left")
ax1.set_title("(a) Step 3: Anomaly Detection", loc='left', fontweight='bold')

# (b) Step 4: Final Cleaned Data
ax2 = axes[1]
hb2 = ax2.hexbin(df4["Red"], df4["logT"], gridsize=60, cmap="viridis", bins='log', mincnt=1)
cb = plt.colorbar(hb2, ax=ax2)
cb.set_label(r"$\log_{10}(\text{Density Counts})$")
ax2.set_xlim(0, 0.5)
ax2.set_xlabel("Red Reflectance")
ax2.set_ylabel(r"$\log_{10}(\text{Turbidity})$[NTU/FNU]")
ax2.set_title("(b) Step 4: Final Cleaned Data", loc='left', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/Fig_S3.png")
plt.close()

print("All tasks completed successfully! Check the SI folder.")