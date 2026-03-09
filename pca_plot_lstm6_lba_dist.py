import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import argparse
from adjustText import adjust_text

# =========================
# Argument Parser
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--trace_num", type=int, default=None)

args = parser.parse_args()
trace_num = args.trace_num

# =========================
# CSV 읽기
# =========================
df = pd.read_csv("test_trace_iotap_stats_lstm6.csv")

# =========================
# Feature 선택 (132개)
# =========================
all_features = df.columns.tolist()
all_features.remove("trace")

features = [
    c for c in all_features
    if (
        c.startswith("rmslba_") or
        c.startswith("hotratio_") or
        c.startswith("hotread_") or
        c.startswith("hotwrite_")
    )
]

print("Selected feature count:", len(features))

X = df[features].values

# =========================
# Standardization
# =========================
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# =========================
# PCA
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)

df_pca = pd.DataFrame({
    "trace": df["trace"],
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1]
})

# =========================
# 기준 trace
# =========================
TARGET_TRACE = "trace_283"
PREFIX = "trace_283_lstm_lba_o_delta_"

target_row = df_pca[df_pca["trace"] == TARGET_TRACE]

if target_row.empty:
    raise ValueError(f"{TARGET_TRACE} not found")

target_x = target_row["PCA1"].values[0]
target_y = target_row["PCA2"].values[0]

# =========================
# distance 계산
# =========================
distances = []

for _, row in df_pca.iterrows():

    name = row["trace"]
    x = row["PCA1"]
    y = row["PCA2"]

    dist = np.sqrt((x - target_x)**2 + (y - target_y)**2)

    distances.append((name, dist))

# =========================
# custom sort
# =========================
def sort_key(item):

    name = item[0]

    if name == TARGET_TRACE:
        return (-1, -1)

    if name.startswith(PREFIX):
        short = name[len(PREFIX):]
    else:
        short = name

    try:
        size, lr = short.split("_lr")
        return (int(size), int(lr))
    except:
        return (9999, 9999)

distances.sort(key=sort_key)

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 8))

texts = []

for _, row in df_pca.iterrows():

    name = row["trace"]
    x = row["PCA1"]
    y = row["PCA2"]

    if name.startswith(PREFIX):
        label_text = name[len(PREFIX):]
    else:
        label_text = name

    texts.append(
        ax.text(x, y, label_text, fontsize=9)
    )

# scatter
ax.scatter(df_pca["PCA1"], df_pca["PCA2"], s=50, alpha=0.7)

# =========================
# Label overlap 해결
# =========================
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
    expand_points=(1.2, 1.4),
    expand_text=(1.2, 1.4),
    force_text=0.8,
    force_points=0.3
)

# =========================
# Distance box 생성
# =========================
distance_text = "Distance from trace_283\n\n"
distance_text += f"{'Model':20s} {'Distance':>20s}\n"
distance_text += "-" * 42 + "\n"

for name, dist in distances:

    if name == TARGET_TRACE:
        continue

    if name.startswith(PREFIX):
        short = name[len(PREFIX):]
    else:
        short = name

    distance_text += f"{short:20s} {dist:20.6f}\n"

ax.text(
    1.05,
    0.5,
    distance_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='center',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# =========================
# Title
# =========================
title = "PCA Result (132 IOTap Features)"

if trace_num is not None:
    title += f" | trace_{trace_num}"

ax.set_title(title)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.grid(True)

# =========================
# Save
# =========================
today = datetime.now().strftime("%Y%m%d")

plt.savefig(
    f"pca132_lstm6_{today}.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("PCA plot 생성 완료")