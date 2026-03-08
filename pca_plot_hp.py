import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
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
df = pd.read_csv("test_trace_iotap_stats_hp.csv")

features = df.columns.tolist()
features.remove("trace")
X = df[features].values

# =========================
# PCA
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame({
    "trace": df["trace"],
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1]
})

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 8))

default_x, default_y = [], []
highlight_x, highlight_y = [], []
texts = []

# label 변환 기준
TARGET_TRACE = "trace_283"
PREFIX = "trace_283_lstm_lba_o_delta_"

for _, row in df_pca.iterrows():

    name = row["trace"]
    x = row["PCA1"]
    y = row["PCA2"]

    # =========================
    # Label 결정
    # =========================
    if name == TARGET_TRACE:
        label_text = name

    elif name.startswith(PREFIX):
        label_text = name[len(PREFIX):]

    else:
        label_text = name

    highlight_x.append(x)
    highlight_y.append(y)

    texts.append(
        ax.text(x, y, label_text, fontsize=9)
    )

# 점 표시
ax.scatter(highlight_x, highlight_y, s=50, alpha=0.7)

# =========================
# Label 겹침 해결
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
# Title
# =========================
title = "PCA Result"

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
plt.savefig(f"pca_hp_{today}.png", dpi=300, bbox_inches='tight')
plt.close()

print("PCA plot 생성 완료")
