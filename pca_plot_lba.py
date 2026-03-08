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
parser.add_argument("--model", type=str, choices=["rnn", "lstm", "gru"], default=None)
parser.add_argument("--sub", type=str,
                    choices=["lba", "lba_delta", "lba_o", "lba_o_delta"],
                    default=None)

args = parser.parse_args()

trace_num = args.trace_num
model_filter = args.model
sub_filter = args.sub

if model_filter is not None:
    sub_filter = None

# =========================
# CSV 읽기
# =========================
df = pd.read_csv("test_trace_iotap_stats.csv")

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

# =========================
# Feature Matrix
# =========================
X = df[features].values

# =========================
# Standardization (매우 중요)
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

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
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 8))

default_x, default_y = [], []
highlight_x, highlight_y = [], []
texts = []

for _, row in df_pca.iterrows():

    name = row["trace"]
    x = row["PCA1"]
    y = row["PCA2"]

    show_label = False
    label_text = None

    if trace_num is not None:

        base = f"trace_{trace_num}"
        prefix = f"trace_{trace_num}_"

        if name == base:
            show_label = True
            label_text = name

        elif name.startswith(prefix):

            suffix = name[len(prefix):]
            parts = suffix.split("_")

            model = parts[0]
            sub = "_".join(parts[1:]) if len(parts) > 1 else None

            if model_filter:
                if model == model_filter:
                    show_label = True
                    label_text = sub

            elif sub_filter:
                if sub == sub_filter:
                    show_label = True
                    label_text = model

    if show_label:
        highlight_x.append(x)
        highlight_y.append(y)
        texts.append(ax.text(x, y, label_text, fontsize=9))
    else:
        default_x.append(x)
        default_y.append(y)

# 기본 점
ax.scatter(default_x, default_y, s=30, alpha=0.3)

# 강조 점
ax.scatter(highlight_x, highlight_y, s=80, alpha=0.9)

# =========================
# 텍스트 겹침 해결
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
title_parts = ["PCA Result (132 IOTap Features)"]

if trace_num is not None:
    title_parts.append(f"trace_{trace_num}")

if model_filter:
    title_parts.append(f"model={model_filter}")

if sub_filter:
    title_parts.append(f"sub={sub_filter}")

ax.set_title(" | ".join(title_parts))
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.grid(True)

# =========================
# 저장
# =========================
today = datetime.now().strftime("%Y%m%d")
plt.savefig(f"pca_filtered_{today}.png", dpi=300, bbox_inches='tight')
plt.close()

print("PCA plot saved")