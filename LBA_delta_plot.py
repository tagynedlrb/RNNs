#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TransformerPlot_IDX_CUM_Offset_v17r0_fixed.py

수정사항
- inference CSV에서 str → numeric 변환 안정화
- mixed dtype 방지
- NaN row 자동 제거
- read_csv low_memory=False
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===== 사용자 설정 =====
ORIGINAL_TRACE = "test_trace_lstm6_plot/trace_283"
INFER_CSV = "test_trace_lstm6_plot/trace_283_lstm_lba_o_delta_lr5"
OUTPUT_DIR = "plot_inference_lstm6"

# ===== 원본 트레이스 로딩 =====
def load_original_trace(path):

    df = pd.read_csv(path, header=None, low_memory=False)
    df.columns = ["device_id", "opcode", "offset", "length", "timestamp"]

    # numeric 변환 (혹시 모를 오류 방지)
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["offset", "length", "timestamp"])

    df["offset"] = df["offset"].astype(np.int64)
    df["length"] = df["length"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # timestamp 상대화
    df["timestamp"] = (df["timestamp"] - df["timestamp"].iloc[0]).astype(np.int64)

    # opcode 변환
    df["opcode"] = df["opcode"].map({"R": 0, "W": 1})

    # 단위 변환
    df["offset"] = df["offset"] // 4096
    df["length"] = df["length"] // 512

    # delta 계산
    df["dt"] = df["timestamp"].diff().fillna(0).clip(lower=0).astype(np.int64)
    df["dlba"] = df["offset"].diff().fillna(0).astype(np.int64)

    return df


# ===== inference 트레이스 로딩 =====
def load_infer_trace(path):

    df = pd.read_csv(path, low_memory=False)

    # numeric 변환
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["offset", "length", "timestamp"])

    df["offset"] = df["offset"].astype(np.int64)
    df["length"] = df["length"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # opcode 변환
    df["opcode"] = df["opcode"].map({"R": 0, "W": 1})

    # 단위 변환
    df["offset"] = df["offset"] // 4096
    df["length"] = df["length"] // 512

    # delta 계산
    df["dt"] = df["timestamp"].diff().fillna(0).clip(lower=0).astype(np.int64)
    df["dlba"] = df["offset"].diff().fillna(0).astype(np.int64)

    return df


def _size_from_length(length_series, global_min, global_max, eps=0.05, scale=10.0):

    rng = (global_max - global_min) if global_max != global_min else 1e-6
    return ((length_series - global_min) / rng + eps) * scale


def plot_idx_vs_y(ax, df, y_field, title, size_minmax, x_start=0,
                  color_read="red", color_write="blue", alpha=0.6, label_prefix=""):

    x = np.arange(len(df)) + x_start

    is_read = (df["opcode"] == 0)
    is_write = ~is_read

    size_norm = _size_from_length(df["length"], *size_minmax)

    ax.scatter(
        x[is_read],
        df.loc[is_read, y_field],
        s=size_norm[is_read],
        alpha=alpha,
        label=(label_prefix + "Read").strip(),
        color=color_read
    )

    ax.scatter(
        x[is_write],
        df.loc[is_write, y_field],
        s=size_norm[is_write],
        alpha=alpha,
        label=(label_prefix + "Write").strip(),
        color=color_write
    )

    ax.set_xlabel("Request Index")

    ylab = {
        "timestamp": "Time",
        "dt": "ΔTime",
        "dlba": "ΔLBA (4KB)",
        "offset": "Offset (4KB LBA)"
    }.get(y_field, y_field)

    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right")


def main():

    # ===== load =====
    original_df = load_original_trace(ORIGINAL_TRACE)
    infer_all = load_infer_trace(INFER_CSV)

    # ===== generated / seed 분리 =====
    gen_df = infer_all[infer_all["type"] == "generated"].copy().reset_index(drop=True)
    seed_df = infer_all[infer_all["type"] == "input"].copy().reset_index(drop=True)

    N = len(gen_df)

    if N == 0:
        raise ValueError("No generated rows found in inference CSV.")

    # ===== original slice =====
    orig_slice = original_df.iloc[:N].copy()

    # ===== size normalize 기준 =====
    combined_lengths = pd.concat(
        [orig_slice["length"], gen_df["length"]],
        ignore_index=True
    )

    size_minmax_combined = (
        combined_lengths.min(),
        combined_lengths.max()
    )

    size_minmax_genonly = (
        gen_df["length"].min(),
        gen_df["length"].max()
    )

    # ===== range 계산 =====
    dt_min_o, dt_max_o = orig_slice["dt"].min(), orig_slice["dt"].max()
    dt_min_g, dt_max_g = gen_df["dt"].min(), gen_df["dt"].max()

    dlba_min_o, dlba_max_o = orig_slice["dlba"].min(), orig_slice["dlba"].max()
    dlba_min_g, dlba_max_g = gen_df["dlba"].min(), gen_df["dlba"].max()

    off_min_o, off_max_o = orig_slice["offset"].min(), orig_slice["offset"].max()
    off_min_g, off_max_g = gen_df["offset"].min(), gen_df["offset"].max()

    # ===== plot =====
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))

    # ===== Original =====
    plot_idx_vs_y(
        axes[0, 0],
        orig_slice,
        "dt",
        f"[Original] Index vs ΔTime (first {N})",
        size_minmax_combined
    )
    axes[0, 0].set_ylim(dt_min_o, dt_max_o)

    plot_idx_vs_y(
        axes[0, 1],
        orig_slice,
        "dlba",
        f"[Original] Index vs ΔLBA (first {N})",
        size_minmax_combined
    )
    axes[0, 1].set_ylim(dlba_min_o, dlba_max_o)

    plot_idx_vs_y(
        axes[0, 2],
        orig_slice,
        "offset",
        "[Original] Index vs Offset (4KB LBA)",
        size_minmax_combined
    )
    axes[0, 2].set_ylim(off_min_o, off_max_o)

    # ===== Generated combined =====
    if len(seed_df) > 0:
        plot_idx_vs_y(
            axes[1, 0],
            seed_df,
            "dt",
            "[Generated] Index vs ΔTime (combined)",
            size_minmax_combined,
            color_read="#888888",
            color_write="#AAAAAA",
            alpha=0.35,
            label_prefix="Seed "
        )

    plot_idx_vs_y(
        axes[1, 0],
        gen_df,
        "dt",
        "[Generated] Index vs ΔTime (combined)",
        size_minmax_combined
    )
    axes[1, 0].set_ylim(dt_min_o, dt_max_o)

    if len(seed_df) > 0:
        plot_idx_vs_y(
            axes[1, 1],
            seed_df,
            "dlba",
            "[Generated] Index vs ΔLBA (combined)",
            size_minmax_combined,
            color_read="#888888",
            color_write="#AAAAAA",
            alpha=0.35,
            label_prefix="Seed "
        )

    plot_idx_vs_y(
        axes[1, 1],
        gen_df,
        "dlba",
        "[Generated] Index vs ΔLBA (combined)",
        size_minmax_combined
    )
    axes[1, 1].set_ylim(dlba_min_o, dlba_max_o)

    if len(seed_df) > 0:
        plot_idx_vs_y(
            axes[1, 2],
            seed_df,
            "offset",
            "[Generated] Index vs Offset (combined)",
            size_minmax_combined,
            color_read="#888888",
            color_write="#AAAAAA",
            alpha=0.35,
            label_prefix="Seed "
        )

    plot_idx_vs_y(
        axes[1, 2],
        gen_df,
        "offset",
        "[Generated] Index vs Offset (combined)",
        size_minmax_combined
    )
    axes[1, 2].set_ylim(off_min_o, off_max_o)

    # ===== Generated only =====
    plot_idx_vs_y(
        axes[2, 0],
        gen_df,
        "dt",
        "[Generated] Index vs ΔTime (generated-only)",
        size_minmax_genonly
    )
    axes[2, 0].set_ylim(dt_min_g, dt_max_g)

    plot_idx_vs_y(
        axes[2, 1],
        gen_df,
        "dlba",
        "[Generated] Index vs ΔLBA (generated-only)",
        size_minmax_genonly
    )
    axes[2, 1].set_ylim(dlba_min_g, dlba_max_g)

    plot_idx_vs_y(
        axes[2, 2],
        gen_df,
        "offset",
        "[Generated] Index vs Offset (generated-only)",
        size_minmax_genonly
    )
    axes[2, 2].set_ylim(off_min_g, off_max_g)

    # ===== save =====
    out_dir = os.path.join(os.path.dirname(INFER_CSV), "..", OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    infer_name = os.path.splitext(os.path.basename(INFER_CSV))[0]

    out_path = os.path.join(
        out_dir,
        f"{infer_name}.png"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
