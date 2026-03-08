#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER CONFIG
# =========================
TRACE_PATH = "test_trace/trace_283"
OUTPUT_DIR = "./trace_283_32k_plots"

GENERATE_LENGTH = 1024 * 32  # 32768


# =========================
# LOAD TRACE
# =========================
def load_trace(path):

    df = pd.read_csv(path, header=None, low_memory=False)

    df.columns = [
        "device_id",
        "opcode",
        "offset",
        "length",
        "timestamp"
    ]

    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["offset", "length", "timestamp"])

    df["offset"] = df["offset"].astype(np.int64)
    df["length"] = df["length"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # opcode
    df["opcode"] = df["opcode"].map({"R": 0, "W": 1})

    # unit convert
    df["offset"] = df["offset"] // 4096
    df["length"] = df["length"] // 512

    # delta
    df["dlba"] = df["offset"].diff().fillna(0).astype(np.int64)

    return df


# =========================
# SIZE NORMALIZATION
# =========================
def size_from_length(length, min_l, max_l):

    eps = 0.05
    scale = 10

    rng = (max_l - min_l) if max_l != min_l else 1e-6

    return ((length - min_l) / rng + eps) * scale


# =========================
# PLOT
# =========================
def plot_window(df, start, end, out_delta, out_original,
                size_minmax):

    sub = df.iloc[start:end].copy()

    x = np.arange(len(sub))

    is_read = sub["opcode"] == 0
    is_write = ~is_read

    size_norm = size_from_length(
        sub["length"],
        size_minmax[0],
        size_minmax[1]
    )

    # =====================
    # DELTA PLOT
    # =====================
    plt.figure(figsize=(10,5))

    plt.scatter(
        x[is_read],
        sub.loc[is_read, "dlba"],
        s=size_norm[is_read],
        alpha=0.6,
        color="red",
        label="Read"
    )

    plt.scatter(
        x[is_write],
        sub.loc[is_write, "dlba"],
        s=size_norm[is_write],
        alpha=0.6,
        color="blue",
        label="Write"
    )

    plt.xlabel("Request Index")
    plt.ylabel("ΔLBA (4KB)")
    plt.title(f"ΔLBA [{start}:{end}]")
    plt.grid(True)
    plt.legend()

    fname = f"delta_{start:06d}_{end:06d}.png"

    plt.tight_layout()
    plt.savefig(os.path.join(out_delta, fname), dpi=120)
    plt.close()

    # =====================
    # ORIGINAL LBA PLOT
    # =====================
    plt.figure(figsize=(10,5))

    plt.scatter(
        x[is_read],
        sub.loc[is_read, "offset"],
        s=size_norm[is_read],
        alpha=0.6,
        color="red",
        label="Read"
    )

    plt.scatter(
        x[is_write],
        sub.loc[is_write, "offset"],
        s=size_norm[is_write],
        alpha=0.6,
        color="blue",
        label="Write"
    )

    plt.xlabel("Request Index")
    plt.ylabel("Offset (4KB LBA)")
    plt.title(f"Original LBA [{start}:{end}]")
    plt.grid(True)
    plt.legend()

    fname = f"original_{start:06d}_{end:06d}.png"

    plt.tight_layout()
    plt.savefig(os.path.join(out_original, fname), dpi=120)
    plt.close()


# =========================
# MAIN
# =========================
def main():

    df = load_trace(TRACE_PATH)

    total = len(df)

    print("Total rows:", total)

    # global size normalization
    size_minmax = (
        df["length"].min(),
        df["length"].max()
    )

    # output dirs
    delta_dir = os.path.join(OUTPUT_DIR, "delta")
    original_dir = os.path.join(OUTPUT_DIR, "original")

    os.makedirs(delta_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    # sliding window
    for start in range(0, total, GENERATE_LENGTH):

        end = min(start + GENERATE_LENGTH, total)

        print(f"Plotting {start} ~ {end}")

        plot_window(
            df,
            start,
            end,
            delta_dir,
            original_dir,
            size_minmax
        )

    print("Done.")


if __name__ == "__main__":
    main()
