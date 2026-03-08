#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np

INPUT_PATH = "./alibaba_trace/ratio.txt"
OUTPUT_PATH = "./percentile.txt"
MIN_TOTAL_REQ = 1_000_000

def parse_ratio_file(path):
    traces = []

    with open(path, "r") as f:
        lines = f.readlines()

    # 헤더 제거
    for line in lines:
        line = line.strip()
        if not line or line.startswith("File name"):
            continue

        # trace_0: 형태 파싱
        # 여러 공백 대응
        parts = re.split(r"\s+", line)

        try:
            trace_name = parts[0].replace(":", "")
            total_req = int(parts[1])
            read_req = int(parts[2])
            write_req = int(parts[3])
            read_ratio = float(parts[4].replace("%", ""))
            write_ratio = float(parts[5].replace("%", ""))
        except:
            continue

        if total_req >= MIN_TOTAL_REQ:
            traces.append({
                "trace": trace_name,
                "total": total_req,
                "read_req": read_req,
                "write_req": write_req,
                "read_ratio": read_ratio,
                "write_ratio": write_ratio,
            })

    return traces


def find_percentile_entries(traces, key_ratio, key_req):
    ratios = np.array([t[key_ratio] for t in traces])

    percentiles = [100, 75, 50, 25, 0]
    results = {}

    for p in percentiles:
        target_value = np.percentile(ratios, p)

        # percentile 값과 가장 가까운 trace 선택
        closest_trace = min(
            traces,
            key=lambda t: abs(t[key_ratio] - target_value)
        )

        results[p] = {
            "trace": closest_trace["trace"],
            "ratio": closest_trace[key_ratio],
            "req": closest_trace[key_req],
        }

    return results


def main():
    traces = parse_ratio_file(INPUT_PATH)

    read_percentiles = find_percentile_entries(
        traces, "read_ratio", "read_req"
    )

    write_percentiles = find_percentile_entries(
        traces, "write_ratio", "write_req"
    )

    with open(OUTPUT_PATH, "w") as f:
        f.write("=== READ RATIO PERCENTILES ===\n")
        for p in [100, 75, 50, 25, 0]:
            entry = read_percentiles[p]
            f.write(
                f"Percentile {p}: "
                f"{entry['trace']} | "
                f"Read Ratio: {entry['ratio']}% | "
                f"Read Request: {entry['req']}\n"
            )

        f.write("\n=== WRITE RATIO PERCENTILES ===\n")
        for p in [100, 75, 50, 25, 0]:
            entry = write_percentiles[p]
            f.write(
                f"Percentile {p}: "
                f"{entry['trace']} | "
                f"Write Ratio: {entry['ratio']}% | "
                f"Write Request: {entry['req']}\n"
            )

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
