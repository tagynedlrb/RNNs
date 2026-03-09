import os
import pandas as pd
import numpy as np
import math
from collections import Counter
from tqdm import tqdm
import re

TRACE_DIR = "./test_trace_lstm6"
OUTPUT_CSV = "./test_trace_iotap_stats_lstm6.csv"

# ============ 헤더 생성(253개) ============
def build_feature_headers():
    names = []
    def add_block(prefix):
        # global + minute[ max,min,q25,q50,q75 ] + second[ max,min,q25,q50,q75 ]
        names.append(f"{prefix}_global")
        for gran in ["minute", "second"]:
            for stat in ["max", "min", "q25", "q50", "q75"]:
                names.append(f"{prefix}_{gran}_{stat}")

    # rwr
    add_block("rwr")

    # transition probability (RR, WR, RW, WW)
    for pat in ["RR", "WR", "RW", "WW"]:
        add_block(f"tp_{pat}")

    # bw (all/W/R)
    for k in ["bw_all", "bw_W", "bw_R"]:
        add_block(k)

    # io size (all/W/R)
    for k in ["iosize_all", "iosize_W", "iosize_R"]:
        add_block(k)

    # rms lba (all/W/R)
    for k in ["rmslba_all", "rmslba_W", "rmslba_R"]:
        add_block(k)

    # hot ratio (p10/p25/p50)
    for p in [10, 25, 50]:
        add_block(f"hotratio_p{p}")

    # hot read ratio (p10/p25/p50)
    for p in [10, 25, 50]:
        add_block(f"hotread_p{p}")

    # hot write ratio (p10/p25/p50)
    for p in [10, 25, 50]:
        add_block(f"hotwrite_p{p}")

    assert len(names) == 253, f"header length {len(names)} != 253"
    return names

NICE_FEATURE_COLS = build_feature_headers()

# ============ 통계 함수(원본 유지) ============
def compute_quantiles(series):
    if len(series) == 0:
        return 0, 0, 0
    return (series.quantile(0.25), series.quantile(0.5), series.quantile(0.75))

def compute_per_window_stat(groups, func):
    return [func(group[1]) for group in groups]

def get_rw_ratio(df):
    rw_count = df['op'].value_counts()
    return rw_count.get('R', 0) / len(df) if len(df) > 0 else 0

def get_transition_probability(df, pattern):
    prev_op = ''
    match = 0
    total = 0
    for op in df['op']:
        if prev_op + op == pattern:
            match += 1
        if prev_op == pattern[0]:
            total += 1
        prev_op = op
    return match / total if total > 0 else 0

def get_avg_bw(df, op_filter=None):
    if len(df) < 2:
        return 0
    duration = (df['ts'].iloc[-1] - df['ts'].iloc[0]) / 1e6
    if duration <= 0:
        return 0
    if op_filter:
        df = df[df['op'] == op_filter]
    return df['ioSize'].sum() / duration if len(df) > 0 else 0

def get_avg_iosize(df, op_filter=None):
    if op_filter:
        df = df[df['op'] == op_filter]
    return df['ioSize'].mean() if len(df) > 0 else 0

def get_rms_lba_shift(df, op_filter=None):
    if op_filter:
        df = df[df['op'] == op_filter]
    if len(df) < 2:
        return 0
    shifted = df['lba'].shift(1).fillna(0) + df['ioSize'].shift(1).fillna(0)
    diff = (df['lba'] - shifted).abs()
    return math.sqrt((diff ** 2).mean()) if not diff.empty else 0

def get_hot_ratio(df, top_percent, op_filter=None):
    if op_filter:
        df = df[df['op'] == op_filter]
    if len(df) == 0:
        return 0
    access_counts = Counter(df['lba'])
    sorted_counts = sorted(access_counts.values(), reverse=True)
    top_n = max(1, int(len(sorted_counts) * top_percent / 100))
    top_access = sum(sorted_counts[:top_n])
    total_access = sum(sorted_counts)
    return top_access / total_access if total_access > 0 else 0

# ============ 공용 DF 기반 피처 추출 (NEW) ============
def _series_stats(s: pd.Series):
    # 빈 시리즈 방어: max/min/quantiles에 0 사용
    if len(s) == 0:
        return 0, 0, 0, 0, 0
    q25, q50, q75 = compute_quantiles(s)
    return s.max(), s.min(), q25, q50, q75

def extract_features_from_df(df: pd.DataFrame):
    """
    df columns: ['ts','op','ioSize','lba']
      - ts: uint64 microseconds
      - op: 'R' or 'W'
      - ioSize: bytes
      - lba: byte offset (absolute)
    반환: 길이 253 리스트(float)
    """
    try:
        df = df[['ts','op','ioSize','lba']].dropna().copy()
        df['ts'] = df['ts'].astype(np.uint64)
        df['ts_second'] = (df['ts'] / 1_000_000).astype(int)
        df['ts_minute'] = (df['ts'] / 60_000_000).astype(int)

        second_groups = df.groupby('ts_second')
        minute_groups = df.groupby('ts_minute')

        features = []

        # rwr
        features.append(get_rw_ratio(df))
        for groups in [minute_groups, second_groups]:
            ratios = compute_per_window_stat(groups, get_rw_ratio)
            s = pd.Series(ratios)
            mx, mn, q25, q50, q75 = _series_stats(s)
            features.extend([mx, mn, q25, q50, q75])

        # transition probability
        for pattern in ['RR', 'WR', 'RW', 'WW']:
            features.append(get_transition_probability(df, pattern))
            for groups in [minute_groups, second_groups]:
                probs = compute_per_window_stat(groups, lambda d: get_transition_probability(d, pattern))
                s = pd.Series(probs)
                mx, mn, q25, q50, q75 = _series_stats(s)
                features.extend([mx, mn, q25, q50, q75])

        # bw
        for op in [None, 'W', 'R']:
            features.append(get_avg_bw(df, op_filter=op))
            for groups in [minute_groups, second_groups]:
                bws = compute_per_window_stat(groups, lambda d: get_avg_bw(d, op))
                s = pd.Series(bws)
                mx, mn, q25, q50, q75 = _series_stats(s)
                features.extend([mx, mn, q25, q50, q75])

        # io size
        for op in [None, 'W', 'R']:
            features.append(get_avg_iosize(df, op_filter=op))
            for groups in [minute_groups, second_groups]:
                sizes = compute_per_window_stat(groups, lambda d: get_avg_iosize(d, op))
                s = pd.Series(sizes)
                mx, mn, q25, q50, q75 = _series_stats(s)
                features.extend([mx, mn, q25, q50, q75])

        # rms lba
        for op in [None, 'W', 'R']:
            features.append(get_rms_lba_shift(df, op_filter=op))
            for groups in [minute_groups, second_groups]:
                rmss = compute_per_window_stat(groups, lambda d: get_rms_lba_shift(d, op))
                s = pd.Series(rmss)
                mx, mn, q25, q50, q75 = _series_stats(s)
                features.extend([mx, mn, q25, q50, q75])

        # hot ratio
        for pct in [10, 25, 50]:
            features.append(get_hot_ratio(df, pct))
            for groups in [minute_groups, second_groups]:
                hots = compute_per_window_stat(groups, lambda d: get_hot_ratio(d, pct))
                s = pd.Series(hots)
                mx, mn, q25, q50, q75 = _series_stats(s)
                features.extend([mx, mn, q25, q50, q75])

        # hot read ratio
        for pct in [10, 25, 50]:
            features.append(get_hot_ratio(df, pct, op_filter='R'))
            for groups in [minute_groups, second_groups]:
                hots = compute_per_window_stat(groups, lambda d: get_hot_ratio(d, pct, op_filter='R'))
                s = pd.Series(hots)
                mx, mn, q25, q50, q75 = _series_stats(s)
                features.extend([mx, mn, q25, q50, q75])

        # hot write ratio
        for pct in [10, 25, 50]:
            features.append(get_hot_ratio(df, pct, op_filter='W'))
            for groups in [minute_groups, second_groups]:
                hots = compute_per_window_stat(groups, lambda d: get_hot_ratio(d, pct, op_filter='W'))
                s = pd.Series(hots)
                mx, mn, q25, q50, q75 = _series_stats(s)
                features.extend([mx, mn, q25, q50, q75])

        # 길이 보증
        if len(features) != 253:
            if len(features) < 253:
                features = features + [0]*(253 - len(features))
            else:
                features = features[:253]
        return features
    except Exception as e:
        print(f"[ERR] extract_features_from_df: {e}")
        return [0.0]*253

# ============ 파일 기반(원본 유지) ============
def extract_features_from_trace(trace_path):
    try:
        df = pd.read_csv(trace_path, header=None)
        df.columns = ['device_id', 'op', 'lba', 'ioSize', 'ts']
        df = df[['ts', 'op', 'ioSize', 'lba']].dropna()
        df['ts'] = df['ts'].astype(np.uint64)
        # 파일 기반 버전은 기존 로직을 그대로 유지
        # 아래는 extract_features_from_df와 동일 동작을 위해 그대로 호출
        return extract_features_from_df(df)
    except Exception as e:
        print(f"Error in {trace_path}: {e}")
        return [None] * 253

# ============ recent → 253D 로컬 IOTAP (NEW) ============
def compute_local_iotap_253(recent, block_bytes=4096, length_unit_bytes=512, dt_us=1000):
    """
    recent: list of dicts.
      허용 키 예:
        {"opcode": 0/1 or 'R'/'W', "dlba": blocks, "length": 512B-count}
        {"opcode": 0/1 or 'R'/'W', "dlba_eff": blocks, "length": 512B-count}
    - ts는 균일 간격(dt_us) 의사 타임스탬프로 생성
    - lba는 Δ 누적으로 절대 바이트 오프셋 구성
    반환: np.ndarray(shape=(253,), dtype=float32)
    """
    try:
        if not recent:
            return np.zeros(253, dtype=np.float32)

        # 키 선택: dlba 또는 dlba_eff
        key = None
        if isinstance(recent[0], dict):
            if "dlba" in recent[0]:
                key = "dlba"
            elif "dlba_eff" in recent[0]:
                key = "dlba_eff"
        if key is None:
            return np.zeros(253, dtype=np.float32)

        # 누적 LBA/TS/OP/size 생성
        lba_abs = []
        cur = 0
        ts_list = []
        ops = []
        sizes = []
        ts = 0
        for r in recent:
            # Δ bytes
            delta_blocks = int(r.get(key, 0))
            delta_bytes = delta_blocks * int(block_bytes)
            cur += delta_bytes
            lba_abs.append(cur)

            # op 매핑
            op_raw = r.get("opcode", 0)
            if isinstance(op_raw, str):
                op_val = 'R' if op_raw.upper().startswith('R') else 'W'
            else:
                op_val = 'R' if int(op_raw) == 0 else 'W'
            ops.append(op_val)

            # ioSize bytes
            sizes.append(int(r.get("length", 0)) * int(length_unit_bytes))

            # ts
            ts += int(dt_us)
            ts_list.append(np.uint64(ts))

        df = pd.DataFrame({
            "ts": np.array(ts_list, dtype=np.uint64),
            "op": ops,
            "ioSize": sizes,
            "lba": lba_abs,
        })

        feats = extract_features_from_df(df)
        return np.asarray(feats, dtype=np.float32)
    except Exception as e:
        print(f"[ERR] compute_local_iotap_253: {e}")
        return np.zeros(253, dtype=np.float32)

# (선택) alias 호환
def build_iotap_from_requests(recent):
    return compute_local_iotap_253(recent)
def main():

    trace_files = []

    for f in os.listdir(TRACE_DIR):
        full_path = os.path.join(TRACE_DIR, f)

        # 파일만 처리 (디렉토리 제외)
        if os.path.isfile(full_path):
            trace_files.append(f)

    trace_files = sorted(trace_files)

    print(f"총 {len(trace_files)}개 파일 발견")

    # =========================
    # 기존 OUTPUT_CSV 읽기
    # =========================
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)

        if (
            "trace" in df.columns
            and df.shape[1] == 254
            and list(df.columns)[1].startswith("feature_")
        ):
            rename_map = {"trace": "trace"}
            feat_cols = [c for c in df.columns if c != "trace"]
            for i, c in enumerate(feat_cols):
                rename_map[c] = NICE_FEATURE_COLS[i]
            df = df.rename(columns=rename_map)

        processed_traces = set(df["trace"].tolist())
    else:
        df = pd.DataFrame(columns=["trace"] + NICE_FEATURE_COLS)
        processed_traces = set()

    # =========================
    # 처리 루프
    # =========================
    for i, trace_file in enumerate(tqdm(trace_files, desc="Processing traces")):

        if trace_file in processed_traces:
            continue

        trace_path = os.path.join(TRACE_DIR, trace_file)
        features = extract_features_from_trace(trace_path)

        if len(features) != 253 or any(v is None for v in features):
            print(f"[Skip] {trace_file}: invalid feature length")
            continue

        row = pd.DataFrame([[trace_file] + features], columns=["trace"] + NICE_FEATURE_COLS)
        df = pd.concat([df, row], ignore_index=True)

        df.to_csv(OUTPUT_CSV, index=False)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(trace_files)} traces")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Saved] {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

