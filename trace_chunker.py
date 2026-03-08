import os
import pandas as pd
import numpy as np
from typing import List
import pickle
from tqdm import tqdm

# 설정
TRACE_DIR = "1m_requests_R"
# ===== 10개 타겟 trace =====
TRACE_LIST = [
    "trace_10",
    "trace_283",
    "trace_372",
    "trace_661",

#    "trace_126",
#    "trace_693",
#    "trace_106",
#    "trace_311",
#    "trace_504",
]

CONDITION_CSV = "1m_requests_iotap_stats.csv"
WINDOW_SIZE = 1024
#WINDOW_SIZE = 128
STRIDE = 128
#STRIDE = 32
EXPECTED_CONDITION_DIM = 253

# 저장 폴더 이름 설정
BASE_OUTPUT_DIR = "processed_chunks_1m_R"
OUTPUT_SUBDIR = f"w{WINDOW_SIZE}_s{STRIDE}"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, OUTPUT_SUBDIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 조건 벡터 로딩
#condition_df = pd.read_csv(CONDITION_CSV)
#condition_df.set_index("trace", inplace=True)


def load_trace(trace_name: str) -> pd.DataFrame:
    path = os.path.join(TRACE_DIR, trace_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace file not found: {path}")

    df = pd.read_csv(path, header=None)
    df.columns = ["device_id", "opcode", "offset", "length", "timestamp"]

    # 시간 상대화: 첫 request 기준으로 timestamp 조정
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]

    # offset을 4KB LBA 단위로 변환
    df["offset"] = df["offset"] // 4096

    # length를 512B 섹터 단위로 변환
    df["length"] = df["length"] // 512

    return df

'''
def get_condition(trace_name: str) -> np.ndarray:
    if trace_name not in condition_df.index:
        raise ValueError(f"Condition not found for trace: {trace_name}")
    condition = condition_df.loc[trace_name].values
    if len(condition) != EXPECTED_CONDITION_DIM:
        raise ValueError(f"Condition for trace {trace_name} has invalid length: {len(condition)} (expected {EXPECTED_CONDITION_DIM})")
    return condition
'''

#def make_chunks(df: pd.DataFrame, condition: np.ndarray, trace_name: str) -> List[dict]:
def make_chunks(df: pd.DataFrame, trace_name: str) -> List[dict]:
    chunks = []
    for i in tqdm(range(0, len(df) - (WINDOW_SIZE + 1) + 1, STRIDE), desc=f"Chunking {trace_name}"):
        chunk = {
            "sequence": df.iloc[i:i + WINDOW_SIZE + 1].to_dict(orient="records"),
            "trace_id": trace_name,
            "start_index": i
        }
        chunks.append(chunk)
    return chunks


for trace_name in tqdm(TRACE_LIST, desc="Processing traces"):
    df = load_trace(trace_name)
    #condition = get_condition(trace_name)
    #chunks = make_chunks(df, condition, trace_name)
    chunks = make_chunks(df, trace_name)

    # 저장 경로
    output_path = os.path.join(OUTPUT_DIR, f"{trace_name}_chunks.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks to {output_path}")
