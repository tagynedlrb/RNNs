# LSTMTraceInferencer_numericCE_noSTART.py

import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from LSTMModel import LSTMModel  # LSTMModel → LSTMModel

# ===== 설정 =====
ORIGINAL_TRACE = "../1m_requests_R/trace_10"
CHUNK_FILE = "../processed_chunks_1m_R/w128_s32/trace_10_chunks.pkl"
MODEL_BASE = "trained_lstm_lba/20260225/v0/trace_10"  # LSTM 모델 경로
MODEL_PATH = os.path.join(MODEL_BASE, "final_model.pt")

SEED = 42
WINDOW_SIZE = 128
GENERATE_LENGTH = 1024 * 32
HIDDEN_DIM = 512

USE_WARMUP = False  # 구조 유지용

# ===== Seed =====
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Sampling 설정 =====
USE_TOPK = False
TOP_K = 10
TEMPERATURE = 1.2   # 1.0 = 기본, >1 = 더 랜덤, <1 = 더 deterministic


def sample_next_token(logits):
    """
    logits: [vocab_size]
    """
    logits = logits / TEMPERATURE
    probs = torch.softmax(logits, dim=-1)

    if USE_TOPK:
        topk_probs, topk_indices = torch.topk(probs, TOP_K)
        topk_probs = topk_probs / torch.sum(topk_probs)
        next_idx = torch.multinomial(topk_probs, 1)
        return topk_indices[next_idx].item()
    else:
        return torch.multinomial(probs, 1).item()


# ============================================================
# =================== 원본 trace 로딩 =========================
# ============================================================

def load_original_trace(path):
    df = pd.read_csv(path, header=None)
    df.columns = ["device_id", "opcode", "offset", "length", "timestamp"]
    df["timestamp"] -= df["timestamp"].iloc[0]
    df["offset"] = df["offset"] // 4096
    return df


# ============================================================
# =================== 초기 입력 하나 선택 =====================
# ============================================================

def load_initial_input():
    with open(CHUNK_FILE, "rb") as f:
        chunks = pickle.load(f)

    idx = random.randint(0, len(chunks) - 1)
    seq = chunks[idx]["sequence"]

    initial_index = int(seq[0]["offset"])
    return initial_index, idx


# ============================================================
# ====================== Inference ===========================
# ============================================================

def run_inference():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    vocab_size = checkpoint["vocab_size"]
    index_to_lba = checkpoint["index_to_lba"]

    model = LSTMModel(
        hidden_dim=HIDDEN_DIM,
        vocab_size=vocab_size
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    initial_index, chunk_idx = load_initial_input()

    # -------------------------------------------------
    # 내부 버퍼 (항상 WINDOW_SIZE 유지)
    # -------------------------------------------------
    generated_indices = [initial_index] * WINDOW_SIZE
    generated_output = []

    with torch.no_grad():
        for _ in range(GENERATE_LENGTH):
            input_window = generated_indices[-WINDOW_SIZE:]

            # [1, T, 1]  (numeric CE 구조)
            input_tensor = torch.tensor(
                input_window,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1).to(device)

            out = model(input_tensor, h=None)  # ������ LSTM forward

            logits = out["lba"]   # [1, T, vocab_size]

            # sampling
            last_logits = logits[:, -1, :].squeeze(0)
            pred_index = sample_next_token(last_logits)

            # argmax 사용시
            # pred_index = torch.argmax(logits[:, -1, :], dim=-1).item()

            generated_indices.append(pred_index)
            generated_output.append(pred_index)

    # index → 실제 LBA 복원
    generated_lba = [index_to_lba[idx] for idx in generated_output]

    return generated_lba, chunk_idx


# ============================================================
# ======================== 저장 ==============================
# ============================================================

def save_logs(generated_seq, chunk_idx):
    today = datetime.now().strftime("%Y%m%d")
    version = MODEL_BASE.split("/")[-2]
    trace_id = MODEL_BASE.split("/")[-1]
    device_id = trace_id.replace("trace_", "")

    out_dir = os.path.join(MODEL_BASE, "inference")
    os.makedirs(out_dir, exist_ok=True)

    infer_path = os.path.join(
        out_dir,
        f"inference_{today}_input{chunk_idx}_gen{GENERATE_LENGTH}_{version}.csv"
    )

    original_df = load_original_trace(ORIGINAL_TRACE)

    with open(infer_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["device_id", "offset", "length", "timestamp", "type"])

        # 생성 구간만 기록
        for lba in generated_seq:
            writer.writerow([
                device_id,
                lba * 4096,
                0,
                0,
                "generated"
            ])

    print("Saved:", infer_path)


# ============================================================
# ======================= MAIN ===============================
# ============================================================

if __name__ == "__main__":
    generated_seq, chunk_idx = run_inference()
    save_logs(generated_seq, chunk_idx)
    print("Inference complete.")
