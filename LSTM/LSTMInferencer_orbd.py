# LSTMInferencer_orbd.py (LBA-only 최신화)

import os
import torch
import pickle
import random
import numpy as np
import csv
from datetime import datetime

from LSTMModel_orbd import LSTMModel

ORBD_BASE = 1024
START_TOKEN = [1024, 1024, 1024, 1024]

# ============================================================
# 설정
# ============================================================

CHUNK_FILE = "../processed_chunks_1m_R/w128_s32/trace_283_chunks.pkl"
MODEL_BASE = "trained_lstm_orbd/20260226/v0/trace_10"
MODEL_PATH = os.path.join(MODEL_BASE, "final_model.pt")

SEED = 42
WINDOW_SIZE = 128
GENERATE_LENGTH = 1024 * 32
HIDDEN_DIM = 1024

# Sampling
USE_TOPK = False
TOP_K = 10
TEMPERATURE = 1.2

# ============================================================
# Seed 고정
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Sampling 함수
# ============================================================

def sample_next_token(logits):
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
# Seed Sequence 로딩
# ============================================================

def load_seed_sequence():
    with open(CHUNK_FILE, "rb") as f:
        chunks = pickle.load(f)

    idx = random.randint(0, len(chunks) - 1)
    seq = chunks[idx]["sequence"][:WINDOW_SIZE]

    offsets = [int(r["offset"]) for r in seq]

    # LBA-only 디코딩 구조
    encoded = []
    for offset in offsets:
        digits = []
        tmp = offset
        for _ in range(4):
            digits.insert(0, tmp % ORBD_BASE)
            tmp //= ORBD_BASE
        encoded.append(digits)

    return encoded, idx


# ============================================================
# Inference
# ============================================================

def run_inference():
    seed_seq, chunk_idx = load_seed_sequence()

    model = LSTMModel(hidden_dim=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    #generated_seq = [START_TOKEN for _ in range(WINDOW_SIZE)]
    generated_seq = [START_TOKEN]

    with torch.no_grad():
        for _ in range(GENERATE_LENGTH):
            input_window = generated_seq[-WINDOW_SIZE:]
            input_tensor = torch.tensor(
                input_window, dtype=torch.long
            ).unsqueeze(0).to(device)  # [1, WINDOW, 4]

            out = model(input_tensor, h=None)  # h=None 고정

            # 마지막 timestep만 사용
            o3_logits = out["offset3"][:, -1, :]
            o2_logits = out["offset2"][:, -1, :]
            o1_logits = out["offset1"][:, -1, :]
            o0_logits = out["offset0"][:, -1, :]

            pred_o3 = sample_next_token(o3_logits)
            pred_o2 = sample_next_token(o2_logits)
            pred_o1 = sample_next_token(o1_logits)
            pred_o0 = sample_next_token(o0_logits)

            next_token = [pred_o3, pred_o2, pred_o1, pred_o0]
            generated_seq.append(next_token)

    return generated_seq, chunk_idx


# ============================================================
# CSV 저장
# ============================================================

def save_logs(generated_seq, chunk_idx):
    today = datetime.now().strftime("%Y%m%d")
    version = MODEL_BASE.split("/")[-2]
    trace_id = MODEL_BASE.split("/")[-1]
    device_id = trace_id.replace("trace_", "")

    out_dir = os.path.join(MODEL_BASE, "inference")
    os.makedirs(out_dir, exist_ok=True)

    lba_path = os.path.join(
        out_dir,
        f"lba_{today}_input{chunk_idx}_gen{GENERATE_LENGTH}_{version}.csv"
    )

    with open(lba_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device_id", "offset", "length", "timestamp", "type"])
        #for token in generated_seq[WINDOW_SIZE:]:
        for token in generated_seq[1:]:
            lba = token[0] * ORBD_BASE**3 + token[1] * ORBD_BASE**2 + token[2] * ORBD_BASE + token[3]
            w.writerow([device_id, int(lba) * 4096, 0, 0, "generated"])

    print("Saved:", lba_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generated_seq, chunk_idx = run_inference()
    save_logs(generated_seq, chunk_idx)
    print("Inference complete.")
