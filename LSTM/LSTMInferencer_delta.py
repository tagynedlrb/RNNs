# LSTMInferencer_deltaCE_sampling.py

import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
import csv
from datetime import datetime

from LSTMModel_delta import LSTMModel


# ============================================================
# ========================= 설정 ==============================
# ============================================================

ORIGINAL_TRACE = "../1m_requests_R/trace_10"
MODEL_BASE = "trained_lstm_lba_delta/20260226/v0/trace_10"
MODEL_PATH = os.path.join(MODEL_BASE, "final_model.pt")

WINDOW_SIZE = 128
SEED = 42
GENERATE_LENGTH = 1024 * 32
HIDDEN_DIM = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# ================= Sampling 설정 =============================
# ============================================================

USE_TOPK = False
TOP_K = 10
TEMPERATURE = 1.2   # 1.0 = 기본 / >1 = 더 랜덤 / <1 = 더 deterministic


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
# =================== 초기 delta 선택 ==========================
# ============================================================

def load_initial_delta(index_to_delta):
    # 첫 delta는 가장 첫 index 사용
    return index_to_delta[0]


# ============================================================
# ====================== Inference ============================
# ============================================================

def run_inference():

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    vocab_size = checkpoint["vocab_size"]
    index_to_delta = checkpoint["index_to_delta"]

    model = LSTMModel(
        hidden_dim=HIDDEN_DIM,
        vocab_size=vocab_size
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    delta_to_index = {v: k for k, v in index_to_delta.items()}

    current_delta = load_initial_delta(index_to_delta)
    current_index = delta_to_index[current_delta]

    generated_indices = []
    input_buffer = [current_index] * WINDOW_SIZE

    with torch.no_grad():

        for _ in range(GENERATE_LENGTH):

            input_window = input_buffer[-WINDOW_SIZE:]

            input_tensor = torch.tensor(
                input_window,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1).to(device)

            out = model(input_tensor)

            logits = out["delta"]   # [1, T, vocab]
            last_logits = logits[:, -1, :].squeeze(0)

            # ������ sampling 적용
            pred_index = sample_next_token(last_logits)

            generated_indices.append(pred_index)
            input_buffer.append(pred_index)

    generated_deltas = [index_to_delta[idx] for idx in generated_indices]

    return generated_deltas


# ============================================================
# ======================== 저장 ===============================
# ============================================================

def save_logs(generated_deltas):

    today = datetime.now().strftime("%Y%m%d")
    out_dir = os.path.join(MODEL_BASE, "inference")
    os.makedirs(out_dir, exist_ok=True)

    base_name = f"inference_{today}_gen{len(generated_deltas)}"

    delta_path = os.path.join(out_dir, f"{base_name}_delta.csv")
    recon_path = os.path.join(out_dir, f"{base_name}_recon.csv")

    device_id = os.path.basename(MODEL_BASE).replace("trace_", "")

    # ============================================================
    # 1️⃣ delta 저장
    # ============================================================

    with open(delta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device_id", "delta_offset_bytes", "length_bytes", "timestamp", "type"])

        for d in generated_deltas:
            w.writerow([
                device_id,
                int(d) * 4096,
                0,
                0,
                "generated"
            ])

    # ============================================================
    # 2️⃣ delta 누적 → offset 복원 (min_cum 방식)
    # ============================================================

    cum_list = []
    cum = 0

    for d in generated_deltas:
        delta_bytes = int(d) * 4096
        cum += delta_bytes
        cum_list.append(cum)

    min_cum = min(cum_list) if cum_list else 0

    with open(recon_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device_id", "offset", "length", "timestamp", "type"])

        for c in cum_list:
            w.writerow([
                device_id,
                c - min_cum,
                0,
                0,
                "generated"
            ])

    print("[Saved]", delta_path)
    print("[Saved]", recon_path)


# ============================================================
# ======================== MAIN ===============================
# ============================================================

if __name__ == "__main__":

    generated_deltas = run_inference()
    save_logs(generated_deltas)

    print("Inference complete.")

