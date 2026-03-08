# RNNInferencer_orbd_delta.py

import os
import torch
import pickle
import random
import numpy as np
import csv
from datetime import datetime

from RNNModel_orbd_delta import RNNModel


# ============================================================
# ===== 설정 =================================================
# ============================================================

CHUNK_FILE = "../processed_chunks_1m_R/w128_s32/trace_283_chunks.pkl"
MODEL_BASE = "trained_rnn_orbd_delta/20260226/v0/trace_10"
MODEL_PATH = os.path.join(MODEL_BASE, "final_model.pt")

SEED = 42
WINDOW_SIZE = 128
GENERATE_LENGTH = 1024 * 32
HIDDEN_DIM = 1024
ORBD_BASE = 1024

START_TOKEN = [2, 1024, 1024, 1024, 1024]
USE_WARMUP = False

# ============================================================
# ===== Sampling 설정 ========================================
# ============================================================

USE_TOPK = False
TOP_K = 10
TEMPERATURE = 1.2


# ============================================================
# ===== Seed 고정 ============================================
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# ===== Sampling 함수 ========================================
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
# ===== ORBD 디코딩 ==========================================
# ============================================================

def recover_magnitude(d3, d2, d1, d0):
    return (
        d3 * ORBD_BASE ** 3 +
        d2 * ORBD_BASE ** 2 +
        d1 * ORBD_BASE +
        d0
    )


def decode_delta(sign, d3, d2, d1, d0):
    mag = recover_magnitude(d3, d2, d1, d0)
    return mag if sign == 0 else (-mag if sign == 1 else 0)


# ============================================================
# ===== Seed Sequence 로딩 ===================================
# ============================================================

def load_seed_sequence():
    with open(CHUNK_FILE, "rb") as f:
        chunks = pickle.load(f)

    idx = random.randint(0, len(chunks) - 1)
    seq = chunks[idx]["sequence"][:WINDOW_SIZE]

    offsets = [int(r["offset"]) for r in seq]

    deltas = []
    prev = offsets[0]
    deltas.append(0)

    for i in range(1, len(offsets)):
        delta = offsets[i] - prev
        deltas.append(delta)
        prev = offsets[i]

    encoded = []

    for d in deltas:
        sign = 0 if d >= 0 else 1
        mag = abs(d)

        digits = []
        tmp = mag
        for _ in range(4):
            digits.insert(0, tmp % ORBD_BASE)
            tmp //= ORBD_BASE

        encoded.append([sign] + digits)

    return encoded, idx


# ============================================================
# ===== Inference ============================================
# ============================================================

def run_inference():

    seed_seq, chunk_idx = load_seed_sequence()

    model = RNNModel(hidden_dim=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    if USE_WARMUP:
        generated_seq = seed_seq.copy()
    else:
        #generated_seq = [START_TOKEN for _ in range(WINDOW_SIZE)]
        generated_seq = [START_TOKEN]

    generated_delta_values = []

    with torch.no_grad():

        for _ in range(GENERATE_LENGTH):

            input_window = generated_seq[-WINDOW_SIZE:]

            input_tensor = torch.tensor(
                input_window,
                dtype=torch.long
            ).unsqueeze(0).to(device)

            out = model(input_tensor)

            sign_logits = out["sign"][:, -1, :].squeeze(0)
            o3_logits   = out["offset3"][:, -1, :].squeeze(0)
            o2_logits   = out["offset2"][:, -1, :].squeeze(0)
            o1_logits   = out["offset1"][:, -1, :].squeeze(0)
            o0_logits   = out["offset0"][:, -1, :].squeeze(0)

            pred_sign = sample_next_token(sign_logits)
            pred_o3   = sample_next_token(o3_logits)
            pred_o2   = sample_next_token(o2_logits)
            pred_o1   = sample_next_token(o1_logits)
            pred_o0   = sample_next_token(o0_logits)

            ## 마지막 timestep만 사용
            #pred_sign = torch.argmax(out["sign"][:, -1, :], dim=-1).item()
            #pred_o3 = torch.argmax(out["offset3"][:, -1, :], dim=-1).item()
            #pred_o2 = torch.argmax(out["offset2"][:, -1, :], dim=-1).item()
            #pred_o1 = torch.argmax(out["offset1"][:, -1, :], dim=-1).item()
            #pred_o0 = torch.argmax(out["offset0"][:, -1, :], dim=-1).item()

            next_token = [pred_sign, pred_o3, pred_o2, pred_o1, pred_o0]

            delta = decode_delta(pred_sign, pred_o3, pred_o2, pred_o1, pred_o0)

            generated_seq.append(next_token)
            generated_delta_values.append(delta)

    # =========================================================
    # ===== min_cum 기반 복원 =================================
    # =========================================================

    cum_list = []
    cum = 0

    for d in generated_delta_values:
        cum += d
        cum_list.append(cum)

    min_cum = min(cum_list) if cum_list else 0
    recon_lba = [c - min_cum for c in cum_list]

    return generated_delta_values, recon_lba, chunk_idx


# ============================================================
# ===== 저장 =================================================
# ============================================================

def save_logs(delta_list, recon_lba, chunk_idx):

    today = datetime.now().strftime("%Y%m%d")
    version = MODEL_BASE.split("/")[-2]
    trace_id = MODEL_BASE.split("/")[-1]
    device_id = trace_id.replace("trace_", "")

    out_dir = os.path.join(MODEL_BASE, "inference")
    os.makedirs(out_dir, exist_ok=True)

    delta_path = os.path.join(
        out_dir,
        f"delta_{today}_input{chunk_idx}_gen{GENERATE_LENGTH}_{version}.csv"
    )

    recon_path = os.path.join(
        out_dir,
        f"recon_{today}_input{chunk_idx}_gen{GENERATE_LENGTH}_{version}.csv"
    )

    # ===== delta 저장 =====
    with open(delta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device_id", "delta_offset_bytes", "length_bytes", "timestamp", "type"])

        for d in delta_list:
            w.writerow([
                device_id,
                int(d) * 4096,
                0,
                0,
                "generated"
            ])

    # ===== reconstructed LBA 저장 =====
    with open(recon_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device_id", "offset", "length", "timestamp", "type"])

        for lba in recon_lba:
            w.writerow([
                device_id,
                int(lba) * 4096,
                0,
                0,
                "generated"
            ])

    print("Saved:")
    print(" -", delta_path)
    print(" -", recon_path)


# ============================================================
# ===== MAIN =================================================
# ============================================================

if __name__ == "__main__":

    delta_list, recon_lba, chunk_idx = run_inference()
    save_logs(delta_list, recon_lba, chunk_idx)

    print("Inference complete.")
