# LSTMTraceInferencer_numeric.py

import os
import torch
import random
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from LSTMModel import LSTMModel


# ============================================================
# 설정
# ============================================================

ORIGINAL_TRACE = "../1m_requests_R/trace_283"

MODEL_BASE = "trained_lstm_numeric/20260306/v1/trace_283"
MODEL_PATH = os.path.join(MODEL_BASE, "final_model.pt")

WINDOW_SIZE = 1024
GENERATE_LENGTH = 1024 * 32

EMBED_DIM = 256
HIDDEN_DIM = 512

SEED = 42


# ============================================================
# Seed
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Original Trace 로드
# ============================================================

def load_original_trace(path):

    df = pd.read_csv(path, header=None)

    df.columns = ["device_id", "opcode", "offset", "length", "timestamp"]

    df["timestamp"] -= df["timestamp"].iloc[0]

    df["offset"] = df["offset"] // 4096

    return df


# ============================================================
# 초기 window
# ============================================================

def load_initial_window():

    df = load_original_trace(ORIGINAL_TRACE)

    offsets = df["offset"].tolist()

    return offsets[:WINDOW_SIZE]


# ============================================================
# Inference
# ============================================================

def run_inference():

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    max_lba = checkpoint["max_lba"]

    model = LSTMModel(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ------------------------------------------------
    # 초기 window
    # ------------------------------------------------

    initial_window = load_initial_window()

    # normalize
    generated_norm = [
        lba / max_lba for lba in initial_window
    ]

    generated_output = []

    h = None

    with torch.no_grad():

        for _ in range(GENERATE_LENGTH):

            window = generated_norm[-WINDOW_SIZE:]

            input_tensor = torch.tensor(
                window,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1).to(device)
            # [1, T, 1]

            out = model(input_tensor, h=None)

            pred = torch.sigmoid(out["lba"])

            next_val = pred[:, -1, 0].item()

            generated_norm.append(next_val)

            # 실제 LBA 복원
            lba = int(next_val * max_lba)

            generated_output.append(lba)

    return generated_output


# ============================================================
# 저장
# ============================================================

def save_logs(generated_seq):

    today = datetime.now().strftime("%Y%m%d")

    version = MODEL_BASE.split("/")[-2]

    trace_id = MODEL_BASE.split("/")[-1]

    device_id = trace_id.replace("trace_", "")

    out_dir = os.path.join(MODEL_BASE, "inference")

    os.makedirs(out_dir, exist_ok=True)

    infer_path = os.path.join(
        out_dir,
        f"inference_{today}_gen{GENERATE_LENGTH}_{version}.csv"
    )

    with open(infer_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["device_id", "offset", "length", "timestamp", "type"])

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

if __name__ == "__main__":

    generated_seq = run_inference()

    save_logs(generated_seq)

    print("Inference complete.")