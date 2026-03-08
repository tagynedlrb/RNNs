# LSTMTrainer_orbd.py (LBA-only 최신화)

import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime

from LSTMModel_orbd import TraceDataset, LSTMModel

# ============================================================
# Seed 고정
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 설정
# ============================================================

CHUNK_FILE = "../processed_chunks_1m_R/w128_s32/trace_661_chunks.pkl"

BATCH_SIZE = 512
EPOCHS = 30
HIDDEN_DIM = 1024
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 학습 함수
# ============================================================

def train_model():
    dataset = TraceDataset(CHUNK_FILE)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = LSTMModel(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer,
        lr_lambda=lambda epoch: 0.9
    )
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # 저장 경로 설정
    # ============================================================

    today = datetime.now().strftime("%Y%m%d")
    base_dir = os.path.join("trained_lstm_orbd", today)
    os.makedirs(base_dir, exist_ok=True)

    existing_versions = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("v")
    ]
    version_numbers = [int(d[1:]) for d in existing_versions if d[1:].isdigit()]
    next_version = f"v{max(version_numbers)+1}" if version_numbers else "v0"

    trace_name = os.path.basename(CHUNK_FILE).split("_")[1]
    model_root = os.path.join(base_dir, next_version, f"trace_{trace_name}")
    checkpoint_dir = os.path.join(model_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_log_path = os.path.join(model_root, "loss.csv")
    test_log_path = os.path.join(model_root, "test_log.csv")

    # ============================================================
    # Training Loop
    # ============================================================

    with open(loss_log_path, "w", newline="") as f_loss:
        writer_loss = csv.writer(f_loss)
        writer_loss.writerow(["epoch", "train_loss"])

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                x = x.to(device)  # [B, T, 4]
                y = y.to(device)  # [B, T, 4]

                out = model(x, h=None)  # h=None 고정

                # target 분해
                y_o3 = y[:, :, 0].long()
                y_o2 = y[:, :, 1].long()
                y_o1 = y[:, :, 2].long()
                y_o0 = y[:, :, 3].long()

                # CE expects [B, C, T]
                loss = (
                    criterion(out["offset3"].transpose(1, 2), y_o3) +
                    criterion(out["offset2"].transpose(1, 2), y_o2) +
                    criterion(out["offset1"].transpose(1, 2), y_o1) +
                    criterion(out["offset0"].transpose(1, 2), y_o0)
                )

                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            writer_loss.writerow([epoch+1, avg_train_loss])
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

            # checkpoint 저장
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_train_loss,
            }, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))

            scheduler.step()

    torch.save(model.state_dict(), os.path.join(model_root, "final_model.pt"))
    print("Model saved to:", model_root)

    # ============================================================
    # Test
    # ============================================================

    model.eval()
    total_test_loss = 0

    with torch.no_grad(), open(test_log_path, "w", newline="") as f_test:
        writer_test = csv.writer(f_test)
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x, h=None)  # h=None 고정

            # target 분해
            y_o3 = y[:, :, 0].long()
            y_o2 = y[:, :, 1].long()
            y_o1 = y[:, :, 2].long()
            y_o0 = y[:, :, 3].long()

            batch_loss_o3 = nn.functional.cross_entropy(out["offset3"].transpose(1,2), y_o3, reduction='none').mean(dim=1)
            batch_loss_o2 = nn.functional.cross_entropy(out["offset2"].transpose(1,2), y_o2, reduction='none').mean(dim=1)
            batch_loss_o1 = nn.functional.cross_entropy(out["offset1"].transpose(1,2), y_o1, reduction='none').mean(dim=1)
            batch_loss_o0 = nn.functional.cross_entropy(out["offset0"].transpose(1,2), y_o0, reduction='none').mean(dim=1)

            total_batch_loss = batch_loss_o3 + batch_loss_o2 + batch_loss_o1 + batch_loss_o0
            total_test_loss += total_batch_loss.mean().item()

            # CSV 저장
            for i in range(x.size(0)):
                writer_test.writerow(["input"] + x[i].cpu().tolist())
                writer_test.writerow(["target"] + y[i].cpu().tolist())

                output_row = []
                for t in range(x.size(1)):
                    pred_o3 = int(torch.argmax(out["offset3"][i, t]).item())
                    pred_o2 = int(torch.argmax(out["offset2"][i, t]).item())
                    pred_o1 = int(torch.argmax(out["offset1"][i, t]).item())
                    pred_o0 = int(torch.argmax(out["offset0"][i, t]).item())
                    output_row.append([pred_o3, pred_o2, pred_o1, pred_o0])

                writer_test.writerow(["output"] + output_row)

                writer_test.writerow([
                    "loss_o3", batch_loss_o3[i].item(),
                    "loss_o2", batch_loss_o2[i].item(),
                    "loss_o1", batch_loss_o1[i].item(),
                    "loss_o0", batch_loss_o0[i].item(),
                    "total_loss", total_batch_loss[i].item()
                ])

    avg_test_loss = total_test_loss / len(test_loader)
    with open(loss_log_path, "a", newline="") as f_loss:
        writer_loss = csv.writer(f_loss)
        writer_loss.writerow(["test", avg_test_loss])

    print(f"Test Loss: {avg_test_loss:.4f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()
