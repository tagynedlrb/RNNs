import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime

from LSTMModel_orbd_delta import TraceDataset, LSTMModel

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

CHUNK_FILE = "../processed_chunks_1m_R/w128_s32/trace_283_chunks.pkl"
BATCH_SIZE = 512
EPOCHS = 20
HIDDEN_DIM = 1024
LR = 1e-4

SIGN_W = 1.0
O3_W   = 1.0
O2_W   = 1.0
O1_W   = 1.0
O0_W   = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 학습
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
    base_dir = os.path.join("trained_lstm_orbd_delta", today)
    os.makedirs(base_dir, exist_ok=True)

    existing_versions = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("v")
    ]

    version_numbers = [int(d[1:]) for d in existing_versions if d[1:].isdigit()]
    next_version = f"v{max(version_numbers) + 1}" if version_numbers else "v0"

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

        writer_loss.writerow([
            "epoch",
            "total_loss",
            "sign_loss",
            "offset3_loss",
            "offset2_loss",
            "offset1_loss",
            "offset0_loss",
            "lr"
        ])

        for epoch in range(EPOCHS):

            model.train()

            total_loss = 0
            total_sign = 0
            total_o3 = 0
            total_o2 = 0
            total_o1 = 0
            total_o0 = 0

            current_lr = optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6e}")

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

                x = x.to(device)
                y = y.to(device)

                out = model(x, h=None)

                y_sign = y[:, :, 0].long()
                y_o3 = y[:, :, 1].long()
                y_o2 = y[:, :, 2].long()
                y_o1 = y[:, :, 3].long()
                y_o0 = y[:, :, 4].long()

                loss_sign = criterion(out["sign"].transpose(1,2), y_sign)
                loss_o3 = criterion(out["offset3"].transpose(1,2), y_o3)
                loss_o2 = criterion(out["offset2"].transpose(1,2), y_o2)
                loss_o1 = criterion(out["offset1"].transpose(1,2), y_o1)
                loss_o0 = criterion(out["offset0"].transpose(1,2), y_o0)

                loss = (
                    SIGN_W * loss_sign +
                    O3_W   * loss_o3 +
                    O2_W   * loss_o2 +
                    O1_W   * loss_o1 +
                    O0_W   * loss_o0
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_sign += loss_sign.item()
                total_o3 += loss_o3.item()
                total_o2 += loss_o2.item()
                total_o1 += loss_o1.item()
                total_o0 += loss_o0.item()

            avg_total = total_loss / len(train_loader)
            avg_sign = total_sign / len(train_loader)
            avg_o3 = total_o3 / len(train_loader)
            avg_o2 = total_o2 / len(train_loader)
            avg_o1 = total_o1 / len(train_loader)
            avg_o0 = total_o0 / len(train_loader)

            writer_loss.writerow([
                epoch + 1,
                avg_total,
                avg_sign,
                avg_o3,
                avg_o2,
                avg_o1,
                avg_o0,
                current_lr
            ])

            print(
                f"Train Loss total={avg_total:.4f} "
                f"sign={avg_sign:.4f} "
                f"o3={avg_o3:.4f} "
                f"o2={avg_o2:.4f} "
                f"o1={avg_o1:.4f} "
                f"o0={avg_o0:.4f}"
            )

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_total,
            }, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))

            scheduler.step()

    torch.save(model.state_dict(), os.path.join(model_root, "final_model.pt"))

    print("Model saved to:", model_root)

    # ============================================================
    # Test Loop
    # ============================================================

    model.eval()

    total_test_loss = 0

    with torch.no_grad(), open(test_log_path, "w", newline="") as f_test:

        writer_test = csv.writer(f_test)

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            y_sign = y[:, :, 0].long()
            y_o3 = y[:, :, 1].long()
            y_o2 = y[:, :, 2].long()
            y_o1 = y[:, :, 3].long()
            y_o0 = y[:, :, 4].long()

            batch_size = x.size(0)

            batch_loss_sign = nn.functional.cross_entropy(
                out["sign"].transpose(1,2),
                y_sign,
                reduction="none"
            ).mean(dim=1)

            batch_loss_o3 = nn.functional.cross_entropy(
                out["offset3"].transpose(1,2),
                y_o3,
                reduction="none"
            ).mean(dim=1)

            batch_loss_o2 = nn.functional.cross_entropy(
                out["offset2"].transpose(1,2),
                y_o2,
                reduction="none"
            ).mean(dim=1)

            batch_loss_o1 = nn.functional.cross_entropy(
                out["offset1"].transpose(1,2),
                y_o1,
                reduction="none"
            ).mean(dim=1)

            batch_loss_o0 = nn.functional.cross_entropy(
                out["offset0"].transpose(1,2),
                y_o0,
                reduction="none"
            ).mean(dim=1)

            total_batch_loss = (
                SIGN_W * batch_loss_sign +
                O3_W   * batch_loss_o3 +
                O2_W   * batch_loss_o2 +
                O1_W   * batch_loss_o1 +
                O0_W   * batch_loss_o0
            )

            total_test_loss += total_batch_loss.mean().item()

            for i in range(batch_size):

                writer_test.writerow(["input"] + x[i].cpu().tolist())
                writer_test.writerow(["target"] + y[i].cpu().tolist())

                output_row = []

                for t in range(x.size(1)):

                    pred_sign = int(torch.argmax(out["sign"][i, t]).item())
                    pred_o3 = int(torch.argmax(out["offset3"][i, t]).item())
                    pred_o2 = int(torch.argmax(out["offset2"][i, t]).item())
                    pred_o1 = int(torch.argmax(out["offset1"][i, t]).item())
                    pred_o0 = int(torch.argmax(out["offset0"][i, t]).item())

                    output_row.append([pred_sign, pred_o3, pred_o2, pred_o1, pred_o0])

                writer_test.writerow(["output"] + output_row)

                writer_test.writerow([
                    "loss_sign", batch_loss_sign[i].item(),
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

if __name__ == "__main__":
    train_model()