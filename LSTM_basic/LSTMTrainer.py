import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime

from LSTMModel import TraceDataset, LSTMModel


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

CHUNK_FILE = "../processed_chunks_1m_R/w1024_s128/trace_283_chunks.pkl"

BATCH_SIZE = 512
EPOCHS = 20

HIDDEN_DIM = 512
EMBED_DIM = 256

LR = 1e-3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# Train
# ============================================================

def train_model():

    dataset = TraceDataset(CHUNK_FILE)

    # ============================================================
    # Train / Test Split
    # ============================================================

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        drop_last=False
    )


    # ============================================================
    # Model
    # ============================================================

    model = LSTMModel(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer,
        lr_lambda=lambda epoch: 0.9
    )

    criterion = nn.MSELoss()


    # ============================================================
    # 저장 경로 생성
    # ============================================================

    today = datetime.now().strftime("%Y%m%d")

    base_dir = os.path.join("trained_lstm_numeric", today)
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
    # TRAIN
    # ============================================================

    with open(loss_log_path, "w", newline="") as f_loss:

        writer_loss = csv.writer(f_loss)
        writer_loss.writerow(["epoch", "train_loss"])

        for epoch in range(EPOCHS):

            model.train()
            total_loss = 0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

                x = x.to(device)
                y = y.to(device)

                out = model(x, h=None)

                pred = torch.sigmoid(out["lba"])

                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            writer_loss.writerow([epoch + 1, avg_train_loss])

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_train_loss,
                "max_lba": dataset.max_lba
            }, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))

            scheduler.step()


    # ============================================================
    # Final Model Save
    # ============================================================

    torch.save({
        "model_state_dict": model.state_dict(),
        "max_lba": dataset.max_lba
    }, os.path.join(model_root, "final_model.pt"))

    print(f"Model saved to {model_root}")


    # ============================================================
    # TEST
    # ============================================================

    model.eval()

    with torch.no_grad(), open(test_log_path, "w", newline="") as f_test:

        writer_test = csv.writer(f_test)

        total_test_loss = 0
        total_samples = 0

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x, h=None)

            pred = torch.sigmoid(out["lba"])

            loss = criterion(pred, y)

            batch_size = x.size(0)

            total_test_loss += loss.item() * batch_size
            total_samples += batch_size

            for i in range(x.size(0)):

                writer_test.writerow(["input"] + x[i].cpu().flatten().tolist())
                writer_test.writerow(["target"] + y[i].cpu().flatten().tolist())
                writer_test.writerow(["output"] + pred[i].cpu().flatten().tolist())
                writer_test.writerow(["loss", loss.item()])


        avg_test_loss = total_test_loss / total_samples


    with open(loss_log_path, "a", newline="") as f_loss:

        writer_loss = csv.writer(f_loss)
        writer_loss.writerow(["test", avg_test_loss])

    print(f"Test Loss: {avg_test_loss:.6f}")


# ============================================================

if __name__ == "__main__":
    train_model()