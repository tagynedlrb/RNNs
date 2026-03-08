import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TraceDataset(Dataset):
    def __init__(self, chunk_path):
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)

        # ============================================================
        # 전체 LBA 수집
        # ============================================================

        all_lba = []

        for chunk in chunks:
            for r in chunk["sequence"]:
                all_lba.append(int(r["offset"]))

        # global max 계산
        self.max_lba = max(all_lba)

        print(f"Max LBA: {self.max_lba}")

        # ============================================================
        # 시퀀스 생성
        # ============================================================

        self.inputs = []
        self.targets = []

        for chunk in chunks:

            seq = chunk["sequence"]

            # LBA → normalization
            offsets = [
                float(int(r["offset"]) / self.max_lba)
                for r in seq
            ]

            input_seq = []
            target_seq = []

            # x_t → x_{t+1}
            for i in range(len(offsets) - 1):

                input_seq.append([offsets[i]])
                target_seq.append([offsets[i + 1]])

            self.inputs.append(torch.tensor(input_seq, dtype=torch.float))
            self.targets.append(torch.tensor(target_seq, dtype=torch.float))

        print(f"Total sequences: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        return self.inputs[idx], self.targets[idx]

class LSTMModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim

        # numeric → embedding
        self.input_embed = nn.Linear(1, embed_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # hidden → LBA prediction
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, h=None):
        """
        x: [B, T, 1]
        """

        x = self.input_embed(x)
        # [B, T, embed_dim]

        out, h_next = self.lstm(x, h)
        # [B, T, hidden_dim]

        pred = self.output_head(out)
        # [B, T, 1]

        return {
            "lba": pred,
            "hidden": h_next
        }