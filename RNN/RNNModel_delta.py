# RNNModel_delta_numericCE.py

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ============================================================
# ===== Dataset (Delta-only, ORBD 없이) =====================
# ============================================================

class TraceDataset(Dataset):
    def __init__(self, chunk_path):
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)

        # -----------------------------
        # 1️⃣ 모든 delta 수집 (음수 포함)
        # -----------------------------
        all_delta = set()

        for chunk in chunks:
            seq = chunk["sequence"]
            offsets = [int(r["offset"]) for r in seq]

            for i in range(1, len(offsets)):
                delta = offsets[i] - offsets[i - 1]
                all_delta.add(delta)

        # -----------------------------
        # 2️⃣ unique 정렬 후 index 매핑
        # -----------------------------
        self.delta_list = sorted(list(all_delta))
        self.delta_to_idx = {d: i for i, d in enumerate(self.delta_list)}
        self.idx_to_delta = {i: d for i, d in enumerate(self.delta_list)}

        self.vocab_size = len(self.delta_list)

        self.inputs = []
        self.targets = []

        # -----------------------------
        # 3️⃣ 시퀀스 생성
        # delta_t → delta_{t+1}
        # -----------------------------
        for chunk in chunks:
            seq = chunk["sequence"]
            offsets = [int(r["offset"]) for r in seq]

            deltas = [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]

            if len(deltas) < 2:
                continue

            input_seq = []
            target_seq = []

            for i in range(len(deltas) - 1):
                # float tensor 입력, CE용 target long tensor
                input_seq.append([float(self.delta_to_idx[deltas[i]])])
                target_seq.append(self.delta_to_idx[deltas[i + 1]])

            self.inputs.append(torch.tensor(input_seq, dtype=torch.float))
            self.targets.append(torch.tensor(target_seq, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================
# ===== Model (RNN + CE) ===================================
# ============================================================

class RNNModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,        # delta index를 float로 넣음
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        """
        x: [B, seq_len, 1] float tensor
        """

        out, h_next = self.rnn(x, h)
        logits = self.output_head(out)  # [B, seq_len, vocab_size]

        return {
            "delta": logits,
            "hidden": h_next,
        }
