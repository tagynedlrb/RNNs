# RNNModel_orbd_delta.py (최신화)

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ORBD_BASE = 1024
SIGN_VOCAB = 3  # 0:+ , 1:- , 2:START

# ============================================================
# ===== Dataset 정의 (ORBD delta RNN) ========================
# ============================================================

class TraceDataset(Dataset):
    def __init__(self, chunk_path):
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)

        self.inputs = []
        self.targets = []

        def to_base_digits(value):
            digits = []
            for _ in range(4):
                digits.insert(0, value % ORBD_BASE)
                value //= ORBD_BASE
            return digits

        for chunk in chunks:
            seq = chunk["sequence"]

            input_seq = []
            target_seq = []

            # delta 계산
            offsets = [int(r["offset"]) for r in seq]
            deltas = [0]  # 첫 delta는 0
            prev = offsets[0]
            for i in range(1, len(offsets)):
                delta = offsets[i] - prev
                deltas.append(delta)
                prev = offsets[i]

            encoded = []
            for d in deltas:
                sign = 0 if d >= 0 else 1
                magnitude = abs(d)
                digits = to_base_digits(magnitude)
                encoded.append([sign] + digits)

            # START token
            input_seq.append([2, 1024, 1024, 1024, 1024])

            # autoregressive input
            for i in range(len(encoded) - 1):
                input_seq.append(encoded[i])

            # target sequence
            for i in range(len(encoded)):
                target_seq.append(encoded[i])

            self.inputs.append(torch.tensor(input_seq, dtype=torch.long))
            self.targets.append(torch.tensor(target_seq, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================
# ===== RNN Model 정의 (ORBD delta) =========================
# ============================================================

class RNNModel(nn.Module):
    def __init__(self, hidden_dim):
        super(RNNModel, self).__init__()

        embed_dim = hidden_dim // 5
        rnn_input_dim = embed_dim * 5  # sign + 4 digits

        # ===== Embedding =====
        self.sign_embed = nn.Embedding(SIGN_VOCAB, embed_dim)
        self.offset3_embed = nn.Embedding(ORBD_BASE + 1, embed_dim)
        self.offset2_embed = nn.Embedding(ORBD_BASE + 1, embed_dim)
        self.offset1_embed = nn.Embedding(ORBD_BASE + 1, embed_dim)
        self.offset0_embed = nn.Embedding(ORBD_BASE + 1, embed_dim)

        # ===== RNN =====
        self.rnn = nn.RNN(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # ===== Output heads =====
        self.sign_head = nn.Linear(hidden_dim, SIGN_VOCAB)
        self.offset3_head = nn.Linear(hidden_dim, ORBD_BASE)
        self.offset2_head = nn.Linear(hidden_dim, ORBD_BASE)
        self.offset1_head = nn.Linear(hidden_dim, ORBD_BASE)
        self.offset0_head = nn.Linear(hidden_dim, ORBD_BASE)

    def forward(self, x, h=None):
        """
        x: [B, T, 5]  -> batch, seq_len, token(5)
        h: hidden state
        """

        # 각 token 분리
        sign = x[:, :, 0].long()
        o3 = x[:, :, 1].long()
        o2 = x[:, :, 2].long()
        o1 = x[:, :, 3].long()
        o0 = x[:, :, 4].long()

        # embedding
        e_sign = self.sign_embed(sign)
        e_o3 = self.offset3_embed(o3)
        e_o2 = self.offset2_embed(o2)
        e_o1 = self.offset1_embed(o1)
        e_o0 = self.offset0_embed(o0)

        embed = torch.cat([e_sign, e_o3, e_o2, e_o1, e_o0], dim=-1)

        # RNN forward
        out, h_next = self.rnn(embed, h)

        # 출력 head
        return {
            "sign": self.sign_head(out),
            "offset3": self.offset3_head(out),
            "offset2": self.offset2_head(out),
            "offset1": self.offset1_head(out),
            "offset0": self.offset0_head(out),
            "hidden": h_next
        }
