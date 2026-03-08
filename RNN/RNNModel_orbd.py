import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ORBD_BASE = 1024

# ============================================================
# Dataset 정의 (LBA-only)
# ============================================================

class TraceDataset(Dataset):
    def __init__(self, chunk_path):
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)

        self.inputs = []
        self.targets = []

        def to_base_digits(offset):
            digits = []
            for _ in range(4):
                digits.insert(0, offset % ORBD_BASE)
                offset //= ORBD_BASE
            return digits

        for chunk in chunks:
            seq = chunk["sequence"]

            input_seq = []
            target_seq = []

            # offset digit만 사용
            offsets = [to_base_digits(int(r["offset"])) for r in seq]

            # START token
            input_seq.append([1024, 1024, 1024, 1024])

            # input: 0 ~ T-2
            for i in range(len(offsets) - 1):
                input_seq.append(offsets[i])

            # target: 0 ~ T-1 전체
            for i in range(len(offsets)):
                target_seq.append(offsets[i])

            self.inputs.append(torch.tensor(input_seq, dtype=torch.long))
            self.targets.append(torch.tensor(target_seq, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================
# RNNModel 정의 (LBA-only)
# ============================================================

class RNNModel(nn.Module):
    def __init__(self, hidden_dim):
        super(RNNModel, self).__init__()

        embed_dim = hidden_dim // 4
        rnn_input_dim = embed_dim * 4  # digit 4개

        # offset embedding
        self.offset3_embed = nn.Embedding(ORBD_BASE + 1, embed_dim)
        self.offset2_embed = nn.Embedding(ORBD_BASE + 1, embed_dim)
        self.offset1_embed = nn.Embedding(ORBD_BASE + 1, embed_dim)
        self.offset0_embed = nn.Embedding(ORBD_BASE + 1, embed_dim)

        # Vanilla RNN
        self.rnn = nn.RNN(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Output heads
        self.offset3_head = nn.Linear(hidden_dim, ORBD_BASE)
        self.offset2_head = nn.Linear(hidden_dim, ORBD_BASE)
        self.offset1_head = nn.Linear(hidden_dim, ORBD_BASE)
        self.offset0_head = nn.Linear(hidden_dim, ORBD_BASE)

    def forward(self, x, h=None):
        """
        x: [B, T, 4]  # batch, sequence_length, feature (digit 4개)
        """
        o3 = x[:, :, 0].long()
        o2 = x[:, :, 1].long()
        o1 = x[:, :, 2].long()
        o0 = x[:, :, 3].long()

        e_o3 = self.offset3_embed(o3)
        e_o2 = self.offset2_embed(o2)
        e_o1 = self.offset1_embed(o1)
        e_o0 = self.offset0_embed(o0)

        embed = torch.cat([e_o3, e_o2, e_o1, e_o0], dim=-1)

        out, h_next = self.rnn(embed, h)

        return {
            "offset3": self.offset3_head(out),
            "offset2": self.offset2_head(out),
            "offset1": self.offset1_head(out),
            "offset0": self.offset0_head(out),
            "hidden": h_next,
        }
