import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ============================================================
# ===== Dataset (START token 없음 / unique mapping) =========
# ============================================================

class TraceDataset(Dataset):
    def __init__(self, chunk_path):
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)

        # ---- 모든 LBA 수집 ----
        all_lba = set()
        for chunk in chunks:
            for r in chunk["sequence"]:
                all_lba.add(int(r["offset"]))

        # ---- unique 정렬 후 index 매핑 ----
        self.lba_list = sorted(list(all_lba))
        self.lba_to_idx = {lba: i for i, lba in enumerate(self.lba_list)}
        self.idx_to_lba = {i: lba for i, lba in enumerate(self.lba_list)}

        self.vocab_size = len(self.lba_list)

        self.inputs = []
        self.targets = []

        # ---- 시퀀스 생성 ----
        for chunk in chunks:
            seq = chunk["sequence"]
            offsets = [int(r["offset"]) for r in seq]

            input_seq = []
            target_seq = []

            # x_t → x_{t+1} 구조
            for i in range(len(offsets) - 1):
                input_seq.append([float(self.lba_to_idx[offsets[i]])])
                target_seq.append(self.lba_to_idx[offsets[i + 1]])

            self.inputs.append(torch.tensor(input_seq, dtype=torch.float))
            self.targets.append(torch.tensor(target_seq, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================
# ===== GRU Model (numeric input + CE output) ============
# ============================================================

class GRUModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim

        # GRU 사용
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # 출력 레이어
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        """
        x: [B, T, 1]
        h: (h_0, c_0) tuple or None
            - h_0: [1, B, hidden_dim]
            - c_0: [1, B, hidden_dim]
        """
        out, h_next = self.gru(x, h)
        # out: [B, T, hidden_dim]
        # h_next: tuple(h_n, c_n), each [1, B, hidden_dim]

        logits = self.output_head(out)
        # logits: [B, T, vocab_size]

        return {
            "lba": logits,
            "hidden": h_next
        }
