"""Tactic-sequence denoising encoder.

Coarse tokenization: each tactic invocation = one vocab token (its head name).
Transformer encoder with a denoising objective (mask 20% of tactic tokens,
predict them). Trained on the constructive-proof subset only, then used to
embed all proofs (constructive + classical) for downstream analysis.

Canonical home for the model class and tokenizer constants. Importers:
    - stage4v3p_encoder.py                       (training runner)
    - stage4v3p_reconstruction.py                (per-proof reconstruction loss)
    - stage4v3p_reconstruction_figures.py        (figure renderer)
    - stage5_multiseed.py                        (multi-seed encoder sweep)
    - scripts/oov_clean_reconstruction.py        (UNK-free reconstruction variant)
    - scripts/prefix_trajectories.py             (per-prefix anomaly trajectory)
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
MASK_ID = 3
UNK_ID = 4
RESERVED = 5  # number of reserved special tokens before vocab IDs start


def encode_proof(heads: list[str], vocab: dict[str, int],
                 max_len: int) -> list[int]:
    ids = [CLS_ID]
    for h in heads:
        v = vocab.get(h)
        if v is None:
            ids.append(UNK_ID)
        else:
            ids.append(v + RESERVED)
        if len(ids) >= max_len - 1:
            break
    ids.append(SEP_ID)
    while len(ids) < max_len:
        ids.append(PAD_ID)
    return ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProofEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 enc_layers: int = 4, dec_layers: int = 2, max_len: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def encode_seq(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        padding_mask = (x == PAD_ID)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return h, padding_mask

    def encode_pool(self, x):
        h, padding_mask = self.encode_seq(x)
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def forward(self, x_corrupt):
        memory, mem_mask = self.encode_seq(x_corrupt)
        B, L = x_corrupt.shape
        positions = torch.arange(L, device=x_corrupt.device).unsqueeze(0).expand(B, L)
        q = self.pos_embed(positions)
        out = self.decoder(q, memory, memory_key_padding_mask=mem_mask)
        return self.head(out)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DenoiseProofDataset(Dataset):
    def __init__(self, tokens: np.ndarray, mask_prob: float = 0.20):
        self.tokens = tokens
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        original = self.tokens[idx].copy()
        corrupted = original.copy()
        # Don't mask special tokens (PAD/CLS/SEP/MASK/UNK)
        non_special = original >= RESERVED
        rand = np.random.random(len(original))
        mask = (rand < self.mask_prob) & non_special
        corrupted[mask] = MASK_ID
        return (torch.from_numpy(corrupted).long(),
                torch.from_numpy(original).long(),
                torch.from_numpy(mask).bool())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_ds, epochs=20, batch_size=128, lr=3e-4, device='cpu'):
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"Training: {len(train_ds)} samples, {len(loader)} batches/epoch")

    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        nb = 0
        model.train()
        for corrupt, target, mask in loader:
            corrupt = corrupt.to(device); target = target.to(device); mask = mask.to(device)
            logits = model(corrupt)
            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, V), target.reshape(-1), reduction='none'
            ).reshape(B, L)
            mloss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
            opt.zero_grad()
            mloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                total_correct += ((preds == target) & mask).sum().item()
                total_masked += mask.sum().item()
            total_loss += mloss.item()
            nb += 1
        sch.step()
        print(f"  Epoch {epoch+1:3d}/{epochs}: loss={total_loss/max(nb,1):.4f} "
              f"masked_acc={total_correct/max(total_masked,1):.3f} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
    return model
