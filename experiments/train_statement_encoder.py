"""
Stage 4v3c: Denoising autoencoder for Lean statements.

v2 used SimCSE-with-dropout and underperformed the length baseline.
The plan recommends a denoising objective: mask 15-30% of tokens, train
encoder + small decoder to reconstruct. The encoder's bottleneck representation
is the embedding. This forces the encoder to retain content (including length
and complexity), not just coarse distributional features.

Trained on S_constructive only (v3 partition). Embeds all matched theorems.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lean.statement_tokenizer import encode  # noqa: E402


# ---------------------------------------------------------------------------
# Model: encoder + small decoder
# ---------------------------------------------------------------------------

class DenoisingEncoderDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 enc_layers: int = 4, dec_layers: int = 2, max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        # Decoder is a smaller transformer encoder that takes positional
        # queries + encoder context, attending via standard cross-attention.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def encode_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool the encoder output."""
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        padding_mask = (x == 0)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def encode_seq(self, x: torch.Tensor):
        """Return full encoder hidden states (for use as decoder memory)."""
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        padding_mask = (x == 0)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return h, padding_mask

    def forward(self, x_corrupt: torch.Tensor, x_target_positions: torch.Tensor):
        """x_corrupt: token IDs with some [MASK] tokens.
        x_target_positions: position indices to predict (B, L).
        Returns logits (B, L, V) for reconstruction."""
        memory, mem_mask = self.encode_seq(x_corrupt)
        B, L = x_corrupt.shape
        positions = torch.arange(L, device=x_corrupt.device).unsqueeze(0).expand(B, L)
        # Decoder query: positional embeddings (we predict each position from
        # encoder context).
        q = self.pos_embed(positions)
        out = self.decoder(q, memory, memory_key_padding_mask=mem_mask)
        return self.head(out)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DenoiseDataset(Dataset):
    def __init__(self, tokens: np.ndarray, mask_id: int, mask_prob: float = 0.20):
        self.tokens = tokens
        self.mask_id = mask_id
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        original = self.tokens[idx].copy()
        corrupted = original.copy()
        # Don't mask padding (id 0). Also avoid masking CLS/SEP if they have
        # known IDs — we'll mask only positions where token != 0 (non-pad).
        non_pad = original != 0
        rand = np.random.random(len(original))
        mask = (rand < self.mask_prob) & non_pad
        corrupted[mask] = self.mask_id
        return (torch.from_numpy(corrupted).long(),
                torch.from_numpy(original).long(),
                torch.from_numpy(mask).bool())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_ds, epochs=10, batch_size=64, lr=3e-4, device='cpu'):
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
        n_batches = 0
        model.train()
        for corrupt, target, mask in loader:
            corrupt = corrupt.to(device)
            target = target.to(device)
            mask = mask.to(device)
            logits = model(corrupt, None)  # (B, L, V)
            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                target.reshape(-1),
                reduction='none',
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
            n_batches += 1
        sch.step()
        acc = total_correct / max(total_masked, 1)
        print(f"  Epoch {epoch+1:3d}/{epochs}: loss={total_loss/max(n_batches,1):.4f} "
              f"masked_acc={acc:.3f} elapsed={time.time()-t0:.1f}s",
              flush=True)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Determinism: fix numpy (split + masking) and torch (weight init /
    # DataLoader shuffling) RNGs for the canonical statement encoder.
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data: leandojo theorem statements + v3 partition
    print("Loading theorem statements + v3 partition...")
    name_to_stmt = {}
    for split in ['train', 'val', 'test']:
        with open(f'results/data/leandojo/{split}.json') as f:
            for thm in json.load(f):
                n = thm.get('full_name', '')
                if n:
                    name_to_stmt[n] = thm.get('theorem_statement', '') or ''
    with open('results/data/stage4v3/theorems_partition.json') as f:
        partition = json.load(f)

    # Reuse v1 vocab + tokenizer
    with open('results/data/stage4/vocab.json') as f:
        vocab = json.load(f)
    with open('results/data/stage4/tokens_meta.json') as f:
        v1_meta = json.load(f)
    MAX_LEN = v1_meta['max_len']
    VOCAB_SIZE = len(vocab)
    MASK_ID = vocab['[MASK]']
    print(f"  vocab={VOCAB_SIZE}, MAX_LEN={MAX_LEN}, MASK_ID={MASK_ID}")

    # Build records with goals + tokens
    from lean.statement_tokenizer import extract_goal
    records = []
    for r in partition:
        if not r['matched']:
            continue
        stmt = name_to_stmt.get(r['full_name'], '')
        goal = extract_goal(stmt)
        if not goal:
            continue
        records.append({
            'full_name': r['full_name'],
            'file_path': r['file_path'],
            'split': r['split'],
            'is_classical': r['is_classical'],
            'goal': goal,
        })
    print(f"  Records with goals: {len(records)}")

    X = np.zeros((len(records), MAX_LEN), dtype=np.int64)
    for i, r in enumerate(records):
        X[i] = encode(r['goal'], vocab, MAX_LEN)
    labels = np.array([int(r['is_classical']) for r in records])
    print(f"  classical={(labels==1).sum():,} constructive={(labels==0).sum():,}")

    # Splits on constructive
    rng = np.random.default_rng(42)
    cons_idx = np.where(labels == 0)[0]
    rng.shuffle(cons_idx)
    n = len(cons_idx)
    train_idx = cons_idx[:int(0.8 * n)]
    val_idx = cons_idx[int(0.8 * n):int(0.9 * n)]
    test_idx = cons_idx[int(0.9 * n):]
    cls_idx = np.where(labels == 1)[0]
    print(f"  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"classical={len(cls_idx)}")

    # Model
    model = DenoisingEncoderDecoder(
        vocab_size=VOCAB_SIZE, d_model=128, nhead=4,
        enc_layers=4, dec_layers=2, max_len=MAX_LEN, dropout=0.1
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Train
    ds = DenoiseDataset(X[train_idx], mask_id=MASK_ID, mask_prob=0.20)
    model = train(model, ds, epochs=8, batch_size=128, lr=3e-4, device=device)

    Path('results/data/stage4v3').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'results/data/stage4v3/encoder.pt')

    # Embed everything (encoder mean-pool on CPU)
    print("Embedding...")
    model = model.to('cpu')
    model.eval()
    reps = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(X), bs):
            batch = torch.from_numpy(X[i:i+bs]).long()
            z = model.encode_pool(batch)
            # Normalize for hull comparability
            z = F.normalize(z, dim=-1)
            reps.append(z.numpy())
    reps = np.concatenate(reps, axis=0)
    print(f"  Shape: {reps.shape}")

    np.savez_compressed('results/data/stage4v3/embeddings.npz',
                        embeddings=reps, labels=labels,
                        train_idx=train_idx, val_idx=val_idx,
                        test_idx=test_idx)
    # Save matched records for analysis
    with open('results/data/stage4v3/records.json', 'w') as f:
        json.dump(records, f)
    print("Saved results/data/stage4v3/encoder.pt, embeddings.npz, records.json")

    # Quick hull AUC
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    ref_idx = rng.choice(train_idx, size=min(5000, len(train_idx)), replace=False)
    pca = PCA(n_components=4).fit(reps[ref_idx])
    rp = pca.transform(reps[ref_idx])
    hull = ConvexHull(rp)
    vert = rp[hull.vertices]
    d_test = cdist(pca.transform(reps[test_idx]), vert).min(axis=1)
    d_cls = cdist(pca.transform(reps[cls_idx]), vert).min(axis=1)
    y = np.concatenate([np.zeros(len(d_test)), np.ones(len(d_cls))])
    s = np.concatenate([d_test, d_cls])
    auc = roc_auc_score(y, s)
    print(f"\nQuick Hull AUC: {auc:.4f}")
    print(f"  test mean: {d_test.mean():.4f}, cls mean: {d_cls.mean():.4f}")


if __name__ == '__main__':
    main()
