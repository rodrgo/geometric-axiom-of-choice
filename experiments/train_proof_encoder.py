"""Stage 4v3p-b: tactic-sequence denoising encoder — training entry point.

The model class, tokenizer constants, dataset, and training loop now live
in `lean/proof_encoder.py`. This script only runs the training pipeline
that produces `data/stage4v3p/encoder.pt` and `embeddings.npz`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lean.proof_encoder import (  # noqa: E402
    PAD_ID, CLS_ID, SEP_ID, MASK_ID, UNK_ID, RESERVED,
    encode_proof, ProofEncoder, DenoiseProofDataset, train,
)


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Determinism: fix every RNG that touches the canonical encoder — the
    # constructive split (np), and weight init / masking / DataLoader
    # shuffling (torch global generator, used by train()).
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading proofs and vocab...")
    with open('results/data/stage4v3p/proofs.json') as f:
        proofs = json.load(f)
    with open('results/data/stage4v3p/vocab.json') as f:
        vocab = json.load(f)
    VOCAB_SIZE = len(vocab) + RESERVED
    MAX_LEN = 64  # 95th percentile of proof length is ~25
    print(f"  vocab={VOCAB_SIZE} (incl {RESERVED} specials), MAX_LEN={MAX_LEN}")
    print(f"  proofs: {len(proofs):,}")

    # Encode all
    X = np.zeros((len(proofs), MAX_LEN), dtype=np.int64)
    for i, p in enumerate(proofs):
        X[i] = encode_proof(p['invocation_heads'], vocab, MAX_LEN)
    labels = np.array([int(p['is_classical']) for p in proofs])
    print(f"  classical={(labels==1).sum():,} constructive={(labels==0).sum():,}")

    # Split constructive
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
    model = ProofEncoder(vocab_size=VOCAB_SIZE, d_model=128, nhead=4,
                         enc_layers=4, dec_layers=2, max_len=MAX_LEN,
                         dropout=0.1).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ds = DenoiseProofDataset(X[train_idx], mask_prob=0.20)
    model = train(model, ds, epochs=20, batch_size=128, lr=3e-4, device=device)

    Path('results/data/stage4v3p').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'results/data/stage4v3p/encoder.pt')
    print("Saved results/data/stage4v3p/encoder.pt")

    # Embed all
    print("Embedding...")
    model = model.to('cpu')
    model.eval()
    reps = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(X), bs):
            batch = torch.from_numpy(X[i:i+bs]).long()
            z = model.encode_pool(batch)
            z = F.normalize(z, dim=-1)
            reps.append(z.numpy())
    reps = np.concatenate(reps, axis=0)
    print(f"  Shape: {reps.shape}")

    np.savez_compressed('results/data/stage4v3p/embeddings.npz',
                        embeddings=reps, labels=labels,
                        train_idx=train_idx, val_idx=val_idx,
                        test_idx=test_idx)

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
    print(f"\nQuick Hull AUC (proof encoder): {auc:.4f}")
    print(f"  test mean: {d_test.mean():.4f}, cls mean: {d_cls.mean():.4f}")


if __name__ == '__main__':
    main()
