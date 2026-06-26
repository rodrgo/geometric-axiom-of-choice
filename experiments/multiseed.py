"""
Step 2.1: Multi-seed encoder runs.

Re-train the proof encoder (train_proof_encoder.py) with 5 different random
seeds and report the mean +- std of hull AUC. Hyperparameters identical to
the published run.

Output: results/data/controls/multi_seed_aucs.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lean.proof_encoder import (  # noqa: E402
    ProofEncoder, DenoiseProofDataset, encode_proof, RESERVED, train,
)


def hull_auc_for(z, labels, train_idx, test_idx, cls_idx, hull_dim=4):
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(42)
    ref_idx = rng.choice(train_idx, size=min(5000, len(train_idx)),
                         replace=False)
    pca = PCA(n_components=hull_dim).fit(z[ref_idx])
    rp = pca.transform(z[ref_idx])
    hull = ConvexHull(rp)
    vert = rp[hull.vertices]
    d_test = cdist(pca.transform(z[test_idx]), vert).min(axis=1)
    d_cls = cdist(pca.transform(z[cls_idx]), vert).min(axis=1)
    y = np.concatenate([np.zeros(len(d_test)), np.ones(len(d_cls))])
    s = np.concatenate([d_test, d_cls])
    return float(roc_auc_score(y, s))


def run_one_seed(seed: int, X: np.ndarray, labels: np.ndarray,
                 vocab_size: int, MAX_LEN: int, device: str):
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    rng = np.random.default_rng(seed)
    cons_idx = np.where(labels == 0)[0]
    rng.shuffle(cons_idx)
    n = len(cons_idx)
    train_idx = cons_idx[:int(0.8 * n)]
    val_idx = cons_idx[int(0.8 * n):int(0.9 * n)]
    test_idx = cons_idx[int(0.9 * n):]
    cls_idx = np.where(labels == 1)[0]

    model = ProofEncoder(vocab_size=vocab_size, d_model=128, nhead=4,
                          enc_layers=4, dec_layers=2, max_len=MAX_LEN,
                          dropout=0.1).to(device)
    ds = DenoiseProofDataset(X[train_idx], mask_prob=0.20)
    t0 = time.time()
    model = train(model, ds, epochs=20, batch_size=128, lr=3e-4, device=device)
    train_t = time.time() - t0

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

    auc = hull_auc_for(reps, labels, train_idx, test_idx, cls_idx)
    print(f"  Hull AUC (seed {seed}): {auc:.4f}  (train {train_t:.0f}s)")
    return {
        'seed': seed,
        'hull_auc': auc,
        'train_time_seconds': train_t,
        'n_train': int(len(train_idx)),
        'n_test': int(len(test_idx)),
        'n_classical': int(len(cls_idx)),
    }


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load proofs and vocab
    with open('results/data/stage4v3p/proofs.json') as f:
        proofs = json.load(f)
    with open('results/data/stage4v3p/vocab.json') as f:
        vocab = json.load(f)
    VOCAB_SIZE = len(vocab) + RESERVED
    MAX_LEN = 64

    # Encode once
    X = np.zeros((len(proofs), MAX_LEN), dtype=np.int64)
    for i, p in enumerate(proofs):
        X[i] = encode_proof(p['invocation_heads'], vocab, MAX_LEN)
    labels = np.array([int(p['is_classical']) for p in proofs])
    print(f"Data: {len(proofs):,} proofs, vocab={VOCAB_SIZE}, MAX_LEN={MAX_LEN}")

    seeds = [42, 123, 456, 789, 1024]
    results = []
    for s in seeds:
        results.append(run_one_seed(s, X, labels, VOCAB_SIZE, MAX_LEN, device))

    aucs = np.array([r['hull_auc'] for r in results])
    summary = {
        'seeds': seeds,
        'per_seed': results,
        'mean_auc': float(aucs.mean()),
        'std_auc': float(aucs.std(ddof=1)),
        'min_auc': float(aucs.min()),
        'max_auc': float(aucs.max()),
        'note': ('Hull AUC of the proof denoising encoder across 5 random '
                 'seeds. Each seed re-shuffles the train/test split and '
                 're-initializes the model. BoW and hand-crafted feature '
                 'baselines are deterministic and not re-run.'),
    }
    Path('results/data/controls').mkdir(parents=True, exist_ok=True)
    with open('results/data/controls/multi_seed_aucs.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Multi-seed summary:")
    print(f"{'='*60}")
    print(f"  Hull AUC: {aucs.mean():.4f} ± {aucs.std(ddof=1):.4f}")
    print(f"  Range:    [{aucs.min():.4f}, {aucs.max():.4f}]")
    print(f"  Per seed: {[(r['seed'], round(r['hull_auc'], 4)) for r in results]}")
    print(f"\nSaved results/data/controls/multi_seed_aucs.json")


if __name__ == '__main__':
    main()
