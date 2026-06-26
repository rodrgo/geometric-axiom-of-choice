"""
Stage 4v3p: Reconstruction loss as a generalization metric.

For every proof, compute the average cross-entropy loss on 20%-masked tokens
under the frozen denoising encoder trained on constructive proofs. Stratify by
BFS distance to Classical.choice and test whether loss scales with depth.

Follows plan_reconstruction_loss.md. Produces summary stats + p-values; figures
are gated behind a CHECKPOINT (plotted only after user confirms the signal).
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lean.proof_encoder import (  # noqa: E402
    ProofEncoder, encode_proof,
    PAD_ID, CLS_ID, SEP_ID, MASK_ID, UNK_ID, RESERVED,
)


def compute_losses_batched(model, X, rng, n_samples=10, mask_prob=0.20,
                           batch_size=256, device='cpu'):
    """Per-proof mean reconstruction loss, averaged over n_samples maskings.

    Returns array of shape (N,). NaN where no maskable token exists.
    """
    N, L = X.shape
    per_sample_losses = np.full((N, n_samples), np.nan, dtype=np.float64)
    non_special = X >= RESERVED  # [N, L] boolean

    model.eval()
    t0 = time.time()
    for s in range(n_samples):
        # Draw a mask for every proof, every token
        rand = rng.random(size=(N, L))
        sample_mask = (rand < mask_prob) & non_special  # [N, L]

        for i in range(0, N, batch_size):
            x_batch = X[i:i+batch_size]
            m_batch = sample_mask[i:i+batch_size]
            target = torch.from_numpy(x_batch).long().to(device)
            mask_t = torch.from_numpy(m_batch).to(device)
            corrupt = target.clone()
            corrupt[mask_t] = MASK_ID

            with torch.no_grad():
                logits = model(corrupt)  # [B, L, V]
            B, Lb, V = logits.shape
            ce = F.cross_entropy(
                logits.reshape(-1, V), target.reshape(-1), reduction='none'
            ).reshape(B, Lb)
            ce = ce * mask_t.float()
            n_masked = mask_t.float().sum(dim=1)  # [B]
            per_proof = ce.sum(dim=1) / n_masked.clamp(min=1)
            per_proof_np = per_proof.cpu().numpy()
            # Record NaN where this sample produced no maskable token
            no_mask = (n_masked == 0).cpu().numpy()
            per_proof_np = np.where(no_mask, np.nan, per_proof_np)
            per_sample_losses[i:i+batch_size, s] = per_proof_np

        print(f"  sample {s+1}/{n_samples} done ({time.time()-t0:.1f}s)", flush=True)

    # Average over samples (ignoring NaN)
    with np.errstate(invalid='ignore'):
        mean_loss = np.nanmean(per_sample_losses, axis=1)
    return mean_loss


def bucket_of(depth):
    if depth is None:
        return None
    if depth <= 2:
        return "depth 2"
    if depth == 3:
        return "depth 3"
    if depth == 4:
        return "depth 4"
    if depth <= 6:
        return "depth 5-6"
    if depth <= 8:
        return "depth 7-8"
    return "depth 9+"


def main():
    device = 'cpu'  # small model + MPS has known issues; CPU is fast enough
    data_dir = Path('results/data/stage4v3p')
    out_dir = Path('results/data/depth_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading proofs, vocab, encoder...")
    with open(data_dir / 'proofs.json') as f:
        proofs = json.load(f)
    with open(data_dir / 'vocab.json') as f:
        vocab = json.load(f)
    with open('results/data/depth_analysis/bfs_distances_full.json') as f:
        bfs = json.load(f)

    VOCAB_SIZE = len(vocab) + RESERVED
    MAX_LEN = 64

    # Encode all proofs (matches encoder script exactly)
    N = len(proofs)
    X = np.zeros((N, MAX_LEN), dtype=np.int64)
    for i, p in enumerate(proofs):
        X[i] = encode_proof(p['invocation_heads'], vocab, MAX_LEN)
    is_classical = np.array([int(p['is_classical']) for p in proofs], dtype=bool)
    names = [p['name'] for p in proofs]
    lengths = np.array([
        int((X[i] >= RESERVED).sum()) for i in range(N)
    ])  # non-special token count

    print(f"  N={N:,}  classical={is_classical.sum():,}  "
          f"constructive={(~is_classical).sum():,}")

    # Load train/val/test split (constructive ML split)
    emb = np.load(data_dir / 'embeddings.npz')
    train_idx = emb['train_idx']
    val_idx = emb['val_idx']
    test_idx = emb['test_idx']
    print(f"  ML split: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}")

    # Load model
    model = ProofEncoder(vocab_size=VOCAB_SIZE, d_model=128, nhead=4,
                         enc_layers=4, dec_layers=2, max_len=MAX_LEN,
                         dropout=0.1).to(device)
    state = torch.load(data_dir / 'encoder.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  encoder params: {sum(p.numel() for p in model.parameters()):,}")

    # Compute reconstruction loss
    print("Computing reconstruction loss (10 mask samples / proof)...")
    rng = np.random.default_rng(0)
    losses = compute_losses_batched(model, X, rng, n_samples=10,
                                    mask_prob=0.20, batch_size=512,
                                    device=device)
    print(f"  losses computed. NaN (unmaskable): {np.isnan(losses).sum()}")

    # Build buckets
    buckets = defaultdict(list)
    buckets_idx = defaultdict(list)
    for i in range(N):
        if np.isnan(losses[i]):
            continue
        if not is_classical[i]:
            # Use only held-out constructive for the baseline
            if i in set(test_idx.tolist()):
                buckets["constructive (test)"].append(losses[i])
                buckets_idx["constructive (test)"].append(i)
            # Also track train-only for sanity
            if i in set(train_idx.tolist()):
                buckets["constructive (train)"].append(losses[i])
                buckets_idx["constructive (train)"].append(i)
        else:
            d = bfs.get(names[i])
            b = bucket_of(d)
            if b is not None:
                buckets[b].append(losses[i])
                buckets_idx[b].append(i)

    bucket_order = [
        "constructive (train)", "constructive (test)",
        "depth 2", "depth 3", "depth 4",
        "depth 5-6", "depth 7-8", "depth 9+",
    ]

    # Report means
    print("\n=== Reconstruction loss by bucket ===")
    summary = {}
    for b in bucket_order:
        ls = np.array(buckets[b])
        if len(ls) == 0:
            print(f"  {b}: empty")
            continue
        mean = ls.mean()
        sem = ls.std(ddof=1) / np.sqrt(len(ls)) if len(ls) > 1 else 0.0
        print(f"  {b:24s}  n={len(ls):6d}  mean={mean:.4f}  sem={sem:.4f}")
        summary[b] = {
            "n": int(len(ls)),
            "mean_loss": float(mean),
            "std_loss": float(ls.std(ddof=1)) if len(ls) > 1 else 0.0,
            "sem_loss": float(sem),
            "median_loss": float(np.median(ls)),
        }

    # Stat tests
    from scipy.stats import mannwhitneyu, spearmanr

    baseline_key = "constructive (test)"
    baseline = np.array(buckets[baseline_key])

    print("\n=== Mann-Whitney U tests (alternative='greater' vs constructive test) ===")
    mw_pvals = {}
    for b in ["depth 2", "depth 3", "depth 4",
              "depth 5-6", "depth 7-8", "depth 9+"]:
        arr = np.array(buckets[b])
        if len(arr) == 0:
            continue
        stat, p_greater = mannwhitneyu(arr, baseline, alternative='greater')
        _, p_two = mannwhitneyu(arr, baseline, alternative='two-sided')
        mw_pvals[b] = {"p_greater": float(p_greater), "p_two_sided": float(p_two),
                       "U": float(stat)}
        print(f"  {b:12s} vs {baseline_key}: "
              f"p(greater)={p_greater:.3e}  p(two-sided)={p_two:.3e}")

    # Spearman among classical
    cls_mask = np.array([
        is_classical[i] and not np.isnan(losses[i]) and bfs.get(names[i]) is not None
        for i in range(N)
    ])
    cls_depths = np.array([bfs[names[i]] for i in range(N) if cls_mask[i]])
    cls_losses = losses[cls_mask]
    rho, p_rho = spearmanr(cls_depths, -cls_losses)
    print(f"\nSpearman(depth, -loss) over classical proofs: "
          f"rho={rho:.3f}, p={p_rho:.3e}, n={len(cls_depths)}")

    # Length control via log-length residualization
    print("\n=== Length-controlled (residualize vs log1p(length) on constructive train) ===")
    from sklearn.linear_model import LinearRegression
    fit_idx = np.array([i for i in train_idx if not np.isnan(losses[i])])
    X_fit = np.log1p(lengths[fit_idx]).reshape(-1, 1)
    y_fit = losses[fit_idx]
    reg = LinearRegression().fit(X_fit, y_fit)
    print(f"  Fit on constructive train (n={len(fit_idx)}): "
          f"slope={reg.coef_[0]:.4f}, intercept={reg.intercept_:.4f}")
    predicted = reg.predict(np.log1p(lengths).reshape(-1, 1))
    residual = losses - predicted

    lc_summary = {}
    print("\n  Residualized loss by bucket:")
    for b in bucket_order:
        idx = buckets_idx[b]
        if len(idx) == 0:
            continue
        r = residual[np.array(idx)]
        r = r[~np.isnan(r)]
        mean = r.mean()
        sem = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
        print(f"    {b:24s}  n={len(r):6d}  mean_res={mean:+.4f}  sem={sem:.4f}")
        lc_summary[b] = {
            "n": int(len(r)),
            "mean_residual": float(mean),
            "sem_residual": float(sem),
        }

    # Length-controlled p-values
    print("\n  Length-controlled Mann-Whitney (residuals):")
    baseline_res = residual[np.array(buckets_idx[baseline_key])]
    baseline_res = baseline_res[~np.isnan(baseline_res)]
    lc_pvals = {}
    for b in ["depth 2", "depth 3", "depth 4",
              "depth 5-6", "depth 7-8", "depth 9+"]:
        idx = buckets_idx[b]
        if len(idx) == 0:
            continue
        arr = residual[np.array(idx)]
        arr = arr[~np.isnan(arr)]
        stat, p_g = mannwhitneyu(arr, baseline_res, alternative='greater')
        _, p_t = mannwhitneyu(arr, baseline_res, alternative='two-sided')
        lc_pvals[b] = {"p_greater": float(p_g), "p_two_sided": float(p_t)}
        print(f"    {b:12s}: p(greater)={p_g:.3e}  p(two-sided)={p_t:.3e}")

    # Persist
    out = {
        "buckets": summary,
        "mannwhitney_vs_constructive_test": mw_pvals,
        "spearman_depth_vs_negloss": {
            "rho": float(rho), "p": float(p_rho), "n": int(len(cls_depths)),
        },
        "length_controlled": {
            "fit_n": int(len(fit_idx)),
            "slope": float(reg.coef_[0]),
            "intercept": float(reg.intercept_),
            "buckets": lc_summary,
            "mannwhitney": lc_pvals,
        },
    }
    out_path = out_dir / 'reconstruction_loss_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Also persist per-proof losses for reproducibility / figures
    np.savez_compressed(
        out_dir / 'reconstruction_loss_per_proof.npz',
        losses=losses,
        lengths=lengths,
        is_classical=is_classical,
        names=np.array(names, dtype=object),
    )
    print(f"Saved {out_dir / 'reconstruction_loss_per_proof.npz'}")


if __name__ == '__main__':
    main()
