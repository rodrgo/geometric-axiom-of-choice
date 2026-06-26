"""Revision 2b: Clean reconstruction loss with OOV tokens excluded from targets.

Matches reconstruction_loss.py's pipeline but narrows the mask to
positions whose TARGET is in-vocabulary (i.e., token id >= RESERVED and
!= UNK_ID). UNK_ID already fails the RESERVED test (UNK=4 < 5), so the
baseline already excludes UNK targets. This "clean" variant additionally
excludes positions that even have UNK tokens in their CONTEXT window (to
address the residual concern that UNK neighbours could bias loss on
in-vocab targets).

Two recomputations per proof:
  (a) standard: 10-sample mean masked-token CE, original mask gate.
  (b) clean: positions retained only if the UNK count in the full
      sequence is zero AND the target is proper-vocab.

We compare depth-wise means and compare the depth-gradient between
the two.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.proof_encoder import (  # noqa: E402
    ProofEncoder, encode_proof,
    PAD_ID, CLS_ID, SEP_ID, MASK_ID, UNK_ID, RESERVED,
)
SEED = 0


def compute_losses(model, X, rng, mode, n_samples=10, mask_prob=0.20,
                   batch_size=512, device="cpu"):
    """mode: 'standard' or 'clean'.

    standard: mask at positions with target >= RESERVED (same as training).
    clean:    additionally exclude any sequence containing UNK anywhere.
    """
    N, L = X.shape
    per_sample = np.full((N, n_samples), np.nan, dtype=np.float64)
    non_special = X >= RESERVED
    has_unk = (X == UNK_ID).any(axis=1)  # per-proof

    model.eval()
    t0 = time.time()
    for s in range(n_samples):
        rand = rng.random(size=(N, L))
        sample_mask = (rand < mask_prob) & non_special
        if mode == "clean":
            sample_mask = sample_mask & (~has_unk)[:, None]

        for i in range(0, N, batch_size):
            x_batch = X[i:i+batch_size]
            m_batch = sample_mask[i:i+batch_size]
            target = torch.from_numpy(x_batch).long().to(device)
            mask_t = torch.from_numpy(m_batch).to(device)
            if mask_t.sum() == 0:
                per_sample[i:i+batch_size, s] = np.nan
                continue
            corrupt = target.clone()
            corrupt[mask_t] = MASK_ID
            with torch.no_grad():
                logits = model(corrupt)
            B, Lb, V = logits.shape
            ce = F.cross_entropy(
                logits.reshape(-1, V), target.reshape(-1), reduction="none"
            ).reshape(B, Lb)
            ce = ce * mask_t.float()
            n_masked = mask_t.float().sum(dim=1)
            per_proof = ce.sum(dim=1) / n_masked.clamp(min=1)
            per_proof_np = per_proof.cpu().numpy()
            no_mask = (n_masked == 0).cpu().numpy()
            per_proof_np = np.where(no_mask, np.nan, per_proof_np)
            per_sample[i:i+batch_size, s] = per_proof_np
        print(f"  [{mode}] sample {s+1}/{n_samples} "
              f"({time.time()-t0:.1f}s)", flush=True)
    with np.errstate(invalid="ignore"):
        return np.nanmean(per_sample, axis=1)


def bucket_of(is_classical, d):
    if not is_classical:
        return "constructive (test)"
    if d is None:
        return None
    if d <= 2:
        return "depth 2"
    if d == 3:
        return "depth 3"
    if d == 4:
        return "depth 4"
    if d <= 6:
        return "depth 5-6"
    if d <= 8:
        return "depth 7-8"
    return "depth 9+"


def main():
    device = "cpu"
    data_dir = ROOT / "results/data/stage4v3p"
    out_dir = ROOT / "results/data/reviewer"

    print("Loading proofs + vocab + encoder...")
    with open(data_dir / "proofs.json") as f:
        proofs = json.load(f)
    with open(data_dir / "vocab.json") as f:
        vocab = json.load(f)
    with open(ROOT / "results/data/depth_analysis/bfs_distances_full.json") as f:
        bfs = json.load(f)

    VOCAB_SIZE = len(vocab) + RESERVED
    MAX_LEN = 64
    print(f"  N={len(proofs)}, vocab={VOCAB_SIZE}")

    N = len(proofs)
    X = np.zeros((N, MAX_LEN), dtype=np.int64)
    for i, p in enumerate(proofs):
        X[i] = encode_proof(p["invocation_heads"], vocab, MAX_LEN)
    is_classical = np.array([int(p["is_classical"]) for p in proofs], dtype=bool)
    names = [p["name"] for p in proofs]

    emb = np.load(data_dir / "embeddings.npz")
    test_idx = set(emb["test_idx"].tolist())
    train_idx = set(emb["train_idx"].tolist())

    model = ProofEncoder(vocab_size=VOCAB_SIZE, d_model=128, nhead=4,
                         enc_layers=4, dec_layers=2, max_len=MAX_LEN,
                         dropout=0.1).to(device)
    state = torch.load(data_dir / "encoder.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rng_std = np.random.default_rng(SEED)
    rng_cln = np.random.default_rng(SEED)

    print("\nComputing STANDARD losses...")
    standard = compute_losses(model, X, rng_std, mode="standard",
                               n_samples=10, mask_prob=0.20,
                               batch_size=512, device=device)
    print(f"  NaN: {np.isnan(standard).sum()}")
    print("\nComputing CLEAN (UNK-free) losses...")
    clean = compute_losses(model, X, rng_cln, mode="clean",
                            n_samples=10, mask_prob=0.20,
                            batch_size=512, device=device)
    print(f"  NaN: {np.isnan(clean).sum()}")

    # Per-bucket summaries
    standard_buckets = defaultdict(list)
    clean_buckets = defaultdict(list)
    standard_buckets_idx = defaultdict(list)
    clean_buckets_idx = defaultdict(list)

    for i in range(N):
        d = bfs.get(names[i])
        if is_classical[i]:
            b = bucket_of(True, d)
        else:
            # only held-out test proofs form the constructive baseline
            if i in test_idx:
                b = "constructive (test)"
            else:
                b = None
        if b is None:
            continue
        if not np.isnan(standard[i]):
            standard_buckets[b].append(standard[i])
            standard_buckets_idx[b].append(i)
        if not np.isnan(clean[i]):
            clean_buckets[b].append(clean[i])
            clean_buckets_idx[b].append(i)

    order = ["constructive (test)", "depth 2", "depth 3", "depth 4",
             "depth 5-6", "depth 7-8", "depth 9+"]

    results = {}
    print("\n=== Bucket means (standard vs clean) ===")
    print(f"{'bucket':<22s} {'n_std':>6s} {'mean_std':>10s} "
          f"{'n_cln':>6s} {'mean_cln':>10s} {'delta':>8s}")
    baseline_std = np.mean(standard_buckets["constructive (test)"])
    baseline_cln = np.mean(clean_buckets["constructive (test)"])
    for b in order:
        s = np.array(standard_buckets[b])
        c = np.array(clean_buckets[b])
        if len(s) == 0:
            continue
        ms = float(s.mean())
        mc = float(c.mean()) if len(c) else float("nan")
        delta = ms - mc
        rel_vs_baseline_std = (ms - baseline_std) / baseline_std
        rel_vs_baseline_cln = (mc - baseline_cln) / baseline_cln if len(c) else float("nan")
        results[b] = {
            "n_standard": int(len(s)),
            "mean_standard": ms,
            "n_clean": int(len(c)),
            "mean_clean": mc,
            "delta_standard_minus_clean": float(delta),
            "pct_above_baseline_standard": float(rel_vs_baseline_std),
            "pct_above_baseline_clean": float(rel_vs_baseline_cln),
        }
        print(f"{b:<22s} {len(s):>6d} {ms:>10.4f} "
              f"{len(c):>6d} {mc:>10.4f} {delta:>+8.4f}")

    # Decision criteria (per plan)
    print("\n=== Depth-2 survival of OOV-clean filter ===")
    d2 = results.get("depth 2", {})
    if d2:
        std_gap = d2["pct_above_baseline_standard"] * 100
        cln_gap = d2["pct_above_baseline_clean"] * 100
        print(f"  Standard: depth-2 loss is {std_gap:+.1f}% vs constructive-test baseline")
        print(f"  Clean   : depth-2 loss is {cln_gap:+.1f}% vs constructive-test baseline")
        if cln_gap >= 15:
            print("  -> CLEAN >= +15%: reconstruction gap is real, report BOTH.")
        elif cln_gap >= 5:
            print("  -> CLEAN >= +5%: attenuated gap, report both and discuss.")
        else:
            print("  -> CLEAN < +5%: reconstruction gap is largely OOV-related.")

    out = {
        "mode_notes": {
            "standard": "Same mask gate as training: target id >= RESERVED. UNK(=4) excluded; classical markers in context are fine.",
            "clean": "Standard gate AND the entire proof sequence contains no UNK tokens (no context contamination)",
        },
        "baseline_standard_mean": float(baseline_std),
        "baseline_clean_mean": float(baseline_cln),
        "buckets": results,
    }
    out_path = out_dir / "oov_clean_reconstruction_loss.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    np.savez_compressed(
        out_dir / "oov_clean_per_proof.npz",
        names=np.array(names, dtype=object),
        is_classical=is_classical,
        standard=standard,
        clean=clean,
    )
    print("Saved per-proof losses to results/data/reviewer/oov_clean_per_proof.npz")


if __name__ == "__main__":
    main()
