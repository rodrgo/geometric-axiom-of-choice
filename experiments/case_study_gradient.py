"""Gradient figure: one representative classical theorem per depth bucket.

For each of five depth buckets (depth 2, 3, 5-6, 7-8, 9+) we:
  1. Take all `is_classical=True` proofs from `stage4v3p/proofs.json` whose
     `invocation_heads` falls in [3, MAX_LEN-2] tokens.
  2. Sample SAMPLES_PER_BUCKET of them with a fixed seed.
  3. Compute per-token reconstruction loss (Signal B: mask one position,
     score CE loss at that position under the frozen denoising encoder).
  4. Pick the proof whose SUM of per-token losses is closest to the
     bucket median; tie-break by proof length closest to the bucket
     median length.

This is a "median saliency, median length" representative -- neither
cherry-picked for maximum visual punch nor randomly drawn. A reviewer
can audit it: the criterion is explicit and the seed is fixed.

We deliberately do NOT filter by CLASSICAL_MARKER presence in either
direction. The point of this figure is to show what the encoder does on
the typical proof in each bucket -- including showing that depth-9+
"typical" proofs read as ordinary constructive math whose
classicalness lives only on non-token (kernel-graph) markers.

Output:
    results/data/stage4v3p/case_study_gradient.json

This JSON is consumed by hero_figure.py to render panel (b) of the hero
figure (per-token saliency by depth).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    STAGE4V3P_DIR, DEPTH_ANALYSIS_DIR,
)
from lean.proof_encoder import (  # noqa: E402
    ProofEncoder, RESERVED,
)
from lean.saliency import signal_b_for_proof  # noqa: E402

MAX_LEN = 64
SAMPLES_PER_BUCKET = 500
SEED = 1

# Deterministic per-bucket offsets so the draw is reproducible across
# Python processes (Python's built-in hash() is randomized per process).
BUCKET_SEED_OFFSET = {
    "two": 11, "three": 23, "four": 37, "five": 53, "six": 71,
    "seven": 89, "nine": 97,
}

# Bucket definitions: (id, label, predicate-on-depth).
# Five rows spanning the detectable range (depths 2-6). Deeper buckets
# (7-8, 9+) are uniformly at-or-below the constructive baseline and don't
# add information to a per-proof figure; the aggregate depth-stratified
# table elsewhere in the paper carries them.
BUCKETS = [
    ("two",   "Depth 2", lambda d: d <= 2),
    ("three", "Depth 3", lambda d: d == 3),
    ("four",  "Depth 4", lambda d: d == 4),
    ("five",  "Depth 5", lambda d: d == 5),
    ("six",   "Depth 6", lambda d: d == 6),
]


def load_inputs():
    print("Loading proofs/encoder/embeddings/bfs...", flush=True)
    proofs = json.loads((STAGE4V3P_DIR / "proofs.json").read_text())
    vocab = json.loads((STAGE4V3P_DIR / "vocab.json").read_text())
    bfs = json.loads((DEPTH_ANALYSIS_DIR / "bfs_distances_full.json").read_text())

    emb_data = np.load(STAGE4V3P_DIR / "embeddings.npz")
    train_idx = emb_data["train_idx"]
    test_idx = emb_data["test_idx"]

    vocab_size = len(vocab) + RESERVED
    model = ProofEncoder(
        vocab_size=vocab_size, d_model=128, nhead=4,
        enc_layers=4, dec_layers=2, max_len=MAX_LEN, dropout=0.1,
    )
    state = torch.load(STAGE4V3P_DIR / "encoder.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return proofs, vocab, bfs, train_idx, test_idx, model


def constructive_baseline(
    proofs: list[dict], vocab: dict, model: ProofEncoder,
    test_idx: np.ndarray, n_sample: int = 300,
) -> dict:
    """Per-token Signal B distribution on a sample of held-out constructive
    proofs. Used to set the white/red threshold (p90)."""
    rng = np.random.default_rng(SEED)
    cons_test = [i for i in test_idx.tolist() if not proofs[i]["is_classical"]]
    sample = rng.choice(cons_test, size=min(n_sample, len(cons_test)), replace=False)
    all_losses: list[float] = []
    for i in sample:
        heads = proofs[int(i)]["invocation_heads"]
        if not (1 <= len(heads) <= MAX_LEN - 2):
            continue
        b = signal_b_for_proof(heads, vocab, model, MAX_LEN)
        all_losses.extend(b.tolist())
    a = np.asarray(all_losses)
    return {
        "n_proofs_sampled": int(len(sample)),
        "n_tokens": int(len(a)),
        "median_signal_B": float(np.median(a)),
        "p90_signal_B": float(np.percentile(a, 90)),
        "p99_signal_B": float(np.percentile(a, 99)),
    }


def bucket_proofs(proofs: list[dict], bfs: dict, pred) -> list[int]:
    out = []
    for i, p in enumerate(proofs):
        if not p.get("is_classical"):
            continue
        d = bfs.get(p["name"])
        if d is None or not pred(d):
            continue
        n = len(p["invocation_heads"])
        if not (3 <= n <= MAX_LEN - 2):
            continue
        out.append(i)
    return out


def select_for_bucket(
    indices: list[int], proofs: list[dict], vocab: dict, model: ProofEncoder,
    bfs: dict, seed: int,
) -> tuple[dict, dict]:
    """Sample SAMPLES_PER_BUCKET indices from the bucket, compute summed
    Signal B per proof, return the proof closest to the bucket median
    (tie-broken by length closest to bucket median length)."""
    rng = np.random.default_rng(seed)
    n = len(indices)
    take = min(SAMPLES_PER_BUCKET, n)
    sampled = rng.choice(indices, size=take, replace=False).tolist()

    records = []
    for idx in sampled:
        p = proofs[int(idx)]
        heads = p["invocation_heads"]
        b = signal_b_for_proof(heads, vocab, model, MAX_LEN)
        records.append({
            "proof_idx": int(idx),
            "name": p["name"],
            "file_path": p["file_path"],
            "depth": bfs[p["name"]],
            "invocation_heads": heads,
            "signal_B": b.tolist(),
            "sum_B": float(b.sum()),
            "len": int(len(heads)),
        })
    sums = np.array([r["sum_B"] for r in records])
    lens = np.array([r["len"] for r in records])
    median_sum = float(np.median(sums))
    median_len = float(np.median(lens))

    def score(r):
        # Primary: distance to median summed Signal B.
        # Secondary tie-break: distance to median length.
        # Tertiary stability: proof_idx.
        return (abs(r["sum_B"] - median_sum),
                abs(r["len"] - median_len),
                r["proof_idx"])

    records.sort(key=score)
    chosen = records[0]
    summary = {
        "n_in_bucket_total": int(n),
        "n_sampled": int(take),
        "median_sum_B": median_sum,
        "median_length": median_len,
    }
    return chosen, summary


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def main():
    t0 = time.time()
    proofs, vocab, bfs, train_idx, test_idx, model = load_inputs()
    print(f"  [{time.time()-t0:.1f}s]")

    print("Computing constructive baseline...", flush=True)
    baseline = constructive_baseline(proofs, vocab, model, test_idx)
    print(f"  baseline: median_B={baseline['median_signal_B']:.3f}  "
          f"p90_B={baseline['p90_signal_B']:.3f}  "
          f"[{time.time()-t0:.1f}s]")

    print()
    selected: list[dict] = []
    for bucket_id, bucket_label, pred in BUCKETS:
        idxs = bucket_proofs(proofs, bfs, pred)
        if not idxs:
            print(f"  {bucket_label}: empty bucket")
            continue
        rec, summary = select_for_bucket(idxs, proofs, vocab, model, bfs,
                                          seed=SEED + BUCKET_SEED_OFFSET[bucket_id])
        rec["bucket_id"] = bucket_id
        rec["bucket_label"] = bucket_label
        rec["bucket_summary"] = summary
        selected.append(rec)
        print(f"  {bucket_label}: chose {rec['name'][:50]:<50s}  "
              f"d={rec['depth']:<3d} len={rec['len']:<3d} "
              f"sum_B={rec['sum_B']:.2f}  "
              f"(bucket median sum_B={summary['median_sum_B']:.2f}, "
              f"median len={summary['median_length']:.1f}, "
              f"n={summary['n_sampled']}/{summary['n_in_bucket_total']})  "
              f"[{time.time()-t0:.1f}s]")
        b_str = " ".join(f"{b:.1f}" for b in rec["signal_B"])
        print(f"      heads:   {rec['invocation_heads']}")
        print(f"      Signal_B {b_str}")

    out_json = STAGE4V3P_DIR / "case_study_gradient.json"
    out_json.write_text(json.dumps({
        "constructive_baseline": baseline,
        "selection_criterion": ("Per bucket: sample SAMPLES_PER_BUCKET classical "
                                "proofs of length 3-62, compute per-token Signal B, "
                                "pick the proof whose summed Signal B is closest to "
                                "the bucket median; tie-break by length closest to "
                                "bucket median length."),
        "samples_per_bucket": SAMPLES_PER_BUCKET,
        "seed": SEED,
        "buckets": selected,
    }, indent=2))
    print(f"\nSaved {out_json.relative_to(ROOT)}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
