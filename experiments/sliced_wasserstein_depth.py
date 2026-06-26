"""Stratification Analysis 3: Depth-stratified sliced Wasserstein + perm test."""
import json, time
from pathlib import Path
import numpy as np
import ot

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "data" / "depth_analysis"

emb_data = np.load(ROOT / "results/data/stage4v3p/embeddings.npz", allow_pickle=True)
emb = emb_data["embeddings"]; labels = emb_data["labels"]
proofs = json.load(open(ROOT / "results/data/stage4v3p/proofs.json"))
distances = json.load(open(OUT / "bfs_distances_full.json"))
depths = np.array(
    [distances.get(p["name"], -1) if p["is_classical"] else -1 for p in proofs], dtype=int
)

BUCKETS = [
    ("depth 2",   2, 2),
    ("depth 3",   3, 3),
    ("depth 4",   4, 4),
    ("depth 5-6", 5, 6),
    ("depth 7-8", 7, 8),
    ("depth 9+",  9, 999),
]
N_SAMPLE = 800   # per class
N_PERMS = 1000
N_PROJ = 500

rng = np.random.default_rng(0)
constr_emb = emb[labels == 0]
# subsample constructive to N_SAMPLE (drawn once — shared across buckets)
C = constr_emb[rng.choice(len(constr_emb), N_SAMPLE, replace=False)]

results = []
for label, lo, hi in BUCKETS:
    mask = (depths >= lo) & (depths <= hi)
    n = int(mask.sum())
    if n < 50:
        results.append({"bucket": label, "n_bucket": n, "skipped": True})
        continue
    B_emb = emb[mask]
    n_s = min(N_SAMPLE, n)
    B = B_emb[rng.choice(n, n_s, replace=False)]
    C_b = C[:n_s]

    t0 = time.time()
    obs = float(ot.sliced.sliced_wasserstein_distance(C_b, B, n_projections=N_PROJ, seed=0))
    # perm test
    combined = np.vstack([C_b, B])
    null = []
    for i in range(N_PERMS):
        p_idx = rng.permutation(len(combined))
        p = combined[p_idx]
        null.append(ot.sliced.sliced_wasserstein_distance(p[:n_s], p[n_s:],
                                                          n_projections=N_PROJ, seed=0))
    null = np.array(null)
    p_val = float((np.sum(null >= obs) + 1) / (N_PERMS + 1))
    z = (obs - null.mean()) / (null.std() + 1e-12)
    results.append({"bucket": label, "depth_range": [lo, hi], "n_bucket": n,
                    "n_used": n_s, "sliced_W_obs": obs,
                    "null_mean": float(null.mean()), "null_std": float(null.std()),
                    "z": float(z), "p_value": p_val,
                    "elapsed_s": time.time() - t0})
    print(f"  {label:<10s} n={n_s:>4d} W={obs:.4f}  null={null.mean():.4f}±{null.std():.4f}  "
          f"z={z:.1f}  p={p_val:.4f}  ({time.time()-t0:.1f}s)")

with open(OUT / "depth_stratified_ot.json", "w") as f:
    json.dump({"n_sample_per_class": N_SAMPLE, "n_perms": N_PERMS,
               "n_projections": N_PROJ, "results": results}, f, indent=2)
print(f"\nSaved: {OUT/'depth_stratified_ot.json'}")
