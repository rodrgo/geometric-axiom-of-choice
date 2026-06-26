"""Aggregate optimal-transport separation of constructive vs classical proof
embeddings (paper Appendix, Table tab:ot_results).

Earth-mover's distance and sliced Wasserstein with a 5000-permutation null,
on a class-balanced 1500-per-class subsample, raw and length-residualized.
"""
import json, time
from pathlib import Path
import numpy as np
import ot
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "data" / "ot"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def sliced_wasserstein(X, Y, n_proj=500, seed=0):
    # POT sliced — fast, high-dim friendly
    return float(ot.sliced.sliced_wasserstein_distance(X, Y, n_projections=n_proj, seed=seed))


def emd2(X, Y):
    M = cdist(X, Y, metric="euclidean")
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    return float(ot.emd2(a, b, M, numItermax=200000))


def perm_test(X, Y, n_perms=500, metric="sliced", verbose=False):
    n = len(X)
    if metric == "sliced":
        obs = sliced_wasserstein(X, Y)
        all_ = np.vstack([X, Y])
        null = []
        for i in range(n_perms):
            idx = rng.permutation(len(all_))
            p = all_[idx]
            null.append(sliced_wasserstein(p[:n], p[n:]))
            if verbose and i % 100 == 0: print(f"  perm {i}/{n_perms}")
        null = np.array(null)
        p = float((np.sum(null >= obs) + 1) / (n_perms + 1))
        return obs, p, null
    else:
        obs = emd2(X, Y)
        all_ = np.vstack([X, Y])
        null = []
        for i in range(n_perms):
            idx = rng.permutation(len(all_))
            p_ = all_[idx]
            null.append(emd2(p_[:n], p_[n:]))
        null = np.array(null)
        p = float((np.sum(null >= obs) + 1) / (n_perms + 1))
        return obs, p, null


N_PERMS = 5000

# ============ Lean — raw ============
print("\n=== Lean (raw) ===")
d2 = np.load(ROOT / "results/data/stage4v3p/embeddings.npz", allow_pickle=True)
emb = d2["embeddings"]; labels = d2["labels"]
constructive = emb[labels == 0]
classical = emb[labels == 1]
n_sub = 1500
C = constructive[rng.choice(len(constructive), n_sub, replace=False)]
Cl = classical[rng.choice(len(classical), n_sub, replace=False)]
t0 = time.time()
emd_lean = emd2(C, Cl)
print(f"  EMD constr vs classical: {emd_lean:.4f} ({time.time()-t0:.1f}s)")
obs_l, p_l, null_l = perm_test(C, Cl, n_perms=N_PERMS, metric="sliced")
print(f"  Sliced W obs={obs_l:.4f}, null mean={null_l.mean():.4f}, p={p_l:.4f}")

# ============ Lean — length-residualized ============
print("\n=== Lean (length-residualized) ===")
with open(ROOT / "results/data/stage4v3p/proofs.json") as f:
    proofs = json.load(f)
lengths = np.array([p["n_invocations"] for p in proofs], dtype=float)
train_idx = d2["train_idx"]
reg = LinearRegression().fit(lengths[train_idx].reshape(-1, 1), emb[train_idx])
emb_r = emb - reg.predict(lengths.reshape(-1, 1))
constr_r = emb_r[labels == 0]; class_r = emb_r[labels == 1]
C2 = constr_r[rng.choice(len(constr_r), n_sub, replace=False)]
Cl2 = class_r[rng.choice(len(class_r), n_sub, replace=False)]
emd_lean_r = emd2(C2, Cl2)
print(f"  EMD residualized: {emd_lean_r:.4f}")
obs_lr, p_lr, null_lr = perm_test(C2, Cl2, n_perms=N_PERMS, metric="sliced")
print(f"  Sliced W residualized obs={obs_lr:.4f}, null mean={null_lr.mean():.4f}, p={p_lr:.4f}")

# Save
out = {
    "lean_raw": {"emd": emd_lean, "sliced_obs": obs_l, "sliced_null_mean": float(null_l.mean()),
                 "sliced_null_std": float(null_l.std()), "p_value": p_l, "n_perms": N_PERMS, "n_sample": n_sub},
    "lean_length_residualized": {"emd": emd_lean_r, "sliced_obs": obs_lr, "sliced_null_mean": float(null_lr.mean()),
                                  "sliced_null_std": float(null_lr.std()), "p_value": p_lr,
                                  "n_perms": N_PERMS, "n_sample": n_sub},
}
with open(OUT / "results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT/'results.json'}")
