"""Step 4.3: superlevel-set containment.

Fit Gaussian KDE on constructive-train embeddings (after PCA to 90%
variance). Calibrate threshold t_q to the q-th percentile of held-out
constructive density. Report containment fraction by depth bucket for
q in {80, 90, 95}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
EMB_ROOT = HERE.parent / "encoder/embeddings"

QUANTILES = [80, 90, 95]
DEPTH_BUCKETS = [
    ("depth_2", lambda d: d == 2),
    ("depth_3", lambda d: d == 3),
    ("depth_4_6", lambda d: 4 <= d <= 6),
    ("depth_7_8", lambda d: 7 <= d <= 8),
    ("depth_9_plus", lambda d: d >= 9),
]


def run_one(seed: int, variant: str) -> dict:
    emb_path = EMB_ROOT / f"seed_{seed}/{variant}.npz"
    d = np.load(emb_path, allow_pickle=True)
    X_all = d["projected"].astype(np.float64)
    splits = d["splits"]
    is_classical = d["is_classical"]
    depth = d["depth"]

    finite_row = np.isfinite(X_all).all(axis=1)
    train_mask = (splits == "train") & (~is_classical) & finite_row
    test_constr_mask = (((splits == "test") | (splits == "val"))
                        & (~is_classical) & finite_row)

    # Zero out NaN rows so scaler/PCA don't choke. These rows are
    # excluded by bucket masks anyway.
    X_all = np.where(finite_row[:, None], X_all, 0.0)
    scaler = StandardScaler().fit(X_all[train_mask])
    Xz = scaler.transform(X_all)

    pca = PCA(n_components=0.90, random_state=42).fit(Xz[train_mask])
    Xp = pca.transform(Xz)
    print(f"  seed={seed} variant={variant}  pca_dim={pca.n_components_}",
          flush=True)

    bw = np.std(Xp[train_mask]) * (
        len(Xp[train_mask]) ** (-1.0 / (pca.n_components_ + 4)))
    kde = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde.fit(Xp[train_mask])

    log_p_ctrl = kde.score_samples(Xp[test_constr_mask])

    out = {"seed": seed, "variant": variant,
           "pca_dim": int(pca.n_components_), "bandwidth": float(bw),
           "n_train": int(train_mask.sum()),
           "n_control": int(test_constr_mask.sum()),
           "buckets": {}}

    for q in QUANTILES:
        t_q = float(np.percentile(log_p_ctrl, 100 - q))
        per_bucket = {}
        for bname, pred in DEPTH_BUCKETS:
            bm = (is_classical & finite_row
                  & np.array([pred(int(x)) for x in depth]))
            X_b = Xp[bm]
            if len(X_b) < 10:
                per_bucket[bname] = {"n": int(len(X_b)),
                                     "fraction_outside": float("nan")}
                continue
            log_p_b = kde.score_samples(X_b)
            frac_outside = float((log_p_b < t_q).mean())
            per_bucket[bname] = {
                "n": int(len(X_b)),
                "fraction_outside": frac_outside,
            }
        # Constructive control's "outside" rate is by construction = (100-q)/100.
        out["buckets"][f"q{q}"] = {
            "threshold_log_p": t_q,
            "constructive_outside_rate": (100 - q) / 100,
            "per_bucket": per_bucket,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+",
                    default=["raw", "stripped", "combined"])
    ap.add_argument("--out", default=str(HERE / "superlevel_containment.json"))
    a = ap.parse_args()

    t0 = time.time()
    runs = []
    for seed in a.seeds:
        for variant in a.variants:
            try:
                runs.append(run_one(seed, variant))
                for q in QUANTILES:
                    fracs = ", ".join(
                        f"{b}={runs[-1]['buckets'][f'q{q}']['per_bucket'][b]['fraction_outside']:.2f}"
                        for b, _ in DEPTH_BUCKETS
                        if not np.isnan(runs[-1]['buckets'][f'q{q}']['per_bucket'][b]['fraction_outside']))
                    print(f"  seed={seed} variant={variant} q={q}: {fracs}",
                          flush=True)
            except FileNotFoundError as e:
                print(f"  skip seed={seed} variant={variant}: {e}",
                      flush=True)

    out = Path(a.out)
    out.write_text(json.dumps({"quantiles": QUANTILES, "runs": runs},
                              indent=2) + "\n")
    print(f"\nwrote {out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
