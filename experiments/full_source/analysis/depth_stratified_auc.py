"""Step 4.1: depth-stratified AUC under k-NN, KDE, Isolation Forest.

For each (seed, variant) pair:
  1. Standardize embeddings (z-score on constructive train).
  2. Fit detectors on constructive-train, scored on:
       a) constructive-test (control distribution)
       b) classical proofs grouped by depth bucket
  3. AUC per depth bucket = ROC(constructive-test vs classical-bucket).

Variants:
  raw       = full tokens, no stripping
  stripped  = atomic strip (headline robustness)
  combined  = combined strip (aggressive)

Output: full_source/analysis/depth_stratified_auc.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EMB_ROOT = HERE.parent / "encoder/embeddings"

DEPTH_BUCKETS = [
    ("depth_2", lambda d: d == 2),
    ("depth_3", lambda d: d == 3),
    ("depth_4_6", lambda d: 4 <= d <= 6),
    ("depth_7_8", lambda d: 7 <= d <= 8),
    ("depth_9_plus", lambda d: d >= 9),
]

KNN_K = 5
ISO_TREES = 200
ISO_SEED = 42


def fit_score(X_train: np.ndarray, X_query: np.ndarray, kind: str
              ) -> np.ndarray:
    """Higher score = more anomalous (further from constructive support)."""
    if kind == "knn":
        nn = NearestNeighbors(n_neighbors=KNN_K)
        nn.fit(X_train)
        d, _ = nn.kneighbors(X_query)
        return d.mean(axis=1)
    elif kind == "iso":
        m = IsolationForest(n_estimators=ISO_TREES, random_state=ISO_SEED,
                            n_jobs=-1)
        m.fit(X_train)
        # IsolationForest returns higher = more normal; flip sign.
        return -m.decision_function(X_query)
    elif kind == "kde":
        # KDE with a fixed reasonable bandwidth. CV is overkill at this
        # scale for the headline number; we report kde as a secondary
        # detector anyway.
        m = KernelDensity(kernel="gaussian",
                          bandwidth=np.std(X_train) * 0.5)
        m.fit(X_train)
        # log-density; flip so higher = more anomalous.
        return -m.score_samples(X_query)
    else:
        raise ValueError(kind)


def auc_for_bucket(scores_control: np.ndarray, scores_bucket: np.ndarray
                   ) -> float:
    if len(scores_bucket) < 10 or len(scores_control) < 10:
        return float("nan")
    y_true = np.concatenate([np.zeros(len(scores_control)),
                             np.ones(len(scores_bucket))])
    y_score = np.concatenate([scores_control, scores_bucket])
    return float(roc_auc_score(y_true, y_score))


def run_one(seed: int, variant: str) -> dict:
    emb_path = EMB_ROOT / f"seed_{seed}/{variant}.npz"
    d = np.load(emb_path, allow_pickle=True)
    X_all = d["projected"]
    splits = d["splits"]
    is_classical = d["is_classical"]
    depth = d["depth"]

    # Filter out rows with NaN embeddings (a tiny number of proofs end
    # up with empty stripped-token sequences; pool over an empty mask
    # produces NaN). At most a few rows out of 42K.
    finite_row = np.isfinite(X_all).all(axis=1)
    train_mask = (splits == "train") & (~is_classical) & finite_row
    test_constr_mask = (((splits == "test") | (splits == "val"))
                        & (~is_classical) & finite_row)

    X_train = X_all[train_mask]
    scaler = StandardScaler().fit(X_train)
    X_train_z = scaler.transform(X_train)
    X_all_z = scaler.transform(X_all)

    X_control = X_all_z[test_constr_mask]
    print(f"  seed={seed} variant={variant}  "
          f"n_train={len(X_train)}  n_control={len(X_control)}",
          flush=True)

    results = {}
    for kind in ["knn", "iso", "kde"]:
        s_control = fit_score(X_train_z, X_control, kind)
        per_bucket = {}
        for bname, pred in DEPTH_BUCKETS:
            bucket_mask = (is_classical & finite_row
                           & np.array([pred(int(x)) for x in depth]))
            X_bucket = X_all_z[bucket_mask]
            if len(X_bucket) < 10:
                per_bucket[bname] = {"auc": float("nan"),
                                     "n_bucket": int(len(X_bucket))}
                continue
            s_bucket = fit_score(X_train_z, X_bucket, kind)
            per_bucket[bname] = {
                "auc": auc_for_bucket(s_control, s_bucket),
                "n_bucket": int(len(X_bucket)),
            }
        results[kind] = per_bucket
        aucs = ", ".join(f"{b}={r['auc']:.3f}" for b, r in per_bucket.items()
                          if not np.isnan(r["auc"]))
        print(f"    {kind:4s}: {aucs}", flush=True)
    return {
        "seed": seed,
        "variant": variant,
        "n_train": int(len(X_train)),
        "n_control": int(len(X_control)),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+",
                    default=["raw", "stripped", "combined"])
    ap.add_argument("--out", default=str(HERE / "depth_stratified_auc.json"))
    a = ap.parse_args()

    t0 = time.time()
    all_results = []
    for seed in a.seeds:
        for variant in a.variants:
            try:
                all_results.append(run_one(seed, variant))
            except FileNotFoundError as e:
                print(f"  skip seed={seed} variant={variant}: {e}",
                      flush=True)

    payload = {
        "knn_k": KNN_K,
        "iso_trees": ISO_TREES,
        "iso_seed": ISO_SEED,
        "buckets": [b for b, _ in DEPTH_BUCKETS],
        "runs": all_results,
    }
    out = Path(a.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
