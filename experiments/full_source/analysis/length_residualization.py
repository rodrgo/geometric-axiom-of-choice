"""Step 5.2: length-residualized depth-stratified AUC.

For each (seed, variant):
  1. Fit linear regression of each embedding dim on log(n_tokens) using
     constructive-train. Subtract prediction → residual embedding.
  2. Optionally standardize residuals.
  3. Recompute k-NN depth-stratified AUC on residuals.

This isolates the contribution of length to the depth signal. Full-source
proofs are longer than head-only sequences, so length residualization
matters here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EMB_ROOT = HERE.parent / "encoder/embeddings"
PARQUET = HERE.parent / "data/full_source.parquet"

KNN_K = 5
DEPTH_BUCKETS = [
    ("depth_2", lambda d: d == 2),
    ("depth_3", lambda d: d == 3),
    ("depth_4_6", lambda d: 4 <= d <= 6),
    ("depth_7_8", lambda d: 7 <= d <= 8),
    ("depth_9_plus", lambda d: d >= 9),
]


def auc_for_bucket(s_ctrl: np.ndarray, s_bkt: np.ndarray) -> float:
    if len(s_bkt) < 10 or len(s_ctrl) < 10:
        return float("nan")
    y = np.concatenate([np.zeros(len(s_ctrl)), np.ones(len(s_bkt))])
    s = np.concatenate([s_ctrl, s_bkt])
    return float(roc_auc_score(y, s))


def run_one(seed: int, variant: str, standardize: bool) -> dict:
    emb_path = EMB_ROOT / f"seed_{seed}/{variant}.npz"
    d = np.load(emb_path, allow_pickle=True)
    X_all = d["projected"].astype(np.float64)
    splits = d["splits"]
    is_classical = d["is_classical"]
    depth = d["depth"]

    t = pq.read_table(PARQUET, columns=["n_tokens"])
    n_tokens = np.asarray(t["n_tokens"].to_pylist(), dtype=np.float64)
    log_len = np.log1p(n_tokens).reshape(-1, 1)

    finite_row = np.isfinite(X_all).all(axis=1)
    train_mask = (splits == "train") & (~is_classical) & finite_row
    test_constr_mask = (((splits == "test") | (splits == "val"))
                        & (~is_classical) & finite_row)

    # Fit length regression on constructive-train.
    reg = LinearRegression()
    reg.fit(log_len[train_mask], X_all[train_mask])
    residuals = X_all - reg.predict(log_len)

    if standardize:
        # Standardize using constructive-train residual std.
        scaler = StandardScaler().fit(residuals[train_mask])
        Xr = scaler.transform(residuals)
    else:
        Xr = residuals

    nn = NearestNeighbors(n_neighbors=KNN_K).fit(Xr[train_mask])
    s_ctrl = nn.kneighbors(Xr[test_constr_mask])[0].mean(axis=1)

    per_bucket = {}
    for bname, pred in DEPTH_BUCKETS:
        bm = (is_classical & finite_row
              & np.array([pred(int(x)) for x in depth]))
        X_b = Xr[bm]
        if len(X_b) < 10:
            per_bucket[bname] = {"auc": float("nan"),
                                 "n_bucket": int(len(X_b))}
            continue
        s_b = nn.kneighbors(X_b)[0].mean(axis=1)
        per_bucket[bname] = {
            "auc": auc_for_bucket(s_ctrl, s_b),
            "n_bucket": int(len(X_b)),
        }
    return {
        "seed": seed,
        "variant": variant,
        "standardized": standardize,
        "n_train": int(train_mask.sum()),
        "n_control": int(test_constr_mask.sum()),
        "results": per_bucket,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+",
                    default=["raw", "stripped", "combined"])
    ap.add_argument("--standardize", action="store_true", default=True,
                    help="Standardize residuals after subtracting length fit.")
    ap.add_argument("--out", default=str(HERE / "length_residualized_auc.json"))
    a = ap.parse_args()

    t0 = time.time()
    runs = []
    for seed in a.seeds:
        for variant in a.variants:
            try:
                runs.append(run_one(seed, variant, a.standardize))
                aucs = ", ".join(
                    f"{b}={runs[-1]['results'][b]['auc']:.3f}"
                    for b, _ in DEPTH_BUCKETS
                    if not np.isnan(runs[-1]['results'][b]['auc']))
                print(f"seed={seed} variant={variant}  {aucs}", flush=True)
            except FileNotFoundError as e:
                print(f"  skip seed={seed} variant={variant}: {e}",
                      flush=True)

    out = Path(a.out)
    out.write_text(json.dumps({
        "knn_k": KNN_K,
        "standardize": a.standardize,
        "buckets": [b for b, _ in DEPTH_BUCKETS],
        "runs": runs,
    }, indent=2) + "\n")
    print(f"\nwrote {out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
