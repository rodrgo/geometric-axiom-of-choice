"""Within-domain depth-stratified k-NN AUC.

For each major Mathlib domain with enough constructive support, restrict
the corpus to that domain and recompute depth-stratified k-NN AUC.

If the gradient persists within a single domain, the deep-bucket
full-source signal is structural. If it collapses, the gradient under
the all-domains analysis is partly topic clustering (classical proofs
are concentrated in MeasureTheory/Analysis/Probability) and the
full-source encoder's deep-bucket AUC is partly a topic-detection
effect that the head-only encoder correctly avoids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EMB_ROOT = HERE.parent / "encoder/embeddings"
PARQUET = HERE.parent / "data/full_source.parquet"

DEPTH_BUCKETS = [
    ("depth_2", lambda d: d == 2),
    ("depth_3", lambda d: d == 3),
    ("depth_4_6", lambda d: 4 <= d <= 6),
    ("depth_7_8", lambda d: 7 <= d <= 8),
    ("depth_9_plus", lambda d: d >= 9),
]

# Domains with enough constructive support for stable within-domain
# k-NN. Threshold: at least 200 constructive-train proofs.
TARGET_DOMAINS = ["Algebra", "Topology", "RingTheory", "CategoryTheory",
                  "LinearAlgebra", "Analysis"]

K = 5
MIN_BUCKET_N = 30


def auc_for(scores_ctrl: np.ndarray, scores_bkt: np.ndarray) -> float:
    if len(scores_bkt) < MIN_BUCKET_N or len(scores_ctrl) < MIN_BUCKET_N:
        return float("nan")
    y = np.concatenate([np.zeros(len(scores_ctrl)),
                        np.ones(len(scores_bkt))])
    s = np.concatenate([scores_ctrl, scores_bkt])
    return float(roc_auc_score(y, s))


def run(seed: int, variant: str) -> dict:
    emb = np.load(EMB_ROOT / f"seed_{seed}/{variant}.npz", allow_pickle=True)
    X_all = emb["projected"].astype(np.float64)
    splits = emb["splits"]
    is_cls = emb["is_classical"]
    depth = emb["depth"]
    finite = np.isfinite(X_all).all(axis=1)

    # Align with parquet domain column (same row order — verified by
    # construction in extract + embed).
    domains = np.asarray(
        pq.read_table(PARQUET, columns=["domain"])["domain"].to_pylist())

    out: dict[str, dict] = {}
    for dom in TARGET_DOMAINS:
        in_dom = (domains == dom)

        train_m = in_dom & (splits == "train") & (~is_cls) & finite
        ctrl_m = (in_dom & ((splits == "test") | (splits == "val"))
                  & (~is_cls) & finite)
        if train_m.sum() < 100 or ctrl_m.sum() < 30:
            out[dom] = {"error": "insufficient constructive support",
                        "n_train": int(train_m.sum()),
                        "n_ctrl": int(ctrl_m.sum())}
            continue

        scaler = StandardScaler().fit(X_all[train_m])
        Xz = scaler.transform(X_all)
        nn = NearestNeighbors(n_neighbors=K).fit(Xz[train_m])
        s_ctrl = nn.kneighbors(Xz[ctrl_m])[0].mean(axis=1)

        per_bucket: dict[str, dict] = {}
        for bname, pred in DEPTH_BUCKETS:
            bm = (in_dom & is_cls & finite
                  & np.array([pred(int(x)) for x in depth]))
            if bm.sum() < MIN_BUCKET_N:
                per_bucket[bname] = {"auc": float("nan"),
                                     "n_bucket": int(bm.sum())}
                continue
            s_b = nn.kneighbors(Xz[bm])[0].mean(axis=1)
            per_bucket[bname] = {
                "auc": auc_for(s_ctrl, s_b),
                "n_bucket": int(bm.sum()),
            }
        out[dom] = {
            "n_train": int(train_m.sum()),
            "n_ctrl": int(ctrl_m.sum()),
            "per_bucket": per_bucket,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variant", default="stripped")
    ap.add_argument("--out", default=str(HERE / "within_domain_auc.json"))
    a = ap.parse_args()

    all_results = []
    for seed in a.seeds:
        all_results.append({
            "seed": seed,
            "variant": a.variant,
            "domains": run(seed, a.variant),
        })
        print(f"seed {seed}:")
        for dom in TARGET_DOMAINS:
            d = all_results[-1]["domains"].get(dom, {})
            if "error" in d:
                print(f"  {dom:18s}  skipped: {d['error']}")
                continue
            line = f"  {dom:18s}  n_tr={d['n_train']:4d}  "
            for bname, _ in DEPTH_BUCKETS:
                v = d["per_bucket"][bname]
                n = v["n_bucket"]
                a_ = v["auc"]
                if np.isnan(a_):
                    line += f"  {bname[:7]}=---({n:>3d})"
                else:
                    line += f"  {bname[:7]}={a_:.3f}({n:>3d})"
            print(line)

    out = Path(a.out)
    out.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nwrote {out}")

    # Aggregate: per (domain, bucket), median over seeds.
    print(f"\n=== median over seeds, variant={a.variant} ===")
    print(f"{'domain':18s}  {'depth_2':>9s} {'depth_3':>9s} "
          f"{'d_4_6':>9s} {'d_7_8':>9s} {'d_9+':>9s}")
    for dom in TARGET_DOMAINS:
        line = f"{dom:18s} "
        for bname, _ in DEPTH_BUCKETS:
            vals = [r["domains"].get(dom, {}).get("per_bucket", {})
                    .get(bname, {}).get("auc", float("nan"))
                    for r in all_results]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                line += f"  {np.median(vals):>7.3f}"
            else:
                line += f"  {'---':>7s}"
        print(line)


if __name__ == "__main__":
    main()
