"""Depth-stratified statement-level k-NN AUC on the matched population.

Same protocol as the matched-pairs statement variant but per-depth-bucket:
  - k=5 NearestNeighbors on the in-domain (constructive-train ∩ matched
    population) statement embeddings.
  - Score = mean Euclidean distance to 5 nearest in-domain neighbours.
  - For each depth bucket d, AUC of (constructive holdout vs.
    classical-at-depth-d) using that score.
  - Both raw and length-residualized variants (LinearRegression(emb ~
    statement_length).fit on the in-domain training subset, subtract).

Buckets match depth_stratified_all_methods.json:
    depth 2, 3, 4, 5, 6, 7-8, 9+.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
KNN_K = 5

BUCKETS = [
    ("depth 2", 2, 2),
    ("depth 3", 3, 3),
    ("depth 4", 4, 4),
    ("depth 5", 5, 5),
    ("depth 6", 6, 6),
    ("depth 7-8", 7, 8),
    ("depth 9+", 9, 999),
]


def knn_auc(z_train: np.ndarray, z_neg: np.ndarray, z_pos: np.ndarray,
            k: int = KNN_K) -> float:
    nn = NearestNeighbors(n_neighbors=k).fit(z_train)
    d_neg = nn.kneighbors(z_neg)[0].mean(axis=1)
    d_pos = nn.kneighbors(z_pos)[0].mean(axis=1)
    y = np.concatenate([np.zeros(len(z_neg)), np.ones(len(z_pos))])
    s = np.concatenate([d_neg, d_pos])
    return float(roc_auc_score(y, s))


def length_residualize(emb: np.ndarray, lengths: np.ndarray,
                       train_mask: np.ndarray) -> np.ndarray:
    Xl = lengths.reshape(-1, 1).astype(float)
    reg = LinearRegression().fit(Xl[train_mask], emb[train_mask])
    return emb - reg.predict(Xl)


def main() -> None:
    # ----- Load matched proof-encoder population -----
    with open(DATA / "stage4v3p" / "proofs.json") as f:
        pop = json.load(f)
    pop_names = {p["name"] for p in pop}
    print(f"Proof-encoder population: {len(pop_names):,} unique names")

    # ----- Load v3 statement embeddings -----
    d = np.load(DATA / "stage4v3" / "embeddings.npz")
    E = d["embeddings"]
    y = d["labels"]                # 0 = constructive, 1 = classical
    train_set = set(d["train_idx"].tolist())

    with open(DATA / "stage4v3" / "records.json") as f:
        recs = json.load(f)
    assert len(recs) == len(E)
    names = np.array([r["full_name"] for r in recs])
    lens = np.array([len(r.get("goal", "")) for r in recs], dtype=float)

    # ----- Load depths -----
    with open(DATA / "depth_analysis" / "bfs_distances_full.json") as f:
        depths_map = json.load(f)

    # ----- Restrict to matched population -----
    keep = np.array([n in pop_names for n in names])
    Em = E[keep]
    ym = y[keep]
    lensm = lens[keep]
    namesm = names[keep]
    is_train_m = np.array([(i in train_set) for i in np.where(keep)[0]])
    print(f"v3 statement embeddings overlap with population: {keep.sum():,}")

    train_mask = is_train_m & (ym == 0)
    constr_hold_mask = (~is_train_m) & (ym == 0)
    print(f"  in-domain train (constructive ∩ original train): "
          f"{train_mask.sum():,}")
    print(f"  constructive holdout: {constr_hold_mask.sum():,}")

    depth_of = np.array([depths_map.get(n, -1) for n in namesm], dtype=int)

    # ----- Aggregate AUCs for reference -----
    classical_mask = (ym == 1)
    auc_agg_raw = knn_auc(Em[train_mask], Em[constr_hold_mask],
                          Em[classical_mask])
    Em_res = length_residualize(Em, lensm, train_mask)
    auc_agg_res = knn_auc(Em_res[train_mask], Em_res[constr_hold_mask],
                          Em_res[classical_mask])
    print(f"\nAggregate statement k-NN AUC (raw):              {auc_agg_raw:.4f}")
    print(f"Aggregate statement k-NN AUC (length-residual.): {auc_agg_res:.4f}")

    # ----- Per-bucket AUCs -----
    print("\nDepth-stratified statement k-NN AUC:")
    print(f"  {'bucket':<10s}  {'n_pos':>6}  {'raw':>7}  {'residualized':>14}")
    rows = []
    for label, lo, hi in BUCKETS:
        bmask = classical_mask & (depth_of >= lo) & (depth_of <= hi)
        n_pos = int(bmask.sum())
        if n_pos < 5:
            print(f"  {label:<10s}  {n_pos:>6}  (skip, n<5)")
            rows.append({"bucket": label, "depth_range": [lo, hi],
                         "n_pos": n_pos, "auc_raw": None,
                         "auc_length_residualized": None})
            continue
        auc_r = knn_auc(Em[train_mask], Em[constr_hold_mask], Em[bmask])
        auc_d = knn_auc(Em_res[train_mask], Em_res[constr_hold_mask],
                        Em_res[bmask])
        print(f"  {label:<10s}  {n_pos:>6}  {auc_r:>7.4f}  {auc_d:>14.4f}")
        rows.append({"bucket": label, "depth_range": [lo, hi],
                     "n_pos": n_pos, "auc_raw": auc_r,
                     "auc_length_residualized": auc_d})

    out = {
        "n_train_in_domain": int(train_mask.sum()),
        "n_constr_holdout": int(constr_hold_mask.sum()),
        "aggregate": {
            "auc_raw": auc_agg_raw,
            "auc_length_residualized": auc_agg_res,
        },
        "buckets": rows,
    }
    out_path = DATA / "reviewer" / "statement_knn_depth_stratified.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
