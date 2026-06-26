"""Reviewer revision -- Concern 2 (topic / mathematical-area confound).

The reviewer's sharp point: topic could live in *proof composition* even when
statements look quiet, so the statement-vs-proof asymmetry (Sec 3.4) does not
settle it.  We hold a *proof-independent* topic proxy (the theorem statement)
fixed and measure the residual proof-side gap.

  (2a) Statement-matched pairs: for each classical proof, find the most
       topically-similar *constructive* proof in statement-embedding space
       (phi_stmt cosine, greedy UNIQUE matching above a similarity threshold),
       then compare the two proofs' phi_proof k-NN anomaly scores (Wilcoxon).
       If, holding topic ~fixed, classical proofs are still more anomalous,
       the signal is proof-compositional, not topical.  We sweep the threshold
       so "same topic" is quantified, not assumed.

  (2b) Finer within-subdomain k-NN: re-run within-area boundary detection at
       Mathlib directory level 2 (e.g. Analysis/Complex) instead of ~10 top
       dirs, to show the signal survives at fine topical granularity.

Writes: results/data/reviewer/topic_confound.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/data/reviewer"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data...")
proofs = json.load(open(ROOT / "results/data/stage4v3p/proofs.json"))
pe = np.load(ROOT / "results/data/stage4v3p/embeddings.npz")
emb_p = pe["embeddings"]
train_idx = pe["train_idx"]
labels_p = pe["labels"]
bfs = json.load(open(ROOT / "results/data/depth_analysis/bfs_distances_full.json"))

se = np.load(ROOT / "results/data/stage4v3/embeddings.npz")
emb_s = se["embeddings"]
records = json.load(open(ROOT / "results/data/stage4v3/records.json"))
stmt_row = {r["full_name"]: i for i, r in enumerate(records)}

# Proof anomaly score (paper protocol).
sc = StandardScaler().fit(emb_p[train_idx])
E = sc.transform(emb_p)
anomaly = NearestNeighbors(n_neighbors=5).fit(E[train_idx]).kneighbors(E)[0].mean(axis=1)
# z-score globally so matched-pair gaps are in SD units (comparable to the
# paper's within-file matched pairs: +1.01 SD at depth 2, etc.).
anomaly = (anomaly - anomaly.mean()) / anomaly.std()

# ---------- (2a) statement-matched pairs ----------
# Intersection of the proof population with the statement encoder, carrying
# phi_stmt (for topic matching) and proof anomaly/label/depth.
inter = [(i, p["name"]) for i, p in enumerate(proofs) if p["name"] in stmt_row]
print(f"phi_stmt / phi_proof intersection: {len(inter)} / {len(proofs)} proofs")

idx_p = np.array([i for i, _ in inter])
S = normalize(emb_s[np.array([stmt_row[n] for _, n in inter])])  # L2 for cosine
is_cls = labels_p[idx_p].astype(int)
anom = anomaly[idx_p]
depth = np.array([bfs.get(proofs[i]["name"], -1) if labels_p[i] else -1 for i in idx_p])

cls_pos = np.where(is_cls == 1)[0]
con_pos = np.where(is_cls == 0)[0]
print(f"  classical={len(cls_pos)}  constructive={len(con_pos)}")

# candidate constructive neighbors (cosine) for each classical statement
K = min(40, len(con_pos))
nn_stmt = NearestNeighbors(n_neighbors=K, metric="cosine").fit(S[con_pos])
dist, nbr = nn_stmt.kneighbors(S[cls_pos])   # cosine distance; sim = 1 - dist


def match_pairs(tau):
    """Greedy unique matching: each constructive used at most once."""
    used = np.zeros(len(con_pos), dtype=bool)
    pairs = []
    # process classical in order of best available similarity (closest first)
    order = np.argsort(dist[:, 0])
    for ci in order:
        for j in range(K):
            sim = 1.0 - dist[ci, j]
            if sim < tau:
                break  # neighbors sorted; no better ahead
            cand = nbr[ci, j]
            if not used[cand]:
                used[cand] = True
                pairs.append((cls_pos[ci], con_pos[cand], sim))
                break
    return pairs


sweep = {}
for tau in [0.0, 0.5, 0.7, 0.8, 0.9]:
    pairs = match_pairs(tau)
    if len(pairs) < 20:
        sweep[f"tau_{tau}"] = {"n_pairs": len(pairs), "note": "too few"}
        continue
    diffs = np.array([anom[c] - anom[k] for c, k, _ in pairs])
    sims = np.array([s for _, _, s in pairs])
    _, p = wilcoxon(diffs, alternative="greater")
    sweep[f"tau_{tau}"] = {
        "n_pairs": len(pairs),
        "mean_stmt_cosine": float(sims.mean()),
        "mean_anomaly_diff": float(diffs.mean()),
        "frac_cls_more_anomalous": float((diffs > 0).mean()),
        "p_greater": float(p),
    }
    print(f"  tau={tau}: n={len(pairs):5d}  mean_cos={sims.mean():.3f}  "
          f"mean_diff={diffs.mean():+.4f}  p={p:.2e}")

# per-depth breakdown at a strict topic threshold
TAU_DEPTH = 0.8
pairs = match_pairs(TAU_DEPTH)
pair_depth = np.array([depth[c] for c, _, _ in pairs])
pair_diff = np.array([anom[c] - anom[k] for c, k, _ in pairs])
per_depth = {}
for lbl, lo, hi in [("d2", 2, 2), ("d3-4", 3, 4), ("d5-6", 5, 6), ("d7+", 7, 999),
                    ("d9+", 9, 999)]:
    m = (pair_depth >= lo) & (pair_depth <= hi)
    if m.sum() >= 20:
        _, pv = wilcoxon(pair_diff[m], alternative="greater")
        per_depth[lbl] = {"n": int(m.sum()), "mean_diff": float(pair_diff[m].mean()),
                          "p_greater": float(pv)}
        print(f"  [tau={TAU_DEPTH}] {lbl:5s} n={int(m.sum()):4d}  "
              f"mean_diff={pair_diff[m].mean():+.4f}  p={pv:.2e}")

# ---------- (2b) finer within-subdomain k-NN ----------
def subdomain(fp, levels=2):
    parts = fp.split("/")
    if parts and parts[0] == "Mathlib":
        return "/".join(parts[1:1 + levels])
    return "unknown"


sub = np.array([subdomain(p["file_path"], 2) for p in proofs])
within = {}
for dom in sorted(set(sub)):
    mask = sub == dom
    cm = mask & (labels_p == 0)
    km = mask & (labels_p == 1)
    n_con, n_cls = int(cm.sum()), int(km.sum())
    if n_con < 30 or n_cls < 30:
        continue
    Xc_tr, Xc_te = train_test_split(emb_p[cm], test_size=0.2, random_state=0)
    if len(Xc_tr) < 5:
        continue
    s = StandardScaler().fit(Xc_tr)
    nn = NearestNeighbors(n_neighbors=5).fit(s.transform(Xc_tr))
    dc = nn.kneighbors(s.transform(Xc_te))[0].mean(1)
    dk = nn.kneighbors(s.transform(emb_p[km]))[0].mean(1)
    y = np.concatenate([np.zeros(len(dc)), np.ones(len(dk))])
    within[dom] = {"n_con": n_con, "n_cls": n_cls,
                   "auc_knn_k5": float(roc_auc_score(y, np.concatenate([dc, dk])))}

aucs = np.array([v["auc_knn_k5"] for v in within.values()])
print(f"\nFiner within-subdomain k-NN: {len(within)} subdomains "
      f"(n_con>=30, n_cls>=30)")
print(f"  AUC median={np.median(aucs):.3f}  mean={aucs.mean():.3f}  "
      f"min={aucs.min():.3f}  max={aucs.max():.3f}  "
      f"frac>0.6={np.mean(aucs>0.6):.2f}")
top = sorted(within.items(), key=lambda kv: -(kv[1]["n_con"] + kv[1]["n_cls"]))[:12]
for dom, r in top:
    print(f"  {dom:<28s} n_con={r['n_con']:>4d} n_cls={r['n_cls']:>4d} "
          f"AUC={r['auc_knn_k5']:.3f}")

json.dump({
    "statement_matched_pairs": {
        "intersection_n": len(inter),
        "n_classical": int(len(cls_pos)),
        "n_constructive": int(len(con_pos)),
        "threshold_sweep": sweep,
        "per_depth_at_tau": {"tau": TAU_DEPTH, "buckets": per_depth},
    },
    "within_subdomain_knn_level2": {
        "n_subdomains": len(within),
        "auc_median": float(np.median(aucs)),
        "auc_mean": float(aucs.mean()),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "frac_above_0.6": float(np.mean(aucs > 0.6)),
        "per_subdomain": within,
    },
}, open(OUT / "topic_confound.json", "w"), indent=2)
print(f"\nSaved {OUT/'topic_confound.json'}")
