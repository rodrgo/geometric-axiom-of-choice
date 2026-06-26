"""Stratification Analysis 1: Depth-stratified separation (Steps 1-5).

- BFS distance from Classical.choice per classical declaration (reverse edges).
- Match to proof-embedded theorems.
- k-NN distance AUC per depth bucket.
- Also run for KDE, OCSVM, Isolation Forest on proof embeddings.
- Save JSONs and figures.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KernelDensity, LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.kernel_graph import load_graph, bfs_classical_depths  # noqa: E402

OUT = ROOT / "results" / "data" / "depth_analysis"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)


# ============ Step 1: BFS distances ============
print("Loading dependency graph…")
fwd, rev, _ = load_graph()

# BFS from Classical.choice via reverse edges (to dependents)
distances = bfs_classical_depths(rev, seeds=("Classical.choice",))

dist_counts = Counter(distances.values())
print(f"  total classical decls: {len(distances)}")
for d in sorted(dist_counts)[:15]:
    print(f"    d={d}: {dist_counts[d]}")

with open(OUT / "bfs_distances.json", "w") as f:
    json.dump({"counts": dict(dist_counts), "summary": {"n": len(distances),
               "max": int(max(distances.values()))}}, f, indent=2)

# full map only needed in memory — too big to dump (171K entries is fine actually)
with open(OUT / "bfs_distances_full.json", "w") as f:
    json.dump(distances, f)


# ============ Step 2: Match to proof-embedded theorems ============
print("\nMatching to proof theorems…")
emb_data = np.load(ROOT / "results/data/stage4v3p/embeddings.npz", allow_pickle=True)
emb = emb_data["embeddings"]
labels = emb_data["labels"]
proofs = json.load(open(ROOT / "results/data/stage4v3p/proofs.json"))
assert len(proofs) == len(emb)

# For each theorem, attach classical_depth (None → not classical; otherwise int)
depths = np.full(len(proofs), -1, dtype=int)  # -1 = constructive; 0+ = classical
for i, p in enumerate(proofs):
    if p["is_classical"]:
        d = distances.get(p["name"])
        if d is not None:
            depths[i] = d
        else:
            depths[i] = -2  # classical but not found in BFS (sanity — shouldn't happen)

# Basic counts
n_constr = int((labels == 0).sum())
n_classical = int((labels == 1).sum())
n_cls_matched = int((depths >= 0).sum())
n_cls_missing = int((depths == -2).sum())
cls_depth_vals = depths[depths >= 0]
print(f"  constructive: {n_constr}")
print(f"  classical: {n_classical} (matched to BFS: {n_cls_matched}, missing: {n_cls_missing})")
print(f"  classical depth: min={cls_depth_vals.min()}, median={int(np.median(cls_depth_vals))}, max={cls_depth_vals.max()}")

# ============ Step 3 + 4: AUC by bucket, multiple methods ============
# Scores for each method need to be comparable across thms. Fit each method
# on constructive training data, score ALL thms, then stratify.
train_idx = emb_data["train_idx"]   # constructive only
test_idx = emb_data["test_idx"]     # constructive only

scaler = StandardScaler().fit(emb[train_idx])
E = scaler.transform(emb)
E_train = E[train_idx]

print("\nFitting anomaly methods on constructive train embeddings…")
t0 = time.time()

# k-NN mean distance
nn = NearestNeighbors(n_neighbors=5).fit(E_train)
knn_d, _ = nn.kneighbors(E)
score_knn = knn_d.mean(axis=1)
print(f"  kNN fit+score: {time.time()-t0:.1f}s")

t0 = time.time()
kde = KernelDensity(bandwidth=1.0, kernel="gaussian").fit(E_train)
score_kde = -kde.score_samples(E)  # higher = more anomalous
print(f"  KDE: {time.time()-t0:.1f}s")

t0 = time.time()
oc = OneClassSVM(kernel="rbf", nu=0.2, gamma="scale").fit(E_train)
score_ocsvm = -oc.decision_function(E)
print(f"  OCSVM: {time.time()-t0:.1f}s")

t0 = time.time()
iso = IsolationForest(contamination="auto", random_state=0, n_estimators=200).fit(E_train)
score_iso = -iso.score_samples(E)
print(f"  IsoForest: {time.time()-t0:.1f}s")

# Also hand-crafted stats and BoW feature k-NN
base = np.load(ROOT / "results/data/stage4v3p/baselines.npz", allow_pickle=True)
X_stats = base["X_stats"]; X_bow = base["X_bow"]
sc_s = StandardScaler().fit(X_stats[train_idx])
sc_b = StandardScaler().fit(X_bow[train_idx])
Es = sc_s.transform(X_stats); Eb = sc_b.transform(X_bow)
nn_s = NearestNeighbors(n_neighbors=5).fit(Es[train_idx])
nn_b = NearestNeighbors(n_neighbors=5).fit(Eb[train_idx])
score_stats = nn_s.kneighbors(Es)[0].mean(1)
score_bow = nn_b.kneighbors(Eb)[0].mean(1)
print(f"  stats/bow kNN scored")

methods = {
    "kNN_encoder": score_knn,
    "KDE_encoder": score_kde,
    "OCSVM_encoder": score_ocsvm,
    "IsoForest_encoder": score_iso,
    "kNN_stats": score_stats,
    "kNN_bow": score_bow,
}

# Buckets
BUCKETS = [
    ("depth 2",   2, 2),
    ("depth 3",   3, 3),
    ("depth 4",   4, 4),
    ("depth 5",   5, 5),
    ("depth 6",   6, 6),
    ("depth 7-8", 7, 8),
    ("depth 9+",  9, 999),
]

# Constructive score (all constructive, not just train — use test_idx to avoid
# the train set, which the anomaly method already saw)
# Actually more conservative: use ALL constructive NOT in train as the "null"
constr_mask = labels == 0
constr_eval_mask = constr_mask.copy()
constr_eval_mask[train_idx] = False   # hold out train
n_constr_eval = int(constr_eval_mask.sum())
print(f"\nUsing {n_constr_eval} held-out constructive theorems as null for AUC")

results = {method: [] for method in methods}
for method_name, scores in methods.items():
    constr_scores = scores[constr_eval_mask]
    for label, lo, hi in BUCKETS:
        bucket_mask = (depths >= lo) & (depths <= hi)
        n_b = int(bucket_mask.sum())
        if n_b < 20:
            results[method_name].append({"bucket": label, "depth_range": [lo, hi],
                                         "n_theorems": n_b, "auc": None})
            continue
        bucket_scores = scores[bucket_mask]
        all_s = np.concatenate([constr_scores, bucket_scores])
        all_y = np.concatenate([np.zeros(len(constr_scores)), np.ones(n_b)])
        auc = float(roc_auc_score(all_y, all_s))
        results[method_name].append({"bucket": label, "depth_range": [lo, hi],
                                     "n_theorems": n_b, "auc": auc})

# Print table
print(f"\n{'bucket':<14s}" + "".join(f"{m:>18s}" for m in methods))
for i, (lbl, _, _) in enumerate(BUCKETS):
    row = results["kNN_encoder"][i]
    n = row["n_theorems"]
    line = f"{lbl:<9s} n={n:<4d}"
    for m in methods:
        a = results[m][i]["auc"]
        line += f"{a:>18.3f}" if a is not None else f"{'—':>18s}"
    print(line)

with open(OUT / "depth_stratified_all_methods.json", "w") as f:
    json.dump({"buckets": BUCKETS, "methods": {m: results[m] for m in methods},
               "n_constructive_null": n_constr_eval}, f, indent=2)

# Also save kNN_encoder-only summary
with open(OUT / "depth_stratified_auc.json", "w") as f:
    json.dump(results["kNN_encoder"], f, indent=2)


# ============ Step 5: Visualize ============
fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(BUCKETS))
colors = {"kNN_encoder": "C0", "KDE_encoder": "C1", "OCSVM_encoder": "C2",
          "IsoForest_encoder": "C3", "kNN_stats": "C4", "kNN_bow": "C5"}
for m in methods:
    ys = [r["auc"] for r in results[m]]
    ax.plot(x, ys, "o-", color=colors[m], label=m, lw=1.5)
ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{b[0]}\nn={results['kNN_encoder'][i]['n_theorems']}"
                    for i, b in enumerate(BUCKETS)], fontsize=9)
ax.set_xlabel("BFS distance from Classical.choice")
ax.set_ylabel("AUC (classical bucket vs constructive)")
ax.set_title("Separation Strength by Depth of Classical Dependence")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.45, max(0.9, max((r["auc"] or 0.5) for m in methods for r in results[m]) + 0.03))
plt.tight_layout()
plt.savefig(FIG / "depth_stratified_auc.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"\nSaved figure: {FIG/'depth_stratified_auc.png'}")


# Figure 2: score distributions by depth bucket (kNN encoder)
fig, ax = plt.subplots(figsize=(9, 5))
parts = []
lbls_v = []
# constructive (null)
constr_scores = methods["kNN_encoder"][constr_eval_mask]
parts.append(constr_scores)
lbls_v.append(f"constructive\nn={len(constr_scores)}")
for label, lo, hi in BUCKETS:
    bmask = (depths >= lo) & (depths <= hi)
    if bmask.sum() < 20:
        continue
    parts.append(methods["kNN_encoder"][bmask])
    lbls_v.append(f"{label}\nn={int(bmask.sum())}")
vp = ax.violinplot(parts, showmedians=True, widths=0.85)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor("gray" if i == 0 else plt.cm.viridis(i / len(parts)))
    body.set_alpha(0.7)
ax.set_xticks(range(1, len(lbls_v) + 1))
ax.set_xticklabels(lbls_v, fontsize=9)
ax.set_ylabel("k-NN anomaly score (mean dist to 5NN in constructive train)")
ax.set_title("Anomaly score distributions by depth of classical dependence")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIG / "depth_anomaly_distributions.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved figure: {FIG/'depth_anomaly_distributions.png'}")
print("\nDONE.")
