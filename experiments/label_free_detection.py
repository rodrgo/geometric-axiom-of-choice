"""Label-free depth-stratified detection + density-superlevel containment.

This is the source of record for the paper's main detection tables, computed
with label-free hyperparameter selection (no classical labels consulted when
fixing any hyperparameter):

  - k-NN: k=5 (pre-registered default)
  - one-class SVM: nu=0.1 (pre-registered default)
  - LOF: k=20 (pre-registered default)
  - KDE bandwidth: chosen by 5-fold CV log-likelihood on the constructive
    training split, over {0.3, 1.0, 3.0}
  - superlevel PCA dim: smallest n_components explaining >=90% of
    constructive-train variance, with a CV-selected KDE bandwidth in that
    PCA space

Produces:
  - Depth-stratified AUC, all methods           -> Table tab:depth_strat
  - Density-superlevel containment by depth      -> Table tab:superlevel_containment
  - Headline label-free Lean support AUCs (raw + length-residualized)

Reads the BFS-distance hub written by depth_knn_auc.py, plus the proof
embeddings/baselines. Output: results/data/reviewer/q2_relabeled_results.json
(consumed by three_measurements_figure.py for the calibrated-curves panel).
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity, LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "data" / "reviewer"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Pre-registered, label-free hyperparameters ----
KNN_K = 5
OCSVM_NU = 0.1
LOF_K = 20
KDE_BW_CANDIDATES = [0.3, 1.0, 3.0]
PCA_VAR_TARGET = 0.90
CV_FOLDS = 5
SEED = 0

rng = np.random.default_rng(SEED)


def cv_bandwidth(X_train, candidates=KDE_BW_CANDIDATES, folds=CV_FOLDS, seed=SEED):
    """Pick KDE bandwidth by mean log-likelihood under K-fold CV on the
    in-support training embeddings only (no classical labels consulted)."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    means = []
    for bw in candidates:
        fold_ll = []
        for tr, te in kf.split(X_train):
            kde = KernelDensity(bandwidth=bw, kernel="gaussian").fit(X_train[tr])
            fold_ll.append(float(kde.score_samples(X_train[te]).mean()))
        means.append(float(np.mean(fold_ll)))
    best_idx = int(np.argmax(means))
    return candidates[best_idx], dict(zip([str(c) for c in candidates], means))


def score_methods(X_train, X_in, X_out, seed=SEED, bw=None):
    """Fit on X_train (in-support). Score X_in + X_out.
    Positive label = out-of-support (classical)."""
    results = {}
    sc = StandardScaler().fit(X_train)
    Xt = sc.transform(X_train)
    Xi = sc.transform(X_in)
    Xo = sc.transform(X_out)
    y = np.concatenate([np.zeros(len(Xi)), np.ones(len(Xo))])

    nn = NearestNeighbors(n_neighbors=KNN_K).fit(Xt)
    di, _ = nn.kneighbors(Xi)
    do, _ = nn.kneighbors(Xo)
    s = np.concatenate([di.mean(1), do.mean(1)])
    results[f"kNN_k{KNN_K}"] = float(roc_auc_score(y, s))

    oc = OneClassSVM(kernel="rbf", nu=OCSVM_NU, gamma="scale").fit(Xt)
    s = np.concatenate([-oc.decision_function(Xi), -oc.decision_function(Xo)])
    results[f"OneClassSVM_nu{OCSVM_NU}"] = float(roc_auc_score(y, s))

    if bw is None:
        bw, _ = cv_bandwidth(Xt)
    kde = KernelDensity(bandwidth=bw, kernel="gaussian").fit(Xt)
    s = np.concatenate([-kde.score_samples(Xi), -kde.score_samples(Xo)])
    results[f"KDE_bwCV{bw}"] = float(roc_auc_score(y, s))

    iso = IsolationForest(contamination="auto", random_state=seed, n_estimators=200).fit(Xt)
    s = np.concatenate([-iso.score_samples(Xi), -iso.score_samples(Xo)])
    results["IsolationForest"] = float(roc_auc_score(y, s))

    if LOF_K < len(Xt):
        lof = LocalOutlierFactor(n_neighbors=LOF_K, novelty=True).fit(Xt)
        s = np.concatenate([-lof.score_samples(Xi), -lof.score_samples(Xo)])
        results[f"LOF_k{LOF_K}"] = float(roc_auc_score(y, s))

    return results, bw


# ============ Lean proof embeddings ============
print("=== Lean proof embeddings ===")
d_lean = np.load(ROOT / "results/data/stage4v3p/embeddings.npz", allow_pickle=True)
emb = d_lean["embeddings"]
labels = d_lean["labels"]
train_idx = d_lean["train_idx"]
val_idx = d_lean["val_idx"]
test_idx = d_lean["test_idx"]
classical_idx = np.where(labels == 1)[0]

constructive_train = emb[train_idx]
constructive_test = emb[test_idx]
classical = emb[classical_idx]
print(f"  constr_train={len(constructive_train)}, constr_test={len(constructive_test)}, classical={len(classical)}")

# subsample classical (only affects LOF scoring; kept for consistency with prior run)
classical_sub = classical[rng.choice(len(classical), min(5000, len(classical)), replace=False)]

# CV KDE bandwidth on standardized constructive train
sc_cn = StandardScaler().fit(constructive_train)
bw_lean, bw_lean_ll = cv_bandwidth(sc_cn.transform(constructive_train))
print(f"  Lean constructive-train KDE bandwidth (CV log-lik): {bw_lean}")
for k, v in bw_lean_ll.items():
    print(f"    bw={k:>4s}  mean_val_loglik={v:.3f}")

lean_raw, _ = score_methods(constructive_train, constructive_test,
                             classical_sub, seed=SEED, bw=bw_lean)
print("  Label-free AUCs (Lean raw):")
for k, v in lean_raw.items():
    print(f"    {k:24s} {v:.3f}")

# Length-residualized
with open(ROOT / "results/data/stage4v3p/proofs.json") as f:
    proofs = json.load(f)
lengths = np.array([p["n_invocations"] for p in proofs], dtype=float)
Xl = lengths.reshape(-1, 1)
reg = LinearRegression().fit(Xl[train_idx], emb[train_idx])
emb_resid = emb - reg.predict(Xl)

ct_tr = emb_resid[train_idx]
ct_te = emb_resid[test_idx]
cl_r = emb_resid[classical_idx]
cl_r_sub = cl_r[rng.choice(len(cl_r), min(5000, len(cl_r)), replace=False)]

sc_ct = StandardScaler().fit(ct_tr)
bw_lean_r, bw_lean_r_ll = cv_bandwidth(sc_ct.transform(ct_tr))
print(f"  Lean length-residualized KDE bandwidth (CV log-lik): {bw_lean_r}")

lean_resid, _ = score_methods(ct_tr, ct_te, cl_r_sub, seed=SEED, bw=bw_lean_r)
print("  Label-free AUCs (Lean length-residualized):")
for k, v in lean_resid.items():
    print(f"    {k:24s} {v:.3f}")


# ============ Depth-stratified AUC (Table tab:depth_strat) ============
print("\n=== Depth-stratified AUC (label-free) ===")
with open(ROOT / "results/data/depth_analysis/bfs_distances_full.json") as f:
    distances = json.load(f)

depths = np.full(len(proofs), -1, dtype=int)
for i, p in enumerate(proofs):
    if p["is_classical"]:
        d = distances.get(p["name"])
        depths[i] = d if d is not None else -2

scaler = StandardScaler().fit(emb[train_idx])
E = scaler.transform(emb)
E_train = E[train_idx]

t0 = time.time()
nn_enc = NearestNeighbors(n_neighbors=KNN_K).fit(E_train)
score_knn = nn_enc.kneighbors(E)[0].mean(axis=1)
print(f"  kNN_encoder: {time.time()-t0:.1f}s")

# use bw_lean (CV-selected bandwidth) rather than a hardcoded value
t0 = time.time()
kde = KernelDensity(bandwidth=bw_lean, kernel="gaussian").fit(E_train)
score_kde = -kde.score_samples(E)
print(f"  KDE_encoder (bw={bw_lean}): {time.time()-t0:.1f}s")

t0 = time.time()
oc = OneClassSVM(kernel="rbf", nu=OCSVM_NU, gamma="scale").fit(E_train)
score_ocsvm = -oc.decision_function(E)
print(f"  OCSVM_encoder (nu={OCSVM_NU}): {time.time()-t0:.1f}s")

t0 = time.time()
iso = IsolationForest(contamination="auto", random_state=SEED, n_estimators=200).fit(E_train)
score_iso = -iso.score_samples(E)
print(f"  IsoForest_encoder: {time.time()-t0:.1f}s")

base = np.load(ROOT / "results/data/stage4v3p/baselines.npz", allow_pickle=True)
X_stats = base["X_stats"]
X_bow = base["X_bow"]
sc_s = StandardScaler().fit(X_stats[train_idx])
sc_b = StandardScaler().fit(X_bow[train_idx])
Es = sc_s.transform(X_stats)
Eb = sc_b.transform(X_bow)
nn_s = NearestNeighbors(n_neighbors=KNN_K).fit(Es[train_idx])
nn_b = NearestNeighbors(n_neighbors=KNN_K).fit(Eb[train_idx])
score_stats = nn_s.kneighbors(Es)[0].mean(1)
score_bow = nn_b.kneighbors(Eb)[0].mean(1)

methods = {
    "kNN_encoder": score_knn,
    "KDE_encoder": score_kde,
    "OCSVM_encoder": score_ocsvm,
    "IsoForest_encoder": score_iso,
    "kNN_stats": score_stats,
    "kNN_bow": score_bow,
}

BUCKETS = [
    ("depth 2",   2, 2),
    ("depth 3",   3, 3),
    ("depth 4",   4, 4),
    ("depth 5",   5, 5),
    ("depth 6",   6, 6),
    ("depth 7-8", 7, 8),
    ("depth 9+",  9, 999),
]

constr_eval_mask = (labels == 0).copy()
constr_eval_mask[train_idx] = False
n_constr_eval = int(constr_eval_mask.sum())
print(f"  Held-out constructive null: n={n_constr_eval}")

depth_results = {m: [] for m in methods}
for mname, scores in methods.items():
    cs = scores[constr_eval_mask]
    for lbl, lo, hi in BUCKETS:
        m = (depths >= lo) & (depths <= hi)
        n_b = int(m.sum())
        if n_b < 20:
            depth_results[mname].append({"bucket": lbl, "n": n_b, "auc": None})
            continue
        bs = scores[m]
        s = np.concatenate([cs, bs])
        y = np.concatenate([np.zeros(len(cs)), np.ones(n_b)])
        depth_results[mname].append({
            "bucket": lbl, "n": n_b,
            "auc": float(roc_auc_score(y, s)),
        })

print(f"\n  {'bucket':<12s}" + "".join(f"{m:>18s}" for m in methods))
for i, (lbl, _, _) in enumerate(BUCKETS):
    n = depth_results["kNN_encoder"][i]["n"]
    row = f"  {lbl:<7s} n={n:<4d}"
    for m in methods:
        a = depth_results[m][i]["auc"]
        row += f"{a:>18.3f}" if a is not None else f"{'--':>18s}"
    print(row)


# ============ Superlevel containment (label-free PCA + KDE) ============
print("\n=== Superlevel containment (label-free) ===")
con_train_emb = emb[train_idx]
pca_full = PCA(random_state=42).fit(con_train_emb)
var_ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(var_ratio)
pca_n = int(np.searchsorted(cum, PCA_VAR_TARGET) + 1)
print(f"  PCA components for {PCA_VAR_TARGET*100:.0f}% variance: n={pca_n} "
      f"(cum_var={float(cum[pca_n-1]):.3f})")

pca = PCA(n_components=pca_n, random_state=42).fit(con_train_emb)
train_pca = pca.transform(con_train_emb)
all_pca = pca.transform(emb)

# CV KDE bandwidth in unstandardized PCA space.
bw_sup, bw_sup_ll = cv_bandwidth(train_pca)
print(f"  KDE bandwidth (CV log-lik) in PCA space: {bw_sup}")

kde_sup = KernelDensity(bandwidth=bw_sup, kernel="gaussian").fit(train_pca)
all_log_d = kde_sup.score_samples(all_pca)
all_d = np.exp(all_log_d)

con_test_d = all_d[test_idx]
QUANTILES = [50, 60, 70, 80, 90, 95, 99]

SUPER_BUCKETS = [
    ("depth 2",   2, 2),
    ("depth 3",   3, 3),
    ("depth 4",   4, 4),
    ("depth 5-6", 5, 6),
    ("depth 7-8", 7, 8),
    ("depth 9+",  9, 999),
]

superlevel_rows = []
for q in QUANTILES:
    t = float(np.percentile(con_test_d, 100 - q))
    con_inside = float(np.mean(con_test_d >= t))
    for lbl, lo, hi in SUPER_BUCKETS:
        m = (labels == 1) & (depths >= lo) & (depths <= hi)
        superlevel_rows.append({
            "threshold_quantile": q,
            "threshold_value": t,
            "depth_bucket": lbl,
            "constructive_fraction_inside": con_inside,
            "classical_fraction_inside": float(np.mean(all_d[m] >= t)) if m.sum() else None,
            "n_classical": int(m.sum()),
        })

print("\n  Containment (% inside superlevel set):")
print(f"  {'bucket':<12s}  " + "  ".join([f"S_{q:>2d}" for q in QUANTILES]))
for lbl, lo, hi in SUPER_BUCKETS:
    row = [f"  {lbl:<12s}"]
    for q in QUANTILES:
        r = [r for r in superlevel_rows
             if r["threshold_quantile"] == q and r["depth_bucket"] == lbl][0]
        v = r["classical_fraction_inside"]
        row.append(f"{v:5.2%}" if v is not None else "  -- ")
    print("  ".join(row))

# Density-rank ("topological depth") AUC; lower rank -> classical.
con_ref_sorted = np.sort(con_test_d)
topo_depth = np.searchsorted(con_ref_sorted, all_d, side="right") / len(con_ref_sorted)
con_test_scores = topo_depth[test_idx]
topo_auc = []
for lbl, lo, hi in SUPER_BUCKETS:
    m = (labels == 1) & (depths >= lo) & (depths <= hi)
    if m.sum() < 20:
        continue
    s = np.concatenate([con_test_scores, topo_depth[m]])
    y = np.concatenate([np.zeros(len(con_test_scores)), np.ones(m.sum())])
    topo_auc.append({"bucket": lbl, "n": int(m.sum()), "auc": float(roc_auc_score(y, -s))})

print("\n  Density-rank AUC (lower rank -> classical):")
for r in topo_auc:
    print(f"  {r['bucket']:10s} n={r['n']:5d}  AUC={r['auc']:.3f}")


# ============ Save ============
out = {
    "method": "label-free hyperparameter selection",
    "hyperparameters": {
        "kNN_k": KNN_K,
        "OCSVM_nu": OCSVM_NU,
        "LOF_k": LOF_K,
        "KDE_bw_candidates": KDE_BW_CANDIDATES,
        "PCA_variance_target": PCA_VAR_TARGET,
        "CV_folds": CV_FOLDS,
    },
    "lean_raw": {
        "aucs": {k: float(v) for k, v in lean_raw.items()},
        "cv_bandwidth": bw_lean,
        "cv_logliks": bw_lean_ll,
    },
    "lean_length_residualized": {
        "aucs": {k: float(v) for k, v in lean_resid.items()},
        "cv_bandwidth": bw_lean_r,
        "cv_logliks": bw_lean_r_ll,
    },
    "depth_stratified": {
        "buckets": [{"label": b[0], "lo": b[1], "hi": b[2]} for b in BUCKETS],
        "methods": depth_results,
        "n_constructive_null": n_constr_eval,
        "kde_bandwidth_used": bw_lean,
    },
    "superlevel": {
        "pca_n_components": pca_n,
        "pca_cum_variance": float(cum[pca_n-1]),
        "kde_bandwidth": bw_sup,
        "cv_logliks": bw_sup_ll,
        "quantiles": QUANTILES,
        "containment_rows": superlevel_rows,
        "topological_depth_auc": topo_auc,
    },
}
with open(OUT / "q2_relabeled_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {OUT/'q2_relabeled_results.json'}")
