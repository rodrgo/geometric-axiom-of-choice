"""Aggregate one-class support estimation on the Lean proof embeddings
(paper Appendix, Table tab:lean_support_full).

Methods: k-NN distance, one-class SVM, Gaussian KDE, Isolation Forest,
Local Outlier Factor. Reports the raw and length-residualized aggregate
AUC sweep used to fix the headline detector hyperparameters.
"""
import json, time
from pathlib import Path
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KernelDensity, LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "data" / "support"; OUT.mkdir(parents=True, exist_ok=True)


def score_methods(X_train, X_in, X_out, seed=0):
    """Fit on X_train (in-support). Score X_in + X_out. Returns dict of AUCs.
    Convention: positive label = out-of-support = gap / classical.
    """
    results = {}
    sc = StandardScaler().fit(X_train)
    Xt = sc.transform(X_train); Xi = sc.transform(X_in); Xo = sc.transform(X_out)
    y = np.concatenate([np.zeros(len(Xi)), np.ones(len(Xo))])

    # k-NN mean distance to k nearest in train
    for k in [1, 5, 10]:
        nn = NearestNeighbors(n_neighbors=k).fit(Xt)
        di, _ = nn.kneighbors(Xi); do, _ = nn.kneighbors(Xo)
        s = np.concatenate([di.mean(1), do.mean(1)])
        results[f"kNN_k{k}"] = roc_auc_score(y, s)

    # One-class SVM
    for nu in [0.05, 0.1, 0.2]:
        oc = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(Xt)
        s = np.concatenate([-oc.decision_function(Xi), -oc.decision_function(Xo)])
        results[f"OneClassSVM_nu{nu}"] = roc_auc_score(y, s)

    # KDE
    for bw in [0.3, 1.0, 3.0]:
        try:
            kde = KernelDensity(bandwidth=bw, kernel="gaussian").fit(Xt)
            s = np.concatenate([-kde.score_samples(Xi), -kde.score_samples(Xo)])
            results[f"KDE_bw{bw}"] = roc_auc_score(y, s)
        except Exception as e:
            results[f"KDE_bw{bw}"] = None

    # Isolation Forest
    iso = IsolationForest(contamination="auto", random_state=seed, n_estimators=200).fit(Xt)
    s = np.concatenate([-iso.score_samples(Xi), -iso.score_samples(Xo)])
    results["IsolationForest"] = roc_auc_score(y, s)

    # Local Outlier Factor (novelty)
    for k in [10, 20, 50]:
        if k >= len(Xt):
            continue
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xt)
        s = np.concatenate([-lof.score_samples(Xi), -lof.score_samples(Xo)])
        results[f"LOF_k{k}"] = roc_auc_score(y, s)

    return results


# ============ Lean proof embeddings ============
print("\n=== Lean proof embeddings ===")
d2 = np.load(ROOT / "results/data/stage4v3p/embeddings.npz", allow_pickle=True)
emb = d2["embeddings"]; labels = d2["labels"]
train_idx = d2["train_idx"]; val_idx = d2["val_idx"]; test_idx = d2["test_idx"]
# labels: 1=classical, 0=constructive. train/val/test are CONSTRUCTIVE indices.
constructive_train = emb[train_idx]
constructive_test = emb[test_idx]
classical_idx = np.where(labels == 1)[0]
classical = emb[classical_idx]
print(f"  constr_train={len(constructive_train)}, constr_test={len(constructive_test)}, classical={len(classical)}")

# For manageability, subsample classical for LOF
rng = np.random.default_rng(0)
classical_sub = classical[rng.choice(len(classical), min(5000, len(classical)), replace=False)]

lean_raw = score_methods(constructive_train, constructive_test, classical_sub, seed=0)
print("Raw embeddings:")
for k, v in lean_raw.items():
    print(f"  {k:24s} {v:.3f}")

# ----- Length-controlled version (residualize against n_invocations) -----
with open(ROOT / "results/data/stage4v3p/proofs.json") as f:
    proofs = json.load(f)
lengths = np.array([p["n_invocations"] for p in proofs], dtype=float)
# regress emb on length
X = lengths.reshape(-1, 1)
reg = LinearRegression().fit(X[train_idx], emb[train_idx])
emb_resid = emb - reg.predict(X)

ct_tr = emb_resid[train_idx]; ct_te = emb_resid[test_idx]
cl_r = emb_resid[classical_idx]
cl_r_sub = cl_r[rng.choice(len(cl_r), min(5000, len(cl_r)), replace=False)]
lean_resid = score_methods(ct_tr, ct_te, cl_r_sub, seed=0)
print("\nLength-residualized:")
for k, v in lean_resid.items():
    print(f"  {k:24s} {v:.3f}")

# Save
out = {
    "lean_raw": {k: float(v) for k, v in lean_raw.items()},
    "lean_length_residualized": {k: float(v) for k, v in lean_resid.items()},
    "notes": {
        "lean": "Train on constructive train embeddings (n=8968); score constructive test (n=1122) vs classical subsample (n=5000)",
    },
}
with open(OUT / "results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT/'results.json'}")
