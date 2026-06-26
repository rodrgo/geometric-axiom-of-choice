"""Fix 3a: Within-domain k-NN AUCs on the proof encoder.

For each Mathlib domain, fit k-NN on constructive proof embeddings and score
classical vs held-out constructive. Replaces the hull-based encoder column.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "data" / "controls"
OUT.mkdir(parents=True, exist_ok=True)


def get_domain(fp):
    parts = fp.split("/")
    if len(parts) >= 2 and parts[0] == "Mathlib":
        return parts[1]
    return parts[0] if parts else "unknown"


d = np.load(ROOT / "results/data/stage4v3p/embeddings.npz", allow_pickle=True)
emb = d["embeddings"]
labels = d["labels"]  # 1 = classical, 0 = constructive
proofs = json.load(open(ROOT / "results/data/stage4v3p/proofs.json"))
assert len(proofs) == len(emb)
domains = np.array([get_domain(p["file_path"]) for p in proofs])

# For each domain with enough data, compute k-NN AUC (k=1, k=5)
results = {}
for domain in sorted(set(domains)):
    mask = domains == domain
    if mask.sum() < 100:
        continue
    con_mask = mask & (labels == 0)
    cls_mask = mask & (labels == 1)
    n_con = con_mask.sum(); n_cls = cls_mask.sum()
    if n_con < 50 or n_cls < 50:
        continue
    X_con = emb[con_mask]; X_cls = emb[cls_mask]
    # 80/20 split constructive
    Xc_tr, Xc_te = train_test_split(X_con, test_size=0.2, random_state=0)
    sc = StandardScaler().fit(Xc_tr)
    Xc_tr_s = sc.transform(Xc_tr); Xc_te_s = sc.transform(Xc_te)
    Xcl_s = sc.transform(X_cls)
    row = {"n_constructive": int(n_con), "n_classical": int(n_cls)}
    for k in [1, 5]:
        if k >= len(Xc_tr_s):
            continue
        nn = NearestNeighbors(n_neighbors=k).fit(Xc_tr_s)
        dc, _ = nn.kneighbors(Xc_te_s); dcl, _ = nn.kneighbors(Xcl_s)
        s = np.concatenate([dc.mean(1), dcl.mean(1)])
        y = np.concatenate([np.zeros(len(Xc_te_s)), np.ones(len(Xcl_s))])
        row[f"auc_knn_k{k}"] = float(roc_auc_score(y, s))
    results[domain] = row

# Sort by total size and report top domains
ranked = sorted(results.items(), key=lambda kv: -(kv[1]["n_constructive"] + kv[1]["n_classical"]))
print(f"{'Domain':<20s} {'n_con':>6s} {'n_cls':>6s} {'kNN k=1':>9s} {'kNN k=5':>9s}")
for dom, r in ranked[:15]:
    print(f"{dom:<20s} {r['n_constructive']:>6d} {r['n_classical']:>6d} "
          f"{r.get('auc_knn_k1', float('nan')):>9.3f} {r.get('auc_knn_k5', float('nan')):>9.3f}")

with open(OUT / "within_domain_knn.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT/'within_domain_knn.json'}")

# Seed stability — recompute k-NN on 5 seeds
print("\n--- Seed stability on k-NN (k=5) ---")
seeds = [42, 123, 456, 789, 1024]
# We only have single embeddings; the 5 seeds originally referred to retrained encoders.
# Approximate by varying the data split.
aucs = []
for seed in seeds:
    classical_idx = np.where(labels == 1)[0]
    constr_idx = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)
    con_tr_idx, con_te_idx = train_test_split(constr_idx, test_size=0.2, random_state=seed)
    cls_sub = classical_idx[rng.choice(len(classical_idx), 5000, replace=False)]
    sc = StandardScaler().fit(emb[con_tr_idx])
    Xtr = sc.transform(emb[con_tr_idx])
    Xte = sc.transform(emb[con_te_idx])
    Xcl = sc.transform(emb[cls_sub])
    nn = NearestNeighbors(n_neighbors=5).fit(Xtr)
    d_te, _ = nn.kneighbors(Xte); d_cl, _ = nn.kneighbors(Xcl)
    s = np.concatenate([d_te.mean(1), d_cl.mean(1)])
    y = np.concatenate([np.zeros(len(Xte)), np.ones(len(Xcl))])
    aucs.append(roc_auc_score(y, s))
    print(f"  seed {seed}: AUC {aucs[-1]:.4f}")
print(f"Mean: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

stab = {"split_seeds": seeds, "auc_knn_k5": [float(a) for a in aucs],
        "mean": float(np.mean(aucs)), "std": float(np.std(aucs)),
        "note": "5 random 80/20 splits of constructive data (single trained encoder)."}
with open(OUT / "seed_stability_knn.json", "w") as f:
    json.dump(stab, f, indent=2)
