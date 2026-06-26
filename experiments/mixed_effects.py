"""Revision 3: Mixed-effects regression with file as random intercept.

Dependent: anomaly_score (k-NN-5 distance in encoder space; std-scaled).
Models:
  (1) is_classical ~ anomaly_score + controls + (1 | file)
  (2) depth-specific indicators: depth2, depth3, depth4, depth56, depth78, depth9p
Also: within-file matched Wilcoxon on (classical - constructive) pairs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/data/reviewer"

print("Loading data...")
with open(ROOT / "results/data/stage4v3p/proofs.json") as f:
    proofs = json.load(f)
emb_data = np.load(ROOT / "results/data/stage4v3p/embeddings.npz")
emb = emb_data["embeddings"]
train_idx = emb_data["train_idx"]
labels = emb_data["labels"]
with open(ROOT / "results/data/depth_analysis/bfs_distances_full.json") as f:
    bfs = json.load(f)
base = np.load(ROOT / "results/data/stage4v3p/baselines.npz")
X_stats = base["X_stats"]

feature_names = [
    "n_invocations", "log_n_invocations",
    "n_distinct_tactics", "n_distinct_per_invocation",
    "frac_structural", "frac_automation", "frac_rewrite", "frac_logical",
    "mean_arg_count", "max_arg_count",
    "has_by_contra", "has_by_cases", "has_choose", "has_exfalso",
    "has_classical", "has_contrapose", "has_push_neg",
    "frac_classical_markers", "log_total_chars",
]
f2i = {n: i for i, n in enumerate(feature_names)}

print(f"  N={len(proofs)}")

# Compute anomaly score: k-NN (k=5) mean distance in standardized embedding
sc = StandardScaler().fit(emb[train_idx])
E = sc.transform(emb)
nn = NearestNeighbors(n_neighbors=5).fit(E[train_idx])
anomaly = nn.kneighbors(E)[0].mean(axis=1)

# Reconstruction loss (pre-computed)
recon_f = np.load(ROOT / "results/data/depth_analysis/reconstruction_loss_per_proof.npz",
                  allow_pickle=True)
recon_losses = recon_f["losses"]
recon_names = list(recon_f["names"])
name_to_recon = dict(zip(recon_names, recon_losses))

# Domain from file_path (first dir after "Mathlib/")
def domain_of(fp: str) -> str:
    parts = fp.split("/")
    if len(parts) >= 2 and parts[0] == "Mathlib":
        return parts[1]
    return "unknown"


# Build dataframe
rows = []
for i, p in enumerate(proofs):
    d = bfs.get(p["name"]) if p["is_classical"] else None
    rows.append({
        "name": p["name"],
        "is_classical": int(p["is_classical"]),
        "depth": d if d is not None else -1,
        "anomaly_score": float(anomaly[i]),
        "recon_loss": float(name_to_recon.get(p["name"], np.nan)),
        "log_proof_length": float(X_stats[i, f2i["log_n_invocations"]]),
        "tactic_diversity": float(X_stats[i, f2i["n_distinct_per_invocation"]]),
        "frac_structural": float(X_stats[i, f2i["frac_structural"]]),
        "frac_automation": float(X_stats[i, f2i["frac_automation"]]),
        "domain": domain_of(p["file_path"]),
        "file": p["file_path"],
    })

df = pd.DataFrame(rows)
print(f"  files={df['file'].nunique()}, domains={df['domain'].nunique()}")
print(f"  classical={df['is_classical'].sum()}, constructive={len(df)-df['is_classical'].sum()}")
print(f"  recon missing: {df['recon_loss'].isna().sum()}")
df = df.dropna(subset=["anomaly_score"])

# Standardize anomaly_score so coefs are interpretable
df["anomaly_score"] = (df["anomaly_score"] - df["anomaly_score"].mean()) / df["anomaly_score"].std()

# ----- Model 1: binary classical -----
print("\n=== Model 1: anomaly_score ~ is_classical + controls | (1|file) ===")
m1 = smf.mixedlm(
    "anomaly_score ~ is_classical + log_proof_length + tactic_diversity "
    "+ frac_structural + frac_automation",
    data=df,
    groups=df["file"],
)
r1 = m1.fit(method="lbfgs")
print(r1.summary())

key = {
    "is_classical_coef": float(r1.params["is_classical"]),
    "is_classical_p": float(r1.pvalues["is_classical"]),
    "log_length_coef": float(r1.params["log_proof_length"]),
    "n": int(r1.nobs),
    "n_files": int(df["file"].nunique()),
}
print(f"\nKEY: is_classical = {key['is_classical_coef']:+.4f} "
      f"(p={key['is_classical_p']:.2e})")

# ----- Model 2: depth-specific -----
print("\n=== Model 2: depth buckets | (1|file) ===")
for col, lo, hi in [("depth2", 2, 2), ("depth3", 3, 3), ("depth4", 4, 4),
                     ("depth56", 5, 6), ("depth78", 7, 8), ("depth9p", 9, 999)]:
    df[col] = ((df["depth"] >= lo) & (df["depth"] <= hi)).astype(int)

m2 = smf.mixedlm(
    "anomaly_score ~ depth2 + depth3 + depth4 + depth56 + depth78 + depth9p "
    "+ log_proof_length + tactic_diversity + frac_structural + frac_automation",
    data=df,
    groups=df["file"],
)
r2 = m2.fit(method="lbfgs")
print(r2.summary())

depth_coefs = {}
for c in ["depth2", "depth3", "depth4", "depth56", "depth78", "depth9p"]:
    depth_coefs[c] = {
        "coef": float(r2.params[c]),
        "p": float(r2.pvalues[c]),
        "se": float(r2.bse[c]),
    }

print("\nDepth-specific coefs:")
for c, v in depth_coefs.items():
    print(f"  {c:<8s} coef={v['coef']:+.4f} se={v['se']:.4f} p={v['p']:.2e}")

# Write summary
with open(OUT / "mixed_effects_results.txt", "w") as f:
    f.write("Revision 3: Mixed-effects regression with file as random intercept\n")
    f.write("=" * 72 + "\n\n")
    f.write("Model 1: anomaly_score ~ is_classical + controls | (1|file)\n\n")
    f.write(str(r1.summary()))
    f.write("\n\nModel 2: depth-specific indicators | (1|file)\n\n")
    f.write(str(r2.summary()))

with open(OUT / "mixed_effects_summary.json", "w") as f:
    json.dump({
        "n_proofs": len(df),
        "n_files": int(df["file"].nunique()),
        "n_domains": int(df["domain"].nunique()),
        "model1": key,
        "model2_depth_coefs": depth_coefs,
    }, f, indent=2)
print(f"\nSaved {OUT/'mixed_effects_results.txt'}")
print(f"Saved {OUT/'mixed_effects_summary.json'}")


# ----- Exact within-file matching + Wilcoxon -----
print("\n=== Within-file exact matching (Wilcoxon on matched pairs) ===")
matched = []
rng = np.random.default_rng(0)
for fpath, sub in df.groupby("file"):
    con = sub[sub["is_classical"] == 0]
    cls = sub[sub["is_classical"] == 1]
    if len(con) < 2 or len(cls) < 2:
        continue
    con_avail = con.copy()
    for _, c_row in cls.iterrows():
        if len(con_avail) == 0:
            break
        diffs = (con_avail["log_proof_length"] - c_row["log_proof_length"]).abs()
        if diffs.min() > 0.3:
            continue
        best_idx = diffs.idxmin()
        matched.append({
            "file": fpath,
            "depth": int(c_row["depth"]),
            "cls_anomaly": float(c_row["anomaly_score"]),
            "con_anomaly": float(con_avail.loc[best_idx, "anomaly_score"]),
        })
        con_avail = con_avail.drop(best_idx)

print(f"  Matched pairs: {len(matched)}")
if len(matched) >= 20:
    diffs = np.array([m["cls_anomaly"] - m["con_anomaly"] for m in matched])
    stat, p = wilcoxon(diffs, alternative="greater")
    print(f"  Overall: mean diff={diffs.mean():+.4f}, Wilcoxon p={p:.2e} (greater)")

    results_match = {"overall": {"n": len(matched), "mean_diff": float(diffs.mean()),
                                 "p_greater": float(p)}}
    for d_label, lo, hi in [("d2", 2, 2), ("d3-4", 3, 4),
                             ("d5-6", 5, 6), ("d7+", 7, 999)]:
        pairs = [m for m in matched if lo <= m["depth"] <= hi]
        if len(pairs) >= 20:
            dd = np.array([m["cls_anomaly"] - m["con_anomaly"] for m in pairs])
            stat, p_val = wilcoxon(dd, alternative="greater")
            results_match[d_label] = {
                "n": len(pairs),
                "mean_diff": float(dd.mean()),
                "p_greater": float(p_val),
            }
            print(f"  {d_label:5s} n={len(pairs):4d}  mean_diff={dd.mean():+.4f}  "
                  f"p={p_val:.2e}")
else:
    results_match = {"overall": {"n": len(matched), "note": "insufficient for test"}}

with open(OUT / "exact_matching_results.json", "w") as f:
    json.dump(results_match, f, indent=2, default=str)
print(f"\nSaved {OUT/'exact_matching_results.json'}")
