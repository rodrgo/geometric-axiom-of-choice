"""Reviewer revision -- Concern 1 (author confound).

Tests whether the classical/constructive anomaly gap survives controlling for the
declaration's author.  Mirrors mixed_effects_regression.py (file random intercept)
but uses a per-declaration author label that *crosscuts files*, so it is not a
strict coarsening of the file control already in the paper.

  (A) author random-intercept mixed model:
        anomaly ~ is_classical + controls | (1 | author)
      + depth-dummy variant.
  (B) within-author, cross-file, length-matched Wilcoxon pairs.

Anomaly score and controls are constructed exactly as in the paper
(k-NN k=5 mean Euclidean distance in the constructive-train StandardScaler space).

Reads:  results/data/reviewer/decl_authors.json  (from confound_authors_extract.py)
Writes: results/data/reviewer/author_confound.json (+ author_confound.txt)
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
proofs = json.load(open(ROOT / "results/data/stage4v3p/proofs.json"))
emb_data = np.load(ROOT / "results/data/stage4v3p/embeddings.npz")
emb = emb_data["embeddings"]
train_idx = emb_data["train_idx"]
bfs = json.load(open(ROOT / "results/data/depth_analysis/bfs_distances_full.json"))
X_stats = np.load(ROOT / "results/data/stage4v3p/baselines.npz")["X_stats"]
authors = json.load(open(OUT / "decl_authors.json"))

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

# Anomaly score: k-NN (k=5) in standardized embedding space (paper protocol).
sc = StandardScaler().fit(emb[train_idx])
E = sc.transform(emb)
anomaly = NearestNeighbors(n_neighbors=5).fit(E[train_idx]).kneighbors(E)[0].mean(axis=1)

rows = []
for i, p in enumerate(proofs):
    a = authors.get(p["name"])
    if a is None:
        continue  # no resolved author
    d = bfs.get(p["name"]) if p["is_classical"] else None
    rows.append({
        "name": p["name"],
        "is_classical": int(p["is_classical"]),
        "depth": d if d is not None else -1,
        "anomaly_score": float(anomaly[i]),
        "log_proof_length": float(X_stats[i, f2i["log_n_invocations"]]),
        "tactic_diversity": float(X_stats[i, f2i["n_distinct_per_invocation"]]),
        "frac_structural": float(X_stats[i, f2i["frac_structural"]]),
        "frac_automation": float(X_stats[i, f2i["frac_automation"]]),
        "author": a["author"],
        "file": p["file_path"],
    })

df = pd.DataFrame(rows)
df["anomaly_score"] = (df["anomaly_score"] - df["anomaly_score"].mean()) / df["anomaly_score"].std()
print(f"  N with author={len(df)}  authors={df['author'].nunique()}  "
      f"files={df['file'].nunique()}")
print(f"  classical={int(df['is_classical'].sum())}  "
      f"constructive={int((df['is_classical']==0).sum())}")
# crosscut diagnostic: authors spanning >1 file, files with >1 author
files_per_author = df.groupby("author")["file"].nunique()
authors_per_file = df.groupby("file")["author"].nunique()
print(f"  authors spanning >1 file: {(files_per_author>1).sum()} "
      f"({100*(files_per_author>1).mean():.0f}%)")
print(f"  files with >1 author:    {(authors_per_file>1).sum()} "
      f"({100*(authors_per_file>1).mean():.0f}%)")

# ---- (A) author random-intercept mixed model ----
print("\n=== Model A1: anomaly ~ is_classical + controls | (1|author) ===")
mA = smf.mixedlm(
    "anomaly_score ~ is_classical + log_proof_length + tactic_diversity "
    "+ frac_structural + frac_automation",
    data=df, groups=df["author"],
)
rA = mA.fit(method="lbfgs")
print(rA.summary())
model_a = {
    "is_classical_coef": float(rA.params["is_classical"]),
    "is_classical_se": float(rA.bse["is_classical"]),
    "is_classical_p": float(rA.pvalues["is_classical"]),
    "n": int(rA.nobs),
    "n_authors": int(df["author"].nunique()),
}
print(f"\nKEY: is_classical = {model_a['is_classical_coef']:+.4f} "
      f"(SE {model_a['is_classical_se']:.4f}, p={model_a['is_classical_p']:.2e})")

print("\n=== Model A2: depth buckets | (1|author) ===")
for col, lo, hi in [("depth2", 2, 2), ("depth3", 3, 3), ("depth4", 4, 4),
                    ("depth56", 5, 6), ("depth78", 7, 8), ("depth9p", 9, 999)]:
    df[col] = ((df["depth"] >= lo) & (df["depth"] <= hi)).astype(int)
mA2 = smf.mixedlm(
    "anomaly_score ~ depth2 + depth3 + depth4 + depth56 + depth78 + depth9p "
    "+ log_proof_length + tactic_diversity + frac_structural + frac_automation",
    data=df, groups=df["author"],
)
rA2 = mA2.fit(method="lbfgs")
depth_coefs = {c: {"coef": float(rA2.params[c]), "se": float(rA2.bse[c]),
                   "p": float(rA2.pvalues[c])}
               for c in ["depth2", "depth3", "depth4", "depth56", "depth78", "depth9p"]}
for c, v in depth_coefs.items():
    print(f"  {c:<8s} coef={v['coef']:+.4f} se={v['se']:.4f} p={v['p']:.2e}")

# ---- (B) within-author, cross-file, length-matched pairs ----
print("\n=== Within-author matched pairs (Wilcoxon, greater) ===")
matched = []
for auth, sub in df.groupby("author"):
    con = sub[sub["is_classical"] == 0]
    cls = sub[sub["is_classical"] == 1]
    if len(con) < 1 or len(cls) < 1:
        continue
    con_avail = con.copy()
    for _, c_row in cls.iterrows():
        if len(con_avail) == 0:
            break
        diffs = (con_avail["log_proof_length"] - c_row["log_proof_length"]).abs()
        if diffs.min() > 0.3:
            continue
        bi = diffs.idxmin()
        matched.append({
            "author": auth,
            "depth": int(c_row["depth"]),
            "cls_anomaly": float(c_row["anomaly_score"]),
            "con_anomaly": float(con_avail.loc[bi, "anomaly_score"]),
            "same_file": bool(c_row["file"] == con_avail.loc[bi, "file"]),
        })
        con_avail = con_avail.drop(bi)

n_pairs = len(matched)
n_cross = sum(not m["same_file"] for m in matched)
print(f"  Matched pairs: {n_pairs}  (cross-file: {n_cross})")
diffs = np.array([m["cls_anomaly"] - m["con_anomaly"] for m in matched])
_, p = wilcoxon(diffs, alternative="greater")
matched_res = {"overall": {"n": n_pairs, "n_cross_file": n_cross,
                           "mean_diff": float(diffs.mean()), "p_greater": float(p)}}
print(f"  Overall: mean diff={diffs.mean():+.4f}  Wilcoxon p={p:.2e}")
for lbl, lo, hi in [("d2", 2, 2), ("d3-4", 3, 4), ("d5-6", 5, 6), ("d7+", 7, 999)]:
    pr = [m for m in matched if lo <= m["depth"] <= hi]
    if len(pr) >= 20:
        dd = np.array([m["cls_anomaly"] - m["con_anomaly"] for m in pr])
        _, pv = wilcoxon(dd, alternative="greater")
        matched_res[lbl] = {"n": len(pr), "mean_diff": float(dd.mean()),
                            "p_greater": float(pv)}
        print(f"  {lbl:5s} n={len(pr):4d}  mean_diff={dd.mean():+.4f}  p={pv:.2e}")

# also: cross-file pairs only (strongest version)
cross = [m for m in matched if not m["same_file"]]
if len(cross) >= 20:
    dd = np.array([m["cls_anomaly"] - m["con_anomaly"] for m in cross])
    _, pv = wilcoxon(dd, alternative="greater")
    matched_res["cross_file_only"] = {"n": len(cross), "mean_diff": float(dd.mean()),
                                      "p_greater": float(pv)}
    print(f"  cross-file-only n={len(cross)}  mean_diff={dd.mean():+.4f}  p={pv:.2e}")

with open(OUT / "author_confound.txt", "w") as f:
    f.write("Author confound: mixed-effects with author random intercept\n")
    f.write("=" * 72 + "\n\nModel A1 (binary):\n\n")
    f.write(str(rA.summary()))
    f.write("\n\nModel A2 (depth buckets):\n\n")
    f.write(str(rA2.summary()))

json.dump({
    "n_proofs_with_author": len(df),
    "n_authors": int(df["author"].nunique()),
    "n_files": int(df["file"].nunique()),
    "authors_spanning_multi_file_frac": float((files_per_author > 1).mean()),
    "files_multi_author_frac": float((authors_per_file > 1).mean()),
    "model_A1_binary": model_a,
    "model_A2_depth_coefs": depth_coefs,
    "matched_pairs": matched_res,
    "paper_file_intercept_beta_cls": 0.402,
}, open(OUT / "author_confound.json", "w"), indent=2)
print(f"\nSaved {OUT/'author_confound.json'}")
print(f"Saved {OUT/'author_confound.txt'}")
