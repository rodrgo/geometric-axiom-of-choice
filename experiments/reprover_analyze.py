"""Analyze ReProver TacGen (single-shot top-K) results, paired with aesop.

Inputs:
  results/data/reviewer/prover_results.jsonl         (aesop, same 251 theorems)
  results/data/reviewer/reprover_tacgen_results.jsonl (this run)
  results/data/reviewer/classical_ablation_results.jsonl (for reference)

Outputs:
  results/data/reviewer/reprover_summary.json
  results/data/reviewer/reprover_anomaly_correlation.json
  results/data/reviewer/reprover_comparison.txt
  results/figures/neural_prover_comparison.png
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]


def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["name"]] = r
    return rows


def per_bucket(rows, name_pred=None):
    # name_pred optional predicate over full rows; defaults to True
    out = {b: {"total": 0, "success": 0} for b in BUCKETS}
    for r in rows.values():
        if name_pred and not name_pred(r):
            continue
        b = r["bucket"]
        if b not in out:
            continue
        out[b]["total"] += 1
        if r.get("success"):
            out[b]["success"] += 1
    return out


def cv_auc(X, y, seed=0):
    if len(np.unique(y)) < 2:
        return np.nan
    cls = np.bincount(y.astype(int))
    n = min(5, int(cls.min()))
    if n < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    try:
        return float(cross_val_score(
            LogisticRegression(max_iter=2000, solver="lbfgs"),
            X, y, scoring="roc_auc", cv=cv,
        ).mean())
    except Exception:
        return np.nan


def boot_auc(df, feats, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sub = df.iloc[idx]
        y = sub["fail"].values.astype(int)
        X = sub[feats].values
        v = cv_auc(X, y)
        if not np.isnan(v):
            aucs.append(v)
    return np.array(aucs)


def summarize_ci(a):
    if len(a) == 0:
        return {"n_boot_valid": 0, "median": None, "ci95": [None, None]}
    return {
        "n_boot_valid": int(len(a)),
        "median": float(np.median(a)),
        "ci95": [float(np.percentile(a, 2.5)),
                 float(np.percentile(a, 97.5))],
    }


def main():
    aesop = load_jsonl(ROOT / "results/data/reviewer/prover_results.jsonl")
    rep   = load_jsonl(ROOT / "results/data/reviewer/reprover_tacgen_results.jsonl")
    common = sorted(set(aesop) & set(rep))
    print(f"paired aesop+reprover: {len(common)}")

    # Per-bucket rates
    aesop_pb = per_bucket(aesop)
    rep_pb   = per_bucket(rep)
    print(f"\n{'Bucket':<15s} {'n':>5s} {'aesop':>10s} {'reprover':>12s}")
    rows = []
    for b in BUCKETS:
        na = aesop_pb[b]["total"]; sa = aesop_pb[b]["success"]
        nr = rep_pb[b]["total"];   sr = rep_pb[b]["success"]
        aesop_r = sa / max(na, 1)
        rep_r   = sr / max(nr, 1)
        rows.append({
            "bucket": b,
            "n_aesop": na, "success_aesop": sa, "rate_aesop": aesop_r,
            "n_reprover": nr, "success_reprover": sr, "rate_reprover": rep_r,
        })
        print(f"{b:<15s} {na:>5d} {sa:>6d} ({100*aesop_r:>5.1f}%) "
              f"{sr:>6d} ({100*rep_r:>5.1f}%)")

    # Combined classical vs constructive
    agg = {"aesop": {}, "reprover": {}}
    for label in ("aesop", "reprover"):
        pb = aesop_pb if label == "aesop" else rep_pb
        cons = pb["constructive"]
        agg[label]["constructive"] = {
            "n": cons["total"], "s": cons["success"],
            "rate": cons["success"] / max(cons["total"], 1),
        }
        c_n = sum(pb[b]["total"] for b in BUCKETS if b != "constructive")
        c_s = sum(pb[b]["success"] for b in BUCKETS if b != "constructive")
        agg[label]["classical_combined"] = {
            "n": c_n, "s": c_s, "rate": c_s / max(c_n, 1),
        }
    print(f"\nCombined:")
    for label in ("aesop", "reprover"):
        c = agg[label]["constructive"]
        cl = agg[label]["classical_combined"]
        print(f"  {label}: constructive {c['s']}/{c['n']} = {100*c['rate']:.1f}%   "
              f"classical {cl['s']}/{cl['n']} = {100*cl['rate']:.1f}%")

    # Anomaly-AUC bootstrap (reprover)
    df = pd.DataFrame([{
        "name": r["name"],
        "bucket": r["bucket"],
        "is_classical": r["bucket"] != "constructive",
        "fail": 0 if r.get("success") else 1,
        "anomaly_score": float(r["anomaly_score"]),
        "proof_length": float(r["proof_length"]),
        "log_proof_length": float(np.log1p(r["proof_length"])),
    } for r in rep.values()
      if r.get("anomaly_score") is not None and r.get("proof_length") is not None])
    print(f"\nReProver anomaly-AUC analysis: N = {len(df)}, fail = {int(df['fail'].sum())}")
    anom_out = {}
    for label, sub in [("overall", df),
                        ("within_classical", df[df["is_classical"]].reset_index(drop=True)),
                        ("within_constructive", df[~df["is_classical"]].reset_index(drop=True))]:
        if len(sub) < 20 or sub["fail"].nunique() != 2:
            anom_out[label] = {"skip_reason": "too few data"}
            print(f"  {label}: skipped (N={len(sub)})")
            continue
        len_only = boot_auc(sub, ["log_proof_length"])
        len_anom = boot_auc(sub, ["log_proof_length", "anomaly_score"])
        n = min(len(len_only), len(len_anom))
        diff = len_anom[:n] - len_only[:n]
        anom_out[label] = {
            "n": len(sub), "n_fail": int(sub["fail"].sum()),
            "length_only": summarize_ci(len_only),
            "length_plus_anomaly": summarize_ci(len_anom),
            "improvement": summarize_ci(diff),
            "prob_improvement_positive": float(np.mean(diff > 0)),
        }
        ci = lambda a: f"[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]"
        print(f"  {label:<20s} len {np.median(len_only):.3f} {ci(len_only)}  "
              f"full {np.median(len_anom):.3f} {ci(len_anom)}  "
              f"Δ {np.median(diff):+.3f} {ci(diff)}  P(Δ>0) {np.mean(diff>0):.3f}")

    # Figure: 3-way bars
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BUCKETS))
    w = 0.4
    a_rates = [rows[i]["rate_aesop"] for i in range(len(BUCKETS))]
    r_rates = [rows[i]["rate_reprover"] for i in range(len(BUCKETS))]
    ax.bar(x - w/2, a_rates, w, label="aesop (single shot)", color="#F44336")
    ax.bar(x + w/2, r_rates, w, label="ReProver TacGen (single shot, top-K=8)", color="#673AB7")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS, rotation=15)
    ax.set_ylabel("Success rate (60s timeout)")
    ax.set_title("aesop vs ReProver TacGen on the same 251 Mathlib theorems")
    ax.legend()
    for i, (ar, rr) in enumerate(zip(a_rates, r_rates)):
        ax.text(x[i] - w/2, ar + 0.01, f"{100*ar:.0f}%", ha="center", fontsize=8)
        ax.text(x[i] + w/2, rr + 0.01, f"{100*rr:.0f}%", ha="center", fontsize=8)
    ax.set_ylim(0, max(max(a_rates), max(r_rates)) * 1.3 + 0.05)
    plt.tight_layout()
    FIG = ROOT / "results" / "figures"
    FIG.mkdir(exist_ok=True)
    plt.savefig(FIG / "neural_prover_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\nsaved {FIG/'neural_prover_comparison.png'}")

    # Save
    with open(ROOT / "results/data/reviewer/reprover_summary.json", "w") as f:
        json.dump({
            "K": 8, "timeout_s": 60,
            "model": "kaiyuy/leandojo-lean4-tacgen-byt5-small",
            "per_bucket": rows, "aggregates": agg,
        }, f, indent=2)
    with open(ROOT / "results/data/reviewer/reprover_anomaly_correlation.json", "w") as f:
        json.dump(anom_out, f, indent=2)

    # Text dump
    lines = ["ReProver TacGen vs aesop — paired comparison",
             "=" * 58, ""]
    lines.append(f"{'Bucket':<15s} {'n':>5s} {'aesop':>12s} {'ReProver':>12s}")
    for r in rows:
        lines.append(f"{r['bucket']:<15s} {r['n_aesop']:>5d} "
                     f"{r['success_aesop']:>3d} ({100*r['rate_aesop']:>5.1f}%)   "
                     f"{r['success_reprover']:>3d} ({100*r['rate_reprover']:>5.1f}%)")
    lines.append("")
    lines.append("Combined:")
    for label in ("aesop", "reprover"):
        c = agg[label]["constructive"]; cl = agg[label]["classical_combined"]
        lines.append(f"  {label}: constructive {c['s']}/{c['n']} = {100*c['rate']:.1f}%   "
                     f"classical {cl['s']}/{cl['n']} = {100*cl['rate']:.1f}%")
    with open(ROOT / "results/data/reviewer/reprover_comparison.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved results/data/reviewer/reprover_comparison.txt")


if __name__ == "__main__":
    main()
