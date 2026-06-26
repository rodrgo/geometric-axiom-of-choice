"""Tier 3 Steps 3.4-3.5: Analyze aesop results and render the figure.

Input: results/data/reviewer/prover_results.jsonl (one JSON per theorem).
Outputs:
  results/data/reviewer/prover_results.json            (aggregated)
  results/data/reviewer/prover_anomaly_correlation.json
  results/figures/prover_success_by_depth.png
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parent.parent


def main():
    path = ROOT / "results/data/reviewer/prover_results.jsonl"
    if not path.exists():
        print(f"MISSING: {path}")
        return
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"loaded {len(results)} results")

    successes = Counter()
    totals = Counter()
    per_bucket = {}
    for r in results:
        totals[r["bucket"]] += 1
        if r.get("success"):
            successes[r["bucket"]] += 1
    buckets_order = ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]

    print("\nAesop success rate by bucket:")
    for b in buckets_order:
        t = totals[b]
        s = successes[b]
        rate = s / max(t, 1)
        per_bucket[b] = {
            "n": t, "n_success": s, "rate": rate,
        }
        print(f"  {b:<14s}  {s}/{t} = {100*rate:.1f}%")

    # Logistic regression: does anomaly_score predict failure beyond length?
    df = pd.DataFrame(results)
    df_clean = df[df["anomaly_score"].notna() & df["proof_length"].notna()].copy()
    df_clean["fail"] = (~df_clean["success"].astype(bool)).astype(int)

    lr_out = {"n": len(df_clean), "n_fail": int(df_clean["fail"].sum())}
    if df_clean["fail"].nunique() == 2 and len(df_clean) >= 50:
        X_len = df_clean[["proof_length"]].values
        X_full = df_clean[["proof_length", "anomaly_score"]].values
        y = df_clean["fail"].values
        auc_len = cross_val_score(
            LogisticRegression(max_iter=1000), X_len, y,
            scoring="roc_auc", cv=5,
        ).mean()
        auc_full = cross_val_score(
            LogisticRegression(max_iter=1000), X_full, y,
            scoring="roc_auc", cv=5,
        ).mean()
        print(f"\n5-fold CV AUC on fail-prediction:")
        print(f"  length only:          {auc_len:.3f}")
        print(f"  length + anomaly:     {auc_full:.3f}")
        print(f"  marginal of anomaly:  {auc_full - auc_len:+.3f}")
        lr_out.update({
            "auc_length_only": float(auc_len),
            "auc_length_plus_anomaly": float(auc_full),
            "marginal_anomaly": float(auc_full - auc_len),
        })
    else:
        print("\n(skipping logistic regression -- too little data or homogeneous outcome)")

    with open(ROOT / "results/data/reviewer/prover_results.json", "w") as f:
        json.dump({
            "per_bucket": per_bucket,
            "overall_success": sum(successes.values()) / max(sum(totals.values()), 1),
            "n_total": sum(totals.values()),
        }, f, indent=2)
    with open(ROOT / "results/data/reviewer/prover_anomaly_correlation.json", "w") as f:
        json.dump(lr_out, f, indent=2)

    # Figure
    FIG = ROOT / "results" / "figures"
    FIG.mkdir(exist_ok=True)

    rates = [per_bucket[b]["rate"] for b in buckets_order]
    counts = [per_bucket[b]["n"] for b in buckets_order]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4CAF50", "#F44336", "#FF9800", "#FFC107", "#9E9E9E"]
    bars = ax.bar(range(len(buckets_order)), rates, color=colors)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={c}", ha="center", fontsize=8)
    ax.set_xticks(range(len(buckets_order)))
    ax.set_xticklabels(buckets_order, rotation=15)
    ax.set_ylabel("aesop success rate (60s timeout)")
    ax.set_title("Prover Success by Classical Depth")
    ax.set_ylim(0, max(rates) * 1.2 if max(rates) > 0 else 0.1)
    plt.tight_layout()
    plt.savefig(FIG / "prover_success_by_depth.png", dpi=300, bbox_inches="tight")
    print(f"\nsaved {FIG/'prover_success_by_depth.png'}")
    print(f"saved results/data/reviewer/prover_results.json")
    print(f"saved results/data/reviewer/prover_anomaly_correlation.json")


if __name__ == "__main__":
    main()
