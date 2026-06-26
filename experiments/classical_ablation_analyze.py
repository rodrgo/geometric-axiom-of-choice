"""Paired analysis of the classical-prefix ablation.

Reads:
  results/data/reviewer/prover_results.jsonl              (aesop)
  results/data/reviewer/classical_ablation_results.jsonl  (classical; aesop)

Writes:
  results/data/reviewer/classical_ablation_results.json   (aggregate per bucket)
  results/data/reviewer/classical_ablation_summary.txt    (printable summary)
  results/figures/classical_prefix_ablation.png           (paired bars)
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent

BUCKETS = ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]


def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["name"]] = r
    return out


def main():
    aesop = load_jsonl(ROOT / "results/data/reviewer/prover_results.jsonl")
    classical = load_jsonl(ROOT / "results/data/reviewer/classical_ablation_results.jsonl")
    # Pair on name
    common = sorted(set(aesop) & set(classical))
    print(f"aesop results: {len(aesop)}  classical;aesop results: {len(classical)}  paired: {len(common)}")

    by_bucket = defaultdict(lambda: {
        "total": 0,
        "aesop_success": 0,
        "classical_aesop_success": 0,
        "rescued": 0,
        "newly_failed": 0,
        "b01": 0,
        "b10": 0,
    })
    paired = []
    for name in common:
        a = aesop[name]
        c = classical[name]
        b = a["bucket"]
        s = by_bucket[b]
        s["total"] += 1
        aes = bool(a.get("success"))
        cla = bool(c.get("success"))
        if aes: s["aesop_success"] += 1
        if cla: s["classical_aesop_success"] += 1
        if not aes and cla:
            s["rescued"] += 1
            s["b01"] += 1
        if aes and not cla:
            s["newly_failed"] += 1
            s["b10"] += 1
        paired.append({
            "name": name, "bucket": b, "depth": a.get("depth"),
            "anomaly_score": a["anomaly_score"],
            "proof_length": a["proof_length"],
            "aesop_success": aes,
            "classical_aesop_success": cla,
            "classical_aesop_time": c["elapsed_s"],
        })

    # Print the headline table
    print()
    print(f"{'Bucket':<15s} {'n':>5s} {'aesop':>8s} {'cls;aes':>10s} {'rescue':>8s} "
          f"{'lost':>6s} {'McN p':>10s}")
    rows = []
    for b in BUCKETS:
        s = by_bucket[b]
        t = s["total"]
        ar = s["aesop_success"] / max(t, 1)
        cr = s["classical_aesop_success"] / max(t, 1)
        denom = t - s["aesop_success"]
        rescue_frac = s["rescued"] / max(denom, 1)
        n01 = s["b01"]; n10 = s["b10"]
        if n01 + n10 > 0:
            p = binomtest(min(n01, n10), n=n01+n10, p=0.5, alternative="two-sided").pvalue
        else:
            p = 1.0
        rows.append({
            "bucket": b, "n": t,
            "aesop_success_rate": ar,
            "classical_aesop_success_rate": cr,
            "rescued": s["rescued"],
            "rescued_of_failed_rate": rescue_frac,
            "newly_failed": s["newly_failed"],
            "mcnemar_p": float(p),
            "b01": n01, "b10": n10,
        })
        print(f"{b:<15s} {t:>5d} {ar*100:>7.1f}% {cr*100:>9.1f}% "
              f"{rescue_frac*100:>7.1f}% {s['newly_failed']:>6d} {p:>10.3e}")

    # Anomaly-score discrimination among classical failures
    print("\nAnomaly-score of rescued vs still-failed (classical-only theorems):")
    rescued_anom = []; stuck_anom = []
    for r in paired:
        if r["bucket"] == "constructive":
            continue
        if r["aesop_success"]:
            continue
        (rescued_anom if r["classical_aesop_success"] else stuck_anom).append(r["anomaly_score"])
    if len(rescued_anom) >= 5 and len(stuck_anom) >= 5:
        u_stat, u_p = mannwhitneyu(rescued_anom, stuck_anom, alternative="two-sided")
        anom_out = {
            "n_rescued": len(rescued_anom),
            "n_stuck": len(stuck_anom),
            "mean_rescued": float(np.mean(rescued_anom)),
            "mean_stuck": float(np.mean(stuck_anom)),
            "median_rescued": float(np.median(rescued_anom)),
            "median_stuck": float(np.median(stuck_anom)),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_p_two_sided": float(u_p),
        }
        print(f"  rescued:     n={len(rescued_anom):3d}  mean={np.mean(rescued_anom):.4f}  median={np.median(rescued_anom):.4f}")
        print(f"  still stuck: n={len(stuck_anom):3d}  mean={np.mean(stuck_anom):.4f}  median={np.median(stuck_anom):.4f}")
        print(f"  Mann-Whitney U p (two-sided) = {u_p:.3e}")
    else:
        anom_out = {"n_rescued": len(rescued_anom), "n_stuck": len(stuck_anom),
                    "note": "insufficient for Mann-Whitney"}
        print(f"  too few for test: rescued={len(rescued_anom)}, stuck={len(stuck_anom)}")

    # Save JSON
    out = {
        "per_bucket": rows,
        "anomaly_rescue": anom_out,
        "n_paired": len(common),
    }
    with open(ROOT / "results/data/reviewer/classical_ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(ROOT / "results/data/reviewer/classical_ablation_summary.txt", "w") as f:
        f.write("Classical Prefix Ablation — paired results\n")
        f.write("=" * 58 + "\n\n")
        f.write(f"Paired theorems: {len(common)}\n\n")
        f.write(f"{'Bucket':<15s} {'n':>5s} {'aesop':>8s} {'cls;aes':>10s} "
                f"{'rescue':>8s} {'lost':>6s} {'McN p':>10s}\n")
        for r in rows:
            f.write(f"{r['bucket']:<15s} {r['n']:>5d} "
                    f"{100*r['aesop_success_rate']:>7.1f}% "
                    f"{100*r['classical_aesop_success_rate']:>9.1f}% "
                    f"{100*r['rescued_of_failed_rate']:>7.1f}% "
                    f"{r['newly_failed']:>6d} {r['mcnemar_p']:>10.3e}\n")
        f.write("\nDiscordant counts (b01 = aesop fail, classical;aesop succeed; b10 = reverse)\n")
        for r in rows:
            f.write(f"  {r['bucket']:<15s} b01={r['b01']:>3d}  b10={r['b10']:>3d}\n")
        f.write("\nAnomaly-score discrimination among classical-only failures:\n")
        f.write(json.dumps(anom_out, indent=2) + "\n")

    # Figure
    FIG = ROOT / "results" / "figures"
    FIG.mkdir(exist_ok=True)
    aesop_rates = [r["aesop_success_rate"] for r in rows]
    cls_rates = [r["classical_aesop_success_rate"] for r in rows]
    x = np.arange(len(BUCKETS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, aesop_rates, w, label="by aesop", color="#F44336")
    ax.bar(x + w/2, cls_rates, w, label="by classical; aesop", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS, rotation=15)
    ax.set_ylabel("Success rate (60s timeout)")
    ax.set_title("Classical Prefix Ablation: Does Activating Classical Context Rescue Failures?")
    ax.legend()
    for i, r in enumerate(rows):
        resc = r["rescued_of_failed_rate"]
        if resc > 0.01:
            ax.text(x[i] + w/2, cls_rates[i] + 0.015,
                    f"+{resc*100:.0f}% rescued", ha="center", fontsize=8,
                    color="#2196F3")
        if r["newly_failed"] > 0:
            ax.text(x[i] - w/2, aesop_rates[i] + 0.015,
                    f"−{r['newly_failed']} lost", ha="center", fontsize=8,
                    color="#F44336")
    max_rate = max(max(aesop_rates), max(cls_rates))
    ax.set_ylim(0, max(0.1, max_rate * 1.3))
    plt.tight_layout()
    plt.savefig(FIG / "classical_prefix_ablation.png", dpi=300, bbox_inches="tight")
    print(f"\nsaved {FIG/'classical_prefix_ablation.png'}")
    print(f"saved results/data/reviewer/classical_ablation_results.json")
    print(f"saved results/data/reviewer/classical_ablation_summary.txt")


if __name__ == "__main__":
    main()
