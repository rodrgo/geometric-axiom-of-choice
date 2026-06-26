"""v2 analysis: combined table across aesop, single-shot ReProver,
top-K validity, and ReProver+aesop hybrid. Also per-bucket McNemar
(single-shot vs hybrid, hybrid vs aesop) and anomaly-AUC bootstrap on
hybrid outcomes.
"""
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "results/data/reviewer/neural_prover_summary_v2.json"
OUT_TXT = ROOT / "results/data/reviewer/neural_prover_v2_comparison.txt"
BUCKETS = ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["name"]] = r
    return rows


def per_bucket(rows, success_key="success"):
    out = {b: {"total": 0, "success": 0} for b in BUCKETS}
    for r in rows.values():
        b = r["bucket"]
        if b not in out:
            continue
        out[b]["total"] += 1
        if r.get(success_key):
            out[b]["success"] += 1
    return out


def mcnemar_discordant(rows_a, rows_b, key_a="success", key_b="success"):
    """Per-bucket (b01, b10, p) for paired outcomes (rows_a vs rows_b)."""
    out = {}
    common = set(rows_a) & set(rows_b)
    for b in BUCKETS:
        b01 = 0  # a fail, b success
        b10 = 0  # a success, b fail
        for n in common:
            a = bool(rows_a[n].get(key_a))
            c = bool(rows_b[n].get(key_b))
            if rows_a[n]["bucket"] != b:
                continue
            if (not a) and c:
                b01 += 1
            if a and (not c):
                b10 += 1
        if b01 + b10 > 0:
            p = binomtest(min(b01, b10), n=b01 + b10, p=0.5,
                           alternative="two-sided").pvalue
        else:
            p = 1.0
        out[b] = {"b01": b01, "b10": b10, "p": float(p)}
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
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sub = df.iloc[idx]
        y = sub["fail"].values.astype(int)
        X = sub[feats].values
        v = cv_auc(X, y)
        if not np.isnan(v):
            out.append(v)
    return np.array(out)


def sci(a):
    return {
        "n": int(len(a)),
        "median": float(np.median(a)) if len(a) else None,
        "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None],
    }


def main():
    aesop   = load(ROOT / "results/data/reviewer/prover_results.jsonl")
    ss_rep  = load(ROOT / "results/data/reviewer/reprover_tacgen_results.jsonl")
    validity = load(ROOT / "results/data/reviewer/reprover_validity_results.jsonl")
    hybrid  = load(ROOT / "results/data/reviewer/reprover_hybrid_results.jsonl")

    # Success-rate tables
    aesop_pb   = per_bucket(aesop)
    ss_pb      = per_bucket(ss_rep)
    hyb_pb     = per_bucket(hybrid, success_key="any_candidate_succeeded")
    # Validity has two columns
    v_top1_pb  = {b: {"total": 0, "success": 0} for b in BUCKETS}
    v_any_pb   = {b: {"total": 0, "success": 0} for b in BUCKETS}
    for r in validity.values():
        b = r["bucket"]
        if b not in v_top1_pb:
            continue
        v_top1_pb[b]["total"] += 1
        v_any_pb[b]["total"]  += 1
        if r.get("top1_valid"):
            v_top1_pb[b]["success"] += 1
        if r.get("top8_any_valid"):
            v_any_pb[b]["success"]  += 1

    rows = []
    print(f"{'Bucket':<15s} {'n':>5s} {'aesop':>8s} {'SS ReP':>8s} "
          f"{'val t1':>8s} {'val t8':>8s} {'hybrid':>8s}")
    for b in BUCKETS:
        n = aesop_pb[b]["total"]
        r = {
            "bucket": b, "n": n,
            "aesop": aesop_pb[b]["success"] / max(n, 1),
            "single_shot_reprover": ss_pb[b]["success"] / max(n, 1),
            "validity_top1":  v_top1_pb[b]["success"] / max(v_top1_pb[b]["total"], 1),
            "validity_top8_any": v_any_pb[b]["success"] / max(v_any_pb[b]["total"], 1),
            "hybrid": hyb_pb[b]["success"] / max(hyb_pb[b]["total"], 1),
        }
        rows.append(r)
        print(f"{b:<15s} {n:>5d} "
              f"{100*r['aesop']:>6.1f}% "
              f"{100*r['single_shot_reprover']:>6.1f}% "
              f"{100*r['validity_top1']:>6.1f}% "
              f"{100*r['validity_top8_any']:>6.1f}% "
              f"{100*r['hybrid']:>6.1f}%")

    # Combined classical vs constructive for every setup
    def agg_cl(pb):
        cons = pb["constructive"]
        c = {"constructive": {"n": cons["total"], "s": cons["success"],
                              "rate": cons["success"] / max(cons["total"], 1)}}
        cn = sum(pb[b]["total"] for b in BUCKETS if b != "constructive")
        cs = sum(pb[b]["success"] for b in BUCKETS if b != "constructive")
        c["classical_combined"] = {"n": cn, "s": cs, "rate": cs / max(cn, 1)}
        return c

    aggs = {
        "aesop": agg_cl(aesop_pb),
        "single_shot_reprover": agg_cl(ss_pb),
        "validity_top8_any": agg_cl(v_any_pb),
        "hybrid": agg_cl(hyb_pb),
    }
    print("\nCombined:")
    for k, v in aggs.items():
        c = v["constructive"]; cl = v["classical_combined"]
        print(f"  {k:<24s} cons {c['s']}/{c['n']}={100*c['rate']:.1f}%  "
              f"class {cl['s']}/{cl['n']}={100*cl['rate']:.1f}%")

    # McNemar paired tests
    mc_aesop_vs_hybrid = mcnemar_discordant(aesop, hybrid, "success", "any_candidate_succeeded")
    mc_ss_vs_hybrid = mcnemar_discordant(ss_rep, hybrid, "success", "any_candidate_succeeded")
    print("\nMcNemar: hybrid vs aesop (per bucket; b01=hybrid-only succ, b10=aesop-only succ):")
    for b in BUCKETS:
        c = mc_aesop_vs_hybrid[b]
        print(f"  {b:<14s} b01={c['b01']:>3d}  b10={c['b10']:>3d}  p={c['p']:.3e}")

    # Anomaly-AUC bootstrap on HYBRID outcomes
    df = pd.DataFrame([{
        "name": r["name"],
        "bucket": r["bucket"],
        "is_classical": r["bucket"] != "constructive",
        "fail": 0 if r.get("any_candidate_succeeded") else 1,
        "anomaly_score": float(r["anomaly_score"]),
        "proof_length": float(r["proof_length"]),
        "log_proof_length": float(np.log1p(r["proof_length"])),
    } for r in hybrid.values()
       if r.get("anomaly_score") is not None and r.get("proof_length") is not None])

    print(f"\nHybrid anomaly-AUC bootstrap (N={len(df)}, fail={int(df['fail'].sum())}):")
    anom_out = {}
    for label, sub in [("overall", df),
                        ("within_classical", df[df["is_classical"]].reset_index(drop=True)),
                        ("within_constructive", df[~df["is_classical"]].reset_index(drop=True))]:
        if len(sub) < 20 or sub["fail"].nunique() != 2:
            anom_out[label] = {"skip_reason": "too few data",
                                "n": len(sub),
                                "n_fail": int(sub["fail"].sum()) if len(sub) else 0}
            print(f"  {label}: skipped (N={len(sub)})")
            continue
        lo = boot_auc(sub, ["log_proof_length"])
        fu = boot_auc(sub, ["log_proof_length", "anomaly_score"])
        n = min(len(lo), len(fu))
        diff = fu[:n] - lo[:n]
        anom_out[label] = {
            "n": len(sub), "n_fail": int(sub["fail"].sum()),
            "length_only": sci(lo),
            "length_plus_anomaly": sci(fu),
            "improvement": sci(diff),
            "prob_improvement_positive": float(np.mean(diff > 0)),
        }
        ci = lambda a: f"[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]"
        print(f"  {label:<20s} len {np.median(lo):.3f} {ci(lo)}  "
              f"full {np.median(fu):.3f} {ci(fu)}  "
              f"Δ {np.median(diff):+.3f} {ci(diff)}  P(Δ>0) {np.mean(diff>0):.3f}")

    # Save JSON
    out = {
        "per_bucket": rows,
        "combined": aggs,
        "mcnemar_hybrid_vs_aesop": mc_aesop_vs_hybrid,
        "mcnemar_hybrid_vs_single_shot": mc_ss_vs_hybrid,
        "hybrid_anomaly_auc": anom_out,
        "note": "All probes paired on the same 251 theorems; 60s timeout; success = exit 0, no sorry, no error: (validity relaxes to allow sorry warning).",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # Text dump
    lines = ["Neural-prover comparison v2", "=" * 70, ""]
    lines.append(f"{'Bucket':<15s} {'n':>5s} {'aesop':>8s} {'SS ReP':>8s} "
                 f"{'val t1':>8s} {'val t8':>8s} {'hybrid':>8s}")
    for r in rows:
        lines.append(f"{r['bucket']:<15s} {r['n']:>5d} "
                     f"{100*r['aesop']:>6.1f}% "
                     f"{100*r['single_shot_reprover']:>6.1f}% "
                     f"{100*r['validity_top1']:>6.1f}% "
                     f"{100*r['validity_top8_any']:>6.1f}% "
                     f"{100*r['hybrid']:>6.1f}%")
    lines.append("")
    lines.append("Combined:")
    for k, v in aggs.items():
        c = v["constructive"]; cl = v["classical_combined"]
        lines.append(f"  {k:<24s} cons {c['s']}/{c['n']}={100*c['rate']:.1f}%  "
                     f"class {cl['s']}/{cl['n']}={100*cl['rate']:.1f}%")
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Figure: 4-way bars
    x = np.arange(len(BUCKETS))
    w = 0.2
    aesop_r  = [r["aesop"] for r in rows]
    ss_r     = [r["single_shot_reprover"] for r in rows]
    val_r    = [r["validity_top8_any"] for r in rows]
    hyb_r    = [r["hybrid"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 1.5*w, aesop_r,  w, label="aesop",                  color="#F44336")
    ax.bar(x - 0.5*w, ss_r,     w, label="ReProver single-shot",   color="#673AB7")
    ax.bar(x + 0.5*w, val_r,    w, label="ReProver top-8 valid",   color="#2196F3")
    ax.bar(x + 1.5*w, hyb_r,    w, label="ReProver + aesop hybrid", color="#009688")
    ax.set_xticks(x); ax.set_xticklabels(BUCKETS, rotation=15)
    ax.set_ylabel("Success rate (60s timeout; validity: no proof closure)")
    ax.set_title("Neural prover probes on the same 251 Mathlib theorems")
    ax.legend(loc="upper right", fontsize=9)
    max_h = max(max(aesop_r), max(ss_r), max(val_r), max(hyb_r))
    ax.set_ylim(0, max_h * 1.15 + 0.05)
    plt.tight_layout()
    FIG = ROOT / "results" / "figures"
    FIG.mkdir(exist_ok=True)
    plt.savefig(FIG / "neural_prover_comparison_v2.png", dpi=300, bbox_inches="tight")
    print(f"\nsaved {FIG/'neural_prover_comparison_v2.png'}")
    print(f"saved {OUT_JSON}\nsaved {OUT_TXT}")


if __name__ == "__main__":
    main()
