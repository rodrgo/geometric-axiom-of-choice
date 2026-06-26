"""Item 1: paired overlap table, aesop vs ReProver+aesop hybrid.

Joins prover_results.jsonl (aesop) and reprover_hybrid_results.jsonl
(hybrid) by theorem name. Produces per-bucket counts of
  both / aesop_only / hybrid_only / neither,
per-bucket McNemar, and a Fisher's-exact on hybrid success
constructive-vs-classical.
"""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]


def load_jsonl_map(path, key):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["name"]] = (bool(r.get(key)), r.get("bucket"))
    return out


def main():
    aesop = load_jsonl_map(ROOT / "results/data/reviewer/prover_results.jsonl", "success")
    hyb   = load_jsonl_map(ROOT / "results/data/reviewer/reprover_hybrid_results.jsonl",
                             "any_candidate_succeeded")
    common = sorted(set(aesop) & set(hyb))
    print(f"aesop={len(aesop)}  hybrid={len(hyb)}  paired={len(common)}")

    table = defaultdict(lambda: {"both": 0, "aesop_only": 0, "hybrid_only": 0,
                                 "neither": 0, "total": 0})
    for name in common:
        a, b_a = aesop[name]
        h, b_h = hyb[name]
        bucket = b_a  # already identical
        tbl = table[bucket]
        tbl["total"] += 1
        if a and h:
            tbl["both"] += 1
        elif a and not h:
            tbl["aesop_only"] += 1
        elif not a and h:
            tbl["hybrid_only"] += 1
        else:
            tbl["neither"] += 1

    classical = {"both": 0, "aesop_only": 0, "hybrid_only": 0,
                 "neither": 0, "total": 0}
    for b in BUCKETS:
        if b == "constructive":
            continue
        for k in classical:
            classical[k] += table[b][k]

    # Print table
    hdr = f"{'Bucket':<15s} {'both':>5s} {'aesop_only':>12s} {'hybrid_only':>12s} {'neither':>8s} {'n':>5s}"
    print(hdr)
    for b in BUCKETS:
        t = table[b]
        print(f"{b:<15s} {t['both']:>5d} {t['aesop_only']:>12d} "
              f"{t['hybrid_only']:>12d} {t['neither']:>8d} {t['total']:>5d}")
    print(f"{'classical all':<15s} {classical['both']:>5d} "
          f"{classical['aesop_only']:>12d} {classical['hybrid_only']:>12d} "
          f"{classical['neither']:>8d} {classical['total']:>5d}")

    # Net change
    a_cons = table["constructive"]["both"] + table["constructive"]["aesop_only"]
    h_cons = table["constructive"]["both"] + table["constructive"]["hybrid_only"]
    a_cls  = classical["both"] + classical["aesop_only"]
    h_cls  = classical["both"] + classical["hybrid_only"]
    print(f"\nAesop successes:  constructive {a_cons}, classical {a_cls}, total {a_cons + a_cls}")
    print(f"Hybrid successes: constructive {h_cons}, classical {h_cls}, total {h_cons + h_cls}")
    print(f"Discordant pairs (hybrid_only − aesop_only):")
    for b in BUCKETS:
        t = table[b]
        print(f"  {b:<15s} hybrid_only={t['hybrid_only']}  aesop_only={t['aesop_only']}  "
              f"net={t['hybrid_only'] - t['aesop_only']:+d}")

    # McNemar per bucket (+ constructive)
    mc = {}
    print(f"\nMcNemar (b01 = hybrid-only, b10 = aesop-only, two-sided):")
    for b in BUCKETS:
        t = table[b]
        n01 = t["hybrid_only"]
        n10 = t["aesop_only"]
        if n01 + n10 > 0:
            p = binomtest(min(n01, n10), n=n01+n10, p=0.5,
                          alternative="two-sided").pvalue
        else:
            p = 1.0
        mc[b] = {"b01": n01, "b10": n10, "p": float(p)}
        print(f"  {b:<15s} b01={n01}  b10={n10}  p={p:.4f}")

    # Fisher's exact on hybrid constructive vs classical
    cons_succ = h_cons
    cons_fail = table["constructive"]["total"] - h_cons
    cls_succ  = h_cls
    cls_fail  = classical["total"] - h_cls
    odds, p_fisher = fisher_exact([[cons_succ, cons_fail], [cls_succ, cls_fail]])
    print(f"\nFisher's exact, HYBRID constructive vs classical success:")
    print(f"  constructive {cons_succ}/{table['constructive']['total']} = "
          f"{100*cons_succ/max(table['constructive']['total'],1):.1f}%")
    print(f"  classical    {cls_succ}/{classical['total']} = "
          f"{100*cls_succ/max(classical['total'],1):.1f}%")
    print(f"  odds ratio = {odds:.3f}, p = {p_fisher:.3e}")

    # Also Fisher on aesop for reference
    a_cons_succ = a_cons
    a_cons_fail = table["constructive"]["total"] - a_cons
    a_cls_succ  = a_cls
    a_cls_fail  = classical["total"] - a_cls
    odds_a, p_fisher_a = fisher_exact([[a_cons_succ, a_cons_fail],
                                         [a_cls_succ, a_cls_fail]])
    print(f"\nFisher's exact, AESOP constructive vs classical success (for reference):")
    print(f"  odds ratio = {odds_a:.3f}, p = {p_fisher_a:.3e}")

    out = {
        "per_bucket": {b: dict(t) for b, t in table.items()},
        "classical_aggregate": classical,
        "n_paired": len(common),
        "aesop_totals": {
            "constructive_success": a_cons,
            "classical_success": a_cls,
            "total_success": a_cons + a_cls,
        },
        "hybrid_totals": {
            "constructive_success": h_cons,
            "classical_success": h_cls,
            "total_success": h_cons + h_cls,
        },
        "mcnemar_per_bucket": mc,
        "fisher_hybrid": {
            "odds_ratio": float(odds),
            "p": float(p_fisher),
            "constructive": [cons_succ, cons_fail],
            "classical": [cls_succ, cls_fail],
        },
        "fisher_aesop": {
            "odds_ratio": float(odds_a),
            "p": float(p_fisher_a),
        },
    }
    with open(ROOT / "results/data/reviewer/aesop_vs_hybrid_overlap.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/data/reviewer/aesop_vs_hybrid_overlap.json")


if __name__ == "__main__":
    main()
