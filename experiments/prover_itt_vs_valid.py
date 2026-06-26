"""QC Check 2: ITT vs valid-only analysis of the prover experiment.

Classifies each of the 251 aesop results into one of:
  success, timeout, aesop_failure, splice_error, unrelated_lean_error, other_failure
Re-aggregates per-bucket success rates under two policies:
  ITT        : all 251 theorems contribute (our current headline)
  valid-only : drop splice_error and unrelated_lean_error

Outputs results/data/reviewer/prover_itt_vs_valid.json.
"""
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]

UNKNOWN_TACTIC_RE = re.compile(r"unknown tactic")
UNEXPECTED_TOKEN_RE = re.compile(r"unexpected token")
AESOP_RE = re.compile(r"aesop")
AESOP_FAILED_RE = re.compile(r"aesop.*(failed|made no progress|exhausted)", re.DOTALL)
TYPE_MISMATCH_RE = re.compile(r"type mismatch")
UNKNOWN_ID_RE = re.compile(r"unknown (identifier|constant|declaration)")


def classify(r):
    if r.get("success"):
        return "success"
    err = (r.get("error_snippet") or "").lower()
    if "timeout" in err or r.get("elapsed_s", 0) >= 59.5:
        # 59.5s threshold because timeout message may be truncated
        if err == "timeout":
            return "timeout"
    if "timeout" == err.strip():
        return "timeout"
    # Splice-placement errors: header located but `by aesop` was grafted
    # into a position where a following tactic produced "unknown tactic" /
    # "unexpected token" — the error is about Lean parsing, not about aesop.
    if (UNKNOWN_TACTIC_RE.search(err) or UNEXPECTED_TOKEN_RE.search(err)) \
            and not AESOP_FAILED_RE.search(err):
        return "splice_error"
    # Genuine aesop failure (most common path)
    if AESOP_FAILED_RE.search(err):
        return "aesop_failure"
    # Other Lean-level diagnostics unrelated to aesop's search outcome
    if TYPE_MISMATCH_RE.search(err) or UNKNOWN_ID_RE.search(err):
        return "unrelated_lean_error"
    # Catch-all
    return "other_failure"


def aggregate(rows, policy):
    by_bucket = defaultdict(lambda: {"total": 0, "success": 0})
    for r in rows:
        if policy == "valid_only" and r["category"] in ("splice_error", "unrelated_lean_error"):
            continue
        by_bucket[r["bucket"]]["total"] += 1
        if r["category"] == "success":
            by_bucket[r["bucket"]]["success"] += 1
    return by_bucket


def main():
    rows = []
    with open(ROOT / "results/data/reviewer/prover_results.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["category"] = classify(r)
            rows.append(r)
    print(f"loaded {len(rows)} aesop results")

    err_per_bucket = defaultdict(Counter)
    for r in rows:
        err_per_bucket[r["bucket"]][r["category"]] += 1
    for b in BUCKETS:
        print(f"  {b}:  " + ", ".join(f"{k}={v}" for k, v in err_per_bucket[b].most_common()))

    itt = aggregate(rows, "itt")
    valid = aggregate(rows, "valid_only")

    print()
    print(f"{'Bucket':<15s} {'ITT n':>6s} {'ITT %':>7s} {'Valid n':>8s} {'Valid %':>8s}")
    out_per_bucket = []
    for b in BUCKETS:
        itt_b = itt[b]
        val_b = valid[b]
        itt_r = 100 * itt_b["success"] / max(itt_b["total"], 1)
        val_r = 100 * val_b["success"] / max(val_b["total"], 1)
        print(f"{b:<15s} {itt_b['total']:>6d} {itt_r:>6.1f}% "
              f"{val_b['total']:>7d} {val_r:>7.1f}%")
        out_per_bucket.append({
            "bucket": b,
            "itt_total": itt_b["total"],
            "itt_success": itt_b["success"],
            "itt_rate": itt_r / 100,
            "valid_total": val_b["total"],
            "valid_success": val_b["success"],
            "valid_rate": val_r / 100,
        })

    # Combined classical ITT vs valid
    c_itt_tot = sum(itt[b]["total"] for b in BUCKETS if b != "constructive")
    c_itt_suc = sum(itt[b]["success"] for b in BUCKETS if b != "constructive")
    c_val_tot = sum(valid[b]["total"] for b in BUCKETS if b != "constructive")
    c_val_suc = sum(valid[b]["success"] for b in BUCKETS if b != "constructive")
    print(f"\nCombined classical: ITT {c_itt_suc}/{c_itt_tot} = {100*c_itt_suc/max(c_itt_tot,1):.2f}%  "
          f"Valid {c_val_suc}/{c_val_tot} = {100*c_val_suc/max(c_val_tot,1):.2f}%")

    out = {
        "policy_notes": {
            "itt": "All 251 theorems contribute; failures include splice errors and unrelated Lean diagnostics.",
            "valid_only": "Drops theorems whose non-success was a splice-location error or a Lean diagnostic unrelated to aesop's search.",
        },
        "per_bucket": out_per_bucket,
        "error_breakdown": {b: dict(err_per_bucket[b]) for b in BUCKETS},
        "classical_combined": {
            "itt": {"total": c_itt_tot, "success": c_itt_suc,
                     "rate": c_itt_suc / max(c_itt_tot, 1)},
            "valid_only": {"total": c_val_tot, "success": c_val_suc,
                            "rate": c_val_suc / max(c_val_tot, 1)},
        },
    }
    with open(ROOT / "results/data/reviewer/prover_itt_vs_valid.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/data/reviewer/prover_itt_vs_valid.json")


if __name__ == "__main__":
    main()
