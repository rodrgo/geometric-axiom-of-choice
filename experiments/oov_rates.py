"""Revision 2a: OOV tactic-head rates by classical depth.

For each stage4v3p proof, computes:
  - total tactic-head tokens (len(invocation_heads))
  - OOV tokens (heads not in the constructive vocabulary)
Aggregates per depth bucket and saves results/data/reviewer/oov_rates_by_depth.json.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def depth_to_bucket(is_classical, depth):
    if not is_classical:
        return "constructive"
    if depth is None:
        return "classical (no BFS)"
    if depth <= 2:
        return "depth 2"
    if depth <= 4:
        return "depth 3-4"
    if depth <= 6:
        return "depth 5-6"
    if depth <= 8:
        return "depth 7-8"
    return "depth 9+"


def main():
    with open(ROOT / "results/data/stage4v3p/proofs.json") as f:
        proofs = json.load(f)
    with open(ROOT / "results/data/stage4v3p/vocab.json") as f:
        vocab = json.load(f)
    with open(ROOT / "results/data/depth_analysis/bfs_distances_full.json") as f:
        bfs = json.load(f)
    vocab_set = set(vocab.keys())

    buckets = defaultdict(lambda: {
        "n_proofs": 0,
        "total_tokens": 0,
        "oov_tokens": 0,
        "proofs_with_any_oov": 0,
        "proofs_with_heavy_oov": 0,  # > 20% OOV
    })
    print(f"Constructive vocab size: {len(vocab_set)}")
    print(f"Total proofs: {len(proofs)}")

    # Per-proof OOV distribution (save alongside for downstream use)
    per_proof_oov = []  # list of (name, bucket, n_total, n_oov)

    for p in proofs:
        heads = p["invocation_heads"]
        n_total = len(heads)
        n_oov = sum(1 for h in heads if h not in vocab_set)
        is_cls = bool(p["is_classical"])
        d = bfs.get(p["name"]) if is_cls else None
        bucket = depth_to_bucket(is_cls, d)

        s = buckets[bucket]
        s["n_proofs"] += 1
        s["total_tokens"] += n_total
        s["oov_tokens"] += n_oov
        if n_oov > 0:
            s["proofs_with_any_oov"] += 1
        if n_total > 0 and n_oov / n_total > 0.20:
            s["proofs_with_heavy_oov"] += 1

        per_proof_oov.append({
            "name": p["name"],
            "bucket": bucket,
            "is_classical": is_cls,
            "depth": d,
            "n_total": n_total,
            "n_oov": n_oov,
        })

    order = ["constructive", "depth 2", "depth 3-4", "depth 5-6",
             "depth 7-8", "depth 9+", "classical (no BFS)"]

    print(f"\n{'bucket':<20s} {'n_proofs':>9s} {'tok_tot':>10s} "
          f"{'oov_tok':>9s} {'tok_rate':>10s} {'any_oov':>10s} {'heavy_oov':>11s}")
    out_buckets = {}
    for b in order:
        s = buckets[b]
        if s["n_proofs"] == 0:
            continue
        tr = s["oov_tokens"] / max(s["total_tokens"], 1)
        pr_any = s["proofs_with_any_oov"] / max(s["n_proofs"], 1)
        pr_heavy = s["proofs_with_heavy_oov"] / max(s["n_proofs"], 1)
        out_buckets[b] = {
            "n_proofs": s["n_proofs"],
            "total_tokens": s["total_tokens"],
            "oov_tokens": s["oov_tokens"],
            "oov_token_rate": tr,
            "proofs_with_any_oov_frac": pr_any,
            "proofs_with_heavy_oov_frac": pr_heavy,
        }
        print(f"{b:<20s} {s['n_proofs']:>9d} {s['total_tokens']:>10d} "
              f"{s['oov_tokens']:>9d} {tr:>10.4%} {pr_any:>10.4%} "
              f"{pr_heavy:>11.4%}")

    out = {
        "vocab_size": len(vocab_set),
        "buckets": out_buckets,
        "notes": {
            "heavy_oov": "OOV fraction > 20% of proof's tactic heads",
            "vocab_source": "results/data/stage4v3p/vocab.json (constructive tactic heads only)",
        },
    }
    out_path = ROOT / "results/data/reviewer/oov_rates_by_depth.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Also save per-proof counts
    import numpy as np
    np.savez_compressed(
        ROOT / "results/data/reviewer/oov_per_proof.npz",
        names=np.array([p["name"] for p in per_proof_oov], dtype=object),
        buckets=np.array([p["bucket"] for p in per_proof_oov], dtype=object),
        is_classical=np.array([p["is_classical"] for p in per_proof_oov]),
        depth=np.array([p["depth"] if p["depth"] is not None else -1 for p in per_proof_oov], dtype=np.int32),
        n_total=np.array([p["n_total"] for p in per_proof_oov], dtype=np.int32),
        n_oov=np.array([p["n_oov"] for p in per_proof_oov], dtype=np.int32),
    )
    print(f"Saved per-proof OOV counts to results/data/reviewer/oov_per_proof.npz")

    # Checkpoint decision per plan
    d2 = out_buckets.get("depth 2", {})
    if d2:
        r = d2["oov_token_rate"]
        print(f"\nCheckpoint: depth-2 OOV token rate = {r:.4%}")
        if r < 0.05:
            print("  < 5%: OOV is not the explanation. Clean-recon experiment is still worth running as a control.")
        elif r < 0.20:
            print("  5-20%: proceed to 2b. Expect some but not dominant OOV effect.")
        else:
            print("  > 20%: proceed to 2b URGENTLY. Result may be largely OOV artefact.")


if __name__ == "__main__":
    main()
