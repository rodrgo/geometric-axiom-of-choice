"""Item 3: shuffled-tactics control for the ReProver+aesop hybrid.

For each of the 251 theorems, pair it with a DIFFERENT theorem in the
same bucket (seed=42, no self-assignment) and run the hybrid pipeline
using the donor's top-8 ReProver tactics. This tests whether the
neural model is providing theorem-specific guidance or just "try a
reasonable opener before aesop."

Splice semantics match scripts/reprover_hybrid.py:
  plain variant: `by\n  <donor_tac>\n  all_goals aesop`
  classical variant (classical buckets only): `by\n  classical\n  <donor_tac>\n  all_goals aesop`
60-s timeout per compile; early-exit on first success.

Stream-writes to results/data/reviewer/reprover_shuffled_results.jsonl; resumes
from partial.
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.lakerunner import try_prove  # noqa: E402
from lean.sources import find_proof_replacement  # noqa: E402
from config import MATHLIB_PATH as MATHLIB  # noqa: E402

OUT_DIR = Path.home() / "prover_eval" / "reprover" / "shuffled"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results/data/reviewer/reprover_shuffled_results.jsonl"
PRIOR = ROOT / "results/data/reviewer/reprover_tacgen_results.jsonl"
TIMEOUT_S = 60
SEED = 42


def render_body(candidate, use_classical=False):
    if use_classical:
        return f"by\n  classical\n  {candidate}\n  all_goals aesop\n"
    return f"by\n  {candidate}\n  all_goals aesop\n"


def try_compile(test_file, timeout_s=TIMEOUT_S):
    success, elapsed, _ = try_prove(test_file, timeout_s=timeout_s)
    return success, elapsed


def done_names():
    if not RESULTS.exists():
        return set()
    out = set()
    with open(RESULTS) as f:
        for line in f:
            try:
                out.add(json.loads(line)["name"])
            except Exception:
                pass
    return out


def build_donor_map(prior):
    """For each theorem, assign a donor from the same bucket (≠ self).

    Uses a random permutation per bucket with a self-swap fixup.
    """
    rng = random.Random(SEED)
    by_bucket = defaultdict(list)
    for name, row in prior.items():
        by_bucket[row["bucket"]].append(name)

    donor = {}
    for bucket, names in by_bucket.items():
        if len(names) < 2:
            # Only one theorem in this bucket; donor = self (degenerate).
            for n in names:
                donor[n] = n
            continue
        shuffled = names.copy()
        rng.shuffle(shuffled)
        # Fix any self-assignments with pairwise swaps.
        for i, n in enumerate(names):
            if shuffled[i] == n:
                j = (i + 1) % len(names)
                # Swap i with j; if j also ends up self-matched, it will
                # be fixed by the next iteration since we're iterating in
                # order over names[] and re-check on j-th position next.
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        # Final sanity check
        for i, n in enumerate(names):
            if shuffled[i] == n:
                # Fall back: pick any other index
                for j in range(len(names)):
                    if j != i and shuffled[j] != names[j] and shuffled[j] != n:
                        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                        break
        for n, d in zip(names, shuffled):
            donor[n] = d
    return donor


def main():
    prior = {}
    with open(PRIOR) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prior[r["name"]] = r
    donor = build_donor_map(prior)
    n_self = sum(1 for k, v in donor.items() if k == v)
    print(f"donor mapping: {len(donor)} theorems; self-mapped: {n_self}")
    if n_self:
        print("  WARNING: some theorems donate to themselves (bucket too small)")

    done = done_names()
    pending = [n for n in prior if n not in done]
    print(f"{len(prior)} theorems; {len(done)} already done; "
          f"{len(pending)} pending")

    from collections import Counter
    tot = Counter(); succ = Counter()
    if RESULTS.exists():
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                tot[r["bucket"]] += 1
                if r.get("any_candidate_succeeded"):
                    succ[r["bucket"]] += 1
            except Exception:
                pass

    t_start = time.time()
    with open(RESULTS, "a") as out:
        for idx, name in enumerate(pending):
            job = prior[name]
            d_name = donor[name]
            donor_tactics = prior.get(d_name, {}).get("tactics") or []
            src_path = MATHLIB / job["source_file"]
            record = {
                "name": name,
                "donor_name": d_name,
                "bucket": job["bucket"],
                "depth": job.get("depth"),
                "anomaly_score": job.get("anomaly_score"),
                "proof_length": job.get("proof_length"),
                "source_file": job["source_file"],
                "any_candidate_succeeded": False,
                "best_candidate": None,
                "best_variant": None,
                "n_candidates_tried": 0,
                "elapsed_total_s": 0.0,
            }
            if not src_path.exists() or not donor_tactics:
                record["skip_reason"] = "src_missing_or_no_donor"
                out.write(json.dumps(record) + "\n")
                out.flush()
                tot[job["bucket"]] += 1
                continue
            source = src_path.read_text()
            short = name.split(".")[-1]
            rep = find_proof_replacement(source, name) \
                or find_proof_replacement(source, short)
            if rep is None:
                record["skip_reason"] = "header_unlocated"
                out.write(json.dumps(record) + "\n")
                out.flush()
                tot[job["bucket"]] += 1
                continue
            body_start, body_end = rep

            variants = [("plain", False)]
            if job["bucket"] != "constructive":
                variants.append(("classical", True))

            elapsed_total = 0.0
            done_thm = False
            for k, cand in enumerate(donor_tactics):
                for vname, use_cls in variants:
                    body = render_body(cand, use_cls)
                    new_source = source[:body_start] + " " + body + source[body_end:]
                    test_file = OUT_DIR / f"shf_{idx:05d}_{k}_{vname}.lean"
                    test_file.write_text(new_source)
                    success, elapsed = try_compile(test_file, TIMEOUT_S)
                    elapsed_total += elapsed
                    record["n_candidates_tried"] += 1
                    try:
                        test_file.unlink()
                    except Exception:
                        pass
                    if success:
                        record["any_candidate_succeeded"] = True
                        record["best_candidate"] = cand
                        record["best_variant"] = vname
                        done_thm = True
                        break
                if done_thm:
                    break
            record["elapsed_total_s"] = elapsed_total

            out.write(json.dumps(record) + "\n")
            out.flush()
            tot[job["bucket"]] += 1
            if record["any_candidate_succeeded"]:
                succ[job["bucket"]] += 1

            if (idx + 1) % 5 == 0 or idx + 1 == len(pending):
                el = time.time() - t_start
                print(f"[{idx+1}/{len(pending)}]  elapsed {el:.1f}s")
                for b in ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]:
                    t = tot[b]; s = succ[b]
                    print(f"  {b:<14s}  {s}/{t}  ({100*s/max(t,1):.1f}%)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
