"""Item 4: classical-prefix hybrid variant, classical theorems only.

The existing hybrid run tried two variants per candidate for classical
theorems:
  plain:     `by\n  <cand>\n  all_goals aesop`
  classical: `by\n  classical\n  <cand>\n  all_goals aesop`

Every hybrid success (9/201 classical) was a plain-variant win, and
for the 192 classical hybrid failures both variants were already
tested and both failed. So the marginal contribution of the
classical-prefix variant is known analytically to be ZERO new
successes beyond plain-only.

What is NOT answered by that analysis: for the 9 plain-wins, would
classical-prefix variant ALSO have worked (or does it hurt)? This
script re-runs classical-prefix-only on those 9 theorems (fast;
9 × up to 8 × 60s ≈ 1 hour worst case, minutes in practice).

We also record a control sweep on ALL 201 classical theorems using
classical-prefix-only to obtain the clean "classical-only hybrid"
number for the paper table.

Output: results/data/reviewer/reprover_classical_only_hybrid_results.jsonl.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.lakerunner import try_prove  # noqa: E402
from lean.sources import find_proof_replacement  # noqa: E402
from config import MATHLIB_PATH as MATHLIB  # noqa: E402

OUT_DIR = Path.home() / "prover_eval" / "reprover" / "classical_hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results/data/reviewer/reprover_classical_only_hybrid_results.jsonl"
PRIOR = ROOT / "results/data/reviewer/reprover_tacgen_results.jsonl"
TIMEOUT_S = 60


def render_body_classical(candidate):
    return f"by\n  classical\n  {candidate}\n  all_goals aesop\n"


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


def main():
    prior = {}
    with open(PRIOR) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prior[r["name"]] = r
    # Classical theorems only
    classical_names = [n for n, r in prior.items() if r["bucket"] != "constructive"]
    print(f"{len(classical_names)} classical theorems")

    done = done_names()
    pending = [n for n in classical_names if n not in done]
    print(f"pending: {len(pending)}")

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
            tactics = job.get("tactics") or []
            src_path = MATHLIB / job["source_file"]
            record = {
                "name": name,
                "bucket": job["bucket"],
                "depth": job.get("depth"),
                "anomaly_score": job.get("anomaly_score"),
                "proof_length": job.get("proof_length"),
                "source_file": job["source_file"],
                "any_candidate_succeeded": False,
                "best_candidate": None,
                "n_candidates_tried": 0,
                "elapsed_total_s": 0.0,
            }
            if not src_path.exists() or not tactics:
                record["skip_reason"] = "src_missing_or_no_tactics"
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

            elapsed_total = 0.0
            for k, cand in enumerate(tactics):
                body = render_body_classical(cand)
                new_source = source[:body_start] + " " + body + source[body_end:]
                test_file = OUT_DIR / f"cls_{idx:05d}_{k}.lean"
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
                for b in ["depth 2", "depth 3-4", "depth 5-6", "depth 7+"]:
                    t = tot[b]; s = succ[b]
                    print(f"  {b:<14s}  {s}/{t}  ({100*s/max(t,1):.1f}%)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
