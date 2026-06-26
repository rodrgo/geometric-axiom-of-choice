"""Experiment B: ReProver candidate-tactic validity probe.

For each of the 251 theorems × 8 previously-generated candidate tactics,
test whether the tactic is Lean-valid at the theorem's initial proof
state, without requiring the proof to close. The Lean source splice is

  := by
    <candidate>
    all_goals sorry

which typechecks if the candidate is a legal tactic at that state.
`sorry` emits a warning but not an `error:`, so our success rule
(`returncode == 0` AND no `error:` in stdout+stderr) lets sorry pass.

Success semantics:
  valid = compile clean (no `error:`). Does not require proof closure.

Per theorem we record:
  top1_valid      — was the single best-scoring candidate valid?
  top8_any_valid  — was at least one of the eight valid?
  n_valid         — how many of the eight were valid?

Uses 30-s per-candidate timeout. Streams to JSONL and supports resume.
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

OUT_DIR = Path.home() / "prover_eval" / "reprover" / "validity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results/data/reviewer/reprover_validity_results.jsonl"
PRIOR = ROOT / "results/data/reviewer/reprover_tacgen_results.jsonl"
TIMEOUT_S = 30


def render_body(candidate):
    # 2-space indent under the existing header's trailing `by`.
    return f"by\n  {candidate}\n  all_goals sorry\n"


def try_compile(test_file, timeout_s=TIMEOUT_S):
    # Validity probe: sorry-emitting tactics are still valid parses.
    valid, elapsed, err = try_prove(test_file, timeout_s=timeout_s,
                                     reject_sorry=False)
    err_short = err[:300]
    return valid, elapsed, err_short


def done_names():
    if not RESULTS.exists():
        return set()
    done = set()
    with open(RESULTS) as f:
        for line in f:
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass
    return done


def main():
    # Load prior ReProver results for (name → tactics) mapping and metadata.
    prior = {}
    with open(PRIOR) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prior[r["name"]] = r
    done = done_names()
    pending = [(name, r) for name, r in prior.items() if name not in done]
    print(f"{len(prior)} theorems with candidates; {len(done)} already done; "
          f"{len(pending)} pending")

    from collections import Counter
    tot = Counter(); top1 = Counter(); any8 = Counter()
    if RESULTS.exists():
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                tot[r["bucket"]] += 1
                if r.get("top1_valid"):
                    top1[r["bucket"]] += 1
                if r.get("top8_any_valid"):
                    any8[r["bucket"]] += 1
            except Exception:
                pass

    t_start = time.time()
    with open(RESULTS, "a") as out:
        for idx, (name, job) in enumerate(pending):
            tactics = job.get("tactics", []) or []
            src_path = MATHLIB / job["source_file"]
            if not src_path.exists() or not tactics:
                out.write(json.dumps({
                    "name": name, "bucket": job["bucket"],
                    "depth": job.get("depth"),
                    "anomaly_score": job.get("anomaly_score"),
                    "proof_length": job.get("proof_length"),
                    "top1_valid": False,
                    "top8_any_valid": False,
                    "n_valid": 0,
                    "validity_per_candidate": [],
                    "skip_reason": "src_missing_or_no_tactics",
                }) + "\n")
                out.flush()
                tot[job["bucket"]] += 1
                continue

            source = src_path.read_text()
            short = name.split(".")[-1]
            rep = find_proof_replacement(source, name) \
                or find_proof_replacement(source, short)
            if rep is None:
                out.write(json.dumps({
                    "name": name, "bucket": job["bucket"],
                    "depth": job.get("depth"),
                    "anomaly_score": job.get("anomaly_score"),
                    "proof_length": job.get("proof_length"),
                    "top1_valid": False,
                    "top8_any_valid": False,
                    "n_valid": 0,
                    "validity_per_candidate": [],
                    "skip_reason": "header_unlocated",
                }) + "\n")
                out.flush()
                tot[job["bucket"]] += 1
                continue
            body_start, body_end = rep

            validity_per_candidate = []
            elapsed_per_candidate = []
            for k, cand in enumerate(tactics):
                new_source = source[:body_start] + " " + render_body(cand) + source[body_end:]
                test_file = OUT_DIR / f"val_{idx:05d}_{k}.lean"
                test_file.write_text(new_source)
                valid, elapsed, _err = try_compile(test_file, TIMEOUT_S)
                validity_per_candidate.append(bool(valid))
                elapsed_per_candidate.append(float(elapsed))
                # Clean up to save disk
                try:
                    test_file.unlink()
                except Exception:
                    pass

            top1v = validity_per_candidate[0] if validity_per_candidate else False
            any8v = any(validity_per_candidate)
            nvalid = sum(validity_per_candidate)

            record = {
                "name": name,
                "bucket": job["bucket"],
                "depth": job.get("depth"),
                "anomaly_score": job.get("anomaly_score"),
                "proof_length": job.get("proof_length"),
                "source_file": job["source_file"],
                "validity_per_candidate": validity_per_candidate,
                "elapsed_per_candidate": elapsed_per_candidate,
                "top1_valid": bool(top1v),
                "top8_any_valid": bool(any8v),
                "n_valid": int(nvalid),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

            tot[job["bucket"]] += 1
            if top1v:
                top1[job["bucket"]] += 1
            if any8v:
                any8[job["bucket"]] += 1

            if (idx + 1) % 5 == 0 or idx + 1 == len(pending):
                el = time.time() - t_start
                print(f"[{idx+1}/{len(pending)}]  elapsed {el:.1f}s  "
                      f"(≈{el/(idx+1):.1f}s/theorem)")
                for b in ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]:
                    t = tot[b]
                    print(f"  {b:<14s}  top-1 {top1[b]}/{t}  "
                          f"top-8-any {any8[b]}/{t}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
