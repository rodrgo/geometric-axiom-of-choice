"""Tier 3 Step 3.3: Run aesop evaluation on each test file.

For each job in results/data/reviewer/prover_jobs.json, run
  lake env lean <test_file>
under a timeout. Exit code 0 and no `sorry`/`error` in output => success.

Writes streaming JSONL to results/data/reviewer/prover_results.jsonl so we can
tail progress and resume if interrupted.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.lakerunner import try_prove  # noqa: E402

JOBS_FILE = ROOT / "results/data/reviewer/prover_jobs.json"
RESULTS_FILE = ROOT / "results/data/reviewer/prover_results.jsonl"
TIMEOUT_S = 60


def already_done() -> set:
    if not RESULTS_FILE.exists():
        return set()
    done = set()
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add(r["name"])
            except json.JSONDecodeError:
                continue
    return done


def main():
    with open(JOBS_FILE) as f:
        data = json.load(f)
    jobs = data["jobs"]
    done = already_done()
    print(f"Total jobs: {len(jobs)};  already done: {len(done)}")
    pending = [j for j in jobs if j["name"] not in done]

    from collections import Counter
    successes = Counter()
    totals = Counter()
    # prime with any prior results
    if RESULTS_FILE.exists():
        for line in open(RESULTS_FILE):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                totals[r["bucket"]] += 1
                if r.get("success"):
                    successes[r["bucket"]] += 1
            except json.JSONDecodeError:
                pass

    t_start = time.time()
    with open(RESULTS_FILE, "a") as out:
        for i, job in enumerate(pending):
            test_file = Path(job["test_file"])
            success, elapsed, err = try_prove(test_file, TIMEOUT_S)
            result = {
                "name": job["name"],
                "bucket": job["bucket"],
                "depth": job.get("depth"),
                "success": bool(success),
                "elapsed_s": float(elapsed),
                "anomaly_score": job["anomaly_score"],
                "proof_length": job["proof_length"],
                "source_file": job["source_file"],
                "error_snippet": err if not success else "",
            }
            out.write(json.dumps(result) + "\n")
            out.flush()
            totals[job["bucket"]] += 1
            if success:
                successes[job["bucket"]] += 1
            if (i + 1) % 10 == 0 or i + 1 == len(pending):
                elapsed_total = time.time() - t_start
                print(f"[{i+1}/{len(pending)}]  total elapsed {elapsed_total:.1f}s")
                for b in ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]:
                    t = totals[b]
                    s = successes[b]
                    print(f"  {b:<14s}  {s}/{t}  ({100*s/max(t,1):.1f}%)")
    print("\nDONE. Results in results/data/reviewer/prover_results.jsonl")


if __name__ == "__main__":
    main()
