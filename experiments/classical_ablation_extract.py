"""Classical prefix ablation: regenerate the 251 prover test files with
`by classical; aesop` in place of `by aesop`. Reuses the header/body
locator from lean.sources so the two runs are paired theorem-by-theorem.

Output test files: ~/prover_eval/classical_ablation/*.lean
Output job spec:   results/data/reviewer/classical_ablation_jobs.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.sources import find_proof_replacement  # noqa: E402
from config import MATHLIB_PATH as MATHLIB  # noqa: E402

OUT_DIR = Path.home() / "prover_eval" / "classical_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lean 4 requires newline-separated tactics inside `by`; `;` at top level
# is a syntax error. Use multi-line form.
TACTIC = " by\n  classical\n  aesop\n"


def main():
    with open(ROOT / "results/data/reviewer/prover_jobs.json") as f:
        jobs = json.load(f)["jobs"]
    print(f"reusing {len(jobs)} prior jobs")

    ablation_jobs = []
    n_skipped = 0
    for idx, t in enumerate(jobs):
        fp = MATHLIB / t["source_file"]
        if not fp.exists():
            n_skipped += 1
            continue
        source = fp.read_text()
        full = t["name"]
        short = full.split(".")[-1]
        rep = find_proof_replacement(source, full)
        if rep is None:
            rep = find_proof_replacement(source, short)
        if rep is None:
            n_skipped += 1
            continue
        body_start, body_end = rep
        if body_end <= body_start:
            n_skipped += 1
            continue

        new_source = source[:body_start] + TACTIC + source[body_end:]
        test_file = OUT_DIR / f"classical_{idx:05d}.lean"
        test_file.write_text(new_source)
        new_t = dict(t)
        new_t["test_file"] = str(test_file)
        ablation_jobs.append(new_t)

    print(f"wrote {len(ablation_jobs)} test files; skipped {n_skipped}")
    out = {
        "tactic": TACTIC.strip(),
        "paired_with": "results/data/reviewer/prover_results.jsonl",
        "jobs": ablation_jobs,
    }
    with open(ROOT / "results/data/reviewer/classical_ablation_jobs.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/data/reviewer/classical_ablation_jobs.json")


if __name__ == "__main__":
    main()
