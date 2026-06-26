"""QC Check 1: positive control for `classical; aesop` vs `aesop`.

Hand-crafted Lean 4 goals that *should* benefit from classical reasoning
(excluded middle, double-negation elimination, Peirce's law,
contraposition, case analysis on an undecidable proposition, plus two
sanity cases that should succeed under both). If `classical; aesop`
rescues any, the pipeline is working and the Mathlib-null from the main
ablation is meaningful. If none, the interpretation of the null has to
be narrowed.

Uses the same multi-line `classical / aesop` splice as the main
ablation (semicolon is a Lean 4 syntax error at the top of `by`), and
the same 60-s timeout.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.lakerunner import try_prove  # noqa: E402

OUT_DIR = Path.home() / "prover_eval" / "positive_control"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT_S = 60

# (label, statement -- everything between `example` and `:=`)
TEST_CASES = [
    ("excluded_middle",   "(P : Prop) : P ∨ ¬P"),
    ("double_negation",   "(P : Prop) (h : ¬¬P) : P"),
    ("contraposition",    "(P Q : Prop) (h : ¬Q → ¬P) : P → Q"),
    ("peirce",            "(P Q : Prop) : ((P → Q) → P) → P"),
    ("case_undecidable",  "(P : Prop) (h : P → 1 = 1) (h' : ¬P → 1 = 1) : 1 = 1"),
    ("not_not_em",        "(P : Prop) : ¬¬(P ∨ ¬P)"),
    # Sanity cases: should succeed in both configurations.
    ("trivial_true",      ": True"),
    ("nat_add_zero",      "(n : Nat) : n + 0 = n"),
]

TACTICS = {
    "aesop": "by aesop",
    "classical_aesop": "by\n  classical\n  aesop",
}


def run_one(source: str, test_path: Path, timeout_s: int = TIMEOUT_S):
    test_path.write_text(source)
    return try_prove(test_path, timeout_s=timeout_s)


def main():
    results = []
    print(f"{'case':<20s}  {'aesop':>9s}  {'cls;aes':>9s}  {'rescued':>8s}")
    for label, stmt in TEST_CASES:
        per_case = {"label": label, "statement": stmt}
        for tac_name, tac_body in TACTICS.items():
            source = f"import Mathlib\n\nexample {stmt} := {tac_body}\n"
            test_path = OUT_DIR / f"{label}__{tac_name}.lean"
            success, elapsed, err = run_one(source, test_path)
            per_case[tac_name] = {
                "success": bool(success),
                "elapsed_s": float(elapsed),
                "error": err,
            }
        per_case["classical_helped"] = (
            not per_case["aesop"]["success"] and per_case["classical_aesop"]["success"]
        )
        results.append(per_case)
        print(f"{label:<20s}  {str(per_case['aesop']['success']):>9s}  "
              f"{str(per_case['classical_aesop']['success']):>9s}  "
              f"{str(per_case['classical_helped']):>8s}")

    n_helped = sum(1 for r in results if r["classical_helped"])
    n_aesop = sum(1 for r in results if r["aesop"]["success"])
    n_cls = sum(1 for r in results if r["classical_aesop"]["success"])
    n_sanity_both = sum(
        1 for r in results
        if r["label"].startswith(("trivial", "nat_"))
        and r["aesop"]["success"] and r["classical_aesop"]["success"]
    )
    print()
    print(f"aesop-only succeeded: {n_aesop}/{len(results)}")
    print(f"classical; aesop succeeded: {n_cls}/{len(results)}")
    print(f"classical prefix RESCUED aesop: {n_helped}/{len(results)}")
    print(f"(of which sanity cases succeeded in both: {n_sanity_both})")

    with open(ROOT / "results/data/reviewer/positive_control.json", "w") as f:
        json.dump({
            "timeout_s": TIMEOUT_S,
            "tactics": TACTICS,
            "n_cases": len(results),
            "n_aesop_success": n_aesop,
            "n_classical_aesop_success": n_cls,
            "n_rescued": n_helped,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved results/data/reviewer/positive_control.json")


if __name__ == "__main__":
    main()
