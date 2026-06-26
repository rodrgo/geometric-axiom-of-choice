"""Reproduce the paper's results from a prepared data directory.

Runs the experiment scripts in canonical dependency order. Each script runs
in this same Python process via ``runpy.run_path(..., run_name='__main__')``.

Before running, the heavy external inputs must be in place (see README,
"Heavy external dependencies"):
  - results/data/leandojo/{train,val,test}.json   (scripts/get_leandojo.py)
  - results/data/stage4v3/decl_graph_raw.jsonl    (scripts/build_decl_graph.py)
  - a compiled Mathlib4 checkout                   (scripts/setup_mathlib.sh)
    — only needed for the `operational` group and axis_comparison.

Stage groups
------------
- data      Kernel partition + proof/statement extraction. Needs the kernel
            graph and LeanDojo splits.
- encoders  Train the proof and statement denoising encoders.
- depth_law The three measurements + the support sweep, statement-vs-proof,
            and axis comparison. depth_knn_auc runs first because it writes
            the BFS-distance hub every later analysis reads.
- robust    Robustness controls (marker ablation, OT, within-domain,
            mixed-effects, OOV, multi-seed).
- figures   Paper figures (hero, three-measurements, proof-representation,
            case-study JSON for the hero panel).
- operational  aesop / ReProver evaluation. Shells out to `lake env lean`
            against the compiled Mathlib checkout — slow; needs Mathlib.
            Skipped with --skip-operational.

Examples
--------
    python experiments/reproduce.py --list
    python experiments/reproduce.py --dry-run
    python experiments/reproduce.py --only-stage depth_law
    python experiments/reproduce.py --from-stage robust
    python experiments/reproduce.py --skip-operational
"""
from __future__ import annotations

import argparse
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
EXP = ROOT / "experiments"

PIPELINE: list[tuple[str, str, list[str]]] = [
    ("data", "Kernel partition + proof/statement extraction", [
        "partition",
        "extract_proofs",
    ]),
    ("encoders", "Train proof + statement denoising encoders", [
        "train_proof_encoder",
        "train_statement_encoder",
    ]),
    ("depth_law", "The depth law: three measurements + comparisons", [
        "depth_knn_auc",          # writes bfs_distances_full.json (hub) + k-NN depth table
        "label_free_detection",   # tab:depth_strat (all methods) + tab:superlevel_containment
        "reconstruction_loss",    # tab:reconstruction_loss
        "reconstruction_figures", # reconstruction_loss_by_depth.png
        "fit_lambdas",            # tab:lambda-implied (mixture weights per bucket)
        "mixture_model_fit",      # sec:lambda-fit: lambda-from-AUC predicts loss + containment
        "support_sweep",          # tab:lean_support_full (full hyperparameter sweep)
        "statement_vs_proof",     # tab:lean_statements
        "axis_comparison",        # tab:axis_comparison (needs kernel graph)
    ]),
    ("robust", "Robustness controls", [
        "marker_ablation",        # tab:depth_ablation + depth_ablation_comparison.png
        "sliced_wasserstein_depth",  # tab:depth_ot
        "optimal_transport",      # tab:ot_results
        "within_domain",          # tab:within_domain
        "mixed_effects",          # App F.4 (writes exact_matching_results.json)
        "oov_rates",              # App F.5
        "oov_reconstruction",     # App F.5
        "multiseed",              # App F.6 (retrains the encoder 5x)
        # Author/topic confound controls (fig:confound_controls). The author
        # step git-blames Mathlib at the LeanDojo commit — needs the clone.
        "confound_authors_extract",
        "confound_author_analysis",
        "confound_topic_analysis",
        "confound_figure",
    ]),
    ("figures", "Paper figures", [
        "case_study_gradient",        # case_study_gradient.json (hero panel b)
        "hero_figure",                # hero_figure.png
        "three_measurements_figure",  # three_measurements_optB.png
        "proof_representation_figure", # proof_representation.png
    ]),
]

OPERATIONAL: list[tuple[str, str, list[str]]] = [
    ("operational", "aesop / ReProver evaluation (needs Mathlib + lake)", [
        "positive_control",
        "prover_extract",
        "prover_run",
        "prover_analyze",
        "prover_itt_vs_valid",
        "anomaly_auc_bootstrap",
        "classical_ablation_extract",
        "classical_ablation_run",
        "classical_ablation_analyze",
        "reprover_eval",
        "reprover_validity",
        "reprover_hybrid",
        "reprover_classical_hybrid",
        "reprover_shuffled_control",
        "reprover_analyze",
        "reprover_v2_analyze",
        "aesop_vs_hybrid_overlap",
    ]),
]


def all_groups(args) -> list[tuple[str, str, list[str]]]:
    out = list(PIPELINE)
    if not args.skip_operational:
        out.extend(OPERATIONAL)
    return out


def filter_groups(groups, args):
    ids = [g[0] for g in groups]
    if args.only_stage:
        if args.only_stage not in ids:
            sys.exit(f"--only-stage: unknown stage {args.only_stage!r}. Known: {', '.join(ids)}")
        return [g for g in groups if g[0] == args.only_stage]
    if args.from_stage:
        if args.from_stage not in ids:
            sys.exit(f"--from-stage: unknown stage {args.from_stage!r}. Known: {', '.join(ids)}")
        return groups[ids.index(args.from_stage):]
    return groups


def run_script(name: str) -> tuple[bool, float, str]:
    script = EXP / f"{name}.py"
    if not script.exists():
        return False, 0.0, f"missing script: {script}"
    t0 = time.time()
    try:
        runpy.run_path(str(script), run_name="__main__")
        return True, time.time() - t0, ""
    except SystemExit as e:
        ok = (e.code is None) or (e.code == 0)
        return ok, time.time() - t0, "" if ok else f"sys.exit({e.code})"
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, f"{type(e).__name__}: {e}"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Reproduce the paper pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="Print the full plan and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print what would run, then exit.")
    p.add_argument("--from-stage", metavar="ID", help="Skip everything before this stage id.")
    p.add_argument("--only-stage", metavar="ID", help="Run only this single stage.")
    p.add_argument("--skip-operational", action="store_true",
                   help="Skip the aesop/ReProver group (needs Mathlib + lake).")
    p.add_argument("--keep-going", action="store_true",
                   help="Don't abort on the first failing script.")
    args = p.parse_args()

    groups = all_groups(args)

    if args.list:
        for sid, label, scripts in groups:
            print(f"[{sid}] {label}")
            for s in scripts:
                print(f"    {s}")
        return 0

    groups = filter_groups(groups, args)
    print(f"Pipeline: {sum(len(g[2]) for g in groups)} scripts across {len(groups)} stages")
    if args.dry_run:
        for sid, label, scripts in groups:
            print(f"  [{sid}] {label}  ({len(scripts)} scripts)")
            for s in scripts:
                print(f"      {s}")
        return 0

    t_start = time.time()
    failures: list[tuple[str, str, str]] = []
    for sid, label, scripts in groups:
        print(f"\n{'='*70}\n[{sid}] {label}\n{'='*70}")
        for s in scripts:
            print(f"\n--- {s} ---")
            ok, elapsed, err = run_script(s)
            if ok:
                print(f"--- {s}: OK ({elapsed:.1f}s) ---")
            else:
                print(f"--- {s}: FAIL ({elapsed:.1f}s) — {err}")
                failures.append((sid, s, err))
                if not args.keep_going:
                    print("\nAborting (use --keep-going to continue past failures).")
                    print(f"Total elapsed: {time.time()-t_start:.1f}s")
                    return 1

    print(f"\n{'='*70}\nDone in {time.time()-t_start:.1f}s. {len(failures)} failure(s).")
    for sid, s, err in failures:
        print(f"  [{sid}] {s}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
