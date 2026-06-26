#!/usr/bin/env bash
#
# run_all.sh — end-to-end reproduction of the paper, from raw inputs to figures.
#
# Pipeline overview (each phase depends on the previous one):
#
#   Phase 0  Environment check.
#   Phase 1  Heavy external inputs (Mathlib, kernel graph, LeanDojo traces).
#   Phase 2  Main pipeline: partition -> encoders -> depth law -> robustness
#            -> figures.  Reproduces every main table/figure of the paper.
#   Phase 3  Full-source appendix experiment (trains 3 BPE encoders on raw
#            proof source; tables tab:full_source_depth / _within_domain).
#   Phase 4  Operational experiments (aesop / ReProver). Needs a working Lean
#            toolchain; OFF by default.
#
# Toggle phases with environment variables (all default to the safe choice):
#
#   RUN_HEAVY=1         also (re)build the kernel graph + fetch LeanDojo in
#                       Phase 1. Default: only checks they exist.
#   SKIP_FULL_SOURCE=1  skip Phase 3 (the ~814 MB encoder experiment).
#   RUN_OPERATIONAL=1   run Phase 4 (needs $MATHLIB_PATH + lake).
#
# Heavy first-time costs: Mathlib checkout (multi-GB), kernel-graph build
# (hours), LeanDojo traces (~940 MB), encoder training. Once the inputs and
# trained encoders exist, the analysis + figures are minutes.
#
# Run from the repo root:  bash run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Prefer the project venv; fall back to whatever `python` is on PATH.
if [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
echo "[run_all] Using interpreter: $PY"

FS="experiments/full_source"

# ---------------------------------------------------------------------------
# Phase 0 — environment
# ---------------------------------------------------------------------------
echo
echo "================ Phase 0: environment ================"
$PY -c "import numpy, sklearn, scipy, torch, matplotlib" \
  || { echo "Missing deps. Run: $PY -m pip install -r requirements.txt"; exit 1; }
echo "[run_all] Core dependencies present."

# ---------------------------------------------------------------------------
# Phase 1 — heavy external inputs
#   - kernel dependency graph  (needs a compiled Mathlib at $MATHLIB_PATH)
#   - LeanDojo Benchmark 4 tactic traces
# ---------------------------------------------------------------------------
echo
echo "================ Phase 1: heavy external inputs ================"
GRAPH="results/data/stage4v3/decl_graph_raw.jsonl"
LEANDOJO="results/data/leandojo/train.json"

if [ "${RUN_HEAVY:-0}" = "1" ]; then
  # Mathlib must exist first (scripts/setup_mathlib.sh clones + caches it).
  echo "[run_all] RUN_HEAVY=1 — building heavy inputs."
  [ -f "$GRAPH" ]    || $PY scripts/build_decl_graph.py --fetch-cache
  [ -f "$LEANDOJO" ] || $PY scripts/get_leandojo.py
else
  echo "[run_all] RUN_HEAVY not set — only checking that inputs exist."
  missing=0
  [ -f "$GRAPH" ]    || { echo "  MISSING $GRAPH  (scripts/build_decl_graph.py)"; missing=1; }
  [ -f "$LEANDOJO" ] || { echo "  MISSING $LEANDOJO  (scripts/get_leandojo.py)"; missing=1; }
  if [ "$missing" = "1" ]; then
    echo "  -> Set up Mathlib (bash scripts/setup_mathlib.sh), then re-run with"
    echo "     RUN_HEAVY=1 bash run_all.sh,  or fetch the inputs manually first."
    exit 1
  fi
  echo "[run_all] Heavy inputs present."
fi

# ---------------------------------------------------------------------------
# Phase 2 — main pipeline (partition -> encoders -> depth law -> robustness -> figures)
# ---------------------------------------------------------------------------
echo
echo "================ Phase 2: main pipeline ================"
$PY experiments/reproduce.py --skip-operational

# ---------------------------------------------------------------------------
# Phase 3 — full-source encoder appendix
# ---------------------------------------------------------------------------
if [ "${SKIP_FULL_SOURCE:-0}" = "1" ]; then
  echo
  echo "[run_all] SKIP_FULL_SOURCE=1 — skipping Phase 3."
else
  echo
  echo "================ Phase 3: full-source encoder appendix ================"
  $PY "$FS/tokenizer/train_bpe.py"
  $PY "$FS/tokenizer/build_strip_list.py"
  $PY "$FS/data/extract_full_source.py"
  for seed in 0 1 2; do
    $PY "$FS/encoder/train.py" --seed "$seed"
  done
  $PY "$FS/encoder/embed.py"            # all seeds x {raw,stripped,combined}
  $PY "$FS/encoder/sanity_check.py"
  $PY "$FS/analysis/depth_stratified_auc.py"
  $PY "$FS/analysis/within_domain_auc.py"
  $PY "$FS/analysis/reconstruction_loss.py"
  $PY "$FS/analysis/length_residualization.py"
  $PY "$FS/analysis/superlevel_containment.py"
  $PY "$FS/analysis/leakage_diagnostic.py"
  $PY "$FS/analysis/aggregate_results.py"
fi

# ---------------------------------------------------------------------------
# Phase 4 — operational experiments (aesop / ReProver). Needs Lean.
# ---------------------------------------------------------------------------
if [ "${RUN_OPERATIONAL:-0}" = "1" ]; then
  echo
  echo "================ Phase 4: operational (aesop / ReProver) ================"
  : "${MATHLIB_PATH:?Set MATHLIB_PATH to a compiled Mathlib checkout for Phase 4}"
  $PY experiments/reproduce.py --only-stage operational
else
  echo
  echo "[run_all] RUN_OPERATIONAL not set — skipping aesop/ReProver (needs Lean)."
fi

echo
echo "================ Done ================"
echo "Tables/data -> results/data/   Figures -> results/figures/"
echo "Full-source results -> $FS/analysis/{results.json,RESULTS.md}"
