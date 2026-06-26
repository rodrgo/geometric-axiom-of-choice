"""Central path configuration.

Every script that reads or writes under ``results/data``, ``results/figures``,
or the external Mathlib checkout imports its paths from here rather than
hardcoding them.

``MATHLIB_PATH`` points *outside* this repository and is overridable with an
environment variable so the code is portable across machines: a compiled
Mathlib4 checkout, needed by ``scripts/build_decl_graph.py`` and by the
operational (prover / ReProver) experiments that shell out to ``lake env
lean``.  Default: ``<repo>/../mathlib4``.

The data root can also be redirected with ``AOC_DATA_ROOT`` (used for
reproducing against a pre-computed data directory).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository roots
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # src/config.py -> repo root
RESULTS_ROOT = ROOT / "results"
DATA_ROOT = Path(os.environ.get("AOC_DATA_ROOT", RESULTS_ROOT / "data"))
FIGURES_ROOT = RESULTS_ROOT / "figures"

# ---------------------------------------------------------------------------
# External dependencies (overridable via environment variables)
# ---------------------------------------------------------------------------

# Compiled Mathlib4 checkout used by `lake env lean` (operational experiments)
# and by scripts/build_decl_graph.py. Override with the MATHLIB_PATH env var.
MATHLIB_PATH = Path(
    os.environ.get("MATHLIB_PATH", ROOT.parent / "mathlib4")
)

# ---------------------------------------------------------------------------
# data/ subdirectories
# ---------------------------------------------------------------------------

LEANDOJO_DIR = DATA_ROOT / "leandojo"

# Lean kernel partition + statement encoder
STAGE4V3_DIR = DATA_ROOT / "stage4v3"
# Lean proof encoder (the main representation in the paper)
STAGE4V3P_DIR = DATA_ROOT / "stage4v3p"

# Analysis outputs
DEPTH_ANALYSIS_DIR = DATA_ROOT / "depth_analysis"
SUPPORT_DIR = DATA_ROOT / "support"
OT_DIR = DATA_ROOT / "ot"
CONTROLS_DIR = DATA_ROOT / "controls"
REVIEWER_DIR = DATA_ROOT / "reviewer"

# Kernel-graph asset and its cached classical-depth map.
DECL_GRAPH_JSONL = STAGE4V3_DIR / "decl_graph_raw.jsonl"
CLASSICAL_DEPTHS_JSON = STAGE4V3_DIR / "classical_depths.json"
