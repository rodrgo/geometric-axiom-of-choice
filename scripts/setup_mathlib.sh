#!/usr/bin/env bash
# Set up the compiled Mathlib4 checkout required by the OPERATIONAL experiments
# (aesop / ReProver evaluation) and by scripts/build_decl_graph.py.
#
# This pins the exact Mathlib commit / Lean toolchain used for the paper so the
# kernel dependency graph and the `lake env lean` prover runs match.
#
# Usage:
#   bash scripts/setup_mathlib.sh [TARGET_DIR]
#
# TARGET_DIR defaults to ../mathlib4 (the default config.MATHLIB_PATH). If you
# put it elsewhere, export MATHLIB_PATH=/abs/path/to/mathlib4 before running the
# experiments.
#
# Prerequisites: git, and elan (the Lean toolchain manager). Install elan from
# https://github.com/leanprover/elan if `lake` is not on your PATH.
set -euo pipefail

# Pinned to the snapshot traced for the paper.
MATHLIB_COMMIT="9f0aee2e9bfe008c35fa9672d28e6dd4411d2971"   # master-2026-03-28
LEAN_TOOLCHAIN="leanprover/lean4:v4.29.0-rc8"

TARGET="${1:-$(cd "$(dirname "$0")/.." && pwd)/../mathlib4}"
TARGET="$(mkdir -p "$TARGET" && cd "$TARGET" && pwd)"

echo "[setup_mathlib] Target: $TARGET"
echo "[setup_mathlib] Lean toolchain: $LEAN_TOOLCHAIN  commit: $MATHLIB_COMMIT"

if ! command -v lake >/dev/null 2>&1; then
  echo "ERROR: 'lake' not found. Install elan: https://github.com/leanprover/elan" >&2
  exit 1
fi

if [ ! -d "$TARGET/.git" ]; then
  echo "[setup_mathlib] Cloning mathlib4..."
  git clone https://github.com/leanprover-community/mathlib4.git "$TARGET"
fi

cd "$TARGET"
echo "[setup_mathlib] Checking out pinned commit..."
git fetch --all --quiet
git checkout "$MATHLIB_COMMIT"

echo "[setup_mathlib] Fetching prebuilt .olean cache (multi-GB, may take a while)..."
lake exe cache get

echo
echo "[setup_mathlib] Done. Point the experiments at this checkout with:"
echo "    export MATHLIB_PATH=\"$TARGET\""
echo "Then build the kernel graph with:"
echo "    python scripts/build_decl_graph.py --fetch-cache"
