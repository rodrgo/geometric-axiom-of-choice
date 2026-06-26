"""Lean tactic-head tokens that name classical-reasoning moves.

A theorem whose tactic-head sequence contains any of these tokens has a
*surface-level* classical signal — visible from the proof text alone,
without needing kernel-level analysis. Used by the saliency / ablation
analyses to distinguish proofs whose classicalness shows in their
tactics from those whose classicalness is only kernel-transitive.

This set is conservative — it captures the explicit, lexical markers a
human reader would also flag, not the full closure of every tactic that
*could* invoke classical reasoning under the hood.
"""

from __future__ import annotations

CLASSICAL_MARKERS: frozenset[str] = frozenset({
    "by_contra", "by_cases", "choose", "exfalso", "classical",
    "contrapose", "push_neg", "tauto", "byContradiction",
    "Classical.em", "Classical.choice", "not_not", "of_not_not",
    "by_contradiction",
})
