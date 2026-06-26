"""Build the exact set of BPE token IDs to remove for the leakage ablation.

For each classical-machinery string, run it through the trained BPE and
record the resulting token IDs. Single-token results are clean
"atomic" strips. Multi-token results are flagged as partial: we still
add every ID to the strip set, but we record which IDs came from a
partial decomposition so the analysis caption can note it.

Methodological note: because the BPE was trained on constructive
proofs only, classical-only n-grams (e.g. ``classical.choice``) tend
to decompose into multiple tokens. This is acceptable — the analogous
behavior at the head level is that markers map to UNK. The partial-strip
flag tells us when a strip is heuristic rather than exact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import normalize_lean_source  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

TOKENIZER_PATH = ROOT / "experiments/full_source/tokenizer/artifacts/bpe_32k.json"
OUT_PATH = ROOT / "experiments/full_source/tokenizer/artifacts/strip_list.json"

# Tactic-level markers (same set as the head-level ablation, except we
# also include the snake_case identifier forms that show up in source).
TACTIC_MARKERS = [
    "by_contra", "by_cases", "choose", "exfalso", "classical",
    "contrapose", "push_neg", "tauto",
]

# Fully-qualified classical-machinery identifiers. These are the names
# that would be cited in a proof body.
NAMED_CLASSICAL = [
    "Classical.em",
    "Classical.choice",
    "Classical.byContradiction",
    "Classical.dec",
    "Classical.decEq",
    "Classical.indefiniteDescription",
    "Classical.propDecidable",
    "Classical.skolem",
    "propext",
    "Quot.exists_rep",
    "Set.indicator",
    "Function.surjInv",
]

# Short tokens we considered but EXCLUDE because of overload risk.
# Keeping the list here as documentation.
EXCLUDED_FOR_OVERLOAD = ["em", "dne", "not_not"]


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))

    entries = []
    atomic_ids: set[int] = set()
    partial_ids: set[int] = set()

    for src in TACTIC_MARKERS + NAMED_CLASSICAL:
        norm = normalize_lean_source(src)
        enc = tok.encode(norm)
        ids = enc.ids
        toks = [tok.id_to_token(i) for i in ids]
        is_atomic = len(ids) == 1
        entries.append({
            "source": src,
            "normalized": norm,
            "token_ids": ids,
            "tokens": toks,
            "atomic": is_atomic,
        })
        if is_atomic:
            atomic_ids.update(ids)
        else:
            partial_ids.update(ids)

    # Two strip variants:
    #   atomic_strip_ids   - IDs from sources that BPE encoded as a
    #                        single token. This is the safe headline:
    #                        every removed ID is a complete classical-
    #                        machinery name, with no overload risk.
    #   combined_strip_ids - atomic ∪ partial. Aggressive: includes
    #                        constituent subwords from multi-token
    #                        decompositions, some of which (".",
    #                        "dec", etc.) appear in non-classical
    #                        contexts. Use as an upper bound on what
    #                        stripping can remove, not as the headline.
    combined_ids = atomic_ids | partial_ids
    payload = {
        "atomic_strip_ids": sorted(atomic_ids),
        "combined_strip_ids": sorted(combined_ids),
        "atomic_strip_count": len(atomic_ids),
        "combined_strip_count": len(combined_ids),
        "entries": entries,
        "excluded_for_overload": EXCLUDED_FOR_OVERLOAD,
        "notes": (
            "Headline ablation uses atomic_strip_ids only. The combined "
            "set includes subwords from partial decompositions (e.g. "
            "the period '.' from 'Classical.choice') which appear in "
            "non-classical contexts and would over-strip. Reported as "
            "an aggressive-strip ablation alongside the MI-based "
            "ablation of Step 5.3."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    n_atomic = sum(1 for e in entries if e["atomic"])
    n_partial = len(entries) - n_atomic
    print(f"wrote {OUT_PATH}")
    print(f"  {len(entries)} source strings")
    print(f"  atomic strip IDs:   {len(atomic_ids)}  (headline)")
    print(f"  combined strip IDs: {len(combined_ids)}  (aggressive)")
    print(f"  atomic sources: {n_atomic}, partial sources: {n_partial}")
    print()
    print("entries:")
    for e in entries:
        flag = "  " if e["atomic"] else "P "
        print(f"  {flag}{e['source']!r:40s} -> {e['tokens']}")


if __name__ == "__main__":
    main()
