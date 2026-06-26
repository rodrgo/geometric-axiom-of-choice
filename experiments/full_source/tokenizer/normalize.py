"""Source normalization for the full-source encoder.

Deterministic, label-free. Applied identically to (a) the BPE training
corpus, (b) the corpus we tokenize for the encoder, and (c) entries on
the strip list.

Design note: we DO NOT space-pad ``.`` or ``_`` even though they are
punctuation, because in Lean they are part of identifier syntax:
``Classical.em`` and ``not_not`` are meant to be single semantic units.
Padding them with spaces would force BPE to learn fragmented tokens for
namespaced names, which would defeat the point of going to 32K vocab
(atomic tokens for classical machinery → exact strip list). Other
structural punctuation IS padded so that things like ``(x`` don't get
merged.
"""
from __future__ import annotations

import re

# Padded: parens/brackets/braces, separators, comparison/arith ops.
# NOT padded: . _ ' (part of Lean identifier syntax).
_PAD_CHARS = r"(){}[]:;,<>=|*+-/\@!?`\""

# Numeric literal: integer, decimal, hex, binary. Conservative — does not
# try to capture every Lean numeric form, just the common ones.
_NUM = re.compile(r"\b(?:0x[0-9a-f]+|0b[01]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b")

# String literal: double-quoted, no escapes inside (Lean source rarely
# contains them in proof bodies; conservative).
_STR = re.compile(r'"[^"\n]*"')

# Whitespace collapse.
_WS = re.compile(r"\s+")

# Build the pad regex once.
_PAD_RE = re.compile("([" + re.escape(_PAD_CHARS) + "])")


def normalize_lean_source(text: str) -> str:
    """Apply normalization. Idempotent on its own output."""
    if not text:
        return ""
    s = text.lower()
    s = _STR.sub(" __str__ ", s)
    s = _NUM.sub(" __num__ ", s)
    s = _PAD_RE.sub(r" \1 ", s)
    s = _WS.sub(" ", s).strip()
    return s


if __name__ == "__main__":
    samples = [
        "have hc : IsCompact (closure U) := hb.isCompact_closure",
        "exact Classical.em p",
        "refine ⟨_, _, rfl⟩",
        "simp [Set.indicator, Function.surjInv, not_not]",
        'rw [show 1 + 2 = 3 from rfl, "x"]',
    ]
    for s in samples:
        print(repr(s))
        print(" ->", repr(normalize_lean_source(s)))
