"""Character-level tokenizer for Lean theorem *statements* (Stage 4 v3).

Used by:
    experiments/stage4v3_encoder.py    — the statement-level denoising encoder

The Stage 4 v1 archive carries its own independent copy of these
functions at `archive/stage4_v1/stage4_tokenize.py` (frozen) so its
re-run still produces the v1-cited numbers verbatim. They are kept in
sync by convention; do not refactor one to import from the other.

Note: this is the *statement* tokenizer (char-level over Lean source).
The *proof* tokenizer for the tactic-sequence encoder lives in
`lean/proof_encoder.py` (`encode_proof`, vocabulary-by-tactic-head).
"""

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Statement normalization
# ---------------------------------------------------------------------------

# theorem_statement looks like:
#   "theorem foo (x : ℕ) : x + 0 = x := by"
#   "lemma bar {α : Type*} [DecidableEq α] ... : ... :="
# We want the goal type (the part after the final `:` at top level, before
# the optional `:=`).

def extract_goal(stmt: str) -> str:
    """Extract the goal type from a theorem_statement string.

    The statement typically has the form:
        theorem <name> <binders> : <type> (:= <proof>?)
    We strip the "theorem/lemma/example NAME" prefix, the binders, and the
    trailing ":=" if present. Falls back to the whole string on failure.
    """
    if not stmt:
        return ''
    s = stmt.strip()
    # Drop the proof: find the leftmost `:=` followed by what looks like a
    # body (by/{/identifier/paren).
    m = re.search(r':=\s*(by\b|\{|\w|\()', s)
    if m:
        s = s[:m.start()].rstrip()
    # Find the top-level `:` that separates binders from type.
    depth = 0
    square = 0
    brace = 0
    colon_positions = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '[':
            square += 1
        elif c == ']':
            square -= 1
        elif c == '{':
            brace += 1
        elif c == '}':
            brace -= 1
        elif c == ':' and depth == 0 and square == 0 and brace == 0:
            if i + 1 < len(s) and s[i + 1] == '=':
                i += 2
                continue
            colon_positions.append(i)
        i += 1
    if colon_positions:
        return s[colon_positions[-1] + 1:].strip()
    # Fallback: drop leading "theorem NAME"
    m = re.match(r'^\s*(theorem|lemma|example|instance|def)\s+\S+\s*', s)
    if m:
        return s[m.end():].strip()
    return s


# ---------------------------------------------------------------------------
# Char-level tokenizer
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']


def build_vocab(texts: list[str], min_count: int = 5) -> dict[str, int]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(t)
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch, c in counter.most_common():
        if c < min_count:
            continue
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    cls = vocab['[CLS]']
    sep = vocab['[SEP]']
    unk = vocab['[UNK]']
    pad = vocab['[PAD]']
    ids = [cls]
    for ch in text[:max_len - 2]:
        ids.append(vocab.get(ch, unk))
    ids.append(sep)
    while len(ids) < max_len:
        ids.append(pad)
    return ids
