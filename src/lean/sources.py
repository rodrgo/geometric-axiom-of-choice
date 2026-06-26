"""Mathlib source-file parsing for theorem-body splicing.

`find_decl_header` and `find_proof_replacement` are used by extract
scripts that build paired test files (`scripts/prover_extract.py`,
`scripts/classical_ablation_extract.py`) — they locate `theorem|lemma|...
<name>` in a Mathlib source file, find the `:=` that closes the header,
then find the next top-level declaration that bounds the proof body.

This is hand-rolled rather than using a Lean AST tool because we only
need to identify boundaries, not parse the body.
"""

import re

_TOP_LEVEL_RE = re.compile(
    r"^(?:@\[|(?:private\s+|protected\s+|noncomputable\s+|nonrec\s+|"
    r"unsafe\s+|partial\s+|scoped\s+|mutual\s+)*"
    r"(?:theorem|lemma|example|def|abbrev|instance|structure|class|inductive|"
    r"coinductive|notation|macro|elab|syntax|opaque|axiom|variable|universe|"
    r"namespace|section|end|open|import|attribute|set_option|initialize)\b)",
    re.MULTILINE,
)


def find_decl_header(source: str, short_name: str):
    """Locate a declaration `theorem|lemma|... <short_name>` in source.

    Returns (start_off, assign_off) where start_off is the beginning of
    the top-level declaration line and assign_off is the char offset
    just after the first `:=` that terminates the header (which may span
    multiple lines).
    Returns None if not found.
    """
    escaped = re.escape(short_name)
    pat = re.compile(
        r"^\s*(?:@\[[^\n]*\]\s*\n)*"
        r"(?:private\s+|protected\s+|noncomputable\s+|nonrec\s+|"
        r"unsafe\s+|partial\s+|scoped\s+|mutual\s+)*"
        r"(?:theorem|lemma|example|def|abbrev|instance)\s+"
        + escaped + r"\b",
        re.MULTILINE,
    )
    m = pat.search(source)
    if m is None:
        return None
    start_off = m.start()
    # From m.end() forward, find the first `:=` at the top level of nesting.
    depth = 0
    i = m.end()
    n = len(source)
    while i < n - 1:
        ch = source[i]
        if ch == "(" or ch == "[" or ch == "{":
            depth += 1
        elif ch == ")" or ch == "]" or ch == "}":
            depth -= 1
        elif depth == 0 and ch == ":" and source[i+1] == "=":
            return start_off, i + 2
        i += 1
    return None


def find_proof_replacement(source: str, short_name: str):
    """Return (body_start_off, body_end_off) of the proof body to replace.

    body_start is just after `:=`. body_end is the start of the next
    top-level declaration (or end of file).
    """
    hdr = find_decl_header(source, short_name)
    if hdr is None:
        return None
    _, body_start = hdr
    m = _TOP_LEVEL_RE.search(source, pos=body_start)
    body_end = m.start() if m is not None else len(source)
    body_trim = body_end
    while body_trim > body_start and source[body_trim - 1] in " \t\n\r":
        body_trim -= 1
    return body_start, body_trim
