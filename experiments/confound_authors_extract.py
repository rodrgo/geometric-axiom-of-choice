"""Reviewer revision: per-declaration author attribution via git blame.

Joins the 42,355-proof population (proofs.json) to LeanDojo source spans
(full_name -> file_path, start, end), then blames each declaration's line span
in the local mathlib4 checkout AT THE LEANDOJO TRACE COMMIT, so blame line
numbers line up with the recorded positions.  The modal author over the spanned
lines is taken as the declaration's author.

This gives an author label that *crosscuts files* (one file -> several authors,
one author -> many files), which is the non-redundant test the reviewers asked
for: the paper's existing file random intercept already nests a file-level
author label.

Output: results/data/reviewer/decl_authors.json
  { name: {author, n_lines, author_frac, file_path, start, end} }

Caveat: git blame credits the *last* modifier; mass refactors can mislabel.
Mitigated with `-w -M -C` (ignore whitespace, follow moves/copies) + modal author
over the proof span.  Unresolved names are dropped, not guessed.
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import MATHLIB_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/data/reviewer"
OUT.mkdir(parents=True, exist_ok=True)
LEANDOJO = ROOT / "results/data/leandojo"
TRACE_COMMIT = "1bc7728a050fc18ca2683f614c531cd7050ff063"


def iter_objects(path):
    """Stream top-level objects out of a big JSON array without loading it as a
    Python structure (the records carry large traced_tactics we don't need)."""
    dec = json.JSONDecoder()
    data = Path(path).read_text()
    i = data.find("[") + 1
    n = len(data)
    while i < n:
        while i < n and data[i] in " \t\r\n,":
            i += 1
        if i >= n or data[i] == "]":
            break
        obj, end = dec.raw_decode(data, i)
        yield obj
        i = end


def blame_authors(file_path):
    """Return {final_line_no: author} for a file at the trace commit, or None."""
    try:
        out = subprocess.run(
            # `-w` ignores whitespace-only reattributions; we deliberately skip
            # `-M -C` (move/copy detection) because cross-file copy detection is
            # pathologically slow (~5x per file with bad outliers) and not needed
            # for a modal-author confound label.
            ["git", "-C", str(MATHLIB_PATH), "blame", "-w",
             "--line-porcelain", TRACE_COMMIT, "--", file_path],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    line_author = {}
    cur_line = None
    for ln in out.stdout.splitlines():
        if len(ln) >= 41 and ln[40] == " " and all(c in "0123456789abcdef" for c in ln[:40]):
            # "<sha> <orig> <final>[ <ngroup>]"
            parts = ln.split()
            cur_line = int(parts[2])
        elif ln.startswith("author ") and cur_line is not None:
            line_author[cur_line] = ln[len("author "):]
    return line_author


def main():
    proofs = json.load(open(ROOT / "results/data/stage4v3p/proofs.json"))
    want = {p["name"] for p in proofs}
    print(f"Population: {len(want)} proofs")

    # 1) Collect source spans from LeanDojo, keyed by full_name.
    spans = {}  # name -> (file_path, start_line, end_line)
    for split in ["train", "val", "test"]:
        path = LEANDOJO / f"{split}.json"
        if not path.exists():
            continue
        n_added = 0
        for rec in iter_objects(path):
            name = rec.get("full_name")
            if name in want and name not in spans:
                start = rec.get("start")
                end = rec.get("end")
                fp = rec.get("file_path")
                if start and end and fp:
                    spans[name] = (fp, int(start[0]), int(end[0]))
                    n_added += 1
        print(f"  {split}: +{n_added} spans (total {len(spans)})")

    print(f"Spans resolved: {len(spans)}/{len(want)}")

    # 2) Group declarations by file; blame each file once.
    by_file = {}
    for name, (fp, s, e) in spans.items():
        by_file.setdefault(fp, []).append((name, s, e))

    print(f"Blaming {len(by_file)} files at {TRACE_COMMIT[:10]} ...")
    authors = {}
    blame_fail = 0
    for fi, (fp, decls) in enumerate(sorted(by_file.items())):
        if fi % 500 == 0:
            print(f"  file {fi}/{len(by_file)} ({len(authors)} resolved)")
        line_author = blame_authors(fp)
        if line_author is None:
            blame_fail += len(decls)
            continue
        for name, s, e in decls:
            span = [line_author[ln] for ln in range(s, e + 1) if ln in line_author]
            if not span:
                continue
            top, cnt = Counter(span).most_common(1)[0]
            authors[name] = {
                "author": top,
                "n_lines": len(span),
                "author_frac": round(cnt / len(span), 4),
                "file_path": fp,
                "start": s,
                "end": e,
            }

    print(f"\nResolved authors for {len(authors)}/{len(want)} proofs "
          f"({100*len(authors)/len(want):.1f}%); blame failed on {blame_fail} decls")
    n_authors = len({a["author"] for a in authors.values()})
    print(f"Distinct authors: {n_authors}")

    # quick top-author / classical breakdown sanity check
    is_cls = {p["name"]: int(p["is_classical"]) for p in proofs}
    per_author = Counter(a["author"] for a in authors.values())
    print("\nTop 10 authors by #proofs (classical frac):")
    for auth, c in per_author.most_common(10):
        names = [n for n, a in authors.items() if a["author"] == auth]
        cf = sum(is_cls[n] for n in names) / len(names)
        print(f"  {auth:<35s} n={c:<5d} classical_frac={cf:.2f}")

    json.dump(authors, open(OUT / "decl_authors.json", "w"), indent=2)
    print(f"\nSaved {OUT/'decl_authors.json'}")


if __name__ == "__main__":
    main()
