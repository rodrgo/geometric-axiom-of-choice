"""Tier 3 Step 3.2: Extract theorem-source snippets for the aesop prover test.

Strategy (Approach A, plan_reviewer_revisions.md):
  For each test theorem we copy its source Mathlib file to a temp path,
  locate the proof body via LeanDojo's `start` / `end` line+column coords,
  and replace everything from `:=` through `end` with `:= by aesop`.

The output is a list of jobs:
  [{
     "name": "...", "bucket": "...", "depth": int, "source_file": "...",
     "test_file": "...", "anomaly_score": float, "proof_length": int,
   }, ...]

`test_file` is the relative path of a file written inside
~/prover_eval/tests/. Each file is an isolated copy of a Mathlib source
file with ONE theorem's proof replaced by `by aesop`; all others are
left intact, so all surrounding context (imports, namespaces, variable
declarations, other theorems) is preserved.

We sample ~60 theorems per depth bucket (held-out from encoder training).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.sources import find_decl_header, find_proof_replacement  # noqa: E402,F401
from config import MATHLIB_PATH as MATHLIB  # noqa: E402

OUT_DIR = Path.home() / "prover_eval" / "tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
N_PER_BUCKET = 60


def depth_bucket(is_classical, d):
    if not is_classical:
        return "constructive"
    if d is None:
        return None
    if d <= 2:
        return "depth 2"
    if d <= 4:
        return "depth 3-4"
    if d <= 6:
        return "depth 5-6"
    return "depth 7+"


def build_test_file(source: str, name: str, body_start: int,
                     body_end: int, idx: int) -> str:
    """Replace the proof body (everything from just after `:=` up to the
    next top-level declaration) with ` by aesop`.
    """
    new_source = source[:body_start] + " by aesop\n" + source[body_end:]
    return new_source


def main():
    random.seed(SEED)
    print("Loading LeanDojo data...")
    leandojo = []
    for split in ("train", "val", "test"):
        with open(ROOT / f"results/data/leandojo/{split}.json") as f:
            leandojo.extend(json.load(f))
    by_name = {thm["full_name"]: thm for thm in leandojo}
    print(f"  total LeanDojo: {len(leandojo)}; unique names: {len(by_name)}")

    print("Loading stage4v3p...")
    with open(ROOT / "results/data/stage4v3p/proofs.json") as f:
        proofs = json.load(f)
    emb_data = np.load(ROOT / "results/data/stage4v3p/embeddings.npz")
    emb = emb_data["embeddings"]
    train_idx = set(emb_data["train_idx"].tolist())
    test_idx = set(emb_data["test_idx"].tolist())
    val_idx = set(emb_data["val_idx"].tolist())
    labels = emb_data["labels"]

    with open(ROOT / "results/data/depth_analysis/bfs_distances_full.json") as f:
        bfs = json.load(f)

    # Compute anomaly scores
    sc = StandardScaler().fit(emb[emb_data["train_idx"]])
    E = sc.transform(emb)
    nn = NearestNeighbors(n_neighbors=5).fit(E[emb_data["train_idx"]])
    anomaly = nn.kneighbors(E)[0].mean(axis=1)

    # Candidate theorems by bucket. For constructive, only held-out
    # (test split) proofs are eligible; for classical, any proof is
    # eligible (they are never in the encoder's training set).
    candidates = defaultdict(list)
    for i, p in enumerate(proofs):
        name = p["name"]
        if name not in by_name:
            continue
        is_cls = bool(p["is_classical"])
        d = bfs.get(name) if is_cls else None
        b = depth_bucket(is_cls, d)
        if b is None:
            continue
        if not is_cls:
            if i not in test_idx and i not in val_idx:
                continue  # constructive must be held out
        candidates[b].append({
            "name": name,
            "bucket": b,
            "is_classical": is_cls,
            "depth": d,
            "proof_length": p["n_invocations"],
            "anomaly_score": float(anomaly[i]),
            "file_path": p["file_path"],
        })
    for b, lst in sorted(candidates.items()):
        print(f"  {b:<14s}  {len(lst):>6d} eligible")

    # Sample
    rng = random.Random(SEED)
    sampled = []
    for b, lst in candidates.items():
        rng.shuffle(lst)
        for i in range(min(N_PER_BUCKET, len(lst))):
            sampled.append(lst[i])
    print(f"\nTotal sampled: {len(sampled)}")

    # Build test files
    jobs = []
    n_skipped = 0
    for idx, t in enumerate(sampled):
        thm = by_name[t["name"]]
        start = thm["start"]
        end = thm["end"]
        fp = MATHLIB / thm["file_path"]
        if not fp.exists():
            n_skipped += 1
            print(f"  [{idx}] {t['name']}: source file missing: {fp}")
            continue
        source = fp.read_text()
        # LeanDojo full_name may be namespace-qualified (e.g. Foo.bar.myThm);
        # the source declaration line uses the short name. Try the full name,
        # then the last dot-segment as a fallback.
        full = t["name"]
        short = full.split(".")[-1]
        rep = find_proof_replacement(source, full)
        if rep is None:
            rep = find_proof_replacement(source, short)
        if rep is None:
            n_skipped += 1
            continue
        body_start, body_end = rep
        if body_end <= body_start:
            n_skipped += 1
            continue
        new_source = build_test_file(source, t["name"], body_start,
                                      body_end, idx)
        test_file = OUT_DIR / f"prove_{idx:05d}.lean"
        test_file.write_text(new_source)
        t["source_file"] = thm["file_path"]
        t["test_file"] = str(test_file)
        t["start"] = start
        t["end"] = end
        jobs.append(t)

    print(f"\nBuilt {len(jobs)} test files; skipped {n_skipped}")
    out = {
        "seed": SEED,
        "n_per_bucket": N_PER_BUCKET,
        "jobs": jobs,
    }
    with open(ROOT / "results/data/reviewer/prover_jobs.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/data/reviewer/prover_jobs.json")


if __name__ == "__main__":
    main()
