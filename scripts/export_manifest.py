"""Export the small, human-inspectable provenance manifest.

Writes the exact study population and operational sample as committed CSVs, so
a reader can see every theorem's classical/constructive label, its distance
from Classical.choice, and its split — without rebuilding the 280 MB kernel
graph or retracing LeanDojo. The full pipeline regenerates these deterministically
(fixed seeds); this just materializes them for inspection / exact reuse.

Outputs (committed under manifest/):
  manifest/proof_population.csv   one row per analyzed theorem:
      name, is_classical, choice_depth, split
      (split is the encoder split: train/val/test for constructive proofs;
       'classical_heldout' for classical proofs — the held-out population.)
  manifest/operational_sample.csv the 251-theorem aesop/ReProver sample:
      name, bucket, is_classical, depth, proof_length, file_path
  manifest/README.md              column docs + provenance (commits, counts).

Run after the main pipeline (so the inputs exist), or point at a precomputed
data dir with AOC_DATA_ROOT=/path/to/results/data python scripts/export_manifest.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import STAGE4V3P_DIR, DEPTH_ANALYSIS_DIR, REVIEWER_DIR  # noqa: E402

OUT = ROOT / "manifest"
# Provenance (see README): the two Mathlib commits behind the inputs.
GRAPH_MATHLIB_COMMIT = "9f0aee2e9bfe008c35fa9672d28e6dd4411d2971"   # v4.29.0-rc8
LEANDOJO_MATHLIB_COMMIT = "1bc7728a050fc18ca2683f614c531cd7050ff063"


def main() -> int:
    proofs = json.loads((STAGE4V3P_DIR / "proofs.json").read_text())
    bfs = json.loads((DEPTH_ANALYSIS_DIR / "bfs_distances_full.json").read_text())
    emb = np.load(STAGE4V3P_DIR / "embeddings.npz", allow_pickle=True)
    train = set(emb["train_idx"].tolist())
    val = set(emb["val_idx"].tolist())
    test = set(emb["test_idx"].tolist())

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- proof population ----
    n_cls = n_con = 0
    split_counts: dict[str, int] = {}
    with (OUT / "proof_population.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "is_classical", "choice_depth", "split"])
        for i, p in enumerate(proofs):
            is_cls = bool(p["is_classical"])
            depth = bfs.get(p["name"], "") if is_cls else ""
            if is_cls:
                split = "classical_heldout"; n_cls += 1
            elif i in train:
                split = "train"; n_con += 1
            elif i in val:
                split = "val"; n_con += 1
            elif i in test:
                split = "test"; n_con += 1
            else:
                split = "constructive_unsplit"; n_con += 1
            split_counts[split] = split_counts.get(split, 0) + 1
            w.writerow([p["name"], int(is_cls), depth, split])
    print(f"proof_population.csv: {len(proofs):,} rows "
          f"({n_cls:,} classical, {n_con:,} constructive)  splits={split_counts}")

    # ---- 251-theorem operational sample ----
    jobs_doc = json.loads((REVIEWER_DIR / "prover_jobs.json").read_text())
    jobs = jobs_doc["jobs"]
    with (OUT / "operational_sample.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "bucket", "is_classical", "depth", "proof_length", "file_path"])
        for j in jobs:
            w.writerow([j["name"], j["bucket"], int(bool(j["is_classical"])),
                        j.get("depth", ""), j.get("proof_length", ""),
                        j.get("file_path", "")])
    print(f"operational_sample.csv: {len(jobs)} rows (seed={jobs_doc.get('seed')})")

    # ---- manifest README ----
    (OUT / "README.md").write_text(
        "# Provenance manifest\n\n"
        "Materialized, human-inspectable view of the exact study population and\n"
        "operational sample. Regenerated deterministically by\n"
        "`scripts/export_manifest.py` from the pipeline outputs.\n\n"
        "## `proof_population.csv`\n"
        f"One row per analyzed theorem ({len(proofs):,} total: {n_cls:,} classical,\n"
        f"{n_con:,} constructive).\n\n"
        "| column | meaning |\n|---|---|\n"
        "| `name` | Lean declaration full name |\n"
        "| `is_classical` | 1 iff the proof term transitively uses `Classical.choice` |\n"
        "| `choice_depth` | shortest dependency distance to `Classical.choice` (classical only) |\n"
        "| `split` | encoder split: `train`/`val`/`test` (constructive) or `classical_heldout` |\n\n"
        "## `operational_sample.csv`\n"
        f"The {len(jobs)} held-out theorems used for the aesop / ReProver evaluation\n"
        f"(sampled with seed {jobs_doc.get('seed')}, up to {jobs_doc.get('n_per_bucket')} per bucket).\n\n"
        "## Provenance\n"
        f"- Kernel dependency graph: Mathlib4 `{GRAPH_MATHLIB_COMMIT}` (v4.29.0-rc8).\n"
        f"- LeanDojo Benchmark 4 traces: Mathlib4 `{LEANDOJO_MATHLIB_COMMIT}`.\n"
        "- Graph and traces are matched by theorem name; see the paper's count table.\n"
    )
    print(f"Wrote manifest/ under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
