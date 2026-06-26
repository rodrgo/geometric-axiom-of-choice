# Provenance manifest

Materialized, human-inspectable view of the exact study population and
operational sample. Regenerated deterministically by
`scripts/export_manifest.py` from the pipeline outputs.

## `proof_population.csv`
One row per analyzed theorem (42,355 total: 31,144 classical,
11,211 constructive).

| column | meaning |
|---|---|
| `name` | Lean declaration full name |
| `is_classical` | 1 iff the proof term transitively uses `Classical.choice` |
| `choice_depth` | shortest dependency distance to `Classical.choice` (classical only) |
| `split` | encoder split: `train`/`val`/`test` (constructive) or `classical_heldout` |

## `operational_sample.csv`
The 251 held-out theorems used for the aesop / ReProver evaluation
(sampled with seed 0, up to 60 per bucket).

## Provenance
- Kernel dependency graph: Mathlib4 `9f0aee2e9bfe008c35fa9672d28e6dd4411d2971` (v4.29.0-rc8).
- LeanDojo Benchmark 4 traces: Mathlib4 `1bc7728a050fc18ca2683f614c531cd7050ff063`.
- Graph and traces are matched by theorem name; see the paper's count table.
