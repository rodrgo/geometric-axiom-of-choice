# Full-source encoder robustness

Auxiliary robustness experiment behind the paper's **"Full-Source Encoder
Robustness"** appendix. The main encoder represents each proof as a sequence
of *tactic heads*; here we instead train denoising Transformer encoders on the
**normalized full proof source** (heads *and* arguments, BPE-tokenized) and
repeat the depth-stratified one-class tests. This shows the shallow
`Classical.choice` boundary is not an artifact of the tactic-head abstraction.

Produces:

- **Table `tab:full_source_depth`** — `analysis/depth_stratified_auc.py` →
  `analysis/aggregate_results.py`
- **Table `tab:full_source_within_domain`** — `analysis/within_domain_auc.py`
- Inline full-source numbers (reconstruction-loss excess, superlevel
  containment, held-out masked-token loss) — the other `analysis/*.py` scripts,
  bundled by `aggregate_results.py` into `analysis/results.json` + `RESULTS.md`.

## Dependencies

Needs the **main repo's data** to exist first (run the main pipeline up to the
proof encoder + depth distances):

- `results/data/leandojo/{train,val,test}.json`
- `results/data/stage4v3p/proofs.json`, `.../embeddings.npz`
- `results/data/depth_analysis/bfs_distances_full.json`

All paths are resolved relative to the repo root (`Path(__file__)...parent`),
so run scripts from the repo root with the project venv.

## Pipeline order

```
# 1. BPE vocab + classical-token strip list (tokenizer artifacts)
python experiments/full_source/tokenizer/train_bpe.py
python experiments/full_source/tokenizer/build_strip_list.py

# 2. Build the normalized full-source proof corpus (parquet, ~26 MB)
python experiments/full_source/data/extract_full_source.py

# 3. Train 3 denoising encoders (constructive proofs only), seeds 0,1,2
python experiments/full_source/encoder/train.py --seed 0
python experiments/full_source/encoder/train.py --seed 1
python experiments/full_source/encoder/train.py --seed 2

# 4. Compute frozen embeddings for raw / stripped / combined variants
python experiments/full_source/encoder/embed.py     # all seeds x variants (~814 MB)

# 5. Per-seed analyses, then aggregate to medians +/- range over seeds
python experiments/full_source/analysis/depth_stratified_auc.py
python experiments/full_source/analysis/within_domain_auc.py
python experiments/full_source/analysis/reconstruction_loss.py
python experiments/full_source/analysis/length_residualization.py
python experiments/full_source/analysis/superlevel_containment.py
python experiments/full_source/analysis/leakage_diagnostic.py
python experiments/full_source/encoder/sanity_check.py
python experiments/full_source/analysis/aggregate_results.py   # -> analysis/results.json, RESULTS.md
```

## Heavy / generated artifacts (git-ignored)

| Artifact | Size | Produced by |
|---|---|---|
| `data/full_source.parquet` | ~26 MB | `data/extract_full_source.py` |
| `tokenizer/artifacts/` (BPE vocab, strip list) | small | `tokenizer/*.py` |
| `encoder/checkpoints/seed_{0,1,2}/` | — | `encoder/train.py` |
| `encoder/embeddings/seed_{i}/{raw,stripped,combined}.npz` | ~814 MB | `encoder/embed.py` |
| `analysis/*.json`, `RESULTS.md` | small | `analysis/*.py` |
