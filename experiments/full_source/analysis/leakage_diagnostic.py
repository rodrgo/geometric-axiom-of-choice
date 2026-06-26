"""Step 5.3: mutual-information leakage diagnostic.

For each token in the BPE vocabulary, compute
    I( token-present-in-proof ; is_classical )
on the VALIDATION split (NOT the test split — using test would
re-introduce label-aware selection on the data that drives the
headline AUC). Rank tokens by MI. Report:

  * Top-50 highest-MI tokens (human-readable form + numeric ID).
  * Intersection with the existing strip list.
  * A separate ablation: depth-stratified AUC after stripping the
    top-50-MI tokens (combined with the original strip list).

Outputs:
  full_source/analysis/leakage_mi.json     -- ranked tokens
  full_source/analysis/leakage_ablation.json -- AUC with top-50-MI stripped
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "../encoder"))
from model import FullSourceEncoder, PAD_ID  # noqa: E402

PARQUET = HERE.parent / "data/full_source.parquet"
TOKENIZER = HERE.parent / "tokenizer/artifacts/bpe_32k.json"
STRIP_LIST = HERE.parent / "tokenizer/artifacts/strip_list.json"
EMB_ROOT = HERE.parent / "encoder/embeddings"

TOP_K_MI = 50
MIN_TOKEN_FREQ = 10  # don't rank tokens that appear in <10 val proofs
                    # (plug-in MI is noisy on rare events)

DEPTH_BUCKETS = [
    ("depth_2", lambda d: d == 2),
    ("depth_3", lambda d: d == 3),
    ("depth_4_6", lambda d: 4 <= d <= 6),
    ("depth_7_8", lambda d: 7 <= d <= 8),
    ("depth_9_plus", lambda d: d >= 9),
]


def token_presence_matrix(token_lists: list[list[int]], V: int
                          ) -> np.ndarray:
    """[N, V] binary: row i, col t = 1 iff token t appears in proof i."""
    N = len(token_lists)
    M = np.zeros((N, V), dtype=np.bool_)
    for i, tl in enumerate(token_lists):
        if not tl:
            continue
        unique = np.unique(np.asarray(tl, dtype=np.int64))
        # Clip out-of-range just in case.
        unique = unique[(unique >= 0) & (unique < V)]
        M[i, unique] = True
    return M


def mutual_info_binary(M: np.ndarray, y: np.ndarray
                       ) -> np.ndarray:
    """Plug-in MI in nats between binary token presence and binary label."""
    N, V = M.shape
    # P(t=1, y=1), P(t=1, y=0), etc.
    p_y1 = y.mean()
    p_y0 = 1.0 - p_y1
    eps = 1e-12

    # token counts per class
    t_in_y1 = M[y].sum(axis=0)  # [V]
    t_in_y0 = M[~y].sum(axis=0)  # [V]
    n_y1 = max(y.sum(), 1)
    n_y0 = max((~y).sum(), 1)

    p_t1_y1 = t_in_y1 / N
    p_t1_y0 = t_in_y0 / N
    p_t0_y1 = p_y1 - p_t1_y1
    p_t0_y0 = p_y0 - p_t1_y0

    p_t1 = p_t1_y1 + p_t1_y0
    p_t0 = 1.0 - p_t1

    def term(p_joint: np.ndarray, p_t: np.ndarray, p_y: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = p_joint * np.log((p_joint + eps) / (p_t * p_y + eps))
        return np.where(p_joint > 0, r, 0.0)

    mi = (term(p_t1_y1, p_t1, p_y1) + term(p_t1_y0, p_t1, p_y0) +
          term(p_t0_y1, p_t0, p_y1) + term(p_t0_y0, p_t0, p_y0))
    # Mask out rare tokens.
    freq = M.sum(axis=0)
    mi[freq < MIN_TOKEN_FREQ] = 0.0
    return mi


def rank_tokens() -> dict:
    print("loading tokens...", flush=True)
    t = pq.read_table(PARQUET, columns=[
        "split", "is_classical", "tokens"])
    splits = np.asarray(t["split"].to_pylist())
    is_classical = np.asarray(t["is_classical"].to_pylist(), dtype=bool)
    tokens = t["tokens"].to_pylist()

    val_mask = (splits == "val")
    val_tokens = [tokens[i] for i in range(len(tokens)) if val_mask[i]]
    val_y = is_classical[val_mask]
    print(f"  val proofs: {val_mask.sum():,} "
          f"(classical: {val_y.sum()}, constructive: {(~val_y).sum()})",
          flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    V = tok.get_vocab_size()

    print(f"  building [N, V] presence matrix (N={len(val_tokens)}, V={V})...",
          flush=True)
    t0 = time.time()
    M = token_presence_matrix(val_tokens, V)
    print(f"  done  [{time.time()-t0:.0f}s]  density={M.mean()*100:.2f}%",
          flush=True)

    print("  computing MI...", flush=True)
    mi = mutual_info_binary(M, val_y)

    order = np.argsort(-mi)
    top_ids = order[:TOP_K_MI].tolist()

    strip = json.loads(STRIP_LIST.read_text())
    existing_strip = set(strip["combined_strip_ids"])

    entries = []
    for idx, tok_id in enumerate(top_ids):
        tok_str = tok.id_to_token(int(tok_id)) or "?"
        freq = int(M[:, tok_id].sum())
        p_pos = float(M[val_y, tok_id].mean()) if val_y.any() else 0.0
        p_neg = float(M[~val_y, tok_id].mean()) if (~val_y).any() else 0.0
        entries.append({
            "rank": idx + 1,
            "token_id": int(tok_id),
            "token": tok_str,
            "mi_nats": float(mi[tok_id]),
            "freq_val": freq,
            "p_in_classical": p_pos,
            "p_in_constructive": p_neg,
            "in_strip_list": int(tok_id) in existing_strip,
        })

    new_in_top_k = [e for e in entries if not e["in_strip_list"]]
    return {
        "top_k": TOP_K_MI,
        "min_token_freq_val": MIN_TOKEN_FREQ,
        "n_val": int(val_mask.sum()),
        "n_classical_val": int(val_y.sum()),
        "top_entries": entries,
        "new_strip_candidates": [e["token_id"] for e in new_in_top_k],
        "n_new_strip_candidates": len(new_in_top_k),
        "top_k_strip_ids": top_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "leakage_mi.json"))
    a = ap.parse_args()

    result = rank_tokens()
    out = Path(a.out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"\ntop-10 highest-MI tokens (val split):")
    for e in result["top_entries"][:10]:
        flag = "S" if e["in_strip_list"] else " "
        print(f"  {flag} #{e['rank']:2d}  {e['token']!r:30s} "
              f"mi={e['mi_nats']:.4f}  "
              f"p(class)={e['p_in_classical']:.3f}  "
              f"p(constr)={e['p_in_constructive']:.3f}")


if __name__ == "__main__":
    main()
