"""Per-token saliency utilities for the proof encoder.

`signal_b_for_proof` returns, for each token position in a proof, the
cross-entropy loss of reconstructing that token when (and only when)
that single position is masked. This is the same denoising objective the
encoder was trained under, evaluated at one position at a time.

High loss => the encoder did not expect this token given its context =>
out-of-distribution relative to the (constructive) training set.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lean.proof_encoder import (
    ProofEncoder, encode_proof, MASK_ID,
)


def encode_proof_array(heads: list[str], vocab: dict, max_len: int) -> np.ndarray:
    """Wrap `encode_proof` to return a numpy int64 array of shape (max_len,)."""
    return np.array(encode_proof(heads, vocab, max_len), dtype=np.int64)


def signal_b_for_proof(
    heads: list[str], vocab: dict, model: ProofEncoder, max_len: int,
) -> np.ndarray:
    """Per-tactic single-position reconstruction loss.

    Returns an array of length len(heads). Position i in the result is the
    CE loss of reconstructing head i when only encoded position i+1 (i.e.
    the head's slot inside the [CLS, t_1, ..., t_T, SEP, PAD...] layout)
    is masked.
    """
    T = len(heads)
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    base = encode_proof_array(heads, vocab, max_len)
    out = np.zeros(T, dtype=np.float32)

    # One forward pass over a batch of T corrupted versions (one per masked position).
    X_corr = np.tile(base, (T, 1)).copy()
    targets = base.copy()
    for i in range(T):
        X_corr[i, i + 1] = MASK_ID
    with torch.no_grad():
        logits = model(torch.from_numpy(X_corr).long())  # (T, max_len, V)
    targets_t = torch.from_numpy(targets).long()
    for i in range(T):
        pos = i + 1
        out[i] = F.cross_entropy(
            logits[i, pos:pos + 1, :], targets_t[pos:pos + 1],
            reduction="mean",
        ).item()
    return out
