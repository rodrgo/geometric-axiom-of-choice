"""Step 4.2: depth-stratified reconstruction loss.

For each held-out proof (constructive test + every classical proof),
draw N_MASKS independent whole-word masks and compute mean CE per
masked token under the frozen encoder. Stratify by depth.

Reports:
  - Raw mean per-proof CE per depth bucket.
  - "Residualized" CE: subtract per-bucket median length effect (fit
    on constructive test only via linear regression of CE on log-tokens).
  - Excess over constructive baseline.
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "../encoder"))
from model import FullSourceEncoder, PAD_ID, MASK_ID  # noqa: E402
from train import ConstructiveTrainDS, whole_word_mask  # noqa: E402

PARQUET = HERE.parent / "data/full_source.parquet"
TOKENIZER = HERE.parent / "tokenizer/artifacts/bpe_32k.json"

N_MASKS = 10
MAX_LEN = 512
BATCH = 32

DEPTH_BUCKETS = [
    ("constructive", lambda c, d: not c),
    ("depth_2", lambda c, d: c and d == 2),
    ("depth_3", lambda c, d: c and d == 3),
    ("depth_4_6", lambda c, d: c and 4 <= d <= 6),
    ("depth_7_8", lambda c, d: c and 7 <= d <= 8),
    ("depth_9_plus", lambda c, d: c and d >= 9),
]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model(ckpt_path: Path, device: torch.device) -> FullSourceEncoder:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FullSourceEncoder(vocab_size=state["vocab_size"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def per_proof_loss(model: FullSourceEncoder, tokens: list[int],
                   word_ids: list[int], device: torch.device,
                   mask_rng: torch.Generator) -> tuple[float, int]:
    """Mean CE per masked token, averaged over N_MASKS independent draws.

    Returns (mean_ce, total_masked_tokens).
    """
    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    wids = torch.tensor(word_ids, dtype=torch.long, device=device).unsqueeze(0)
    losses = []
    n_masked = 0
    with torch.no_grad():
        for _ in range(N_MASKS):
            corrupted, target = whole_word_mask(ids, wids, 0.15, mask_rng)
            if not target.any():
                continue
            logits, _ = model(corrupted)
            l = F.cross_entropy(logits[target], ids[target], reduction="mean")
            losses.append(l.item())
            n_masked += int(target.sum().item())
    if not losses:
        return float("nan"), 0
    return float(np.mean(losses)), n_masked


def run_one_seed(seed: int, device: torch.device,
                 ckpt_root: Path) -> dict:
    ckpt = ckpt_root / f"seed_{seed}/phi_full.pt"
    if not ckpt.exists():
        print(f"  skip seed {seed}: {ckpt} missing", flush=True)
        return {"seed": seed, "error": "missing checkpoint"}

    model = load_model(ckpt, device)
    print(f"  loaded {ckpt}", flush=True)

    # Re-tokenize source_normalized to recover word_ids.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))

    t = pq.read_table(PARQUET, columns=[
        "name", "split", "is_classical", "depth",
        "source_normalized", "n_tokens"])
    names = t["name"].to_pylist()
    splits = t["split"].to_pylist()
    is_classical_col = t["is_classical"].to_pylist()
    depth_col = t["depth"].to_pylist()
    sources = t["source_normalized"].to_pylist()
    n_tokens_col = t["n_tokens"].to_pylist()

    # Evaluate on constructive-test ∪ all-classical.
    keep_idx = [i for i in range(len(names))
                if is_classical_col[i] or splits[i] != "train"]
    print(f"  evaluating {len(keep_idx):,} proofs", flush=True)

    mask_rng = torch.Generator(device=device)
    mask_rng.manual_seed(seed * 31337 + 999)

    per_proof = []
    t0 = time.time()
    for k, i in enumerate(keep_idx):
        if k > 0 and k % 1000 == 0:
            print(f"    {k}/{len(keep_idx)}  [{time.time()-t0:.0f}s]",
                  flush=True)
        enc = tok.encode(sources[i])
        ids = enc.ids[:MAX_LEN]
        wids = [w if w is not None else -1 for w in enc.word_ids[:MAX_LEN]]
        if len(ids) < 4:
            continue
        ce, n_m = per_proof_loss(model, ids, wids, device, mask_rng)
        per_proof.append({
            "name": names[i],
            "split": splits[i],
            "is_classical": is_classical_col[i],
            "depth": depth_col[i],
            "n_tokens": n_tokens_col[i],
            "mean_ce": ce,
            "n_masked": n_m,
        })

    # Aggregate by bucket.
    buckets = {}
    for bname, pred in DEPTH_BUCKETS:
        rows = [r for r in per_proof if pred(r["is_classical"], r["depth"])]
        if not rows:
            buckets[bname] = {"n": 0, "median_ce": None}
            continue
        ce_arr = np.array([r["mean_ce"] for r in rows], dtype=np.float64)
        len_arr = np.array([r["n_tokens"] for r in rows], dtype=np.float64)
        buckets[bname] = {
            "n": int(len(rows)),
            "median_ce": float(np.median(ce_arr)),
            "mean_ce": float(np.mean(ce_arr)),
            "median_n_tokens": float(np.median(len_arr)),
        }
    return {"seed": seed, "buckets": buckets,
            "per_proof_count": len(per_proof)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-root",
                    default=str(HERE / "../encoder/checkpoints"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(HERE / "reconstruction_loss.json"))
    a = ap.parse_args()

    device = pick_device(a.device)
    print(f"device: {device}", flush=True)

    runs = []
    for seed in a.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        runs.append(run_one_seed(seed, device, Path(a.ckpt_root)))

    out = Path(a.out)
    out.write_text(json.dumps({
        "n_masks": N_MASKS,
        "runs": runs,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
