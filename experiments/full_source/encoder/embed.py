"""Step 3.5: compute embeddings for every theorem under a frozen
encoder checkpoint.

Outputs:
  full_source/encoder/embeddings/seed_{i}/raw.npz       <-- using `tokens`
  full_source/encoder/embeddings/seed_{i}/stripped.npz  <-- `tokens_stripped`
  full_source/encoder/embeddings/seed_{i}/combined.npz  <-- `tokens_stripped_combined`

Each .npz holds:
  names     : str[N]
  pooled    : float32[N, 256]   pre-projection
  projected : float32[N, 128]   projected
  splits    : str[N]
  is_classical : bool[N]
  depth     : int[N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import FullSourceEncoder, PAD_ID  # noqa: E402

PARQUET = HERE.parent / "data/full_source.parquet"
DEFAULT_CKPT_ROOT = HERE / "checkpoints"
EMB_ROOT = HERE / "embeddings"

BATCH = 32


def load_model(ckpt_path: Path, device: torch.device) -> FullSourceEncoder:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FullSourceEncoder(vocab_size=state["vocab_size"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def embed_one_column(model: FullSourceEncoder, ids_col: list[list[int]],
                     device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    N = len(ids_col)
    pooled_out = np.zeros((N, model.d_model), dtype=np.float32)
    proj_out = np.zeros((N, model.proj.out_features), dtype=np.float32)
    t0 = time.time()
    for i in range(0, N, BATCH):
        batch = ids_col[i:i + BATCH]
        L = max(len(x) for x in batch)
        arr = np.full((len(batch), L), PAD_ID, dtype=np.int64)
        for j, ids in enumerate(batch):
            arr[j, :len(ids)] = ids
        x = torch.from_numpy(arr).to(device)
        with torch.no_grad():
            pooled, projected = model.embed(x)
        pooled_out[i:i + len(batch)] = pooled.cpu().numpy()
        proj_out[i:i + len(batch)] = projected.cpu().numpy()
        if (i // BATCH) % 200 == 0 and i > 0:
            print(f"    {i}/{N}  [{time.time()-t0:.0f}s]", flush=True)
    return pooled_out, proj_out


def run_one_seed(seed: int, ckpt_root: Path, device: torch.device) -> None:
    ckpt_path = ckpt_root / f"seed_{seed}/phi_full.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = load_model(ckpt_path, device)
    print(f"seed {seed}: loaded {ckpt_path}", flush=True)

    t = pq.read_table(PARQUET, columns=[
        "name", "split", "is_classical", "depth",
        "tokens", "tokens_stripped", "tokens_stripped_combined",
    ])
    names = np.asarray(t["name"].to_pylist())
    splits = np.asarray(t["split"].to_pylist())
    is_classical = np.asarray(t["is_classical"].to_pylist(), dtype=bool)
    depth = np.asarray(t["depth"].to_pylist(), dtype=np.int32)

    out_dir = EMB_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind, colname in [("raw", "tokens"),
                          ("stripped", "tokens_stripped"),
                          ("combined", "tokens_stripped_combined")]:
        print(f"  embedding [{kind}]...", flush=True)
        col = [list(x) for x in t[colname].to_pylist()]
        pooled, projected = embed_one_column(model, col, device)
        out = out_dir / f"{kind}.npz"
        np.savez_compressed(out, names=names, splits=splits,
                            is_classical=is_classical, depth=depth,
                            pooled=pooled, projected=projected)
        print(f"  wrote {out}  ({out.stat().st_size / (1<<20):.1f} MB)",
              flush=True)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = pick_device(a.device)
    print(f"device: {device}")
    for seed in a.seeds:
        run_one_seed(seed, Path(a.ckpt_root), device)


if __name__ == "__main__":
    main()
