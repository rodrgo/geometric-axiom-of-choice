"""Train the full-source denoising encoder under one seed.

Local-runnable (for smoke tests) and Modal-callable (for the real run).
See modal_app.py for the cloud entrypoint.

Whole-word masking: with probability 0.15 we mask all subword tokens
belonging to a single pre-tokenized word. The tokenizer's Encoding
object provides word_ids, so we compute them at dataset construction
and store them alongside tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Path bootstrap so we can import model.py whether invoked locally or
# from the Modal sandbox.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import (FullSourceEncoder, PAD_ID, MASK_ID,  # noqa: E402
                   count_parameters)

# Defaults (repo-relative; HERE = full_source/encoder/).
_FS = HERE.parent  # full_source/
DEFAULT_PARQUET = str(_FS / "data" / "full_source.parquet")
DEFAULT_TOKENIZER = str(_FS / "tokenizer" / "artifacts" / "bpe_32k.json")
DEFAULT_OUT_DIR = str(HERE / "checkpoints")

MASK_PROB = 0.15
MAX_LEN = 512


@dataclass
class TrainCfg:
    seed: int = 0
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    epochs: int = 30
    grad_clip: float = 1.0
    val_fraction: float = 0.10
    log_every: int = 50
    eval_every_epochs: int = 1
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# -----------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------

class ConstructiveTrainDS(Dataset):
    """Constructive training proofs, tokenized with word_ids for
    whole-word masking. We re-tokenize from source_normalized to
    recover word_ids (parquet only stored token IDs)."""

    def __init__(self, parquet_path: str, tokenizer_path: str,
                 split: str = "train", max_len: int = MAX_LEN) -> None:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = tok.get_vocab_size()

        t = pq.read_table(parquet_path,
                          columns=["split", "is_classical",
                                   "source_normalized"])
        is_classical = t["is_classical"].to_pylist()
        splits = t["split"].to_pylist()
        sources_all = t["source_normalized"].to_pylist()
        sources = [s for s, c, sp in zip(sources_all, is_classical, splits)
                   if (not c) and sp == split]
        encs = tok.encode_batch(sources)

        self.token_ids: list[np.ndarray] = []
        self.word_ids: list[np.ndarray] = []
        skipped = 0
        for e in encs:
            ids = e.ids[:max_len]
            wids_raw = e.word_ids[:max_len]
            # word_ids may contain None for special tokens; we don't
            # add specials so this should not occur, but be defensive.
            wids = np.array([w if w is not None else -1 for w in wids_raw],
                            dtype=np.int32)
            if len(ids) < 4:
                skipped += 1
                continue
            self.token_ids.append(np.asarray(ids, dtype=np.int32))
            self.word_ids.append(wids)
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.token_ids[i], self.word_ids[i]


def collate(batch: list[tuple[np.ndarray, np.ndarray]]
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to the batch's max length; returns (ids, word_ids)."""
    L = max(len(x[0]) for x in batch)
    B = len(batch)
    ids = np.full((B, L), PAD_ID, dtype=np.int64)
    wids = np.full((B, L), -1, dtype=np.int64)
    for i, (tok, w) in enumerate(batch):
        ids[i, :len(tok)] = tok
        wids[i, :len(w)] = w
    return torch.from_numpy(ids), torch.from_numpy(wids)


# -----------------------------------------------------------------
# Whole-word masking
# -----------------------------------------------------------------

def whole_word_mask(ids: torch.Tensor, wids: torch.Tensor,
                    p: float, rng: torch.Generator
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (corrupted_ids, target_mask).

    target_mask is True at positions whose containing word was chosen
    for masking. corrupted_ids has MASK_ID at exactly those positions
    (we don't use the 80/10/10 BERT mix; mask-only is fine for our
    purposes and clearer for the reconstruction-loss metric).
    """
    device = ids.device
    B, L = ids.shape
    corrupted = ids.clone()
    target = torch.zeros_like(ids, dtype=torch.bool)

    for b in range(B):
        wb = wids[b]
        valid = wb >= 0
        unique_words = torch.unique(wb[valid])
        if unique_words.numel() == 0:
            continue
        # Per word, flip a coin.
        flips = torch.rand(unique_words.shape, generator=rng, device=device) < p
        chosen = unique_words[flips]
        if chosen.numel() == 0:
            continue
        mask = torch.isin(wb, chosen)
        target[b] = mask
        corrupted[b, mask] = MASK_ID
    return corrupted, target


# -----------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------

def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(cfg: TrainCfg, parquet_path: str, tokenizer_path: str,
          out_dir: str, max_steps: int | None = None) -> dict:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    print(f"device: {device}", flush=True)

    print("loading datasets...", flush=True)
    t0 = time.time()
    full_train = ConstructiveTrainDS(parquet_path, tokenizer_path,
                                     split="train")
    print(f"  constructive-train: {len(full_train):,} proofs"
          f"  (skipped: {full_train.skipped})"
          f"  vocab: {full_train.vocab_size}"
          f"  [{time.time()-t0:.1f}s]", flush=True)

    # Split off val from train deterministically.
    n_val = int(round(cfg.val_fraction * len(full_train)))
    rng = np.random.default_rng(cfg.seed * 7919 + 13)
    idx = rng.permutation(len(full_train))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    train_ds = torch.utils.data.Subset(full_train, train_idx)
    val_ds = torch.utils.data.Subset(full_train, val_idx)
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=collate,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, collate_fn=collate,
                            num_workers=2, drop_last=False)

    model = FullSourceEncoder(vocab_size=full_train.vocab_size,
                              max_len=MAX_LEN).to(device)
    n_params = count_parameters(model)
    print(f"params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    print(f"steps/epoch: {steps_per_epoch}  total: {total_steps}", flush=True)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, cfg.warmup_steps, total_steps))

    # Reproducible mask RNG on the training device.
    mask_rng = torch.Generator(device=device)
    mask_rng.manual_seed(cfg.seed * 31337 + 1)

    step = 0
    losses = []
    val_history = []
    best_val = float("inf")
    out_path = Path(out_dir) / f"seed_{cfg.seed}"
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        for ids, wids in train_loader:
            ids = ids.to(device, non_blocking=True)
            wids = wids.to(device, non_blocking=True)
            corrupted, target = whole_word_mask(ids, wids, MASK_PROB, mask_rng)

            logits, _ = model(corrupted)
            if target.any():
                loss = F.cross_entropy(logits[target], ids[target])
            else:
                # No words masked in this batch (unlikely with p=0.15
                # and any reasonable batch). Skip.
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1

            if step % cfg.log_every == 0:
                lr_now = sched.get_last_lr()[0]
                print(f"  step {step}/{total_steps}  "
                      f"epoch {epoch}  loss {np.mean(losses[-50:]):.3f}  "
                      f"lr {lr_now:.2e}  "
                      f"elapsed {time.time()-t0:.0f}s", flush=True)
            if max_steps is not None and step >= max_steps:
                break

        # Validation.
        if (epoch + 1) % cfg.eval_every_epochs == 0:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for ids, wids in val_loader:
                    ids = ids.to(device); wids = wids.to(device)
                    corrupted, target = whole_word_mask(
                        ids, wids, MASK_PROB, mask_rng)
                    if not target.any():
                        continue
                    logits, _ = model(corrupted)
                    v = F.cross_entropy(logits[target], ids[target])
                    vlosses.append(v.item())
            val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
            val_history.append((epoch, val_loss))
            print(f"  [epoch {epoch}] val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "vocab_size": full_train.vocab_size,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, out_path / "phi_full.pt")
                print(f"    -> saved (val_loss {val_loss:.4f})", flush=True)

        if max_steps is not None and step >= max_steps:
            break

    # Final save (whatever the last state is) too, distinct from best.
    torch.save({
        "model": model.state_dict(),
        "cfg": asdict(cfg),
        "vocab_size": full_train.vocab_size,
        "epoch": cfg.epochs - 1,
        "final_val_loss": val_history[-1][1] if val_history else None,
    }, out_path / "phi_full_last.pt")

    summary = {
        "seed": cfg.seed,
        "n_params": n_params,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "total_steps": step,
        "best_val_loss": best_val,
        "val_history": val_history,
        "elapsed_s": time.time() - t0,
    }
    (out_path / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"DONE seed={cfg.seed}  best_val={best_val:.4f}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Cap total optimizer steps (for smoke testing).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    a = ap.parse_args()
    cfg = TrainCfg(seed=a.seed, batch_size=a.batch_size, epochs=a.epochs,
                   device=a.device)
    train(cfg, a.parquet, a.tokenizer, a.out_dir, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
