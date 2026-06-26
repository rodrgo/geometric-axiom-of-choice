"""Step 3.4: sanity check — did the encoder learn anything useful?

Three diagnostics per seed:

  1) Reconstruction loss on held-out constructive proofs (whole-word
     masking, 10 masks per proof). Pass: mean CE < 3.0 nats. We also
     report the uniform-vocab baseline log(V) for context.

  2) Domain probe. Logistic regression on the frozen 128-d projected
     embeddings predicting the top-level Mathlib directory across
     domains with >= 200 theorems each. 5-fold CV. Pass: accuracy
     >= 0.20. Reports the most-frequent-class baseline + a
     label-shuffled baseline as sanity floor.

  3) Length probe. Ridge regression on the 128-d embeddings predicting
     log(n_tokens). 5-fold CV. Pass: R^2 >= 0.3.

If ALL three pass, the seed's downstream results are reliable. If any
fail, mark the seed unreliable and proceed with the others.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import FullSourceEncoder, PAD_ID, MASK_ID  # noqa: E402
from train import (ConstructiveTrainDS, whole_word_mask,  # noqa: E402
                   collate, MASK_PROB)

PARQUET = HERE.parent / "data/full_source.parquet"
TOKENIZER = HERE.parent / "tokenizer/artifacts/bpe_32k.json"

# Pre-committed thresholds (from plan_full_data.md Step 3.4).
THRESH_RECON_LOSS = 3.0
THRESH_DOMAIN_ACC = 0.20
THRESH_LENGTH_R2 = 0.30
MIN_DOMAIN_N = 200

N_VAL_MASKS = 10  # masks per held-out proof


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


def diag_reconstruction(model: FullSourceEncoder, device: torch.device,
                        seed: int) -> dict:
    """Mean CE on held-out constructive proofs under whole-word masking."""
    ds = ConstructiveTrainDS(str(PARQUET), str(TOKENIZER), split="train")
    # Same val split partition as training.
    rng = np.random.default_rng(seed * 7919 + 13)
    idx = rng.permutation(len(ds))
    n_val = int(round(0.10 * len(ds)))
    val_idx = idx[:n_val].tolist()

    val_subset = torch.utils.data.Subset(ds, val_idx)
    loader = torch.utils.data.DataLoader(
        val_subset, batch_size=32, shuffle=False, collate_fn=collate)

    mask_rng = torch.Generator(device=device)
    mask_rng.manual_seed(seed * 31337 + 42)

    losses = []
    with torch.no_grad():
        for ids, wids in loader:
            ids = ids.to(device); wids = wids.to(device)
            for _ in range(N_VAL_MASKS):
                corrupted, target = whole_word_mask(
                    ids, wids, MASK_PROB, mask_rng)
                if not target.any():
                    continue
                logits, _ = model(corrupted)
                l = F.cross_entropy(logits[target], ids[target])
                losses.append(l.item())
    mean_loss = float(np.mean(losses))
    return {
        "mean_recon_loss_nats": mean_loss,
        "uniform_baseline_nats": float(math.log(model.vocab_size)),
        "n_eval_batches": len(losses),
        "passes": mean_loss < THRESH_RECON_LOSS,
    }


def diag_domain_probe(emb_path: Path, seed: int) -> dict:
    """5-fold logistic regression accuracy on Mathlib top-level domain."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    d = np.load(emb_path, allow_pickle=True)
    X_all = d["projected"]
    is_classical = d["is_classical"]
    splits = d["splits"]
    domains = np.load(emb_path.parent / "_domains.npy", allow_pickle=True) \
        if (emb_path.parent / "_domains.npy").exists() else None
    if domains is None:
        # Read domains from parquet, aligned by name (which preserves order).
        t = pq.read_table(PARQUET, columns=["domain"])
        domains = np.asarray(t["domain"].to_pylist())

    # Use constructive proofs only (avoid leakage from classical proofs
    # being concentrated in certain domains).
    mask = (~is_classical) & (splits != "test")
    X = X_all[mask]; y = domains[mask]

    # Filter to domains with >= MIN_DOMAIN_N theorems.
    unique, counts = np.unique(y, return_counts=True)
    big_domains = set(unique[counts >= MIN_DOMAIN_N])
    keep = np.array([d in big_domains for d in y])
    X = X[keep]; y = y[keep]
    if len(X) < 100:
        return {"error": "too few samples after filtering",
                "passes": False}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(X, y):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    mean_acc = float(np.mean(accs))

    # Baselines.
    _, c2 = np.unique(y, return_counts=True)
    mfc = float(c2.max() / c2.sum())
    uniform = 1.0 / len(set(y))

    # Shuffled-label baseline.
    rng = np.random.default_rng(seed + 1)
    y_shuf = rng.permutation(y)
    shuf_accs = []
    for tr, te in cv.split(X, y_shuf):
        clf = LogisticRegression(max_iter=500, n_jobs=-1)
        clf.fit(X[tr], y_shuf[tr])
        shuf_accs.append(float(clf.score(X[te], y_shuf[te])))
    mean_shuf = float(np.mean(shuf_accs))

    return {
        "mean_cv_accuracy": mean_acc,
        "shuffled_baseline_acc": mean_shuf,
        "most_frequent_class_baseline": mfc,
        "uniform_baseline": uniform,
        "n_domains_kept": int(len(set(y))),
        "n_samples": int(len(X)),
        "passes": mean_acc > THRESH_DOMAIN_ACC and mean_acc > mean_shuf + 0.05,
    }


def diag_length_probe(emb_path: Path, seed: int) -> dict:
    """5-fold ridge R^2 predicting log token count."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    d = np.load(emb_path, allow_pickle=True)
    X_all = d["projected"]
    is_classical = d["is_classical"]
    splits = d["splits"]
    t = pq.read_table(PARQUET, columns=["n_tokens"])
    n_tokens = np.asarray(t["n_tokens"].to_pylist(), dtype=np.float32)

    mask = (~is_classical) & (splits != "test")
    X = X_all[mask]
    y = np.log1p(n_tokens[mask])
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in cv.split(X):
        m = Ridge(alpha=1.0)
        m.fit(X[tr], y[tr])
        r2s.append(float(m.score(X[te], y[te])))
    mean_r2 = float(np.mean(r2s))
    return {
        "mean_cv_r2": mean_r2,
        "n_samples": int(len(X)),
        "passes": mean_r2 >= THRESH_LENGTH_R2,
    }


def run_one_seed(seed: int, ckpt_root: Path, emb_root: Path,
                 device: torch.device) -> dict:
    print(f"\n=== seed {seed} ===", flush=True)
    ckpt = ckpt_root / f"seed_{seed}/phi_full.pt"
    emb_path = emb_root / f"seed_{seed}/raw.npz"
    for p in (ckpt, emb_path):
        if not p.exists():
            print(f"  MISSING: {p}", flush=True)
            return {"seed": seed, "error": f"missing {p}",
                    "all_pass": False}

    model = load_model(ckpt, device)
    print(f"  loaded {ckpt}", flush=True)

    print(f"  diag 1: reconstruction loss...", flush=True)
    d1 = diag_reconstruction(model, device, seed)
    print(f"    recon_loss={d1['mean_recon_loss_nats']:.3f}  "
          f"(baseline={d1['uniform_baseline_nats']:.2f})  "
          f"pass={d1['passes']}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"  diag 2: domain probe...", flush=True)
    d2 = diag_domain_probe(emb_path, seed)
    print(f"    acc={d2.get('mean_cv_accuracy', float('nan')):.3f}  "
          f"shuf={d2.get('shuffled_baseline_acc', float('nan')):.3f}  "
          f"mfc={d2.get('most_frequent_class_baseline', float('nan')):.3f}  "
          f"pass={d2.get('passes', False)}", flush=True)

    print(f"  diag 3: length probe...", flush=True)
    d3 = diag_length_probe(emb_path, seed)
    print(f"    R^2={d3['mean_cv_r2']:.3f}  pass={d3['passes']}", flush=True)

    return {
        "seed": seed,
        "reconstruction": d1,
        "domain_probe": d2,
        "length_probe": d3,
        "all_pass": d1["passes"] and d2.get("passes", False) and d3["passes"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckpt-root", default=str(HERE / "checkpoints"))
    ap.add_argument("--emb-root", default=str(HERE / "embeddings"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(HERE.parent /
                                          "analysis/sanity_results.json"))
    a = ap.parse_args()

    device = pick_device(a.device)
    print(f"device: {device}")
    results = []
    for seed in a.seeds:
        results.append(run_one_seed(seed, Path(a.ckpt_root),
                                     Path(a.emb_root), device))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "thresholds": {
            "recon_loss_nats": THRESH_RECON_LOSS,
            "domain_accuracy": THRESH_DOMAIN_ACC,
            "length_r2": THRESH_LENGTH_R2,
        },
        "per_seed": results,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")
    for r in results:
        print(f"  seed {r['seed']}: all_pass={r.get('all_pass', False)}")


if __name__ == "__main__":
    main()
