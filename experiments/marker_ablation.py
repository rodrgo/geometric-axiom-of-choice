"""Depth-stratified ablation: strip classical markers, re-embed with frozen
encoder, recompute depth-stratified k-NN AUC.

Exactly matches depth_knn_auc.py's pipeline and buckets, but with ablated
tactic sequences.
"""
import json, time
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lean.proof_encoder import ProofEncoder, encode_proof, PAD_ID, RESERVED  # noqa: E402
from lean.markers import CLASSICAL_MARKERS  # noqa: E402

OUT = ROOT / "results" / "data" / "depth_analysis"
FIG = ROOT / "results" / "figures"

# ---- Load data ----
proofs = json.load(open(ROOT / "results/data/stage4v3p/proofs.json"))
vocab = json.load(open(ROOT / "results/data/stage4v3p/vocab.json"))
MAX_LEN = 64
VOCAB_SIZE = len(vocab) + RESERVED

# Load distances (already saved from depth_knn_auc.py)
distances = json.load(open(OUT / "bfs_distances_full.json"))

# Load original labels + splits
emb_data = np.load(ROOT / "results/data/stage4v3p/embeddings.npz", allow_pickle=True)
labels = emb_data["labels"]
train_idx = emb_data["train_idx"]

# Which vocab markers are actually present
vocab_markers = [m for m in CLASSICAL_MARKERS if m in vocab]
vocab_marker_ids = {vocab[m] + RESERVED for m in vocab_markers}
print(f"CLASSICAL_MARKERS in vocab: {vocab_markers}")

# ---- Strip markers ----
print("Stripping markers and re-tokenizing…")
def strip(heads):
    return [h for h in heads if h not in CLASSICAL_MARKERS]

X_ablated = np.zeros((len(proofs), MAX_LEN), dtype=np.int64)
X_original = np.zeros((len(proofs), MAX_LEN), dtype=np.int64)
n_changed = 0
tokens_removed = np.zeros(len(proofs), dtype=int)
for i, p in enumerate(proofs):
    orig_heads = p["invocation_heads"]
    ab_heads = strip(orig_heads)
    if len(ab_heads) != len(orig_heads):
        n_changed += 1
    tokens_removed[i] = len(orig_heads) - len(ab_heads)
    X_ablated[i] = encode_proof(ab_heads, vocab, MAX_LEN)
    X_original[i] = encode_proof(orig_heads, vocab, MAX_LEN)
print(f"  proofs changed: {n_changed}/{len(proofs)} "
      f"({n_changed/len(proofs)*100:.1f}%)")
print(f"  tokens removed total: {tokens_removed.sum()}, "
      f"mean per changed proof: {tokens_removed[tokens_removed>0].mean():.2f}")

# ---- Load frozen encoder ----
print("Loading frozen encoder…")
device = "cpu"
model = ProofEncoder(vocab_size=VOCAB_SIZE, d_model=128, nhead=4,
                     enc_layers=4, dec_layers=2, max_len=MAX_LEN, dropout=0.1)
state = torch.load(ROOT / "results/data/stage4v3p/encoder.pt",
                    map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

def embed_all(X):
    reps = []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(X), bs):
            batch = torch.from_numpy(X[i:i+bs]).long()
            z = model.encode_pool(batch)
            z = F.normalize(z, dim=-1)
            reps.append(z.numpy())
    return np.concatenate(reps, axis=0)

print("Embedding ablated…")
t0 = time.time()
E_ab = embed_all(X_ablated)
print(f"  done in {time.time()-t0:.1f}s, shape={E_ab.shape}")
# Sanity: re-embed original and compare to saved embeddings
print("Embedding original (sanity)…")
E_orig_new = embed_all(X_original)
E_orig_saved = emb_data["embeddings"]
max_dev = np.abs(E_orig_new - E_orig_saved).max()
print(f"  max abs dev from saved embeddings: {max_dev:.5f} "
      f"({'OK' if max_dev < 1e-3 else 'WARN'})")

# ---- Depth-stratified AUC ----
depths = np.array([distances.get(p["name"], -1) if p["is_classical"] else -1
                   for p in proofs], dtype=int)

BUCKETS = [
    ("depth 2",   2, 2),
    ("depth 3",   3, 3),
    ("depth 4",   4, 4),
    ("depth 5",   5, 5),
    ("depth 6",   6, 6),
    ("depth 7-8", 7, 8),
    ("depth 9+",  9, 999),
]

def auc_table(E, name):
    sc = StandardScaler().fit(E[train_idx])
    Es = sc.transform(E)
    nn = NearestNeighbors(n_neighbors=5).fit(Es[train_idx])
    score = nn.kneighbors(Es)[0].mean(axis=1)
    # held-out constructive = all constructive not in train
    constr_eval = (labels == 0).copy()
    constr_eval[train_idx] = False
    cs = score[constr_eval]
    rows = []
    for lbl, lo, hi in BUCKETS:
        mask = (depths >= lo) & (depths <= hi)
        n_b = int(mask.sum())
        if n_b < 20:
            rows.append({"bucket": lbl, "n": n_b, "auc": None})
            continue
        bs = score[mask]
        y = np.concatenate([np.zeros(len(cs)), np.ones(n_b)])
        s = np.concatenate([cs, bs])
        rows.append({"bucket": lbl, "n": n_b, "auc": float(roc_auc_score(y, s))})
    return rows

print("\n=== Ablated (markers stripped) ===")
ab_rows = auc_table(E_ab, "ablated")
print("=== Original (re-embedded, sanity) ===")
orig_rows = auc_table(E_orig_new, "original")

# Marker frequency per bucket
print("\n=== Marker frequency per bucket ===")
marker_stats = []
for lbl, lo, hi in BUCKETS:
    mask = (depths >= lo) & (depths <= hi)
    if mask.sum() == 0:
        continue
    has = 0; total_tokens = 0
    for i in np.where(mask)[0]:
        heads = proofs[i]["invocation_heads"]
        markers = [h for h in heads if h in CLASSICAL_MARKERS]
        if markers:
            has += 1
        total_tokens += len(markers)
    n = int(mask.sum())
    frac = has / n
    avg = total_tokens / n
    marker_stats.append({"bucket": lbl, "n": n,
                         "frac_with_marker": frac,
                         "avg_markers_per_proof": avg})
    print(f"  {lbl:<10s} n={n:>5d}  frac_with_marker={frac:.3f}  avg={avg:.3f}")

# ---- Save + print summary ----
combined = []
for i, b in enumerate(BUCKETS):
    row = {
        "bucket": b[0], "depth_range": [b[1], b[2]],
        "n": ab_rows[i]["n"],
        "original_knn_auc": orig_rows[i]["auc"],
        "ablated_knn_auc": ab_rows[i]["auc"],
        "frac_with_marker": marker_stats[i]["frac_with_marker"],
        "avg_markers_per_proof": marker_stats[i]["avg_markers_per_proof"],
    }
    if row["original_knn_auc"] is not None and row["ablated_knn_auc"] is not None:
        row["drop"] = row["original_knn_auc"] - row["ablated_knn_auc"]
    combined.append(row)

print("\n" + "=" * 86)
print(f"{'bucket':<10s} {'n':>5s} {'orig AUC':>9s} {'ablated':>9s} {'drop':>7s} "
      f"{'frac_markers':>13s} {'avg':>6s}")
for r in combined:
    print(f"{r['bucket']:<10s} {r['n']:>5d} "
          f"{r['original_knn_auc']:>9.3f} {r['ablated_knn_auc']:>9.3f} "
          f"{r.get('drop',0):>7.3f} "
          f"{r['frac_with_marker']:>13.3f} {r['avg_markers_per_proof']:>6.3f}")

with open(OUT / "depth_ablation_results.json", "w") as f:
    json.dump({"markers": sorted(CLASSICAL_MARKERS),
               "vocab_markers": vocab_markers,
               "buckets": combined,
               "notes": {"proofs_changed_frac": n_changed/len(proofs),
                         "embedding_sanity_max_dev": float(max_dev)}}, f, indent=2)

# ---- Figure ----
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(BUCKETS))
w = 0.38
orig_aucs = [r["original_knn_auc"] for r in combined]
ab_aucs = [r["ablated_knn_auc"] for r in combined]
ax.bar(x - w/2, orig_aucs, w, label="original", color="C0")
ax.bar(x + w/2, ab_aucs, w, label="markers stripped", color="C3")
for i, r in enumerate(combined):
    ax.text(i, max(orig_aucs[i], ab_aucs[i]) + 0.01,
            f"{r['frac_with_marker']*100:.0f}%",
            ha="center", fontsize=8, color="gray")
ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels([f"{b['bucket']}\nn={b['n']}" for b in combined], fontsize=9)
ax.set_ylabel("k-NN AUC vs held-out constructive")
ax.set_title("Depth-stratified ablation: removing classical tactic markers"
             "\n(% above bars = fraction of bucket proofs containing a marker)")
ax.legend()
ax.set_ylim(0.45, 0.9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIG / "depth_ablation_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"\nSaved: {FIG/'depth_ablation_comparison.png'}")
print(f"Saved: {OUT/'depth_ablation_results.json'}")
