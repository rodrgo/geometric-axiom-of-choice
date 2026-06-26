"""Figures for the reconstruction-loss generalization experiment.

Reads the per-proof losses saved by reconstruction_loss.py and produces:
  - results/figures/reconstruction_loss_by_depth.png         (raw means, THE headline figure)
  - results/figures/reconstruction_loss_violins.png          (distribution per bucket)
  - results/figures/reconstruction_loss_length_controlled.png (residualized means)
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


BUCKET_ORDER = [
    "constructive (test)",
    "depth 2", "depth 3", "depth 4",
    "depth 5-6", "depth 7-8", "depth 9+",
]


def bucket_of(depth):
    if depth is None:
        return None
    if depth <= 2:
        return "depth 2"
    if depth == 3:
        return "depth 3"
    if depth == 4:
        return "depth 4"
    if depth <= 6:
        return "depth 5-6"
    if depth <= 8:
        return "depth 7-8"
    return "depth 9+"


def main():
    fig_dir = Path('results/figures'); fig_dir.mkdir(exist_ok=True)
    out_dir = Path('results/data/depth_analysis')

    npz = np.load(out_dir / 'reconstruction_loss_per_proof.npz',
                  allow_pickle=True)
    losses = npz['losses']
    lengths = npz['lengths']
    is_classical = npz['is_classical']
    names = npz['names']

    with open('results/data/depth_analysis/bfs_distances_full.json') as f:
        bfs = json.load(f)
    emb = np.load('results/data/stage4v3p/embeddings.npz')
    train_idx = set(emb['train_idx'].tolist())
    test_idx = set(emb['test_idx'].tolist())

    # Raw buckets
    buckets = defaultdict(list)
    buckets_idx = defaultdict(list)
    for i in range(len(losses)):
        if np.isnan(losses[i]):
            continue
        if not is_classical[i]:
            if i in test_idx:
                buckets["constructive (test)"].append(losses[i])
                buckets_idx["constructive (test)"].append(i)
        else:
            d = bfs.get(names[i])
            b = bucket_of(d)
            if b is not None:
                buckets[b].append(losses[i])
                buckets_idx[b].append(i)

    # Length-residualize (fit on constructive train)
    fit_idx = np.array([i for i in train_idx if not np.isnan(losses[i])])
    reg = LinearRegression().fit(
        np.log1p(lengths[fit_idx]).reshape(-1, 1), losses[fit_idx])
    predicted = reg.predict(np.log1p(lengths).reshape(-1, 1))
    residual = losses - predicted

    # ---- Figure A: mean loss by bucket ----
    means = [np.mean(buckets[b]) for b in BUCKET_ORDER]
    sems = [np.std(buckets[b], ddof=1) / np.sqrt(len(buckets[b]))
            for b in BUCKET_ORDER]
    ns = [len(buckets[b]) for b in BUCKET_ORDER]

    colors = ['#2E7D32'] + plt.cm.Reds(np.linspace(0.85, 0.30, 6)).tolist()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(range(len(BUCKET_ORDER)), means, yerr=sems, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.6)
    ax.set_xticks(range(len(BUCKET_ORDER)))
    ax.set_xticklabels(BUCKET_ORDER, rotation=20, ha='right')
    ax.set_ylabel('Mean reconstruction loss\n(cross-entropy on 20% masked tokens)')
    ax.set_title('Generalization gap: reconstruction loss scales with depth to Classical.choice')
    ax.axhline(y=means[0], color='#2E7D32', linestyle='--', alpha=0.55,
               label=f'constructive baseline ({means[0]:.2f})')
    # annotate n
    y_top = max(m + s for m, s in zip(means, sems))
    for i, (m, s, n) in enumerate(zip(means, sems, ns)):
        ax.text(i, m + s + 0.03, f'n={n:,}', ha='center', va='bottom',
                fontsize=8, color='#333')
    ax.set_ylim(2.15, y_top + 0.22)
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = fig_dir / 'reconstruction_loss_by_depth.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {p}")

    # ---- Figure B: violin plots ----
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    data = [buckets[b] for b in BUCKET_ORDER]
    parts = ax.violinplot(data, positions=range(len(BUCKET_ORDER)),
                          showmeans=False, showmedians=True, widths=0.82)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.75); pc.set_edgecolor('black')
    for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
        if key in parts:
            parts[key].set_color('black'); parts[key].set_linewidth(0.8)
    ax.set_xticks(range(len(BUCKET_ORDER)))
    ax.set_xticklabels(BUCKET_ORDER, rotation=20, ha='right')
    ax.set_ylabel('Reconstruction loss')
    ax.set_title('Distribution of reconstruction loss by depth')
    ax.axhline(y=means[0], color='#2E7D32', linestyle='--', alpha=0.55,
               label=f'constructive baseline ({means[0]:.2f})')
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = fig_dir / 'reconstruction_loss_violins.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {p}")

    # ---- Figure C: length-controlled (residualized) ----
    res_buckets = {b: residual[np.array(buckets_idx[b])] for b in BUCKET_ORDER}
    res_buckets = {b: r[~np.isnan(r)] for b, r in res_buckets.items()}
    res_means = [np.mean(res_buckets[b]) for b in BUCKET_ORDER]
    res_sems = [np.std(res_buckets[b], ddof=1) / np.sqrt(len(res_buckets[b]))
                for b in BUCKET_ORDER]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(range(len(BUCKET_ORDER)), res_means, yerr=res_sems, capsize=4,
           color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(y=0.0, color='black', linewidth=0.6)
    ax.set_xticks(range(len(BUCKET_ORDER)))
    ax.set_xticklabels(BUCKET_ORDER, rotation=20, ha='right')
    ax.set_ylabel('Length-residualized reconstruction loss\n(excess loss vs.\\ '
                  'log-length fit on constructive train)')
    ax.set_title('Depth gradient survives length control')
    y_top = max(m + s for m, s in zip(res_means, res_sems))
    for i, (m, s, n) in enumerate(zip(res_means, res_sems, ns)):
        va = 'bottom' if m >= 0 else 'top'
        offset = 0.02 if m >= 0 else -0.02
        ax.text(i, m + s + offset if m >= 0 else m - s + offset, f'n={n:,}',
                ha='center', va=va, fontsize=8, color='#333')
    ax.set_ylim(-0.10, y_top + 0.12)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = fig_dir / 'reconstruction_loss_length_controlled.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {p}")


if __name__ == '__main__':
    main()
