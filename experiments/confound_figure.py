"""Author & topic confound controls figure (paper fig:confound_controls).

Panel (a): the depth-stratified classical-vs-constructive anomaly gap survives
three matched controls -- same file (paper), same author, same statement/topic.
Panel (b): finer-grained within-subdomain (level-2) k-NN AUC distribution.

Reads author_confound.json, topic_confound.json, and exact_matching_results.json
(from confound_author_analysis.py, confound_topic_analysis.py, mixed_effects.py).
Produces results/figures/confound_controls.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "results/data/reviewer"
FIG = ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)

author = json.load(open(REV / "author_confound.json"))
topic = json.load(open(REV / "topic_confound.json"))
filep = json.load(open(REV / "exact_matching_results.json"))

BUCKETS = ["d2", "d3-4", "d5-6", "d7+"]


def gaps(src, default=np.nan):
    return [src.get(b, {}).get("mean_diff", default) for b in BUCKETS]


file_gap = gaps(filep)
author_gap = gaps(author["matched_pairs"])
topic_gap = gaps(topic["statement_matched_pairs"]["per_depth_at_tau"]["buckets"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

# Palette matched to the paper's three-measurement figures
# (hero_figure.panel_c_lines / three_measurements): blue / red / green.
BLUE, RED, GREEN = "#1565C0", "#C62828", "#2E7D32"

# --- panel (a): grouped matched-pair bars ---
x = np.arange(len(BUCKETS))
w = 0.26
ax1.bar(x - w, file_gap, w, label="same file (paper)", color=BLUE)
ax1.bar(x, author_gap, w, label="same author", color=RED)
ax1.bar(x + w, topic_gap, w, label="same statement / topic", color=GREEN)
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(["depth 2", "depth 3-4", "depth 5-6", "depth 7+"])
ax1.set_ylabel("anomaly gap, classical $-$ constructive (SD)")
ax1.set_title("(a) Gap survives matched controls")
ax1.legend(frameon=False, fontsize=9)

# --- panel (b): within-subdomain AUC distribution ---
sub = topic["within_subdomain_knn_level2"]
aucs = np.array([v["auc_knn_k5"] for v in sub["per_subdomain"].values()])
ax2.hist(aucs, bins=np.arange(0.45, 0.96, 0.05), color=BLUE,
         edgecolor="white")
ax2.axvline(0.5, color="k", ls=":", lw=1, label="chance")
ax2.axvline(np.median(aucs), color=RED, ls="--", lw=1.5,
            label=f"median {np.median(aucs):.3f}")
ax2.set_xlabel("within-subdomain $k$-NN AUC")
ax2.set_ylabel("# subdomains")
ax2.set_title(f"(b) {sub['n_subdomains']} level-2 subdomains")
ax2.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(FIG / "confound_controls.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG/'confound_controls.png'}")

