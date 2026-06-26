"""Hero figure: three-panel summary of the paper's depth law.

Panel A: depth schematic. Stacked depth strata fanning out from
  ``Classical.choice``. Width of each stratum is proportional to the
  number of Mathlib theorems at that depth; layout has no arbitrary
  angles.

Panel B: case-study gradient. Five representative classical proofs,
  one per depth bucket (2 through 6), with per-tactic reconstruction
  loss rendered as cell intensity.

Panel C: three depth-stratified metrics (k-NN AUC, reconstruction
  loss excess, fraction outside the q=90 superlevel set), each
  rescaled into a normalized "boundary strength" axis. Same depth
  buckets as Panel B and the rest of the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
FIG = ROOT / "results" / "figures"

CMAP = plt.cm.plasma_r


def load_inputs():
    with open(DATA / "stage4v3p" / "proofs.json") as f:
        proofs = json.load(f)
    with open(DATA / "depth_analysis" / "bfs_distances_full.json") as f:
        bfs = json.load(f)
    return proofs, bfs


# -----------------------------------------------------------------
# Panel A: depth schematic
# -----------------------------------------------------------------

def panel_a_strata(ax, proofs, bfs):
    """Stacked depth strata. Width = bucket population (log-scaled to
    avoid the depth-9+ tail vanishing); no random angles.
    Preserved as an alternative; not used by default."""
    # Population counts per bucket from the actual data.
    pop = {"2": 0, "3": 0, "4-6": 0, "7-8": 0, "9+": 0}
    for p in proofs:
        if not p["is_classical"]:
            continue
        d = bfs.get(p["name"])
        if d is None:
            continue
        if d == 2: pop["2"] += 1
        elif d == 3: pop["3"] += 1
        elif 4 <= d <= 6: pop["4-6"] += 1
        elif 7 <= d <= 8: pop["7-8"] += 1
        elif d >= 9: pop["9+"] += 1

    labels = ["2", "3", "4-6", "7-8", "9+"]
    counts = [pop[l] for l in labels]
    # Linear scaling normalised so the widest band fills the panel.
    max_count = max(counts)
    widths = [0.18 + 0.78 * c / max_count for c in counts]
    colour_vals = [0.10, 0.30, 0.50, 0.72, 0.92]

    # Axiom node
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.20, 0.88), 0.60, 0.07,
        boxstyle="round,pad=0.012",
        facecolor="#222", edgecolor="black"))
    ax.text(0.50, 0.915, "Classical.choice", ha="center", va="center",
            color="white", fontsize=9, family="monospace",
            fontweight="bold")

    # Strata
    y_top = 0.78
    band_h = 0.115
    gap = 0.02
    for i, (lbl, w, cv, n) in enumerate(zip(labels, widths, colour_vals, counts)):
        y = y_top - i * (band_h + gap) - band_h
        x_left = 0.5 - w / 2
        rect = mpatches.FancyBboxPatch(
            (x_left, y), w, band_h, boxstyle="round,pad=0.005",
            facecolor=CMAP(cv), edgecolor="black", linewidth=0.6,
            alpha=0.90)
        ax.add_patch(rect)
        # Depth label on left
        ax.text(x_left - 0.015, y + band_h / 2, f"depth {lbl}",
                ha="right", va="center", fontsize=8.5)
        # Count on right
        ax.text(x_left + w + 0.015, y + band_h / 2, f"n={n:,}",
                ha="left", va="center", fontsize=8)


    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("(a) Mathlib stratified by depth from the axiom",
                 fontsize=10.5)


def panel_a_distances(ax, proofs, bfs):
    """k-NN distance distributions, depth-stratified.

    Mirrors the style of the Stage-3 classifier hull-distance panel:
    overlaid semi-transparent histograms, one per depth bucket plus
    the constructive-test baseline. Distance = mean Euclidean distance
    to the 5 nearest constructive-training embeddings, on standardized
    features (matches tab:depth_strat methodology)."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    emb_data = np.load(DATA / "stage4v3p" / "embeddings.npz")
    X = emb_data["embeddings"].astype(np.float64)
    train_idx = emb_data["train_idx"]
    test_idx = emb_data["test_idx"]
    val_idx = emb_data["val_idx"]

    scaler = StandardScaler().fit(X[train_idx])
    Xz = scaler.transform(X)
    nn = NearestNeighbors(n_neighbors=5).fit(Xz[train_idx])
    dists = nn.kneighbors(Xz)[0].mean(axis=1)

    held_out = set(test_idx.tolist()) | set(val_idx.tolist())
    depths = np.array([bfs.get(p["name"], -1) for p in proofs])
    is_cls = np.array([bool(p["is_classical"]) for p in proofs])

    def select(predicate):
        return np.array([i for i, p in enumerate(proofs) if predicate(i, p)])

    # Four groups: constructive baseline + the three extremes of the
    # depth gradient. Fewer groups keep the overlay readable; the full
    # six-bucket breakdown is reported in tab:depth_strat.
    # Low alpha (~0.28) so overlapping curves remain individually
    # visible; the curve outlines below carry most of the silhouette.
    groups = [
        ("Constructive (test)",
         select(lambda i, p: (not p["is_classical"]) and i in held_out),
         "#888888", 0.28),
        ("Classical, depth 2",
         select(lambda i, p: p["is_classical"] and depths[i] == 2),
         CMAP(0.10), 0.30),
        ("Classical, depth 4-6",
         select(lambda i, p: p["is_classical"] and 4 <= depths[i] <= 6),
         CMAP(0.55), 0.28),
        ("Classical, depth 9+",
         select(lambda i, p: p["is_classical"] and depths[i] >= 9),
         CMAP(0.92), 0.28),
    ]

    # Smooth KDE curves rather than stepped bars: the histogram version
    # loses readability when distributions overlap. Common evaluation
    # grid across the [0.5%, 99.5%] quantile range of pooled distances.
    from scipy.stats import gaussian_kde
    all_d = np.concatenate([dists[idx] for _, idx, _, _ in groups
                            if len(idx) > 0])
    x_grid = np.linspace(np.quantile(all_d, 0.005),
                         np.quantile(all_d, 0.995), 400)

    bulk_peaks = []
    for label, idx, colour, alpha in groups:
        if len(idx) < 2:
            continue
        kde = gaussian_kde(dists[idx], bw_method=0.20)
        y = kde(x_grid)
        ax.fill_between(x_grid, 0, y, alpha=alpha, color=colour,
                        linewidth=0.0,
                        label=f"{label} (n={len(idx):,})")
        # Slightly stronger outline carries the silhouette where the
        # fills are very transparent.
        ax.plot(x_grid, y, color=colour, linewidth=1.4, alpha=0.95)
        # Cap on the bulk so the constructive spike at the left edge
        # doesn't blow the y-axis.
        bulk_peaks.append(np.percentile(y, 97))

    ax.set_xlabel("mean distance to 5 nearest constructive proofs",
                  fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title("(a) k-NN distance distribution by depth", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    if bulk_peaks:
        ax.set_ylim(0, 1.5 * max(bulk_peaks))


# -----------------------------------------------------------------
# Panel B: case-study gradient
# -----------------------------------------------------------------

def panel_b(ax):
    """Five depth-bucket representatives with per-token reconstruction
    loss as cell color. Each row has TWO sub-rows: a header line with
    the depth label + theorem name, and a cell line with colored
    tactic-head boxes and the Sigma summary at the end."""
    data = json.loads((DATA / "stage4v3p"
                       / "case_study_gradient.json").read_text())
    buckets = data["buckets"]
    global_peak = max(max(r["signal_B"]) for r in buckets)

    n_rows = len(buckets)
    n_cols_max = max(len(r["invocation_heads"]) for r in buckets)

    # Vertical layout: each row gets a header sub-row (~25% of height)
    # and a cell sub-row (~70%); 5% gap below.
    row_h = 1.0 / n_rows
    head_frac = 0.28
    cell_frac = 0.62
    gap_frac = 1.0 - head_frac - cell_frac

    # Horizontal layout: cells use the full row width; Sigma lives in
    # the header row to keep the cell strip narrow.
    cell_w = 1.0 / n_cols_max
    pad_x = 0.003

    for i, r in enumerate(buckets):
        y_row_top = 1.0 - i * row_h
        y_header = y_row_top - head_frac * row_h / 2
        y_cell_top = y_row_top - head_frac * row_h - cell_frac * row_h

        # Header: "Depth N  <theorem>   Σ=X.X" on a single line. Σ
        # sits in the header rather than alongside the cells, so the
        # cell strip can use the full row width.
        nm = r["name"]
        if len(nm) > 48:
            nm = nm[:46] + ".."
        ax.text(0.0, y_header,
                f"{r['bucket_label']}   {nm}",
                ha="left", va="center", fontsize=8.2, color="#222")
        ax.text(1.0, y_header,
                rf"$\Sigma$={r['sum_B']:.1f}",
                ha="right", va="center", fontsize=8.2, color="#222")

        # Cells
        heads = r["invocation_heads"]
        sigB = r["signal_B"]
        for j, (h, b) in enumerate(zip(heads, sigB)):
            # gamma correction (1.5): low-saliency cells go nearly white,
            # high-saliency cells stay strongly red. No alpha floor.
            x = max(0.0, min(1.0, b / global_peak))
            alpha = 0.95 * (x ** 1.5)
            xc = j * cell_w + pad_x
            rect = mpatches.Rectangle(
                (xc, y_cell_top), cell_w - 2 * pad_x,
                cell_frac * row_h,
                facecolor=(0.84, 0.10, 0.10, alpha),
                edgecolor="#777", linewidth=0.5)
            ax.add_patch(rect)
            txt = h if len(h) <= 10 else h[:9] + "."
            ax.text(xc + (cell_w - 2 * pad_x) / 2,
                    y_cell_top + cell_frac * row_h / 2, txt,
                    ha="center", va="center", fontsize=7.2,
                    family="monospace")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.06)
    ax.axis("off")
    ax.set_title("(b) Per-token saliency by depth (red = high recon. loss)",
                 fontsize=10.5)


# -----------------------------------------------------------------
# Panel C: three measurements
# -----------------------------------------------------------------

def panel_c_lines(ax):
    """k-NN AUC, reconstruction loss excess, fraction-outside-q90
    on a shared rescaled axis.

    Preserved as the legacy panel C; the live hero now uses
    panel_c_beeswarm. The standalone three-measurements figure
    (three_measurements.png) still uses this function."""
    auc = json.loads((DATA / "depth_analysis"
                      / "depth_stratified_auc.json").read_text())
    rec = json.loads((DATA / "depth_analysis"
                      / "reconstruction_loss_results.json").read_text())
    # Containment: the label-free superlevel numbers (PCA-by-CV-variance +
    # CV bandwidth on constructive-train), so the figure agrees with
    # tab:superlevel_containment in the paper.
    q2 = json.loads((DATA / "reviewer"
                     / "q2_relabeled_results.json").read_text())
    cont = q2["superlevel"]["containment_rows"]

    # Bucket pool: depth 2, 3, 4-6, 7-8, 9+ (collapse 5/6 into 4-6).
    auc_by = {r["bucket"]: r for r in auc}
    n4 = auc_by["depth 4"]["n_theorems"]; a4 = auc_by["depth 4"]["auc"]
    n5 = auc_by["depth 5"]["n_theorems"]; a5 = auc_by["depth 5"]["auc"]
    n6 = auc_by["depth 6"]["n_theorems"]; a6 = auc_by["depth 6"]["auc"]
    a_4_6 = (n4 * a4 + n5 * a5 + n6 * a6) / (n4 + n5 + n6)

    auc_vals = np.array([
        auc_by["depth 2"]["auc"], auc_by["depth 3"]["auc"], a_4_6,
        auc_by["depth 7-8"]["auc"], auc_by["depth 9+"]["auc"],
    ])

    # Reconstruction loss has buckets [d2, d3, d4, d5-6, d7-8, d9+];
    # we collapse d4 and d5-6 into a "depth 4-6" by sample-weighted mean.
    r4 = rec["buckets"]["depth 4"]
    r56 = rec["buckets"]["depth 5-6"]
    n4_r, n56_r = r4["n"], r56["n"]
    loss_4_6 = ((r4["mean_loss"] * n4_r + r56["mean_loss"] * n56_r)
                / (n4_r + n56_r))
    recon_vals = np.array([
        rec["buckets"]["depth 2"]["mean_loss"],
        rec["buckets"]["depth 3"]["mean_loss"],
        loss_4_6,
        rec["buckets"]["depth 7-8"]["mean_loss"],
        rec["buckets"]["depth 9+"]["mean_loss"],
    ])
    cons_test_loss = rec["buckets"]["constructive (test)"]["mean_loss"]

    s90 = {r["depth_bucket"]: r["classical_fraction_inside"]
           for r in cont if r["threshold_quantile"] == 90}
    # Same collapse for containment if needed.
    cont_buckets = ["depth 2", "depth 3", "depth 4-6", "depth 7-8",
                    "depth 9+"]
    if all(b in s90 for b in cont_buckets):
        outside_vals = np.array([1.0 - s90[b] for b in cont_buckets])
    else:
        # Fallback: weighted collapse from depth 4 and depth 5-6.
        outside_vals = np.array([
            1.0 - s90["depth 2"], 1.0 - s90["depth 3"],
            1.0 - (s90["depth 4"] * n4_r + s90["depth 5-6"] * n56_r)
                  / (n4_r + n56_r),
            1.0 - s90["depth 7-8"], 1.0 - s90["depth 9+"],
        ])

    auc_norm = (auc_vals - 0.5) / (auc_vals.max() - 0.5)
    recon_norm = (recon_vals - cons_test_loss) / (recon_vals.max() - cons_test_loss)
    outside_norm = (outside_vals - 0.10) / (outside_vals.max() - 0.10)

    labels = ["2", "3", "4-6", "7-8", "9+"]
    x = np.arange(len(labels))

    ax.plot(x, auc_norm, "o-", color="#1565C0", linewidth=2.4, markersize=8,
            label=rf"$k$-NN AUC: {auc_vals[0]:.2f} $\to$ {auc_vals[-1]:.2f}")
    ax.plot(x, recon_norm, "s-", color="#C62828", linewidth=2.4, markersize=8,
            label=rf"Recon. loss: {recon_vals[0]:.2f} $\to$ {recon_vals[-1]:.2f}")
    ax.plot(x, outside_norm, "^-", color="#2E7D32", linewidth=2.4, markersize=8,
            label=rf"Outside $S_{{90}}$: {outside_vals[0]:.2f} $\to$ {outside_vals[-1]:.2f}")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("depth from Classical.choice", fontsize=10)
    ax.set_ylabel("normalized boundary strength", fontsize=10)
    ax.set_title("(c) Three measurements, one gradient", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.set_ylim(-0.15, 1.15)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8.5)


# -----------------------------------------------------------------
# Panel C (current): beeswarm of high-D log-density by depth bucket
# -----------------------------------------------------------------

def panel_c_beeswarm(ax):
    """Strip plot of per-proof log-density under the constructive
    high-D KDE, stratified by depth bucket. Median tick per column;
    horizontal line at the q=90 threshold.

    Pipeline matches the q2-rework superlevel protocol so the
    fraction-outside annotations agree exactly with
    tab:superlevel_containment: PCA on raw (unstandardized)
    constructive-train embeddings to ≥90% variance, KDE on the
    unscaled PCA output with bandwidth 0.3 (CV-selected on
    constructive train), threshold at the 10th percentile of
    constructive-test log-density."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity

    emb_data = np.load(DATA / "stage4v3p" / "embeddings.npz")
    X = emb_data["embeddings"].astype(np.float64)
    train_idx = emb_data["train_idx"]
    test_idx = set(emb_data["test_idx"].tolist())
    val_idx = set(emb_data["val_idx"].tolist())
    with open(DATA / "stage4v3p" / "proofs.json") as f:
        proofs = json.load(f)
    with open(DATA / "depth_analysis"
              / "bfs_distances_full.json") as f:
        bfs = json.load(f)
    is_classical = np.array([bool(p["is_classical"]) for p in proofs])
    depths = np.array([bfs.get(p["name"], -1) for p in proofs])

    train_constr = np.array([i for i in train_idx if not is_classical[i]])
    con_train_emb = X[train_constr]
    # Determine PCA dim from cum-variance on constructive train; then
    # refit at that dim. Matches q2_rework.py:303-310.
    pca_full = PCA(random_state=42).fit(con_train_emb)
    pca_n = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_),
                                 0.90) + 1)
    pca = PCA(n_components=pca_n, random_state=42).fit(con_train_emb)
    Xp = pca.transform(X)
    kde = KernelDensity(kernel="gaussian",
                        bandwidth=0.3).fit(pca.transform(con_train_emb))
    log_p = kde.score_samples(Xp)

    # q=90 threshold: 10th percentile of held-out constructive test
    # log-density (matches q2_rework using test_idx only).
    test_only_constr = np.array([i for i in emb_data["test_idx"]
                                  if not is_classical[i]])
    t_q90 = float(np.percentile(log_p[test_only_constr], 10))
    # held_constr is the test+val constructive sample we visualize in
    # the leftmost beeswarm column (and use for sub-sampling).
    held_constr = np.array([i for i, c in enumerate(is_classical)
                            if (not c) and (i in test_idx or i in val_idx)])

    buckets = [
        ("Constructive\n(test+val)", held_constr, "#888888"),
        ("d=2", np.where(is_classical & (depths == 2))[0], CMAP(0.10)),
        ("d=3", np.where(is_classical & (depths == 3))[0], CMAP(0.30)),
        ("d=4-6",
         np.where(is_classical & (depths >= 4) & (depths <= 6))[0],
         CMAP(0.55)),
        ("d=7-8",
         np.where(is_classical & (depths >= 7) & (depths <= 8))[0],
         CMAP(0.78)),
        ("d=9+", np.where(is_classical & (depths >= 9))[0], CMAP(0.95)),
    ]

    rng = np.random.default_rng(0)
    sub_n = 1500  # cap per bucket so densest columns stay readable
    fracs_outside = []
    for x_c, (label, idx, colour) in enumerate(buckets):
        if len(idx) == 0:
            fracs_outside.append(None)
            continue
        keep = (rng.choice(idx, size=sub_n, replace=False)
                if len(idx) > sub_n else idx)
        y = log_p[keep]
        jitter = rng.uniform(-0.32, 0.32, size=len(keep))
        ax.scatter(x_c + jitter, y, s=3, color=colour, alpha=0.45,
                   rasterized=True, edgecolors="none")
        median_y = float(np.median(log_p[idx]))
        ax.plot([x_c - 0.4, x_c + 0.4], [median_y, median_y],
                color="#222", linewidth=1.8, solid_capstyle="butt",
                zorder=10)
        fracs_outside.append(float((log_p[idx] < t_q90).mean()))

    # q=90 threshold line
    ax.axhline(t_q90, color="#C62828", linestyle="--", linewidth=1.4,
               alpha=0.9, zorder=5)
    ax.text(len(buckets) - 1 + 0.45, t_q90, " $q{=}90$",
            color="#C62828", fontsize=8.5, va="center",
            ha="left", fontweight="bold")

    # Fraction-outside annotation just below the threshold line, inside
    # the axes so it isn't clipped by bbox_inches='tight'.
    q01 = float(np.percentile(log_p, 0.5))
    q99 = float(np.percentile(log_p, 99.5))
    ax.set_ylim(q01, q99)
    y_ann = q01 + 0.04 * (q99 - q01)
    for x_c, frac in enumerate(fracs_outside):
        if frac is None:
            continue
        ax.text(x_c, y_ann, f"{100*frac:.0f}%",
                ha="center", va="bottom", fontsize=8.5,
                color="#222",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.2))

    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([b[0] for b in buckets], fontsize=8.5)
    ax.set_ylabel("log density under constructive KDE (high-D)",
                  fontsize=9)
    ax.set_xlabel("depth bucket  (fraction outside $S_{90}$ below)",
                  fontsize=9)
    ax.set_title("(c) Per-proof log-density by depth",
                 fontsize=10.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.6, len(buckets) - 0.4)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    proofs, bfs = load_inputs()

    plt.rcParams["text.usetex"] = False
    # Three landscape panels.
    fig = plt.figure(figsize=(16.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.2],
                          wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a_distances(ax_a, proofs, bfs)
    panel_b(ax_b)
    panel_c_beeswarm(ax_c)

    out = FIG / "hero_figure.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
