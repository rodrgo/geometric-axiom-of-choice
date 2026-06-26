"""Illustration of the proof-representation pipeline.

Five stages laid out left to right with NO arrows between them; each
box is content-fitted (width = longest line at the chosen font size,
plus a uniform padding). Real Mathlib proof:
CategoryTheory.GrothendieckTopology.arrow_trans.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"

# Real proof: theorem arrow_trans (Mathlib/CategoryTheory/Sites/Grothendieck.lean).
THEOREM = "GrothendieckTopology.arrow_trans"
PROOF_LINES = [
    ("intro", " k"),
    ("apply", " J.transitive h"),
    ("intro", " Z g hg"),
    ("rw",    " [<- Sieve.pullback_comp]"),
    ("apply", " k (g >> f) hg"),
]
HEADS = [h for h, _ in PROOF_LINES]
SEQUENCE = ["[CLS]"] + HEADS + ["[SEP]"]
MASKED_POSITIONS = {2, 5}  # mask second token ("apply") and fifth ("rw")


HEAD_COLOR = "#1565C0"
MASK_COLOR = "#C62828"
BOX_FACE = "#fafafa"
BOX_EDGE = "#555"
ENC_FACE = "#fff3e0"
ENC_EDGE = "#ef6c00"


# Approximate character widths in axes-fraction units at the figure size
# we use (14, 4). A 9-point monospace char is ~0.0095 axes units; a 9-pt
# proportional char is ~0.0072. These are tuned by trial below.
CW_MONO = 0.0072
CW_PROP = 0.0058


def text_width(s: str, mono: bool = True, fontsize: float = 9.0) -> float:
    """Approximate axes-fraction width of a string at our default font size.
    Scaled linearly by font size."""
    cw = (CW_MONO if mono else CW_PROP) * (fontsize / 9.0)
    return cw * len(s)


# -----------------------------------------------------------------
# Drawing primitives
# -----------------------------------------------------------------

def fitted_box(ax, x_left, y_bottom, width, height, **kwargs):
    rect = mpatches.FancyBboxPatch(
        (x_left, y_bottom), width, height,
        boxstyle="round,pad=0.010",
        facecolor=kwargs.pop("facecolor", BOX_FACE),
        edgecolor=kwargs.pop("edgecolor", BOX_EDGE),
        linewidth=kwargs.pop("linewidth", 0.9),
        **kwargs)
    ax.add_patch(rect)


def title_for(stage_num: int, title: str) -> str:
    return f"({stage_num}) {title}"


# -----------------------------------------------------------------
# Stage builders. Each returns the box width and renders into ax.
# Position by passing x_left; y axis is shared.
# -----------------------------------------------------------------

def stage1_source(ax, x_left, y_top, y_bottom):
    title = title_for(1, "Mathlib proof")
    body_lines = [(h, args) for h, args in PROOF_LINES]
    # Real signature from Mathlib (CategoryTheory/Sites/Grothendieck.lean),
    # ASCII-substituted to avoid font glyph issues. The "..." stands in
    # for the universally-quantified hypothesis to keep the box compact.
    header_lines = [
        "theorem arrow_trans (f : Y -> X)",
        "    (h : J.Covers S f) :",
        "    ... -> J.Covers R f := by",
    ]
    header_fs = 8.0
    body_fs = 9.5
    # Empirically, matplotlib renders monospace ~12% narrower than our
    # CW_MONO estimate, so the boxes look loose. Apply a fudge factor
    # here rather than changing CW_MONO globally (which would also
    # shrink the token cells in stages 3 and 4).
    TIGHT = 0.87
    longest_body_w = max(
        text_width(h, mono=True, fontsize=body_fs)
        + text_width(a, mono=True, fontsize=body_fs)
        for h, a in body_lines) * TIGHT + 0.012  # body indent
    longest_header_w = max(
        text_width(h, mono=True, fontsize=header_fs)
        for h in header_lines) * TIGHT
    title_w = text_width(title + "  ", mono=False, fontsize=10) * TIGHT
    pad_x = 0.010
    width = max(longest_body_w, longest_header_w, title_w) + 2 * pad_x

    fitted_box(ax, x_left, y_bottom, width, y_top - y_bottom)
    # Title
    ax.text(x_left + pad_x, y_top - 0.04, title,
            fontsize=10, fontweight="bold", ha="left", va="top")
    # Header lines (real theorem signature, smaller)
    line_y = y_top - 0.12
    dy_h = 0.06
    for hline in header_lines:
        ax.text(x_left + pad_x, line_y, hline,
                fontsize=header_fs, family="monospace", color="#888",
                ha="left", va="top")
        line_y -= dy_h
    line_y -= 0.02
    dy = 0.07
    for h, args in body_lines:
        ax.text(x_left + pad_x + 0.012, line_y, h,
                fontsize=9.5, family="monospace", color=HEAD_COLOR,
                fontweight="bold", ha="left", va="top")
        ax.text(x_left + pad_x + 0.012
                + text_width(h, mono=True, fontsize=9.5),
                line_y, args,
                fontsize=9.5, family="monospace", color="#555",
                ha="left", va="top")
        line_y -= dy
    return width


def stage2_heads(ax, x_left, y_top, y_bottom):
    title = title_for(2, "Tactic heads")
    pad_x = 0.010
    TIGHT = 0.87
    # Width: max of (title at fs=10 proportional, longest head at the
    # rendered fs=10.5 monospace), shrunk to match actual rendering.
    title_w = text_width(title, mono=False, fontsize=10) * TIGHT
    longest_head = max(HEADS, key=len)
    head_w = text_width(longest_head, mono=True, fontsize=10.5) * TIGHT
    width = max(title_w, head_w) + 2 * pad_x
    fitted_box(ax, x_left, y_bottom, width, y_top - y_bottom)
    ax.text(x_left + width / 2, y_top - 0.04, title,
            fontsize=10, fontweight="bold", ha="center", va="top")
    ax.text(x_left + width / 2, y_top - 0.10,
            "first identifier", ha="center", va="top",
            fontsize=7.8, color="#666", style="italic")
    ax.text(x_left + width / 2, y_top - 0.155,
            "of each call",
            ha="center", va="top", fontsize=7.8, color="#666",
            style="italic")
    line_y = y_top - 0.24
    dy = 0.08
    for h in HEADS:
        ax.text(x_left + width / 2, line_y, h,
                fontsize=10.5, family="monospace", color=HEAD_COLOR,
                fontweight="bold", ha="center", va="top")
        line_y -= dy
    return width


def stage3_sequence(ax, x_left, y_top, y_bottom, masked: bool):
    """Token strip. y_top - y_bottom is the FULL allowed height; we
    draw one row of cells centered vertically in that band."""
    title = title_for(3 if not masked else 4,
                       "Token sequence" if not masked
                       else "Masked input (20%)")
    pad_x = 0.012
    cell_pad = 0.003
    inner_fs = 7.8
    # Cell width = widest token + small padding
    tokens = SEQUENCE
    max_tok = max(tokens, key=lambda t: len(t))
    cell_inner = text_width(max_tok, mono=True, fontsize=inner_fs) + 0.012
    cell_w = cell_inner + 2 * cell_pad
    n = len(tokens)
    inner_w = n * cell_w
    width = inner_w + 2 * pad_x
    fitted_box(ax, x_left, y_bottom, width, y_top - y_bottom)
    ax.text(x_left + width / 2, y_top - 0.04, title,
            fontsize=10, fontweight="bold", ha="center", va="top")
    # Center cells vertically
    cell_h = 0.14
    cell_y = y_bottom + ((y_top - y_bottom) - 0.10 - cell_h) / 2.0
    for j, tok in enumerate(tokens):
        x = x_left + pad_x + j * cell_w
        is_mask = masked and (j in MASKED_POSITIONS)
        face = ("#ffcdd2" if is_mask
                else ("#e3f2fd" if tok not in {"[CLS]", "[SEP]"}
                      else "#e0e0e0"))
        cell = mpatches.FancyBboxPatch(
            (x + cell_pad, cell_y), cell_inner, cell_h,
            boxstyle="round,pad=0.004",
            facecolor=face, edgecolor="#888", linewidth=0.6)
        ax.add_patch(cell)
        display = "[MASK]" if is_mask else tok
        ax.text(x + cell_pad + cell_inner / 2,
                cell_y + cell_h / 2, display,
                fontsize=inner_fs, family="monospace", ha="center",
                va="center",
                color=(MASK_COLOR if is_mask
                       else (HEAD_COLOR if tok not in {"[CLS]", "[SEP]"}
                             else "#444")))
    return width


def stage5_encoder(ax, x_left, y_top, y_bottom):
    title = title_for(5, "Denoising encoder")
    pad_x = 0.015
    # Width: take the widest line at its rendered font size; the loss
    # formula and "constructive training only" dominate.
    width_candidates = [
        text_width("constructive training only", mono=False, fontsize=8.0),
        text_width("Denoising encoder", mono=False, fontsize=10),
        text_width("predict masked tokens", mono=False, fontsize=8.5),
        0.150,  # rough estimate for the math-mode loss formula
    ]
    width = max(width_candidates) + 2 * pad_x
    fitted_box(ax, x_left, y_bottom, width, y_top - y_bottom,
               facecolor=ENC_FACE, edgecolor=ENC_EDGE, linewidth=1.4)
    cx = x_left + width / 2

    # ----- title -----
    y = y_top - 0.04
    ax.text(cx, y, title, fontsize=10, fontweight="bold",
            ha="center", va="top")

    # ----- encoder block -----
    y -= 0.07
    ax.text(cx, y, r"$\phi_{\rm proof}$",
            fontsize=12.5, ha="center", va="top", color="#222")
    y -= 0.07
    ax.text(cx, y, "Transformer", fontsize=8.0, ha="center", va="top",
            style="italic", color="#5d4037")
    y -= 0.04
    ax.text(cx, y, "constructive training only",
            fontsize=8.0, ha="center", va="top", style="italic",
            color="#5d4037")

    # ----- arrow to embedding -----
    y -= 0.035
    ax.annotate("", xy=(cx, y - 0.040), xytext=(cx, y - 0.005),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))

    # ----- embedding -----
    y -= 0.075
    ax.text(cx, y, r"$z \in \mathbb{R}^{128}$",
            fontsize=11.0, ha="center", va="top",
            color="#222", fontweight="bold")
    y -= 0.045
    ax.text(cx, y, "(proof embedding)",
            fontsize=7.5, ha="center", va="top", color="#666",
            style="italic")

    # ----- arrow to decoder -----
    y -= 0.035
    ax.annotate("", xy=(cx, y - 0.040), xytext=(cx, y - 0.005),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    ax.text(cx + 0.020, y - 0.022, "MLM head",
            fontsize=7.5, ha="left", va="center",
            color="#888", style="italic")

    # ----- decoder output -----
    y -= 0.075
    ax.text(cx, y, r"predict $t_i$ at masked $i$",
            fontsize=8.5, ha="center", va="top", color="#222")

    # ----- reconstruction loss -----
    y -= 0.075
    ax.text(cx, y,
            r"$\mathcal{L}_{\rm recon} = -\!\sum_{i\in M}\!\log p_\theta(t_i \mid z)$",
            fontsize=9.0, ha="center", va="top", color="#C62828")

    return width


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def _layout(ax, x_start: float, y_top: float, y_bottom: float,
            gap: float) -> float:
    """Render all five stages starting at x_start. Returns the
    right edge of stage 5 (i.e. total occupied width = right - x_start)."""
    # Stage 1
    w1 = stage1_source(ax, x_start, y_top, y_bottom)
    x = x_start + w1 + gap
    # Stage 2
    w2 = stage2_heads(ax, x, y_top, y_bottom)
    x += w2 + gap
    # Stages 3 & 4: stacked vertically, share horizontal slot
    band_top_y = y_top
    band_bot_y = y_bottom + (y_top - y_bottom) / 2 + 0.012
    w3 = stage3_sequence(ax, x, band_top_y, band_bot_y, masked=False)
    band_top_y2 = y_bottom + (y_top - y_bottom) / 2 - 0.012
    band_bot_y2 = y_bottom
    w4 = stage3_sequence(ax, x, band_top_y2, band_bot_y2, masked=True)
    x += max(w3, w4) + gap
    # Stage 5
    w5 = stage5_encoder(ax, x, y_top, y_bottom)
    return x + w5


def main() -> None:
    plt.rcParams["text.usetex"] = False
    gap = 0.008
    y_top = 0.92
    y_bottom = 0.08

    # ----- Pass 1: probe widths on a throwaway figure -----
    probe_fig, probe_ax = plt.subplots(figsize=(14.0, 3.6))
    probe_ax.set_xlim(0, 1)
    probe_ax.set_ylim(0, 1)
    probe_ax.axis("off")
    probe_right = _layout(probe_ax, 0.0, y_top, y_bottom, gap)
    plt.close(probe_fig)

    # ----- Pass 2: render centered -----
    total = probe_right
    x_offset = max(0.0, (1.0 - total) / 2.0)
    fig, ax = plt.subplots(figsize=(14.0, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    final_right = _layout(ax, x_offset, y_top, y_bottom, gap)
    if final_right > 0.99:
        print(f"warn: layout right edge = {final_right:.3f}; may clip")

    fig.tight_layout()
    out = FIG / "proof_representation.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}  total={total:.3f}  offset={x_offset:.3f}")


if __name__ == "__main__":
    main()
