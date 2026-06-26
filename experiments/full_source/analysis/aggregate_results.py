"""Step 6: aggregate per-seed analyses into the headline summary.

Reads:
  sanity_results.json
  depth_stratified_auc.json
  reconstruction_loss.json
  length_residualized_auc.json
  superlevel_containment.json
  leakage_mi.json

Writes:
  results.json   -- structured per-bucket numbers with median + range
  RESULTS.md     -- one-page human-readable summary
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

BUCKETS = ["depth_2", "depth_3", "depth_4_6", "depth_7_8", "depth_9_plus"]


def median_range(xs: list[float]) -> dict:
    if not xs:
        return {"median": None, "range": [None, None], "per_seed": []}
    return {
        "median": float(statistics.median(xs)),
        "range": [float(min(xs)), float(max(xs))],
        "per_seed": [float(x) for x in xs],
    }


def collect_auc_across_seeds(path: Path, detector: str
                              ) -> dict[str, dict[str, dict]]:
    """{variant: {bucket: {median, range, per_seed}}}."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    by_variant: dict[str, dict[str, list[float]]] = {}
    for run in data["runs"]:
        v = run["variant"]
        by_variant.setdefault(v, {b: [] for b in BUCKETS})
        for b in BUCKETS:
            r = run["results"][detector][b]
            if r["auc"] is not None and not _is_nan(r["auc"]):
                by_variant[v][b].append(r["auc"])
    return {v: {b: median_range(by_variant[v][b]) for b in BUCKETS}
            for v in by_variant}


def collect_length_resid(path: Path) -> dict[str, dict[str, dict]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    by_variant: dict[str, dict[str, list[float]]] = {}
    for run in data["runs"]:
        v = run["variant"]
        by_variant.setdefault(v, {b: [] for b in BUCKETS})
        for b in BUCKETS:
            r = run["results"][b]
            if r["auc"] is not None and not _is_nan(r["auc"]):
                by_variant[v][b].append(r["auc"])
    return {v: {b: median_range(by_variant[v][b]) for b in BUCKETS}
            for v in by_variant}


def collect_recon_from_files(seed_dirs: list[Path]) -> dict[str, dict]:
    """Aggregate per-seed recon_loss.json files (written to checkpoints/)."""
    by_bucket: dict[str, list[float]] = {b: [] for b in BUCKETS}
    constructive_per_seed = []
    for sd in seed_dirs:
        f = sd / "recon_loss.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for b in BUCKETS:
            bd = data.get("buckets", {}).get(b, {})
            v = bd.get("median_ce")
            if v is not None and not _is_nan(v):
                by_bucket[b].append(v)
        c = data.get("buckets", {}).get("constructive", {})
        v = c.get("median_ce")
        if v is not None and not _is_nan(v):
            constructive_per_seed.append(v)
    return {
        "constructive_baseline": median_range(constructive_per_seed),
        "per_bucket": {b: median_range(by_bucket[b]) for b in BUCKETS},
    }


def collect_containment(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict = {}
    for run in data["runs"]:
        v = run["variant"]
        out.setdefault(v, {})
        for q_key, q_data in run["buckets"].items():
            out[v].setdefault(q_key, {b: [] for b in BUCKETS})
            for b in BUCKETS:
                pb = q_data["per_bucket"].get(b, {})
                if pb.get("fraction_outside") is not None and \
                        not _is_nan(pb["fraction_outside"]):
                    out[v][q_key][b].append(pb["fraction_outside"])
    return {v: {q: {b: median_range(out[v][q][b]) for b in BUCKETS}
                for q in out[v]} for v in out}


def collect_sanity(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data


def _is_nan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False


def head_level_baseline() -> dict:
    """Load head-level k-NN AUC for direct comparison. The on-disk
    schema is a list of {bucket, depth_range, n_theorems, auc}; we
    map it onto our bucket names."""
    p = (HERE.parent.parent.parent / "results/data/depth_analysis/"
         "depth_stratified_auc.json")
    if not p.exists():
        return {}
    rows = json.loads(p.read_text())
    by_bucket: dict[str, float] = {}
    # Head-level uses depth 2 / 3 / 4 / 5 / 6 / 7-8 / 9+; we group
    # 4,5,6 → depth_4_6 by averaging weighted by n.
    grouped_4_6 = []
    for r in rows:
        d_range = r["depth_range"]
        n = r["n_theorems"]
        auc = r["auc"]
        if d_range == [2, 2]:
            by_bucket["depth_2"] = auc
        elif d_range == [3, 3]:
            by_bucket["depth_3"] = auc
        elif d_range[0] in (4, 5, 6) and d_range[1] in (4, 5, 6):
            grouped_4_6.append((n, auc))
        elif d_range == [7, 8]:
            by_bucket["depth_7_8"] = auc
        elif d_range[0] == 9:
            by_bucket["depth_9_plus"] = auc
    if grouped_4_6:
        total_n = sum(n for n, _ in grouped_4_6)
        by_bucket["depth_4_6"] = sum(n * a for n, a in grouped_4_6) / total_n
    return by_bucket


def main() -> None:
    auc_knn = collect_auc_across_seeds(HERE / "depth_stratified_auc.json",
                                        "knn")
    auc_iso = collect_auc_across_seeds(HERE / "depth_stratified_auc.json",
                                        "iso")
    auc_kde = collect_auc_across_seeds(HERE / "depth_stratified_auc.json",
                                        "kde")
    length_resid = collect_length_resid(
        HERE / "length_residualized_auc.json")
    seed_dirs = [HERE.parent / "encoder/checkpoints" / f"seed_{s}"
                 for s in (0, 1, 2)]
    recon = collect_recon_from_files(seed_dirs)
    containment = collect_containment(HERE / "superlevel_containment.json")
    sanity = collect_sanity(HERE / "sanity_results.json")
    head_base = head_level_baseline()

    results = {
        "head_level_reference": head_base,
        "sanity": sanity,
        "depth_stratified_knn_auc": auc_knn,
        "depth_stratified_iso_auc": auc_iso,
        "depth_stratified_kde_auc": auc_kde,
        "length_residualized_knn_auc": length_resid,
        "reconstruction_loss": recon,
        "superlevel_containment": containment,
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")

    # Human-readable summary.
    md = []
    md.append("# Full-Source Encoder: Results Summary\n")
    md.append("This document is auto-generated by aggregate_results.py.\n")

    if sanity:
        md.append("\n## Sanity check (Step 3.4)\n")
        for r in sanity.get("per_seed", []):
            md.append(f"- seed {r['seed']}: all_pass = "
                      f"**{r.get('all_pass', False)}**")
            rc = r.get("reconstruction", {})
            dp = r.get("domain_probe", {})
            lp = r.get("length_probe", {})
            md.append(f"  - recon_loss = {rc.get('mean_recon_loss_nats', '?'):.3f} nats "
                      f"(threshold < {sanity['thresholds']['recon_loss_nats']:.2f}, "
                      f"pass={rc.get('passes', False)})")
            md.append(f"  - domain probe acc = {dp.get('mean_cv_accuracy', float('nan')):.3f} "
                      f"vs shuffled {dp.get('shuffled_baseline_acc', float('nan')):.3f} "
                      f"(threshold > {sanity['thresholds']['domain_accuracy']:.2f}, "
                      f"pass={dp.get('passes', False)})")
            md.append(f"  - length probe R^2 = {lp.get('mean_cv_r2', float('nan')):.3f} "
                      f"(threshold >= {sanity['thresholds']['length_r2']:.2f}, "
                      f"pass={lp.get('passes', False)})")
        md.append("")

    md.append("\n## Depth-stratified k-NN AUC (median over 3 seeds)\n")
    md.append("| Bucket | head-only | full raw | full stripped | full combined |")
    md.append("|---|---|---|---|---|")
    for b in BUCKETS:
        row = [b]
        h = f"{head_base[b]:.3f}" if b in head_base else "—"
        row.append(h)
        for variant in ("raw", "stripped", "combined"):
            v = auc_knn.get(variant, {}).get(b, {})
            if v.get("median") is not None:
                lo, hi = v["range"]
                row.append(f"{v['median']:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("\n## Length-residualized k-NN AUC\n")
    md.append("| Bucket | raw | stripped | combined |")
    md.append("|---|---|---|---|")
    for b in BUCKETS:
        row = [b]
        for variant in ("raw", "stripped", "combined"):
            v = length_resid.get(variant, {}).get(b, {})
            if v.get("median") is not None:
                lo, hi = v["range"]
                row.append(f"{v['median']:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("\n## Reconstruction loss (whole-word, median nats over 3 seeds)\n")
    if recon:
        c = recon.get("constructive_baseline", {})
        md.append(f"Constructive baseline: {c.get('median', '?'):.3f}")
        md.append("")
        md.append("| Bucket | median CE |")
        md.append("|---|---|")
        for b in BUCKETS:
            v = recon["per_bucket"].get(b, {})
            if v.get("median") is not None:
                lo, hi = v["range"]
                md.append(f"| {b} | {v['median']:.3f} [{lo:.3f}, {hi:.3f}] |")
            else:
                md.append(f"| {b} | — |")
        md.append("")

    md.append("\n## Superlevel containment q=90 (fraction outside)\n")
    md.append("| Bucket | raw | stripped | combined |")
    md.append("|---|---|---|---|")
    for b in BUCKETS:
        row = [b]
        for variant in ("raw", "stripped", "combined"):
            v = containment.get(variant, {}).get("q90", {}).get(b, {})
            if v.get("median") is not None:
                row.append(f"{v['median']:.3f}")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md_path = HERE / "RESULTS.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
