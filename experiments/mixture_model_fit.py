"""Fit the mixture-law lambda_d from AUC gradient, predict reconstruction loss
and superlevel-set containment, and compare with observations.

The model (Proposition, paper):
    Q_d = (1 - lambda_d) P + lambda_d R
where P = constructive distribution, R = directly-classical (~depth 2).

Identities:
    AUC(s; P, Q_d)     = 0.5 (1 - lambda_d) + lambda_d * AUC(s; P, R)
    E[loss | Q_d]      = (1 - lambda_d) E[loss | P] + lambda_d E[loss | R]
    Pr[Q_d in S_q]     = (1 - lambda_d) q + lambda_d Pr[R in S_q]  (for S_q set from P)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
OUT = DATA / "mixture_model"
OUT.mkdir(exist_ok=True)


# Both AUC and superlevel containment come from the label-free detection
# output (label_free_detection.py), so the mixture predictions match the
# paper's tab:depth_strat / tab:superlevel_containment.
_Q2 = None


def _q2():
    global _Q2
    if _Q2 is None:
        _Q2 = json.loads((DATA / "reviewer" / "q2_relabeled_results.json").read_text())
    return _Q2


def load_auc():
    # head-encoder k-NN AUC, per fine bucket (incl. depth 5 and 6 separately)
    return {r["bucket"]: r["auc"]
            for r in _q2()["depth_stratified"]["methods"]["kNN_encoder"]}


def load_recon():
    payload = json.loads(
        (DATA / "depth_analysis" / "reconstruction_loss_results.json").read_text()
    )
    return payload["buckets"]


def load_containment():
    # (quantile, bucket) -> (cons_inside, cls_inside), label-free superlevel
    return {
        (r["threshold_quantile"], r["depth_bucket"]): (
            r["constructive_fraction_inside"],
            r["classical_fraction_inside"],
        )
        for r in _q2()["superlevel"]["containment_rows"]
    }


# Bucketing: AUC and recon use slightly different buckets. For comparison we
# align them. AUC: depth 2, 3, 4, 5, 6, 7-8, 9+. Recon: depth 2, 3, 4, 5-6, 7-8, 9+.
# Containment: depth 2, 3, 4, 5-6, 7-8, 9+.
# We use the coarser common set: depth 2, 3, 4, 5-6, 7-8, 9+
# For AUC, merge depth 5 and depth 6 by sample-size weighting.

AUC_BUCKETS_N = {
    "depth 2": 3680,
    "depth 3": 8686,
    "depth 4": 8299,
    "depth 5": 4393,
    "depth 6": 2580,
    "depth 7-8": 2604,
    "depth 9+": 872,
}


def merge_auc_5_6(aucs):
    a5, a6 = aucs["depth 5"], aucs["depth 6"]
    n5, n6 = AUC_BUCKETS_N["depth 5"], AUC_BUCKETS_N["depth 6"]
    merged = (n5 * a5 + n6 * a6) / (n5 + n6)
    return {
        "depth 2": aucs["depth 2"],
        "depth 3": aucs["depth 3"],
        "depth 4": aucs["depth 4"],
        "depth 5-6": merged,
        "depth 7-8": aucs["depth 7-8"],
        "depth 9+": aucs["depth 9+"],
    }


def main():
    auc_raw = load_auc()
    recon = load_recon()
    cont = load_containment()

    aucs = merge_auc_5_6(auc_raw)

    # Treat depth 2 bucket as our estimate of AUC(s; P, R), E[loss | R],
    # and Pr[R in S_q]. lambda_2 := 1 by construction.
    auc_pr = aucs["depth 2"]
    loss_p = recon["constructive (test)"]["mean_loss"]
    loss_r = recon["depth 2"]["mean_loss"]

    buckets = ["depth 2", "depth 3", "depth 4", "depth 5-6", "depth 7-8", "depth 9+"]

    results = {"buckets": {}, "setup": {
        "auc_P_R_estimate": auc_pr,
        "loss_P_estimate": loss_p,
        "loss_R_estimate": loss_r,
    }}

    for q in [80, 90, 95]:
        cons_q, _ = cont[(q, "depth 2")]
        _, r_q = cont[(q, "depth 2")]
        results["setup"][f"pr_P_in_S{q}"] = cons_q
        results["setup"][f"pr_R_in_S{q}_estimate"] = r_q

    print(f"AUC(P,R) estimate (depth 2): {auc_pr:.3f}")
    print(f"E[loss|P] (cons test): {loss_p:.3f}")
    print(f"E[loss|R] (depth 2):   {loss_r:.3f}")
    print()

    header = (
        f"{'bucket':<10} {'lam_d':>6} "
        f"{'loss_obs':>9} {'loss_pred':>9} {'loss_err':>9} "
        f"{'S80_obs':>8} {'S80_pred':>9} "
        f"{'S90_obs':>8} {'S90_pred':>9} "
        f"{'S95_obs':>8} {'S95_pred':>9}"
    )
    print(header)
    print("-" * len(header))

    for b in buckets:
        auc = aucs[b]
        # Solve for lambda from AUC
        lam = (auc - 0.5) / (auc_pr - 0.5)
        lam = float(np.clip(lam, 0.0, 1.0))

        # Predict reconstruction loss
        loss_obs = recon[b]["mean_loss"]
        loss_pred = (1 - lam) * loss_p + lam * loss_r
        loss_err = loss_pred - loss_obs

        row = {"auc": auc, "lambda": lam,
               "loss_obs": loss_obs, "loss_pred": loss_pred,
               "loss_err": loss_err}

        cont_line = ""
        for q in [80, 90, 95]:
            _, cls_inside = cont[(q, b)]
            pr_p = cont[(q, "depth 2")][0]   # set from P: q/100
            pr_r = cont[(q, "depth 2")][1]
            pred = (1 - lam) * pr_p + lam * pr_r
            row[f"S{q}_obs"] = cls_inside
            row[f"S{q}_pred"] = pred
            row[f"S{q}_err"] = pred - cls_inside
            cont_line += f" {cls_inside:8.3f} {pred:9.3f}"

        print(
            f"{b:<10} {lam:6.3f} "
            f"{loss_obs:9.3f} {loss_pred:9.3f} {loss_err:+9.3f}"
            + cont_line
        )

        results["buckets"][b] = row

    out_path = OUT / "lambda_fit.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
