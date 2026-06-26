"""Descriptive post-hoc calibration of the depth law: three implied
mixture-weight estimates per depth bucket from the head-encoder
$\\phi_{\\rm proof}$.

For each depth bucket d we compute three estimates of the mixture
weight \\lambda_d under the two-population model
  Q_d = (1 - \\lambda_d) P + \\lambda_d R,
using held-out constructive as \\lambda = 0 and depth 2 as the
empirical frontier \\lambda = 1:

  \\hat\\lambda^{AUC}_d  = (AUC_d - 0.5) / (AUC_2 - 0.5)
  \\hat\\lambda^{loss}_d = (ell_d - ell_con) / (ell_2 - ell_con)
  \\hat\\lambda^{S90}_d  = (0.90 - c_d) / (0.90 - c_2)

where c_d is the fraction of depth-d proofs inside the constructive
q=90 superlevel set.

Inputs (head-encoder, $\\phi_{\\rm proof}$):
  - depth-stratified k-NN AUC: results/data/reviewer/q2_relabeled_results.json
  - reconstruction loss by bucket: results/data/depth_analysis/reconstruction_loss_results.json
  - superlevel containment: results/data/reviewer/q2_relabeled_results.json (superlevel section)

Outputs:
  results/data/reviewer/fit_lambdas.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"


# -----------------------------------------------------------
# Constants (the post-hoc calibration anchors)
# -----------------------------------------------------------
AUC_BASELINE = 0.5            # chance
S90_REFERENCE = 0.90          # design quantile (held-out constructive inside ~ 0.8993)


# -----------------------------------------------------------
# Load AUC by depth (head encoder, k-NN k=5)
# -----------------------------------------------------------
with open(DATA / "reviewer" / "q2_relabeled_results.json") as f:
    q2 = json.load(f)

auc_by_depth_fine: dict[str, dict] = {}
for row in q2["depth_stratified"]["methods"]["kNN_encoder"]:
    auc_by_depth_fine[row["bucket"]] = {"auc": row["auc"], "n": row["n"]}

# The reconstruction-loss + containment data use a coarser bucketing
# (depth 5-6 instead of 5 and 6 separately). Aggregate AUC accordingly.
n5 = auc_by_depth_fine["depth 5"]["n"]; auc5 = auc_by_depth_fine["depth 5"]["auc"]
n6 = auc_by_depth_fine["depth 6"]["n"]; auc6 = auc_by_depth_fine["depth 6"]["auc"]
auc56 = (auc5 * n5 + auc6 * n6) / (n5 + n6)
n56 = n5 + n6

AUC = {
    "depth 2":   (auc_by_depth_fine["depth 2"]["auc"],   auc_by_depth_fine["depth 2"]["n"]),
    "depth 3":   (auc_by_depth_fine["depth 3"]["auc"],   auc_by_depth_fine["depth 3"]["n"]),
    "depth 4":   (auc_by_depth_fine["depth 4"]["auc"],   auc_by_depth_fine["depth 4"]["n"]),
    "depth 5-6": (auc56, n56),
    "depth 7-8": (auc_by_depth_fine["depth 7-8"]["auc"], auc_by_depth_fine["depth 7-8"]["n"]),
    "depth 9+":  (auc_by_depth_fine["depth 9+"]["auc"],  auc_by_depth_fine["depth 9+"]["n"]),
}


# -----------------------------------------------------------
# Reconstruction loss by depth (head encoder)
# -----------------------------------------------------------
with open(DATA / "depth_analysis" / "reconstruction_loss_results.json") as f:
    rl = json.load(f)
loss_bk = rl["buckets"]
ELL_CON = loss_bk["constructive (test)"]["mean_loss"]   # held-out constructive
LOSS = {b: (loss_bk[b]["mean_loss"], loss_bk[b]["n"])
        for b in ["depth 2", "depth 3", "depth 4",
                  "depth 5-6", "depth 7-8", "depth 9+"]}


# -----------------------------------------------------------
# Superlevel containment (q=90), inside fractions
# -----------------------------------------------------------
C = {}
for row in q2["superlevel"]["containment_rows"]:
    if row["threshold_quantile"] != 90:
        continue
    C[row["depth_bucket"]] = {
        "c_inside": row["classical_fraction_inside"],
        "n": row["n_classical"],
        "constr_inside_baseline": row["constructive_fraction_inside"],
    }
constr_inside_empirical = next(iter(C.values()))["constr_inside_baseline"]


# -----------------------------------------------------------
# Compute lambdas
# -----------------------------------------------------------
AUC_d2 = AUC["depth 2"][0]
ELL_d2 = LOSS["depth 2"][0]
C_d2   = C["depth 2"]["c_inside"]

auc_denom  = AUC_d2 - AUC_BASELINE
loss_denom = ELL_d2 - ELL_CON
s90_denom  = S90_REFERENCE - C_d2

print(f"Anchors:")
print(f"  AUC: baseline=0.5, frontier (d=2) = {AUC_d2:.4f}, denom = {auc_denom:.4f}")
print(f"  Loss: baseline (constructive test) = {ELL_CON:.4f} nats, frontier = {ELL_d2:.4f}, denom = {loss_denom:.4f}")
print(f"  S90: reference = 0.90 (empirical {constr_inside_empirical:.4f}), frontier c_2 = {C_d2:.4f}, denom = {s90_denom:.4f}")
print()

rows = []
for b in ["depth 2", "depth 3", "depth 4", "depth 5-6", "depth 7-8", "depth 9+"]:
    auc, n_auc = AUC[b]
    ell, n_ell = LOSS[b]
    c   = C[b]["c_inside"]
    n_c = C[b]["n"]
    lam_auc  = (auc - AUC_BASELINE) / auc_denom
    lam_loss = (ell - ELL_CON) / loss_denom
    lam_s90  = (S90_REFERENCE - c) / s90_denom
    rows.append({
        "bucket": b,
        "n_auc": n_auc, "auc": auc, "lambda_auc": lam_auc,
        "n_loss": n_ell, "loss_nats": ell, "lambda_loss": lam_loss,
        "n_containment": n_c, "c_inside": c, "lambda_s90": lam_s90,
    })

print(f"{'bucket':<10s} | {'AUC':>6s} {'λ_AUC':>7s} | "
      f"{'loss':>6s} {'λ_loss':>7s} | {'c_in':>6s} {'λ_S90':>7s}")
print("-" * 70)
for r in rows:
    print(f"{r['bucket']:<10s} | {r['auc']:>6.3f} {r['lambda_auc']:>7.3f} | "
          f"{r['loss_nats']:>6.3f} {r['lambda_loss']:>7.3f} | "
          f"{r['c_inside']:>6.3f} {r['lambda_s90']:>7.3f}")

# Sanity: depth-2 lambdas must all equal 1.
assert abs(rows[0]["lambda_auc"]  - 1.0) < 1e-9
assert abs(rows[0]["lambda_loss"] - 1.0) < 1e-9
assert abs(rows[0]["lambda_s90"]  - 1.0) < 5e-3  # small drift from using 0.90 vs empirical

out = {
    "anchors": {
        "auc_baseline": AUC_BASELINE,
        "auc_frontier_depth2": AUC_d2,
        "loss_baseline_construct_test": ELL_CON,
        "loss_frontier_depth2": ELL_d2,
        "s90_reference": S90_REFERENCE,
        "s90_constr_inside_empirical": constr_inside_empirical,
        "s90_frontier_depth2": C_d2,
    },
    "rows": rows,
}
out_path = DATA / "reviewer" / "fit_lambdas.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {out_path}")
