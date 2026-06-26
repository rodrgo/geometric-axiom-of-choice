"""QC Check 3: bootstrap CIs + within-group analysis for anomaly-score
prediction of aesop failure.

Fits logistic regression predicting failure (= 1 - success) from either
log_proof_length alone or log_proof_length + anomaly_score. Reports
median 5-fold-CV AUC with 95% percentile intervals from 1000 bootstrap
resamples, overall and within each group (classical, constructive).

Output: results/data/reviewer/anomaly_auc_bootstrap.json.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 1000
SEED = 0


def cv_auc(X, y, seed=0):
    """5-fold stratified CV AUC using logistic regression."""
    if len(np.unique(y)) < 2:
        return np.nan
    cls_counts = np.bincount(y.astype(int))
    n_splits = min(5, int(cls_counts.min()))
    if n_splits < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        scores = cross_val_score(
            LogisticRegression(max_iter=2000, solver="lbfgs"),
            X, y, scoring="roc_auc", cv=cv,
        )
        return float(np.mean(scores))
    except Exception:
        return np.nan


def boot_auc(df, feats, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sub = df.iloc[idx]
        y = sub["fail"].values.astype(int)
        X = sub[feats].values
        auc = cv_auc(X, y, seed=0)
        if not np.isnan(auc):
            aucs.append(auc)
    return np.array(aucs)


def summarize(samples, label=""):
    if len(samples) == 0:
        return {"n_valid_boot": 0, "label": label,
                "median": None, "ci95": [None, None]}
    return {
        "n_valid_boot": int(len(samples)),
        "label": label,
        "median": float(np.median(samples)),
        "mean": float(np.mean(samples)),
        "ci95": [float(np.percentile(samples, 2.5)),
                 float(np.percentile(samples, 97.5))],
    }


def report_group(df, tag):
    print(f"\n=== {tag} ===")
    print(f"  N = {len(df)}  failures = {int(df['fail'].sum())}  "
          f"successes = {int(len(df) - df['fail'].sum())}")
    if len(df) < 20 or df["fail"].nunique() != 2:
        print("  insufficient data")
        return None
    aucs_len = boot_auc(df, ["log_proof_length"])
    aucs_full = boot_auc(df, ["log_proof_length", "anomaly_score"])
    # match bootstrap length (same seed -> paired if same draws; we just
    # align by position, so take min length for the diff)
    n = min(len(aucs_len), len(aucs_full))
    diffs = aucs_full[:n] - aucs_len[:n]
    ci = lambda a: f"[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]"
    print(f"  length only:          median {np.median(aucs_len):.3f}  95% CI {ci(aucs_len)}")
    print(f"  length + anomaly:     median {np.median(aucs_full):.3f}  95% CI {ci(aucs_full)}")
    print(f"  improvement:          median {np.median(diffs):+.3f}  95% CI {ci(diffs)}")
    print(f"  P(improvement > 0):   {np.mean(diffs > 0):.3f}")
    return {
        "n": len(df),
        "n_fail": int(df["fail"].sum()),
        "n_success": int(len(df) - df["fail"].sum()),
        "length_only": summarize(aucs_len, "length"),
        "length_plus_anomaly": summarize(aucs_full, "length+anomaly"),
        "improvement": summarize(diffs, "full-length"),
        "p_improvement_positive": float(np.mean(diffs > 0)),
    }


def main():
    rows = []
    with open(ROOT / "results/data/reviewer/prover_results.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("anomaly_score") is None or r.get("proof_length") is None:
                continue
            rows.append({
                "name": r["name"],
                "bucket": r["bucket"],
                "is_classical": r["bucket"] != "constructive",
                "fail": 0 if r.get("success") else 1,
                "anomaly_score": float(r["anomaly_score"]),
                "proof_length": float(r["proof_length"]),
                "log_proof_length": float(np.log1p(r["proof_length"])),
            })
    df = pd.DataFrame(rows)

    print(f"Total N = {len(df)}  fail = {int(df['fail'].sum())}  "
          f"succ = {int(len(df) - df['fail'].sum())}")

    overall = report_group(df, "OVERALL")
    cls = report_group(df[df["is_classical"]].reset_index(drop=True),
                        "WITHIN CLASSICAL")
    con = report_group(df[~df["is_classical"]].reset_index(drop=True),
                        "WITHIN CONSTRUCTIVE")

    out = {
        "n_boot": N_BOOT,
        "seed": SEED,
        "overall": overall,
        "within_classical": cls,
        "within_constructive": con,
    }
    with open(ROOT / "results/data/reviewer/anomaly_auc_bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/data/reviewer/anomaly_auc_bootstrap.json")


if __name__ == "__main__":
    main()
