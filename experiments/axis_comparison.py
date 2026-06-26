"""Conditional axis comparison (paper Sec. 4 / Appendix, Table tab:axis_comparison).

The depth law is steepest for Classical.choice. To show it is not a generic
axiom-dependence effect, we condition on the Classical.choice-unreached class
and ask whether depth from the other two kernel-tracked axioms (propext,
Quot.sound) still predicts anomaly within that constructive class.

For each conditional axis, k-NN is trained on doubly-unreached proofs (no
dependence on Classical.choice or the axis) and scores depth-stratified
buckets of Classical.choice-unreached proofs by their depth from the axis.
The Classical.choice column of the table is the head-level k-NN AUC from
`depth_knn_auc.py` (Table tab:depth_strat).

Requires the kernel dependency graph (results/data/stage4v3/decl_graph_raw.jsonl);
see README "Heavy external dependencies".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import STAGE4V3P_DIR, DEPTH_ANALYSIS_DIR  # noqa: E402
from lean.kernel_graph import load_graph, bfs_classical_depths  # noqa: E402

PROOFS = STAGE4V3P_DIR / "proofs.json"
EMB = STAGE4V3P_DIR / "embeddings.npz"
OUT = DEPTH_ANALYSIS_DIR / "axis_comparison.json"

BUCKETS = [
    ("depth_2", lambda d: d == 2),
    ("depth_3", lambda d: d == 3),
    ("depth_4_6", lambda d: 4 <= d <= 6),
    ("depth_7_8", lambda d: 7 <= d <= 8),
    ("depth_9_plus", lambda d: d >= 9),
]
K = 5
MIN_BUCKET = 30


def main() -> None:
    fwd, rev, _ = load_graph()
    cc = bfs_classical_depths(rev, seeds=("Classical.choice",))
    pe = bfs_classical_depths(rev, seeds=("propext",))
    qs = bfs_classical_depths(rev, seeds=("Quot.sound",))
    proofs = json.loads(PROOFS.read_text())
    emb = np.load(EMB, allow_pickle=True)
    X = emb["embeddings"].astype(np.float64)
    train_idx = set(emb["train_idx"].tolist())
    test_idx = set(emb["test_idx"].tolist())
    val_idx = set(emb["val_idx"].tolist())
    names = [p["name"] for p in proofs]
    n = len(names)

    cc_dep = np.array([cc.get(nm) is not None for nm in names])
    pe_d = np.array([pe.get(nm, -1) for nm in names], dtype=np.int32)
    qs_d = np.array([qs.get(nm, -1) for nm in names], dtype=np.int32)

    out: dict = {"conditioning": "Classical.choice-unreached", "axes": {}}

    for axis_name, dep_arr in [("propext", pe_d), ("Quot.sound", qs_d)]:
        cc_unreached = ~cc_dep
        # Doubly-unreached: the k-NN training subset for this conditional axis.
        train_m = np.array([
            i in train_idx and cc_unreached[i] and dep_arr[i] == -1
            for i in range(n)])
        ctrl_m = np.array([
            (i in test_idx or i in val_idx)
            and cc_unreached[i] and dep_arr[i] == -1
            for i in range(n)])

        scaler = StandardScaler().fit(X[train_m])
        Xz = scaler.transform(X)
        nn = NearestNeighbors(n_neighbors=K).fit(Xz[train_m])
        s_ctrl = nn.kneighbors(Xz[ctrl_m])[0].mean(axis=1)

        per_bucket: dict[str, dict] = {}
        for bname, pred in BUCKETS:
            bm = np.array([
                cc_unreached[i] and pred(int(dep_arr[i]))
                for i in range(n)])
            nb = int(bm.sum())
            if nb < MIN_BUCKET:
                per_bucket[bname] = {"auc": float("nan"), "n_bucket": nb}
                continue
            s_b = nn.kneighbors(Xz[bm])[0].mean(axis=1)
            y = np.concatenate([np.zeros(len(s_ctrl)), np.ones(len(s_b))])
            s = np.concatenate([s_ctrl, s_b])
            per_bucket[bname] = {
                "auc": float(roc_auc_score(y, s)),
                "n_bucket": nb,
            }

        out["axes"][axis_name] = {
            "n_train_doubly_unreached": int(train_m.sum()),
            "n_ctrl_doubly_unreached": int(ctrl_m.sum()),
            "per_bucket": per_bucket,
        }
        print(f"\n{axis_name} (conditional on Classical.choice-unreached):")
        print(f"  train={out['axes'][axis_name]['n_train_doubly_unreached']}  "
              f"ctrl={out['axes'][axis_name]['n_ctrl_doubly_unreached']}")
        for bname, _ in BUCKETS:
            b = per_bucket[bname]
            if np.isnan(b["auc"]):
                print(f"  {bname:14s}  n={b['n_bucket']:>5,}  (too few)")
            else:
                print(f"  {bname:14s}  n={b['n_bucket']:>5,}  AUC={b['auc']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
