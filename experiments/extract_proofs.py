"""
Stage 4v3p-a: Extract proofs aligned with v3 partition, compute baselines.

Approaches:
  2. Bag-of-words over tactic names (no neural network).
  3. Hand-crafted proof statistics.
  + Logistic regression calibration on (BoW + stats).

DECISION POINT: if logistic regression AUC < 0.55, the proof signal does
not exist at the tactic level — stop and report. If ≥ 0.60, proceed to
encoder training.
"""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Tactic-head extraction (reuse heuristic from stage4_extract.py)
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*'?")
_CHUNK_SEP_RE = re.compile(
    r"<;>|;|\||·|\bby\b|\btry\b|\bfirst\b|\ball_goals\b|"
    r"\bany_goals\b|\bfocus\b|\brepeat\b|\biterate\b"
)


def extract_heads(tactic_str: str) -> list[str]:
    if not tactic_str:
        return []
    s = tactic_str.replace('\n', ' ').strip()
    chunks = _CHUNK_SEP_RE.split(s)
    heads = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _IDENT_RE.match(chunk)
        if m:
            head = m.group(0)
            end = m.end()
            if end < len(chunk) and chunk[end] == '!':
                head = head + '!'
            heads.append(head)
    return heads


# Classical-marker tactics for ablation reference
CLASSICAL_MARKERS = {
    'by_contra', 'by_contra!', 'by_contra\'',
    'by_cases', 'by_cases!',
    'classical', 'choose', 'choose!',
    'exfalso', 'contrapose', 'contrapose!',
    'push_neg', 'tauto', 'itauto',
}


# ---------------------------------------------------------------------------
# Step 1: Load and align
# ---------------------------------------------------------------------------

def main():
    print("Loading v3 partition...")
    with open('results/data/stage4v3/theorems_partition.json') as f:
        partition = json.load(f)
    label_map = {r['full_name']: r for r in partition if r.get('matched')}
    print(f"  matched theorems: {len(label_map):,}")

    print("Loading LeanDojo tactic data...")
    proofs = []
    n_loaded = 0
    for split in ['train', 'val', 'test']:
        with open(f'results/data/leandojo/{split}.json') as f:
            data = json.load(f)
        for thm in data:
            n_loaded += 1
            name = thm.get('full_name', '')
            p = label_map.get(name)
            if not p:
                continue
            tactic_strings = [t.get('tactic', '') for t in thm.get('traced_tactics', [])]
            if not tactic_strings:
                continue
            # Coarse tokens: head per tactic invocation. We pick the FIRST head
            # (the outermost tactic) for each invocation. A multi-head invocation
            # like "rw [...]; ring" gets two heads — we keep them all so the
            # bag-of-words sees both.
            all_heads = []
            invocation_heads = []  # one representative head per invocation
            for ts in tactic_strings:
                hs = extract_heads(ts)
                if hs:
                    invocation_heads.append(hs[0])
                    all_heads.extend(hs)
            if not invocation_heads:
                continue
            proofs.append({
                'name': name,
                'is_classical': p['is_classical'],
                'split': p['split'],
                'file_path': p['file_path'],
                'n_invocations': len(tactic_strings),
                'invocation_heads': invocation_heads,  # ordered, one per step
                'all_heads': all_heads,  # ordered, possibly multiple per step
            })
    print(f"  Read {n_loaded:,} theorems, kept {len(proofs):,} with non-empty tactic seqs")

    # Filter degenerate proofs
    n_before = len(proofs)
    proofs = [p for p in proofs
              if 2 <= len(p['invocation_heads']) <= 200]
    print(f"  After length filter (2–200 invocations): {len(proofs):,}")

    n_classical = sum(1 for p in proofs if p['is_classical'])
    n_constructive = len(proofs) - n_classical
    print(f"\nPartition (filtered):")
    print(f"  classical:    {n_classical:>7,} "
          f"({100*n_classical/len(proofs):.1f}%)")
    print(f"  constructive: {n_constructive:>7,} "
          f"({100*n_constructive/len(proofs):.1f}%)")

    if n_constructive < 5000 or n_classical < 1000:
        print("WARNING: small partition after filtering")

    # ------------------------------------------------------------------
    # Build vocabulary from constructive only
    # ------------------------------------------------------------------
    print("\nBuilding tactic vocabulary from constructive proofs only...")
    constructive_tactic_counts = Counter()
    for p in proofs:
        if not p['is_classical']:
            constructive_tactic_counts.update(p['all_heads'])
    # Keep tactics with count >= 5 in constructive set
    vocab_list = [t for t, c in constructive_tactic_counts.most_common()
                  if c >= 5]
    vocab = {t: i for i, t in enumerate(vocab_list)}
    UNK = len(vocab)
    print(f"  Vocab size (constructive ≥5 occurrences): {len(vocab):,}")
    print(f"  Top 20 constructive tactics: "
          f"{[(t, constructive_tactic_counts[t]) for t in vocab_list[:20]]}")

    # Sanity: which classical-marker tactics are in the constructive vocab?
    in_vocab_markers = [t for t in CLASSICAL_MARKERS if t in vocab]
    print(f"  Classical-marker tactics seen in constructive proofs: "
          f"{in_vocab_markers}")

    # ------------------------------------------------------------------
    # Approach 2: Bag-of-words feature vectors
    # ------------------------------------------------------------------
    print("\n[Approach 2] Bag-of-words over constructive vocabulary")
    V = len(vocab)
    X_bow = np.zeros((len(proofs), V), dtype=np.float32)
    for i, p in enumerate(proofs):
        c = Counter(p['all_heads'])
        total = sum(c.values())
        if total == 0:
            continue
        for tac, cnt in c.items():
            j = vocab.get(tac)
            if j is not None:
                X_bow[i, j] = cnt / total

    labels = np.array([int(p['is_classical']) for p in proofs])

    # Hull AUC on raw bag-of-words feature vectors (no encoder, just features)
    auc_bow_hull, _ = compute_hull_auc(X_bow, labels)
    print(f"  Hull AUC (BoW vectors): {auc_bow_hull:.4f}")

    # Logistic regression on BoW
    auc_bow_lr = lr_auc(X_bow, labels)
    print(f"  Logistic regression AUC (BoW): {auc_bow_lr:.4f}")

    # ------------------------------------------------------------------
    # Approach 3: Hand-crafted proof statistics
    # ------------------------------------------------------------------
    print("\n[Approach 3] Hand-crafted proof statistics")
    STRUCTURAL = {'intro', 'apply', 'exact', 'refine', 'have', 'show',
                  'suffices', 'use', 'constructor', 'cases', 'rcases',
                  'obtain', 'rintro'}
    AUTOMATION = {'simp', 'simpa', 'simp_all', 'simp_rw', 'ring', 'ring_nf',
                  'omega', 'norm_num', 'linarith', 'nlinarith', 'positivity',
                  'aesop', 'tauto', 'decide', 'gcongr', 'fun_prop',
                  'field_simp', 'polyrith', 'grind', 'lia'}
    REWRITE = {'rw', 'rwa', 'simp_rw', 'rewrite', 'nth_rw', 'erw'}
    LOGICAL = {'apply', 'exact', 'refine', 'intro', 'intros', 'rintro',
               'have', 'suffices', 'cases', 'induction', 'left', 'right',
               'constructor', 'use'}

    feats_list = []
    feature_names = [
        'n_invocations', 'log_n_invocations',
        'n_distinct_tactics', 'n_distinct_per_invocation',
        'frac_structural', 'frac_automation', 'frac_rewrite', 'frac_logical',
        'mean_arg_count', 'max_arg_count',
        'has_by_contra', 'has_by_cases', 'has_choose', 'has_exfalso',
        'has_classical', 'has_contrapose', 'has_push_neg',
        'frac_classical_markers', 'log_total_chars',
    ]
    for p in proofs:
        heads = p['invocation_heads']
        all_heads = p['all_heads']
        ni = len(heads)
        c = Counter(heads)
        n_distinct = len(c)
        # Argument count proxy: number of identifiers in each tactic - 1.
        # We don't have the raw strings here per tactic — approximate using
        # all_heads / invocation_heads ratio (chunks per invocation).
        arg_proxy = len(all_heads) / max(ni, 1) - 1  # extra heads per inv

        s = {
            'n_invocations': ni,
            'log_n_invocations': np.log1p(ni),
            'n_distinct_tactics': n_distinct,
            'n_distinct_per_invocation': n_distinct / max(ni, 1),
            'frac_structural':  sum(1 for h in heads if h in STRUCTURAL) / ni,
            'frac_automation':  sum(1 for h in heads if h in AUTOMATION) / ni,
            'frac_rewrite':     sum(1 for h in heads if h in REWRITE) / ni,
            'frac_logical':     sum(1 for h in heads if h in LOGICAL) / ni,
            'mean_arg_count':   arg_proxy,
            'max_arg_count':    arg_proxy,  # we don't track per-step here
            'has_by_contra':    int(any(h.startswith('by_contra') for h in heads)),
            'has_by_cases':     int(any(h.startswith('by_cases') for h in heads)),
            'has_choose':       int(any(h.startswith('choose') for h in heads)),
            'has_exfalso':      int('exfalso' in c),
            'has_classical':    int('classical' in c),
            'has_contrapose':   int(any(h.startswith('contrapose') for h in heads)),
            'has_push_neg':     int('push_neg' in c),
            'frac_classical_markers':
                sum(1 for h in heads if h in CLASSICAL_MARKERS) / ni,
            'log_total_chars':  np.log1p(sum(len(h) for h in all_heads)),
        }
        feats_list.append([s[k] for k in feature_names])
    X_stats = np.array(feats_list, dtype=np.float32)

    # Per-feature AUC
    print(f"  {'feature':<28s} {'mean_cons':>10s} {'mean_cls':>10s} {'AUC':>6s}")
    for j, n in enumerate(feature_names):
        v = X_stats[:, j]
        a = roc_auc_score(labels, v)
        a = max(a, 1 - a)
        if a > 0.55:
            print(f"  {n:<28s} {v[labels==0].mean():>10.3f} "
                  f"{v[labels==1].mean():>10.3f} {a:>6.3f}")

    auc_stats_hull, _ = compute_hull_auc(X_stats, labels)
    print(f"  Hull AUC (stats):              {auc_stats_hull:.4f}")
    auc_stats_lr = lr_auc(X_stats, labels)
    print(f"  Logistic regression AUC (stats): {auc_stats_lr:.4f}")

    # ------------------------------------------------------------------
    # Calibration: combined BoW + stats
    # ------------------------------------------------------------------
    print("\n[Calibration] Combined BoW + stats logistic regression")
    X_combined = np.hstack([X_bow, X_stats])
    auc_combined_lr = lr_auc(X_combined, labels)
    print(f"  AUC (combined logistic): {auc_combined_lr:.4f}")

    # Decision point
    print(f"\n{'='*60}")
    print("DECISION POINT")
    print(f"{'='*60}")
    print(f"  Combined logistic regression AUC: {auc_combined_lr:.4f}")
    if auc_combined_lr < 0.55:
        decision = "STOP — signal too weak in proofs"
    elif auc_combined_lr >= 0.60:
        decision = "PROCEED — train tactic-sequence encoder"
    else:
        decision = "MARGINAL — proceed cautiously"
    print(f"  Decision: {decision}")

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    Path('results/data/stage4v3p').mkdir(parents=True, exist_ok=True)
    np.savez_compressed('results/data/stage4v3p/baselines.npz',
                        X_bow=X_bow, X_stats=X_stats, labels=labels)
    with open('results/data/stage4v3p/proofs.json', 'w') as f:
        json.dump(proofs, f)
    with open('results/data/stage4v3p/vocab.json', 'w') as f:
        json.dump(vocab, f, ensure_ascii=False)

    stats = {
        'n_proofs': len(proofs),
        'n_classical': int(n_classical),
        'n_constructive': int(n_constructive),
        'vocab_size': len(vocab),
        'auc_bow_hull': float(auc_bow_hull),
        'auc_bow_lr': float(auc_bow_lr),
        'auc_stats_hull': float(auc_stats_hull),
        'auc_stats_lr': float(auc_stats_lr),
        'auc_combined_lr': float(auc_combined_lr),
        'decision': decision,
        'feature_names': feature_names,
    }
    with open('results/data/stage4v3p/baseline_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/data/stage4v3p/{baselines.npz,proofs.json,vocab.json,baseline_stats.json}")


def compute_hull_auc(X: np.ndarray, labels: np.ndarray, hull_dim: int = 4):
    """Hull AUC where positive=classical (label 1)."""
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA

    cons_idx = np.where(labels == 0)[0]
    cls_idx = np.where(labels == 1)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(cons_idx)
    n_train = int(0.8 * len(cons_idx))
    train_idx = cons_idx[:n_train]
    test_idx = cons_idx[int(0.9 * len(cons_idx)):]
    ref_idx = rng.choice(train_idx, size=min(5000, len(train_idx)),
                         replace=False)

    # Need numerical stability when input has zero variance in some dims
    # PCA handles it fine.
    pca = PCA(n_components=min(hull_dim, X.shape[1])).fit(X[ref_idx])
    rp = pca.transform(X[ref_idx])
    try:
        hull = ConvexHull(rp)
        vert = rp[hull.vertices]
    except Exception as e:
        # Fall back to centroid-distance metric
        c = rp.mean(axis=0)
        d_test = np.linalg.norm(pca.transform(X[test_idx]) - c, axis=1)
        d_cls = np.linalg.norm(pca.transform(X[cls_idx]) - c, axis=1)
        y = np.concatenate([np.zeros(len(d_test)), np.ones(len(d_cls))])
        s = np.concatenate([d_test, d_cls])
        return float(roc_auc_score(y, s)), None
    d_test = cdist(pca.transform(X[test_idx]), vert).min(axis=1)
    d_cls = cdist(pca.transform(X[cls_idx]), vert).min(axis=1)
    y = np.concatenate([np.zeros(len(d_test)), np.ones(len(d_cls))])
    s = np.concatenate([d_test, d_cls])
    return float(roc_auc_score(y, s)), hull


def lr_auc(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold cross-validated logistic regression AUC."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        Xs = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=500, n_jobs=1)
        clf.fit(Xs.transform(X[tr]), y[tr])
        proba = clf.predict_proba(Xs.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(aucs))


if __name__ == '__main__':
    main()
