"""Stage 4v3a: Build classical/constructive partition from kernel decl graph.

Algorithm:
  1. Load forward / reverse dependency graph (via lean.kernel_graph).
  2. BFS through reverse edges starting from CLASSICAL_SEEDS.
  3. Everything reachable is "classical"; everything else "constructive".
  4. Match against LeanDojo theorem names.

Optionally exclude propext / Quot.sound from the seed set since they're
nearly universal in Mathlib (built into elementary equality/quotient
machinery). Plan recommends starting with just Classical.choice.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lean.kernel_graph import load_graph, bfs_classical_depths  # noqa: E402


CLASSICAL_SEEDS = ('Classical.choice',)


def main():
    print("Loading decl graph...")
    fwd, rev, _modules = load_graph()
    print(f"  fwd entries: {len(fwd):,}, rev entries: {len(rev):,}")

    # Verify seed exists in graph
    print(f"\nChecking seeds...")
    for seed in CLASSICAL_SEEDS:
        in_fwd = seed in fwd
        n_users = len(rev.get(seed, []))
        print(f"  {seed}: in fwd={in_fwd}, n_direct_users={n_users}")

    # BFS on reverse graph
    print("\nBFS on reverse graph...")
    depths = bfs_classical_depths(rev, seeds=CLASSICAL_SEEDS)
    classical_set = set(depths)
    all_decls = set(fwd.keys())
    constructive_set = all_decls - classical_set

    print(f"\nPartition over {len(all_decls):,} declarations:")
    print(f"  classical:    {len(classical_set):,} "
          f"({100*len(classical_set)/len(all_decls):.1f}%)")
    print(f"  constructive: {len(constructive_set):,} "
          f"({100*len(constructive_set)/len(all_decls):.1f}%)")

    # ------------------------------------------------------------------
    # Match against LeanDojo theorems
    # ------------------------------------------------------------------
    print("\nMatching against LeanDojo theorem statements...")
    leandojo_records = []
    for split in ['train', 'val', 'test']:
        with open(f'results/data/leandojo/{split}.json') as f:
            for thm in json.load(f):
                leandojo_records.append({
                    'full_name': thm.get('full_name', ''),
                    'file_path': thm.get('file_path', ''),
                    'theorem_statement': thm.get('theorem_statement', ''),
                    'split': split,
                })
    print(f"  LeanDojo theorems: {len(leandojo_records):,}")

    matched_classical = 0
    matched_constructive = 0
    unmatched = 0
    for r in leandojo_records:
        n = r['full_name']
        if not n:
            unmatched += 1
            continue
        if n in classical_set:
            matched_classical += 1
            r['is_classical'] = True
            r['matched'] = True
        elif n in constructive_set:
            matched_constructive += 1
            r['is_classical'] = False
            r['matched'] = True
        else:
            unmatched += 1
            r['is_classical'] = None
            r['matched'] = False

    n_matched = matched_classical + matched_constructive
    n_total = len(leandojo_records)
    match_rate = n_matched / n_total
    print(f"  Matched: {n_matched:,} / {n_total:,} ({100*match_rate:.1f}%)")
    print(f"    classical:    {matched_classical:,} "
          f"({100*matched_classical/max(n_matched,1):.1f}% of matched)")
    print(f"    constructive: {matched_constructive:,} "
          f"({100*matched_constructive/max(n_matched,1):.1f}% of matched)")
    print(f"  Unmatched: {unmatched:,}")

    if match_rate < 0.6:
        print("\n  ! WARNING: low match rate. Sample of unmatched names:")
        for r in leandojo_records[:20]:
            if not r['matched']:
                print(f"    {r['full_name']}")

    # Spot-check
    print("\nSample classical theorems (first 10 from val):")
    cnt = 0
    for r in leandojo_records:
        if r['split'] == 'val' and r.get('is_classical') is True:
            print(f"  {r['full_name']}  ({r['file_path']})")
            cnt += 1
            if cnt >= 10:
                break

    print("\nSample constructive theorems (first 10 from val):")
    cnt = 0
    for r in leandojo_records:
        if r['split'] == 'val' and r.get('is_classical') is False:
            print(f"  {r['full_name']}  ({r['file_path']})")
            cnt += 1
            if cnt >= 10:
                break

    # Compare with v2 partition
    print("\nComparing with Stage 4v2 partition...")
    try:
        with open('results/data/stage4v2/theorems_partition.json') as f:
            v2 = {r['full_name']: r['is_classical'] for r in json.load(f)}
        v3 = {r['full_name']: r.get('is_classical') for r in leandojo_records
              if r.get('matched')}
        common = set(v2) & set(v3)
        v2_classical = {n for n in common if v2[n]}
        v3_classical = {n for n in common if v3[n]}
        agree = sum(1 for n in common if v2[n] == v3[n])
        print(f"  Common theorems: {len(common):,}")
        print(f"  v2 classical: {len(v2_classical):,}, "
              f"v3 classical: {len(v3_classical):,}")
        print(f"  Agreement: {agree:,} ({100*agree/max(len(common),1):.1f}%)")
        print(f"  v2 classical ∩ v3 classical: "
              f"{len(v2_classical & v3_classical):,}")
        print(f"  v3 classical \\ v2 classical (newly classical): "
              f"{len(v3_classical - v2_classical):,}")
        print(f"  v2 classical \\ v3 classical (no longer classical): "
              f"{len(v2_classical - v3_classical):,}")
    except FileNotFoundError:
        print("  (No v2 partition found)")

    # Domain distribution
    def domain(p):
        parts = p.split('/')
        return parts[1] if len(parts) >= 2 and parts[0] == 'Mathlib' else 'Other'

    dom_class = Counter()
    dom_cons = Counter()
    for r in leandojo_records:
        if not r.get('matched'):
            continue
        d = domain(r['file_path'])
        if r['is_classical']:
            dom_class[d] += 1
        else:
            dom_cons[d] += 1
    print(f"\nDomain distribution (% classical, top 15 by total):")
    all_doms = set(dom_class) | set(dom_cons)
    rows = []
    for d in all_doms:
        total = dom_class[d] + dom_cons[d]
        if total >= 100:
            pct = 100 * dom_class[d] / total
            rows.append((d, total, pct, dom_class[d]))
    rows.sort(key=lambda x: -x[1])
    for d, t, pct, nc in rows[:15]:
        print(f"  {d:<22s} n_total={t:>6,}  n_classical={nc:>6,}  "
              f"({pct:>5.1f}%)")

    # Save
    Path('results/data/stage4v3').mkdir(parents=True, exist_ok=True)
    with open('results/data/stage4v3/theorems_partition.json', 'w') as f:
        json.dump([
            {
                'full_name': r['full_name'],
                'file_path': r['file_path'],
                'split': r['split'],
                'is_classical': r.get('is_classical'),
                'matched': r.get('matched', False),
            }
            for r in leandojo_records
        ], f)

    stats = {
        'total_decls_in_graph': len(all_decls),
        'classical_in_graph': len(classical_set),
        'constructive_in_graph': len(constructive_set),
        'leandojo_total': n_total,
        'leandojo_matched': n_matched,
        'match_rate': match_rate,
        'matched_classical': matched_classical,
        'matched_constructive': matched_constructive,
        'classical_pct_of_matched': 100 * matched_classical / max(n_matched, 1),
        'unmatched': unmatched,
        'seed_set': sorted(CLASSICAL_SEEDS),
    }
    with open('results/data/stage4v3/partition_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/data/stage4v3/{theorems_partition,partition_stats}.json")


if __name__ == '__main__':
    main()
