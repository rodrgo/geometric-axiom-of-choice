"""Kernel-level declaration dependency graph and classical-depth BFS.

The graph is `results/data/stage4v3/decl_graph_raw.jsonl` (~471K records,
one JSON object per line, terminator `[DONE]`), produced from a compiled
Mathlib checkout by `scripts/build_decl_graph.py`. See README.md ("Heavy
external dependencies").

Each line has:
    n: declaration name
    m: module
    d: list of dependencies (names this decl uses)

`load_graph()` returns (fwd, rev, modules) where fwd[n] is the dep list,
rev[n] is the list of declarations that depend on n, and modules[n] is
the module string.

`bfs_classical_depths(rev, seeds)` returns {decl_name: depth} for every
declaration reachable from a seed in `seeds` via reverse edges (= via
"used by"). A declaration's depth is its shortest-path distance from
any seed. Declarations not in the dict are constructive (unreachable).

The classical-depth map is cached to `data/stage4v3/classical_depths.json`
so Stage 6 / reviewer scripts don't have to re-parse the 471K-line file.
"""

import json
import time
from collections import defaultdict, deque

from config import DECL_GRAPH_JSONL, CLASSICAL_DEPTHS_JSON


CLASSICAL_SEEDS_DEFAULT = ('Classical.choice',)


def load_graph():
    """Parse decl_graph_raw.jsonl into (fwd, rev, modules)."""
    t0 = time.time()
    fwd: dict[str, list[str]] = {}
    rev: dict[str, list[str]] = defaultdict(list)
    modules: dict[str, str] = {}
    with open(DECL_GRAPH_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line or line == '[DONE]':
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec['n']
            deps = rec.get('d', [])
            fwd[name] = deps
            modules[name] = rec.get('m', '')
            for dep in deps:
                rev[dep].append(name)
    print(f"  Loaded {len(fwd):,} declarations in {time.time()-t0:.1f}s")
    return fwd, rev, modules


def bfs_classical_depths(rev: dict, seeds=CLASSICAL_SEEDS_DEFAULT) -> dict:
    """BFS on reverse edges from `seeds`. Returns {name: shortest-depth}.

    depth 0 = seed itself, depth 1 = direct user of a seed, etc.
    Declarations not reached are constructive (omit from dict).
    """
    depths: dict[str, int] = {s: 0 for s in seeds}
    queue = deque((s, 0) for s in seeds)
    while queue:
        node, d = queue.popleft()
        for user in rev.get(node, []):
            if user not in depths:
                depths[user] = d + 1
                queue.append((user, d + 1))
    return depths


def load_classical_depths(*, rebuild: bool = False,
                          seeds=CLASSICAL_SEEDS_DEFAULT) -> dict:
    """Load `classical_depths.json` if cached, otherwise compute and cache.

    Pass rebuild=True to force a recomputation (e.g. if the seed set
    changes).
    """
    if not rebuild and CLASSICAL_DEPTHS_JSON.exists():
        with open(CLASSICAL_DEPTHS_JSON) as f:
            return json.load(f)
    print(f"Computing classical depths (seeds={list(seeds)})...")
    _, rev, _ = load_graph()
    depths = bfs_classical_depths(rev, seeds=seeds)
    CLASSICAL_DEPTHS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASSICAL_DEPTHS_JSON, 'w') as f:
        json.dump(depths, f)
    print(f"  Cached {len(depths):,} entries to {CLASSICAL_DEPTHS_JSON}")
    return depths
