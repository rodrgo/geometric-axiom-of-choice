"""Loaders for the vendored LeanDojo Benchmark 4 split files.

`data/leandojo/{train,val,test}.json` are JSON arrays of theorem records.
Each record has at least: full_name, file_path, theorem_statement, start,
end, plus tactic / proof annotations from LeanDojo.
"""

import json

from config import LEANDOJO_DIR


def load_split(split: str) -> list[dict]:
    """Load one of {train, val, test}."""
    with open(LEANDOJO_DIR / f"{split}.json") as f:
        return json.load(f)


def load_all_splits() -> list[dict]:
    """Concatenate train + val + test, in that order. Each record gets a
    `split` field added in-place."""
    records = []
    for split in ('train', 'val', 'test'):
        for thm in load_split(split):
            thm.setdefault('split', split)
            records.append(thm)
    return records


def by_full_name(records: list[dict]) -> dict[str, dict]:
    """Index by `full_name` for O(1) lookup. Later entries overwrite earlier."""
    return {thm['full_name']: thm for thm in records if thm.get('full_name')}
