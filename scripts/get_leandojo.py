"""Fetch the LeanDojo Benchmark 4 tactic-trace splits.

The learned-representation experiments read traced tactic proofs from
``results/data/leandojo/{train,val,test}.json`` (~940 MB total). Each record
has at least ``full_name``, ``file_path``, ``theorem_statement`` and the
``traced_tactics`` list (with ``state_before`` / ``state_after``).

These come from LeanDojo Benchmark 4 (Yang et al., NeurIPS 2023), random
split. The exact revision used for the paper was traced against Mathlib4
commit `1bc7728a050fc18ca2683f614c531cd7050ff063` — every trace record carries
this commit in its `url`/`commit` field, and this script verifies it (see
EXPECTED_MATHLIB_COMMIT below) so a wrong/!= version is caught immediately.

There are two ways to obtain the splits:

1. Download a prepared copy (recommended). Set LEANDOJO_URL to a tarball/zip
   that unpacks to train.json / val.json / test.json and run this script with
   --download. We do not hardcode a URL because the official distribution is
   versioned; see https://leandojo.org and the dataset release on Zenodo
   (search "LeanDojo Benchmark 4"). Pick the release traced against the commit
   above (or re-trace it, option 2) so depth labels line up theorem-for-theorem.

2. Re-trace locally with the `lean-dojo` package against the pinned Mathlib
   checkout (scripts/setup_mathlib.sh). This is the most faithful route but
   requires a full Mathlib trace (hours). See the LeanDojo docs:
   https://leandojo.readthedocs.io/

This script only verifies/asserts the expected layout, and optionally
downloads + unpacks from LEANDOJO_URL. It deliberately does no tracing.

Usage:
    python scripts/get_leandojo.py            # check what is present
    LEANDOJO_URL=<tarball-url> python scripts/get_leandojo.py --download
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import LEANDOJO_DIR  # noqa: E402

SPLITS = ("train", "val", "test")

# The traces used for the paper were generated against this Mathlib4 commit.
# Each record embeds it in its `commit`/`url` field; verify_provenance()
# checks a downloaded copy matches so depth labels line up theorem-for-theorem.
EXPECTED_MATHLIB_COMMIT = "1bc7728a050fc18ca2683f614c531cd7050ff063"


def status() -> bool:
    LEANDOJO_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for s in SPLITS:
        p = LEANDOJO_DIR / f"{s}.json"
        if p.is_file():
            mb = p.stat().st_size / 1e6
            print(f"  [ok]      {p}  ({mb:,.0f} MB)")
        else:
            print(f"  [MISSING] {p}")
            ok = False
    return ok


def verify_provenance() -> bool:
    """Check that the traces carry EXPECTED_MATHLIB_COMMIT.

    Each record embeds the commit in its `url`/`commit` field near the start of
    the file, so we scan only the first few KB rather than loading the whole
    (hundreds-of-MB) split.
    """
    p = LEANDOJO_DIR / "test.json"
    if not p.is_file():
        return False
    with p.open() as f:
        head = f.read(8000)
    if EXPECTED_MATHLIB_COMMIT in head:
        print(f"  [ok]      provenance: Mathlib4 commit {EXPECTED_MATHLIB_COMMIT[:12]}…")
        return True
    print(f"  [warn] provenance: expected Mathlib4 commit {EXPECTED_MATHLIB_COMMIT[:12]}… "
          f"not found in test.json head — depth labels may not align with the paper.")
    return False


def download(url: str) -> None:
    LEANDOJO_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if url.lower().endswith(".zip") else ".tar.gz"
    archive = LEANDOJO_DIR / f"_leandojo_download{suffix}"
    print(f"[get_leandojo] Downloading {url} -> {archive}")
    urllib.request.urlretrieve(url, archive)
    print("[get_leandojo] Unpacking...")
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(LEANDOJO_DIR)
    else:
        with tarfile.open(archive) as t:
            t.extractall(LEANDOJO_DIR)
    archive.unlink(missing_ok=True)
    # Flatten if the archive unpacked into a subdirectory.
    for s in SPLITS:
        if not (LEANDOJO_DIR / f"{s}.json").is_file():
            hits = list(LEANDOJO_DIR.rglob(f"{s}.json"))
            if hits:
                hits[0].replace(LEANDOJO_DIR / f"{s}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="Download + unpack from the LEANDOJO_URL environment variable.")
    args = ap.parse_args()

    if args.download:
        import os
        url = os.environ.get("LEANDOJO_URL")
        if not url:
            sys.exit("Set LEANDOJO_URL to a tarball/zip of the LeanDojo Benchmark 4 "
                     "splits, or obtain them per the module docstring.")
        download(url)

    print(f"LeanDojo splits under {LEANDOJO_DIR}:")
    if status():
        verify_provenance()
        print("[get_leandojo] All splits present.")
        return 0
    print("\n[get_leandojo] Missing splits. See this script's docstring for how to "
          "obtain LeanDojo Benchmark 4 (https://leandojo.org).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
