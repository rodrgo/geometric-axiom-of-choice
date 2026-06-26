"""Subprocess wrapper for `lake env lean <test_file>`.

Shared by the aesop / classical-prefix / ReProver evaluation scripts.
The standard success rule is:
    exit code 0
    AND "error:" not in (stdout + stderr).lower()
    AND "sorry" not in (stdout + stderr).lower()  [unless reject_sorry=False]

reject_sorry=False is for the candidate-tactic *validity* probe
(scripts/reprover_validity.py) — sorry-emitting tactics are still valid
parses; we just want to know whether Lean accepted them.
"""

import subprocess
import time
from pathlib import Path

from config import MATHLIB_PATH

DEFAULT_TIMEOUT_S = 60


def try_prove(
    test_file: Path,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    *,
    mathlib: Path = MATHLIB_PATH,
    reject_sorry: bool = True,
) -> tuple[bool, float, str]:
    """Run `lake env lean <test_file>` and report (success, elapsed, err).

    success: True if Lean accepted the file (per the rule above).
    elapsed: wall time in seconds.
    err: first 400 chars of stderr (or stdout if stderr empty) when not
         successful; empty string on success or timeout returns "timeout".
    """
    start = time.time()
    try:
        r = subprocess.run(
            ["lake", "env", "lean", str(test_file)],
            cwd=mathlib,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.time() - start
        out = (r.stdout + r.stderr).lower()
        success = (r.returncode == 0) and ("error:" not in out)
        if reject_sorry:
            success = success and ("sorry" not in out)
        if success:
            return True, elapsed, ""
        return False, elapsed, (r.stderr[:400] or r.stdout[:400])
    except subprocess.TimeoutExpired:
        return False, timeout_s, "timeout"
