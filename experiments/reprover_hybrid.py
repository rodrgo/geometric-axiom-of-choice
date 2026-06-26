"""Experiment A: ReProver + aesop hybrid.

For each theorem and each of its 8 previously-generated ReProver
candidate tactics, test whether

      by
        <candidate>
        all_goals aesop

closes the theorem. For classical theorems (non-constructive bucket)
we also try

      by
        classical
        <candidate>
        all_goals aesop

as a second variant per candidate. Early-exit on the first success per
theorem; record which candidate/variant won. Timeout: 60s per check.

Semantics (paired with aesop and single-shot ReProver experiments):
  success = `lake env lean` exit 0 AND no `error:` AND no `sorry` in
            combined stdout+stderr.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.lakerunner import try_prove  # noqa: E402
from lean.sources import find_proof_replacement  # noqa: E402
from config import MATHLIB_PATH as MATHLIB  # noqa: E402

OUT_DIR = Path.home() / "prover_eval" / "reprover" / "hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results/data/reviewer/reprover_hybrid_results.jsonl"
PRIOR = ROOT / "results/data/reviewer/reprover_tacgen_results.jsonl"
TIMEOUT_S = 60


def render_body(candidate, use_classical=False):
    if use_classical:
        return f"by\n  classical\n  {candidate}\n  all_goals aesop\n"
    return f"by\n  {candidate}\n  all_goals aesop\n"


def try_compile(test_file, timeout_s=TIMEOUT_S):
    success, elapsed, _ = try_prove(test_file, timeout_s=timeout_s)
    return success, elapsed


def done_names():
    if not RESULTS.exists():
        return set()
    done = set()
    with open(RESULTS) as f:
        for line in f:
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass
    return done


def main():
    prior = {}
    with open(PRIOR) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prior[r["name"]] = r
    done = done_names()
    pending = [(n, r) for n, r in prior.items() if n not in done]
    print(f"{len(prior)} theorems with candidates; {len(done)} already done; "
          f"{len(pending)} pending")

    from collections import Counter
    tot = Counter(); succ = Counter()
    if RESULTS.exists():
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                tot[r["bucket"]] += 1
                if r.get("any_candidate_succeeded"):
                    succ[r["bucket"]] += 1
            except Exception:
                pass

    t_start = time.time()
    with open(RESULTS, "a") as out:
        for idx, (name, job) in enumerate(pending):
            tactics = job.get("tactics") or []
            src_path = MATHLIB / job["source_file"]
            record = {
                "name": name,
                "bucket": job["bucket"],
                "depth": job.get("depth"),
                "anomaly_score": job.get("anomaly_score"),
                "proof_length": job.get("proof_length"),
                "source_file": job["source_file"],
                "any_candidate_succeeded": False,
                "best_candidate": None,
                "best_variant": None,
                "n_candidates_tried": 0,
                "elapsed_total_s": 0.0,
            }
            if not src_path.exists() or not tactics:
                record["skip_reason"] = "src_missing_or_no_tactics"
                out.write(json.dumps(record) + "\n")
                out.flush()
                tot[job["bucket"]] += 1
                continue
            source = src_path.read_text()
            short = name.split(".")[-1]
            rep = find_proof_replacement(source, name) \
                or find_proof_replacement(source, short)
            if rep is None:
                record["skip_reason"] = "header_unlocated"
                out.write(json.dumps(record) + "\n")
                out.flush()
                tot[job["bucket"]] += 1
                continue
            body_start, body_end = rep

            # Variant list: for constructive only plain; else plain + classical.
            variants = [("plain", False)]
            if job["bucket"] != "constructive":
                variants.append(("classical", True))

            elapsed_total = 0.0
            done_thm = False
            for k, cand in enumerate(tactics):
                for variant_name, use_cls in variants:
                    body = render_body(cand, use_cls)
                    new_source = source[:body_start] + " " + body + source[body_end:]
                    test_file = OUT_DIR / f"hyb_{idx:05d}_{k}_{variant_name}.lean"
                    test_file.write_text(new_source)
                    success, elapsed = try_compile(test_file, TIMEOUT_S)
                    elapsed_total += elapsed
                    record["n_candidates_tried"] += 1
                    try:
                        test_file.unlink()
                    except Exception:
                        pass
                    if success:
                        record["any_candidate_succeeded"] = True
                        record["best_candidate"] = cand
                        record["best_variant"] = variant_name
                        done_thm = True
                        break
                if done_thm:
                    break
            record["elapsed_total_s"] = elapsed_total

            out.write(json.dumps(record) + "\n")
            out.flush()
            tot[job["bucket"]] += 1
            if record["any_candidate_succeeded"]:
                succ[job["bucket"]] += 1

            if (idx + 1) % 5 == 0 or idx + 1 == len(pending):
                el = time.time() - t_start
                print(f"[{idx+1}/{len(pending)}]  elapsed {el:.1f}s")
                for b in ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]:
                    t = tot[b]; s = succ[b]
                    print(f"  {b:<14s}  {s}/{t}  ({100*s/max(t,1):.1f}%)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
