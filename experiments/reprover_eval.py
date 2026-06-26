"""Run single-shot top-K TacGen evaluation on the 251-theorem aesop sample.

Pipeline per theorem (paired with aesop/classical-prefix experiments):

  1. Fetch the initial proof state `state_before` from LeanDojo's
     pre-traced data (data/leandojo/{train,val,test}.json). Because we
     sourced the theorem sample from there in the first place, every
     theorem has a state.
  2. Generate K=8 beam-search tactic candidates from
     kaiyuy/leandojo-lean4-tacgen-byt5-small.
  3. Copy the theorem's Mathlib source file, splice in
        := by
          first
            | tac_1
            | tac_2
            | ...
            | tac_K
     where tac_i are the K generated tactics.
  4. Compile with `lake env lean` in the Mathlib project dir, 60-s
     timeout. Success = exit 0, no `error:` or `sorry` in combined
     stdout+stderr.

This is a single-shot top-K protocol: the model does NOT see
intermediate proof states, only the initial goal. It is therefore
weaker than full best-first search, but isolates the quality of the
tactic generator in exactly the same evaluation format as the aesop
experiment (one wall-clock compile per theorem, same source contexts,
same timeout).

Stream-writes to results/data/reviewer/reprover_tacgen_results.jsonl and can
resume from partial results.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lean.lakerunner import try_prove as _try_prove  # noqa: E402
from lean.sources import find_proof_replacement  # noqa: E402
from config import MATHLIB_PATH as MATHLIB  # noqa: E402
OUT_DIR = Path.home() / "prover_eval" / "reprover"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = ROOT / "results/data/reviewer/reprover_tacgen_results.jsonl"
JOBS_FILE = ROOT / "results/data/reviewer/prover_jobs.json"

MODEL_NAME = "kaiyuy/leandojo-lean4-tacgen-byt5-small"
K = 8
TIMEOUT_S = 60


def build_state_index():
    """Returns {full_name: state_before} covering our 251 theorems."""
    need_names = set()
    with open(JOBS_FILE) as f:
        for j in json.load(f)["jobs"]:
            need_names.add(j["name"])
    print(f"need states for {len(need_names)} theorems")
    idx = {}
    for split in ("train", "val", "test"):
        with open(ROOT / f"results/data/leandojo/{split}.json") as f:
            data = json.load(f)
        for thm in data:
            name = thm.get("full_name")
            if name in need_names and name not in idx:
                tts = thm.get("traced_tactics", [])
                if tts:
                    idx[name] = tts[0]["state_before"]
        print(f"  after {split}: {len(idx)} states found")
    missing = need_names - set(idx)
    print(f"missing: {len(missing)}")
    return idx


def already_done():
    if not RESULTS_FILE.exists():
        return set()
    done = set()
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["name"])
            except json.JSONDecodeError:
                continue
    return done


def render_first_block(tactics):
    body = "by\n    first\n"
    for t in tactics:
        body += f"      | {t}\n"
    return body


def try_compile(test_file, timeout_s=TIMEOUT_S):
    return _try_prove(test_file, timeout_s=timeout_s)


def main():
    print("Loading jobs...")
    with open(JOBS_FILE) as f:
        jobs = json.load(f)["jobs"]
    done = already_done()
    pending = [j for j in jobs if j["name"] not in done]
    print(f"{len(jobs)} total, {len(done)} done, {len(pending)} pending")

    print("Building state index from LeanDojo...")
    states = build_state_index()
    coverage = sum(1 for j in jobs if j["name"] in states)
    print(f"state coverage: {coverage}/{len(jobs)}")

    print(f"Loading {MODEL_NAME}...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()

    from collections import Counter
    succ = Counter(); totals = Counter()
    if RESULTS_FILE.exists():
        for line in open(RESULTS_FILE):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            totals[r["bucket"]] += 1
            if r.get("success"):
                succ[r["bucket"]] += 1

    t_start = time.time()
    with open(RESULTS_FILE, "a") as out:
        for i, job in enumerate(pending):
            name = job["name"]
            state = states.get(name)
            if state is None:
                out.write(json.dumps({
                    "name": name,
                    "bucket": job["bucket"],
                    "depth": job.get("depth"),
                    "success": False,
                    "elapsed_s": 0.0,
                    "anomaly_score": job["anomaly_score"],
                    "proof_length": job["proof_length"],
                    "source_file": job["source_file"],
                    "tactics": [],
                    "skip_reason": "no_state",
                }) + "\n")
                out.flush()
                totals[job["bucket"]] += 1
                continue

            # Generate top-K tactics
            g0 = time.time()
            import torch
            with torch.no_grad():
                inputs = tok(state, return_tensors="pt", truncation=True, max_length=2300)
                outs = model.generate(
                    inputs.input_ids,
                    max_length=256,
                    num_beams=K,
                    do_sample=False,
                    num_return_sequences=K,
                    length_penalty=0.0,
                    early_stopping=False,
                )
            tactics = tok.batch_decode(outs, skip_special_tokens=True)
            gen_s = time.time() - g0

            # Splice + compile
            fp = MATHLIB / job["source_file"]
            if not fp.exists():
                out.write(json.dumps({
                    "name": name,
                    "bucket": job["bucket"],
                    "depth": job.get("depth"),
                    "success": False,
                    "elapsed_s": 0.0,
                    "gen_s": gen_s,
                    "tactics": tactics,
                    "skip_reason": "src_missing",
                    "anomaly_score": job["anomaly_score"],
                    "proof_length": job["proof_length"],
                    "source_file": job["source_file"],
                }) + "\n")
                out.flush()
                totals[job["bucket"]] += 1
                continue
            source = fp.read_text()
            full = name
            short = full.split(".")[-1]
            rep = find_proof_replacement(source, full)
            if rep is None:
                rep = find_proof_replacement(source, short)
            if rep is None:
                out.write(json.dumps({
                    "name": name, "bucket": job["bucket"], "depth": job.get("depth"),
                    "success": False, "elapsed_s": 0.0, "gen_s": gen_s,
                    "tactics": tactics, "skip_reason": "header_unlocated",
                    "anomaly_score": job["anomaly_score"],
                    "proof_length": job["proof_length"],
                    "source_file": job["source_file"],
                }) + "\n")
                out.flush()
                totals[job["bucket"]] += 1
                continue
            body_start, body_end = rep
            new_source = source[:body_start] + " " + render_first_block(tactics) + source[body_end:]
            test_file = OUT_DIR / f"reprover_{i:05d}_{name.replace('.', '_').replace('/', '_')[-80:]}.lean"
            test_file.write_text(new_source)
            success, elapsed, err = try_compile(test_file, TIMEOUT_S)

            result = {
                "name": name,
                "bucket": job["bucket"],
                "depth": job.get("depth"),
                "success": bool(success),
                "elapsed_s": float(elapsed),
                "gen_s": float(gen_s),
                "anomaly_score": job["anomaly_score"],
                "proof_length": job["proof_length"],
                "source_file": job["source_file"],
                "tactics": tactics,
                "error_snippet": err if not success else "",
            }
            out.write(json.dumps(result) + "\n")
            out.flush()
            totals[job["bucket"]] += 1
            if success:
                succ[job["bucket"]] += 1

            if (i + 1) % 10 == 0 or i + 1 == len(pending):
                el = time.time() - t_start
                print(f"[{i+1}/{len(pending)}]  elapsed {el:.1f}s")
                for b in ["constructive", "depth 2", "depth 3-4", "depth 5-6", "depth 7+"]:
                    t = totals[b]; s = succ[b]
                    print(f"  {b:<14s}  {s}/{t}  ({100*s/max(t,1):.1f}%)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
