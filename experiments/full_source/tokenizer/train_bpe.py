"""Train BPE on the constructive training corpus.

Vocabulary: 32K. Special tokens: <pad> <unk> <mask> <bos> <eos>.

Constructive training corpus is determined by (is_classical=False, split='train')
in results/data/stage4v3p/proofs.json, joined with the LeanDojo splits
to recover the raw ``tactic`` strings. The encoder later sees the same
corpus.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import normalize_lean_source  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.pre_tokenizers import WhitespaceSplit  # noqa: E402
from tokenizers.trainers import BpeTrainer  # noqa: E402

VOCAB_SIZE = 32_768
SPECIAL = ["<pad>", "<unk>", "<mask>", "<bos>", "<eos>"]

PROOFS_JSON = ROOT / "results/data/stage4v3p/proofs.json"
LEANDOJO_TRAIN = ROOT / "results/data/leandojo/train.json"
LEANDOJO_VAL = ROOT / "results/data/leandojo/val.json"
LEANDOJO_TEST = ROOT / "results/data/leandojo/test.json"

OUT_TOKENIZER = ROOT / "experiments/full_source/tokenizer/artifacts/bpe_32k.json"
OUT_CORPUS = ROOT / "experiments/full_source/tokenizer/artifacts/train_corpus.txt"


def proof_body(theorem: dict) -> str:
    """Concatenate the traced tactic strings of a LeanDojo theorem."""
    return "\n".join(t["tactic"] for t in theorem.get("traced_tactics", []))


def build_corpus() -> list[str]:
    t0 = time.time()
    print("loading head-level proof manifest...", flush=True)
    proofs = json.loads(PROOFS_JSON.read_text())
    constructive_train = {p["name"] for p in proofs
                          if not p["is_classical"] and p["split"] == "train"}
    print(f"  constructive-train: {len(constructive_train):,} proofs"
          f"  [{time.time()-t0:.1f}s]")

    print("loading LeanDojo train+val+test (need to join on name)...", flush=True)
    by_name: dict[str, dict] = {}
    for p in [LEANDOJO_TRAIN, LEANDOJO_VAL, LEANDOJO_TEST]:
        data = json.loads(p.read_text())
        for t in data:
            by_name[t["full_name"]] = t
    print(f"  LeanDojo theorems indexed: {len(by_name):,}"
          f"  [{time.time()-t0:.1f}s]")

    print("building corpus...", flush=True)
    corpus: list[str] = []
    missing = 0
    for name in sorted(constructive_train):
        t = by_name.get(name)
        if t is None:
            missing += 1
            continue
        body = proof_body(t)
        if not body.strip():
            continue
        corpus.append(normalize_lean_source(body))
    print(f"  corpus: {len(corpus):,} proofs"
          f"  (missing in LeanDojo: {missing})"
          f"  [{time.time()-t0:.1f}s]")
    return corpus


def main() -> None:
    corpus = build_corpus()
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    OUT_CORPUS.write_text("\n".join(corpus) + "\n")
    print(f"wrote {OUT_CORPUS}")

    print("training BPE...", flush=True)
    t0 = time.time()
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = WhitespaceSplit()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL,
                         show_progress=False)
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(str(OUT_TOKENIZER))
    print(f"  trained, vocab={tok.get_vocab_size()}  [{time.time()-t0:.1f}s]")
    print(f"wrote {OUT_TOKENIZER}")

    # Smoke test on a few classical-machinery strings.
    print("\nsmoke test (tokens for representative strings):")
    for s in ["classical.em", "classical.choice", "not_not", "set.indicator",
              "function.surjinv", "by_contra", "by_cases", "have",
              "exact", "fun_prop", "hb.iscompact_closure"]:
        ids = tok.encode(s).ids
        toks = [tok.id_to_token(i) for i in ids]
        print(f"  {s!r:30s} -> {len(ids)} tokens: {toks}")


if __name__ == "__main__":
    main()
