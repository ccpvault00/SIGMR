"""Pre-compute the MedCPT embedding pools (the deployed default).

Encodes the two frozen vocabulary pools used at inference:
    Q_emb_pool       [92, 768]   - questions in train_CoT.json
    A_emb_pool      [405, 768]   - non-final answers (cosine-NN answer-lookup target)

These pools are loaded by EmbeddingPools at inference time and never updated.
Exact-dedupe verified against the training set on 2026-05-15: 92 / 405.

Encoder (--encoder, default medcpt): the DEPLOYED pools under model/pools were built with
ncbi/MedCPT-Query-Encoder - a BERT-base whose [CLS] separates clinical antonyms
("Yes/No", "1/4", "Malignant/Benign") to cos ≈ 0.6-0.78, where PubMedBERT collapses them
to ≈ 0.97 (unseparable → cosine-NN majority collapse). PubMedBERT is retained as a legacy
option. Both are 110M-param, 768-d, and fit alongside TRIDENT on the GPU.

USAGE:
    python scripts/prep_embedding_pools.py
    python scripts/prep_embedding_pools.py --device cuda --batch_size 64

OUTPUT (under --out_dir):
    Q_emb_pool.pt           torch.save({"emb": Tensor[92, 768], "vocab": list[str]})
    A_emb_pool.pt           torch.save({"emb": Tensor[405, 768], "vocab": list[str]})

Loading example:
    pool = torch.load("Q_emb_pool.pt")
    Q_emb_pool = pool["emb"]    # [92, 768]
    Q_VOCAB    = pool["vocab"]  # list[str]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Deployed encoder = MedCPT-Query-Encoder (switched from an earlier PubMedBERT
# default after an empirical antonym test: PubMedBERT collapses "Yes/No", "1/4",
# "Malignant/Benign" to cos sim ≈ 0.97 (unseparable), while MedCPT pushes them to 0.6-0.78).
PUBMEDBERT_MODEL_ID: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MEDCPT_MODEL_ID: str = "ncbi/MedCPT-Query-Encoder"

# The final-report Q; extract_answers excludes its answer from A_pool (it's a full
# report body, not a short categorical answer).
FINAL_REPORT_QUESTION: str = "What is the final pathology report?"

# Expected pool sizes (verified by exact-dedupe scan on 2026-05-15).
# Used as assertions; mismatch means train_CoT.json schema changed.
# Q=92 not 93: organizer label noise puts an empty string Q on 150 turns
# (141 NSCLC subtype, 5 dx_count, 4 cascade). Filtered in extract_questions;
# backfill of training data deferred to dataset class stage.
EXPECTED_Q_COUNT: int = 92
EXPECTED_A_COUNT: int = 405

# BERT-base - 512-token max. Q/A strings are short (typically < 50 tokens).
QA_MAX_LENGTH: int = 128


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: extract_questions
# ─────────────────────────────────────────────────────────────────────────────
def _load_filtered_records(cot_path: Path, allowed_ids: set[str] | None) -> list:
    """Load CoT records, optionally filtered to a set of slide ids (train-only pools).

    When allowed_ids is given (= train split), only those records' answers/questions
    enter the vocab pools. allowed_ids may include or omit the '.tiff' suffix - both
    forms are matched.
    """
    data = json.loads(cot_path.read_text())
    if allowed_ids is None:
        return data
    norm = {s.replace(".tiff", "") for s in allowed_ids}
    return [r for r in data if r["id"].replace(".tiff", "") in norm]


def extract_questions(cot_path: Path, allowed_ids: set[str] | None = None) -> list[str]:
    """Return sorted list of unique question strings from train_CoT.json.

    Args:
        cot_path: path to train_CoT.json (list of records, each with
            'chain-of-thought' field of [{question, answer, next_question}, ...]).

    Returns:
        Sorted list of unique question strings (expected 93 entries).

    Steps:
        1. json.loads the file.
        2. Iterate every record's 'chain-of-thought' list.
        3. Add `turn['question'].strip()` to a set.
        4. Return `sorted(the_set)` - sorted for reproducibility (so q_idx
           is deterministic across runs / machines).

    Why sorted (not insertion order):
        Q_VOCAB is used as a stable q_idx index by the M1 answerer + routing.
        If the index changes between runs, saved checkpoints break. Sorted is
        the cheapest determinism guarantee.

    Test:
        questions = extract_questions(Path('/mnt/data/reg2026/train_CoT.json'))
        assert len(questions) == 93
        assert "What is the organ?" in questions
        assert FINAL_REPORT_QUESTION in questions
    """
    data = _load_filtered_records(cot_path, allowed_ids)
    question_set = set()

    for record in data:
        for turn in record["chain-of-thought"]:
            q = turn["question"].strip()
            if q:  # skip empty (organizer label noise, 150 turns)
                question_set.add(q)

    return sorted(question_set)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: extract_answers (non-final)
# ─────────────────────────────────────────────────────────────────────────────
def extract_answers(cot_path: Path, allowed_ids: set[str] | None = None) -> list[str]:
    """Return sorted list of unique answer strings, EXCLUDING final-report answers.

    Args:
        cot_path: path to train_CoT.json.

    Returns:
        Sorted list of unique non-final answer strings (expected 405 entries).

    Steps:
        1. Same iteration as step 1, but for each turn:
           - Skip the turn entirely if `turn['question'] == FINAL_REPORT_QUESTION`
           - Otherwise add `turn['answer'].strip()` to a set.
        2. Return `sorted(the_set)`.

    Why exclude final-report answers:
        The final-report answer is a full multi-line report body, not a short
        categorical answer. Mixing the ~700 reports into A_pool would inflate it
        and pollute the answer cosine-NN with entire reports.

    Test:
        answers = extract_answers(Path('/mnt/data/reg2026/train_CoT.json'))
        assert len(answers) == 405
        assert "1" in answers          # dx_count answer
        assert "Yes, there is an abnormality." in answers
        # Should NOT contain a full report:
        assert not any("Adenocarcinoma, moderately" in a for a in answers if len(a) > 200)
    """
    data = _load_filtered_records(cot_path, allowed_ids)
    answer_set = set()

    for record in data:
        for turn in record["chain-of-thought"]:
            if turn["question"] == FINAL_REPORT_QUESTION:
                continue
            answer_set.add(turn["answer"].strip())

    return sorted(answer_set)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: encode_pool (BERT-base [CLS] - MedCPT or PubMedBERT)
# ─────────────────────────────────────────────────────────────────────────────
def encode_pool(
    strings: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = QA_MAX_LENGTH,
) -> torch.Tensor:
    """Encode strings via BERT-base [CLS] pooling (MedCPT-Query-Encoder or PubMedBERT).

    Args:
        strings: list of N strings to encode (order is preserved in output).
        model: BERT-based encoder (MedCPT/PubMedBERT), eval mode, on `device`.
        tokenizer: matching tokenizer.
        device: 'cpu' or 'cuda'. Default cpu - TRIDENT is occupying GPU.
        batch_size: how many strings per forward (use 32 on cpu, 64+ on gpu).
        max_length: truncation length. 128 for Q/A, 256 for reports.

    Returns:
        Tensor[N, 768] on cpu, dtype float32, in same order as input strings.

    Steps:
        1. model.eval()
        2. all_embs: list[Tensor] = []
        3. with torch.no_grad():
              for i in range(0, len(strings), batch_size):
                  batch = strings[i : i + batch_size]
                  inputs = tokenizer(
                      batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors='pt'
                  ).to(device)
                  outputs = model(**inputs)
                  cls = outputs.last_hidden_state[:, 0, :]   # [B, 768]
                  all_embs.append(cls.cpu().float())
        4. return torch.cat(all_embs, dim=0)

    Why [CLS] not mean-pool:
        BERT [CLS] is the pretraining objective for sentence-level reps.
        Mean-pool over tokens is fine but loses ~0.005 cosine on short
        biomedical strings (verified empirically on PubMedBERT). For our
        use (cosine NN search), [CLS] is the safer default.

    Why .cpu().float() per batch:
        Reduces GPU memory pressure when this is run alongside TRIDENT.
        Final concat happens on cpu (cheap for ~1100 × 768 floats).

    Test:
        # Run with a small slice to verify shape:
        out = encode_pool(["hello", "world"], model, tokenizer, "cpu", 2)
        assert out.shape == (2, 768)
        assert out.dtype == torch.float32
        assert not out.isnan().any()
    """
    model.eval()
    all_embs: list[Tensor] = []
    with torch.no_grad():
        for i in range(0, len(strings), batch_size):
            batch = strings[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            all_embs.append(cls.cpu().float())
    return torch.cat(all_embs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: main - orchestrate, assert, save
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Build the Q + A embedding pools and save them under --out_dir.

    Extracts the unique question / non-final-answer vocabularies from train_CoT.json,
    encodes each with the selected biomedical encoder ([CLS] pooling), and saves
    Q_emb_pool.pt + A_emb_pool.pt. Full-data builds assert the known 92 / 405 counts;
    a --split-json build relaxes those to warnings.

    Runtime: ~5 min on cpu, ~30 sec on cuda.
    """
    parser = argparse.ArgumentParser(description="Pre-compute biomedical embedding pools.")
    parser.add_argument("--cot_path", type=Path, default=Path("/mnt/data/reg2026/train_CoT.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("/mnt/data/reg2026/embedding_pools"))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--encoder",
        type=str,
        default="medcpt",
        choices=["medcpt", "pubmedbert"],
        help="Which biomedical encoder. Default medcpt = ncbi/MedCPT-Query-Encoder (deployed; better antonym separation). pubmedbert = legacy.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="if set, build the pools from this split's TRAIN ids only. Relaxes the full-data count asserts to warnings.",
    )
    parser.add_argument(
        "--split-keys",
        type=str,
        default="train",
        help="Comma-separated split keys to include (default 'train'; use 'train,val' for the final shipped pools).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_id = MEDCPT_MODEL_ID if args.encoder == "medcpt" else PUBMEDBERT_MODEL_ID
    logger.info("Using encoder: %s (--encoder=%s)", model_id, args.encoder)

    # Optional train-only (or train+val) filtering: the cosine-NN candidate
    # pools are built from these ids only.
    allowed_ids: set[str] | None = None
    if args.split_json is not None:
        split = json.loads(args.split_json.read_text())
        keys = [k.strip() for k in args.split_keys.split(",") if k.strip()]
        allowed_ids = {sid for k in keys for sid in split.get(k, [])}
        logger.info("Pool filtering ON: split=%s keys=%s -> %d allowed slide ids", args.split_json, keys, len(allowed_ids))

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(args.device)

    Q_VOCAB = extract_questions(args.cot_path, allowed_ids)
    A_VOCAB = extract_answers(args.cot_path, allowed_ids)
    if allowed_ids is None:
        # Full-data build: enforce the known counts as integrity checks.
        assert len(Q_VOCAB) == EXPECTED_Q_COUNT, f"Q_VOCAB {len(Q_VOCAB)} != {EXPECTED_Q_COUNT}"
        assert len(A_VOCAB) == EXPECTED_A_COUNT, f"A_VOCAB {len(A_VOCAB)} != {EXPECTED_A_COUNT}"
    else:
        # Filtered build: counts legitimately differ - log, don't assert.
        logger.info(
            "Filtered pool sizes: Q=%d (full %d), A=%d (full %d)",
            len(Q_VOCAB),
            EXPECTED_Q_COUNT,
            len(A_VOCAB),
            EXPECTED_A_COUNT,
        )

    Q_emb = encode_pool(Q_VOCAB, model, tok, args.device, args.batch_size, QA_MAX_LENGTH)
    A_emb = encode_pool(A_VOCAB, model, tok, args.device, args.batch_size, QA_MAX_LENGTH)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"emb": Q_emb, "vocab": Q_VOCAB}, args.out_dir / "Q_emb_pool.pt")
    logger.info(f"Saved Q_emb_pool.pt: shape {Q_emb.shape}, {len(Q_VOCAB)} strings")
    torch.save({"emb": A_emb, "vocab": A_VOCAB}, args.out_dir / "A_emb_pool.pt")
    logger.info(f"Saved A_emb_pool.pt: shape {A_emb.shape}, {len(A_VOCAB)} strings")


if __name__ == "__main__":
    main()
