"""EmbeddingPools - runtime lookup service for the 2 MedCPT pools.

Loads Q_emb_pool.pt / A_emb_pool.pt (written by scripts/prep_embedding_pools.py).
These hold continuous MedCPT-Query-Encoder embeddings for cosine-NN question/answer lookup.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

DEFAULT_POOL_DIR: Path = Path(os.environ.get("REG_POOL_DIR", "/mnt/data/reg2026/embedding_pools"))

# State-machine sentinels for the multi-diagnosis deterministic loop.
DX_COUNT_QUESTION: str = "What is the number of diagnoses to includes?"
DX_COUNT_ANCHORS: tuple[str, ...] = ("1", "2", "3", "4")

EXPECTED_EMB_DIM: int = 768
EXPECTED_Q_COUNT: int = 92
EXPECTED_A_COUNT: int = 405

Q_ALIAS_MAP: dict[str, str] = {
    "What is histologic type of lesion?": "What is the histologic type of lesion?",
    "Is there any abnormality present": "Is there any abnormality present?",
}


def canonicalize_q(q_text: str) -> str:
    """Normalize Q text by collapsing known typo variants to canonical form."""
    return Q_ALIAS_MAP.get(q_text, q_text)


class EmbeddingPools:
    """Eager-loaded 2-pool service for inference.

    Attributes set by __init__:
        Q_VOCAB, Q_EMB              : 92 × 768
        A_VOCAB, A_EMB              : 405 × 768
    """

    def __init__(self, pool_dir: Path = DEFAULT_POOL_DIR) -> None:
        """Load the Q + A pools, build reverse dicts, and sanity-assert shapes."""
        q_payload = torch.load(pool_dir / "Q_emb_pool.pt", weights_only=False)
        a_payload = torch.load(pool_dir / "A_emb_pool.pt", weights_only=False)

        self.Q_VOCAB = q_payload["vocab"]
        self.Q_EMB = q_payload["emb"]
        self.A_VOCAB = a_payload["vocab"]
        self.A_EMB = a_payload["emb"]

        self._q_to_idx = {q: i for i, q in enumerate(self.Q_VOCAB)}
        self._a_to_idx = {a: i for i, a in enumerate(self.A_VOCAB)}

        _ALLOWED_EMB_DIMS = (EXPECTED_EMB_DIM, 1024)
        assert self.Q_EMB.shape[1] in _ALLOWED_EMB_DIMS, f"Q emb dim {self.Q_EMB.shape[1]} not in {_ALLOWED_EMB_DIMS}"
        assert self.A_EMB.shape[1] in _ALLOWED_EMB_DIMS, f"A emb dim {self.A_EMB.shape[1]} not in {_ALLOWED_EMB_DIMS}"
        assert self.Q_EMB.shape[0] == len(self.Q_VOCAB), "Q emb/vocab length mismatch"
        assert self.A_EMB.shape[0] == len(self.A_VOCAB), "A emb/vocab length mismatch"
        assert all(s in self._a_to_idx for s in DX_COUNT_ANCHORS), "dx-count anchors missing from A_VOCAB"

    def q_idx(self, q_text: str) -> int:
        """Q text → Q_VOCAB index. Raises KeyError if unknown."""
        q_text = canonicalize_q(q_text)
        try:
            return self._q_to_idx[q_text]
        except KeyError:
            raise KeyError(f"Q not in vocab: {q_text!r} (vocab size {len(self.Q_VOCAB)})") from None

    def a_text(self, idx: int) -> str:
        """A_VOCAB[idx] with bounds check."""
        if not 0 <= idx < len(self.A_VOCAB):
            raise IndexError(f"a_text idx {idx} out of range [0, {len(self.A_VOCAB)}]")
        return self.A_VOCAB[idx]

    def dx_count_anchor_indices(self) -> list[int]:
        """A_VOCAB indices for DX_COUNT_ANCHORS, used by the multi-diagnosis state machine."""
        return [self._a_to_idx[s] for s in DX_COUNT_ANCHORS]

    def __repr__(self) -> str:
        try:
            return f"EmbeddingPools(Q={len(self.Q_VOCAB)}, A={len(self.A_VOCAB)})"
        except AttributeError:
            return "EmbeddingPools(<not yet loaded>)"
