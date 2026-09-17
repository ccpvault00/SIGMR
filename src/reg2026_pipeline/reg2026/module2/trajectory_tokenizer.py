"""Module 2 - Trajectory tokenizer.

Vocab construction:
  0-4   : 5 special tokens (PAD, BOS, EOS, NO_EDGE, UNK)
  5-96  : 92 Q tokens (indexed by Q_VOCAB position + 5)
  97-501: 405 A tokens (indexed by A_VOCAB position + 97)

Total vocab size: 502 tokens.

A trajectory (Q_0, A_0), (Q_1, A_1), ..., (Q_T, A_T) is encoded as:
  [BOS, Q_token_0, A_token_0, Q_token_1, A_token_1, ..., Q_token_T, A_token_T, EOS]

The next_Q target is encoded separately as a multilabel vector over the 92-Q output
space (i.e., raw Q_VOCAB indices, not token IDs).
"""

from __future__ import annotations

from dataclasses import dataclass

from reg2026.data.embedding_pools import EmbeddingPools

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
NO_EDGE_ID = 3
UNK_ID = 4
N_SPECIAL = 5

Q_TOKEN_OFFSET = N_SPECIAL  # 5
A_TOKEN_OFFSET = N_SPECIAL + 92  # 97


@dataclass
class TrajectoryTokenizer:
    """Tokenize (Q, A) pairs to ids using EmbeddingPools Q/A vocabularies."""

    pools: EmbeddingPools

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + len(self.pools.Q_VOCAB) + len(self.pools.A_VOCAB)

    @property
    def n_q_vocab(self) -> int:
        return len(self.pools.Q_VOCAB)

    @property
    def n_a_vocab(self) -> int:
        return len(self.pools.A_VOCAB)

    def encode_q(self, q_text: str) -> int:
        """Q text → token id (UNK_ID if not in vocab)."""
        try:
            q_idx = self.pools.q_idx(q_text)
            return Q_TOKEN_OFFSET + q_idx
        except KeyError:
            return UNK_ID

    def encode_a(self, a_text: str) -> int:
        """A text → token id via exact match in A_VOCAB (UNK_ID otherwise).

        At inference, Module 1 outputs A_emb → nearest-neighbor → a_idx directly;
        we then map a_idx → A_TOKEN_OFFSET + a_idx without text lookup.
        """
        # Strict exact match first
        for idx, candidate in enumerate(self.pools.A_VOCAB):
            if candidate == a_text:
                return A_TOKEN_OFFSET + idx
        # Fallback: case-insensitive trim
        a_norm = (a_text or "").strip().lower()
        for idx, candidate in enumerate(self.pools.A_VOCAB):
            if candidate.strip().lower() == a_norm:
                return A_TOKEN_OFFSET + idx
        return UNK_ID

    def encode_a_idx(self, a_idx: int) -> int:
        """Direct A_VOCAB index → token id (used at inference after nn_search)."""
        if 0 <= a_idx < self.n_a_vocab:
            return A_TOKEN_OFFSET + a_idx
        return UNK_ID

    def encode_trajectory(self, history: list[tuple[str, str]], add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """history = [(q_text, a_text), ...] → list of token ids.

        Args:
            add_bos: prepend BOS_ID
            add_eos: append EOS_ID
        """
        tokens: list[int] = [BOS_ID] if add_bos else []
        for q, a in history:
            tokens.append(self.encode_q(q))
            tokens.append(self.encode_a(a))
        if add_eos:
            tokens.append(EOS_ID)
        return tokens

    def decode_token(self, token_id: int) -> str:
        """Token id → human-readable string (for debug)."""
        if token_id == PAD_ID:
            return "<PAD>"
        if token_id == BOS_ID:
            return "<BOS>"
        if token_id == EOS_ID:
            return "<EOS>"
        if token_id == NO_EDGE_ID:
            return "<NO_EDGE>"
        if token_id == UNK_ID:
            return "<UNK>"
        if Q_TOKEN_OFFSET <= token_id < A_TOKEN_OFFSET:
            return f"Q[{self.pools.Q_VOCAB[token_id - Q_TOKEN_OFFSET]}]"
        if A_TOKEN_OFFSET <= token_id < A_TOKEN_OFFSET + self.n_a_vocab:
            return f"A[{self.pools.A_VOCAB[token_id - A_TOKEN_OFFSET]}]"
        return f"<INVALID:{token_id}>"
