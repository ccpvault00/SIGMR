"""Module 2 - Trajectory Transformer (text-only DAG reasoning).

Small encoder-only Transformer that takes a sequence of (Q, A) tokens and
predicts a multilabel distribution over the 92 Q vocabulary (next-Qs to emit
from the LAST visited Q in the history).

Sequence layout:
  [BOS, Q_0, A_0, Q_1, A_1, ..., Q_t, A_t]   ← input
  output: multilabel logits over 92 Qs (children of Q_t)

Training target: BCE with pos_weight against the set of next_Q indices observed
in GT for this (Q_t, history) state.

Sizing:
  - vocab_size 502 → embedding ~64-128 dim sufficient
  - 4 layers, 4 heads, d_model=128, ff_dim=256 → ~600k params
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrajectoryModelConfig:
    vocab_size: int = 502
    n_q_vocab: int = 92  # output dim (multilabel)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_dim: int = 256
    max_seq_len: int = 64  # 32 turns × 2 tokens + BOS = 65 (round to 64 for safety)
    dropout: float = 0.1
    learn_q_threshold: bool = False  # if True, add per-Q learned threshold (subtracted from logits)
    rel_pos: bool = False  # REVERSE position (distance from current/last token) - relative CoT distance, not absolute index
    attn_pool: bool = False  # learned-query attention pool over full history (added to last-position; not just last-pos bottleneck)


class TrajectoryTransformer(nn.Module):
    """Encoder-only Transformer for DAG-trajectory next-Q prediction."""

    def __init__(self, config: TrajectoryModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or TrajectoryModelConfig()
        c = self.config

        self.token_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(c.max_seq_len, c.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.ff_dim,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.n_layers)
        self.norm = nn.LayerNorm(c.d_model)

        # learned-query attention pool over the full (Q,A) history.
        if c.attn_pool:
            self.pool_query = nn.Parameter(torch.randn(c.d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(c.d_model, c.n_heads, dropout=c.dropout, batch_first=True)

        # Output head: multilabel over 92 Q vocab
        self.next_q_head = nn.Linear(c.d_model, c.n_q_vocab)

        # Optional: learned per-Q logit-space threshold (decision boundary).
        # Subtracted from logits before sigmoid; trained via BCE gradient.
        # Equivalent to learning the optimal threshold per Q automatically;
        # tackles M2 over-emission on low-precision Qs (e.g. Nottingham subscores).
        if c.learn_q_threshold:
            self.q_threshold = nn.Parameter(torch.zeros(c.n_q_vocab))
        else:
            self.register_parameter("q_threshold", None)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Args:
            tokens: [B, L] long token ids (padded with PAD_ID=0)
            attention_mask: [B, L] bool - True for valid positions, False for PAD.
                If None, treat all positions as valid (assumes no PAD).

        Returns:
            next_q_logits: [B, n_q_vocab] - multilabel logits for next Q from
                          the LAST non-PAD position.
        """
        B, L = tokens.shape
        device = tokens.device

        valid = attention_mask if attention_mask is not None else (tokens != 0)
        seq_lens = (valid.sum(dim=1) - 1).clamp(min=0)  # last non-PAD index per sample
        if self.config.rel_pos:
            # REVERSE position - distance from the current (last non-PAD) token. Current=0,
            # immediate parent=1, ... -> relative CoT distance, length-general, not absolute index.
            positions = (seq_lens.unsqueeze(1) - torch.arange(L, device=device).unsqueeze(0)).clamp(0, self.config.max_seq_len - 1)
        else:
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(tokens) + self.pos_emb(positions)

        # PyTorch nn.TransformerEncoder expects src_key_padding_mask: True=PAD
        src_key_padding_mask = ~valid
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        batch_idx = torch.arange(B, device=device)
        last_hidden = x[batch_idx, seq_lens]  # [B, d_model]
        if self.config.attn_pool:
            # aggregate the full history via a learned query (not just the last position).
            q = self.pool_query.view(1, 1, -1).expand(B, 1, -1)
            pooled, _ = self.pool_attn(q, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)  # [B,1,d]
            last_hidden = last_hidden + pooled.squeeze(1)

        logits = self.next_q_head(last_hidden)  # [B, n_q_vocab]
        if self.q_threshold is not None:
            logits = logits - self.q_threshold  # per-Q learned decision boundary
        return logits
