"""B-QCond: Q-conditional cross-attention WSI integration.

Replaces B's Q-agnostic saliency + top-K self-attn with a Perceiver / Q-Former
style cross-attention where q_context_seq queries ALL tile features. Each Q
position attends to tiles via learned attention (no fixed top-K, no fixed
learnable queries - query is the Q-text-conditioned context from the text
transformer).

Architecture:
    visual_context[i] = CrossAttn(query=q_context[i], key/value=all_tiles)
    Stacked n_layers, each with FFN.

N-immunity: each q attends only to tiles (no q-q via this path; q-q via text
transformer's DAG mask). Tiles do not attend back to q. So per-position output
depends only on q_context[i] (which itself depends only on strict ancestors in
text DAG) + the fixed tile set → N-immune.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from reg2026.data.embedding_pools import EmbeddingPools
from reg2026.module1.network_base import ModuleOneConfig, ModuleOneBase


class ModuleOneBQCond(ModuleOneBase):
    """B-QCond: Q-conditional cross-attention (Perceiver-style) for WSI integration."""

    def __init__(
        self,
        config: ModuleOneConfig,
        pools: EmbeddingPools,
        abmil: nn.Module,
        aux_heads: nn.Module,
        n_wsi_layers: int = 2,
    ) -> None:
        super().__init__(config, pools, abmil, aux_heads)
        self.n_wsi_layers = n_wsi_layers

        # Project tiles into d_model space
        self.tile_proj = nn.Linear(config.tile_emb_dim, config.d_model)
        self.tile_norm = nn.LayerNorm(config.d_model)

        # Cross-attention layers (q queries tiles)
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(config.d_model, num_heads=config.n_heads, dropout=config.dropout, batch_first=True) for _ in range(n_wsi_layers)])
        self.cross_ln_q = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(n_wsi_layers)])
        self.cross_ln_kv = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(n_wsi_layers)])
        self.cross_ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(config.d_model, config.ffn_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.ffn_dim, config.d_model),
                )
                for _ in range(n_wsi_layers)
            ]
        )

        self._init_weights()

    def forward_wsi(
        self,
        tile_features: Tensor,
        tile_pad_mask: Tensor,
        q_context_seq: Tensor,
        qq_dag_mask: Tensor | None = None,  # unused - q-q in text transformer
    ) -> Tensor:
        """Cross-attention: each q queries all tiles.

        Args:
            tile_features: [B, N_tiles, tile_emb_dim]
            tile_pad_mask: [B, N_tiles] True = real tile, False = pad
            q_context_seq: [B, N, d_model] - text-side q output per turn
            qq_dag_mask: unused (DAG enforced in text transformer; not needed here)

        Returns:
            visual_context_seq: [B, N, d_model]
        """
        # Project tiles
        tile_repr = self.tile_norm(self.tile_proj(tile_features))  # [B, N_tiles, d_model]
        # key_padding_mask: True = ignore (padding)
        key_pad = ~tile_pad_mask

        q = q_context_seq
        for attn, ln_q, ln_kv, ffn in zip(self.cross_attn, self.cross_ln_q, self.cross_ln_kv, self.cross_ffn, strict=True):
            q_n = ln_q(q)
            kv_n = ln_kv(tile_repr)
            attn_out, _ = attn(q_n, kv_n, kv_n, key_padding_mask=key_pad)
            q = q + attn_out
            q = q + ffn(q)
        return q  # [B, N, d_model]
