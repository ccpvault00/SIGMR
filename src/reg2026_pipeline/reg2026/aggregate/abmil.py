"""Multi-head non-gated ABMIL aggregator.
Pools tile features [B, N, F] into slide embeddings [B, H*D] via attention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ABMILConfig:
    """Frozen config for ABMIL aggregator.

    Attributes:
        in_dim: tile feature dim. Default 1536 = H-Optimus-1 output.
        embed_dim: per-head latent dim D. Default 256.
        num_heads: attention heads H. Default 4.
        encoder_layers: depth of pre-attention MLP. Default 3.
        dropout: applied to concat slide embedding. Default 0.5.
        use_batchnorm: BN after each encoder Linear. Default True.
        attention_latent: Tanh attention bottleneck dim. None → (embed_dim+1)//2.
    """

    in_dim: int = 1536
    embed_dim: int = 256
    num_heads: int = 4
    encoder_layers: int = 3
    dropout: float = 0.5
    use_batchnorm: bool = True
    attention_latent: int | None = None

    @property
    def output_dim(self) -> int:
        """Slide embedding dim after concat = num_heads × embed_dim."""
        return self.num_heads * self.embed_dim

    @property
    def resolved_attention_latent(self) -> int:
        """Default attention bottleneck = (embed_dim + 1) // 2."""
        return self.attention_latent or (self.embed_dim + 1) // 2


class ABMIL(nn.Module):
    """Multi-head non-gated ABMIL aggregator.

    Input:
        bags: Tensor [B, N, in_dim] - N tile features per slide.
        mask: Tensor [B, N] bool - True = valid tile, False = padding (or None).

    Output:
        slide_emb: Tensor [B, num_heads * embed_dim]
        attn: Tensor [B, num_heads, N] - post-softmax attention scores.

    Architecture:
        encoder: Linear(F→D) → LeakyReLU → BN, repeated `encoder_layers` times
        per-head attention: Linear(D → D/2) → Tanh → Linear(D/2 → 1)
        masked softmax over N (fp16-safe: mask value -1e4, not -1e8)
        weighted sum per head → stack → reshape → dropout
    """

    def __init__(self, config: ABMILConfig | None = None) -> None:
        super().__init__()
        self.config = config or ABMILConfig()
        c = self.config
        self.encoder = nn.Sequential(
            nn.Linear(c.in_dim, c.embed_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(c.embed_dim) if c.use_batchnorm else nn.Identity(),
            *[
                nn.Sequential(
                    nn.Linear(c.embed_dim, c.embed_dim),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(c.embed_dim) if c.use_batchnorm else nn.Identity(),
                )
                for _ in range(c.encoder_layers - 1)
            ],
        )

        self.attention_heads = nn.ModuleList([_make_attention_head(c.embed_dim, c.resolved_attention_latent) for _ in range(c.num_heads)])
        self.dropout = nn.Dropout(c.dropout)

        self._initialize_weights()

    def forward(
        self,
        bags: Tensor,
        mask: Tensor | None = None,
        p_lesional: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Pool tile bags into slide embeddings (optionally gated by P(lesional)).

        Args:
            bags: [B, N, in_dim] tile features.
            mask: [B, N] bool mask. True = valid, False = padding.
                  If None, all tiles are treated as valid.
            p_lesional: [B, N] float in [0, 1]

        Returns:
            slide_emb: [B, num_heads * embed_dim]
            attn: [B, num_heads, N] post-softmax attention scores (post-gating
                if p_lesional given).
        """
        if bags.dim() != 3 or bags.shape[2] != self.config.in_dim:
            raise ValueError(f"Expected [B, N, {self.config.in_dim}], got {tuple(bags.shape)}")
        B, N, F = bags.shape

        if mask is None:
            mask = torch.ones(bags.shape[:2], dtype=torch.bool, device=bags.device)
        elif mask.shape != bags.shape[:2] or mask.dtype != torch.bool:
            raise ValueError(f"mask must be bool [B, N], got dtype={mask.dtype} shape={tuple(mask.shape)}")

        if p_lesional is not None and p_lesional.shape != bags.shape[:2]:
            raise ValueError(f"p_lesional must be [B, N] = {tuple(bags.shape[:2])}, got {tuple(p_lesional.shape)}")
        flat = bags.reshape(-1, F)
        embeddings = self.encoder(flat)
        embeddings = embeddings.reshape(B, N, -1)

        per_head_sums: list[Tensor] = []
        per_head_attn: list[Tensor] = []

        for head in self.attention_heads:
            raw_scores = head(embeddings)
            if p_lesional is not None:
                raw_scores = raw_scores * p_lesional.unsqueeze(-1)  # [B, N, 1]
            attn_weights = self._masked_softmax(raw_scores, mask)
            weighted_sum = (attn_weights * embeddings).sum(dim=1)

            per_head_sums.append(weighted_sum)
            per_head_attn.append(attn_weights.squeeze(-1))

        stacked = torch.stack(per_head_sums, dim=1)
        slide_emb = stacked.reshape(B, -1)
        slide_emb = self.dropout(slide_emb)
        attn = torch.stack(per_head_attn, dim=1)

        return slide_emb, attn

    @staticmethod
    def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
        """fp16-safe masked softmax over dim 1.

        Args:
            scores: [B, N, 1] raw attention scores.
            mask: [B, N] bool. True = keep, False = mask out.

        Returns:
            [B, N, 1] post-softmax weights summing to 1 along dim 1
            (over only the valid entries).
        """
        mask_3d = mask.unsqueeze(-1)
        mask_value = -1e4 if scores.dtype == torch.float16 else -1e8
        masked = torch.where(mask_3d, scores, torch.full_like(scores, mask_value))
        return torch.softmax(masked, dim=1)

    def _initialize_weights(self) -> None:
        """Xavier normal on Linear weights, zero on bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def _make_attention_head(embed_dim: int, attention_latent: int) -> nn.Sequential:
    """Single non-gated attention head: produces [B, N, 1] raw scores."""
    return nn.Sequential(
        nn.Linear(embed_dim, attention_latent),
        nn.Tanh(),
        nn.Linear(attention_latent, 1),
    )
