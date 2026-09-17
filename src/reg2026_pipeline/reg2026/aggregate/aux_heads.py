"""Auxiliary multi-task prediction heads.

Two output paths coexist:

1. Slide-level: `forward(slide_emb)` - categorical heads (organ, scanner, …)
   + optional `dx_prior` multi-label BCE. Input: slide_emb [B, in_dim] from
   ABMIL output (default in_dim=1024).

2. Tile-level: `forward_tile(tile_features)` - a tissue_severity 3-class head
   (non-tissue / non-lesional / lesional). Architecture: Linear(tile_in_dim,
   hidden) → GELU → Linear(hidden, 3), ~0.4M params with defaults (1536→256→3).
   Input: tile_features [B, N, tile_in_dim] from H-Optimus directly
   (default tile_in_dim=1536; N=many for a WSI, N=1 for a single ROI).

The 3-class head trains end-to-end from the classification losses (it gates
ABMIL attention via P(lesional)) - no pseudo-labeling, no separate tissue GT.

Head set: site / extraction_method / histtype / grade / scanner / organ
(categorical, CE loss) + dx_prior (multi-label BCE DDx prior). All slide-level
heads share the same input slide_emb [B, in_dim]; each is a single Linear.
tissue_severity[..., 2] (P(lesional)) gates the ABMIL aggregator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuxHeadsConfig:
    """Frozen config for AuxHeads.

    Attributes:
        in_dim: slide embedding dim. Default 1024 (ABMIL 4-head × 256 output).
        categorical_heads: mapping head name → num_classes for single-label CE
            classification. Example: {"organ": 7, "scanner": 6}. num_classes
            for some heads (histtype, grade, scanner, site, extraction_method)
            depends on the REG2026 label set.
        dx_num_classes: number of diagnosis classes for `dx_prior` multi-label
            BCE head. None = no dx_prior head built (toggleable).
        dropout: applied between input and each head's Linear.

    Tile-level tissue_severity head:
        tile_in_dim: per-tile feature dim from H-Optimus. Default 1536.
            Independent of `in_dim` since the slide-level path goes through
            ABMIL (1536 → 1024) but the tile-level path uses raw H-Optimus output.
        tissue_severity_num_classes: number of classes. None (default) = head
            not built (saves params + memory). 3 = enable the 3-class head
            (non-tissue / non-lesional / lesional).
        tissue_severity_hidden_dim: hidden dim of the 2-layer MLP. Default 256
            (Linear 1536→256 → GELU → Linear 256→3, ~0.4M params). Only used
            when tissue_severity_num_classes is set.
        tissue_severity_dropout: dropout before the tissue_severity MLP.
            Independent of `dropout` to allow per-head tuning if needed.
    """

    in_dim: int = 1024
    categorical_heads: dict[str, int] = field(default_factory=dict)
    dx_num_classes: int | None = None
    dropout: float = 0.1
    # tile-level head
    tile_in_dim: int = 1536
    tissue_severity_num_classes: int | None = None
    tissue_severity_hidden_dim: int = 256
    tissue_severity_dropout: float = 0.1


class AuxHeads(nn.Module):
    """Multi-task auxiliary heads from slide embedding.

    Input: slide_emb [B, in_dim] (typically ABMIL output [B, 1024]).
    Output: dict with one entry per head:
        - categorical heads: logits [B, num_classes]
        - "dx_prior" head (if dx_num_classes set): logits [B, dx_num_classes]
          (apply sigmoid externally for multi-label probability)

    Architecture: single Dropout shared across heads + per-head Linear.
    """

    def __init__(self, config: AuxHeadsConfig | None = None) -> None:
        super().__init__()
        self.config = config or AuxHeadsConfig()
        c = self.config

        # === slide-level path ===
        self.dropout = nn.Dropout(c.dropout)
        self.categorical_heads = nn.ModuleDict({name: nn.Linear(c.in_dim, num_classes) for name, num_classes in c.categorical_heads.items()})
        self.dx_prior_head: nn.Linear | None = nn.Linear(c.in_dim, c.dx_num_classes) if c.dx_num_classes is not None else None

        # === Tile-level path ===
        # 2-layer MLP: Linear → GELU → Linear. None when disabled.
        self.tissue_severity_head: nn.Sequential | None = (
            nn.Sequential(
                nn.Linear(c.tile_in_dim, c.tissue_severity_hidden_dim),
                nn.GELU(),
                nn.Linear(c.tissue_severity_hidden_dim, c.tissue_severity_num_classes),
            )
            if c.tissue_severity_num_classes is not None
            else None
        )
        # tile_dropout always built (cheap); only applied when head is enabled.
        self.tile_dropout = nn.Dropout(c.tissue_severity_dropout)

        self._initialize_weights()

    def forward(self, slide_emb: Tensor) -> dict[str, Tensor]:
        """Compute slide-level head logits.

        Args:
            slide_emb: [B, in_dim] slide-level embedding from ABMIL output.

        Returns:
            dict mapping head name → logits tensor.
              - categorical: [B, num_classes]
              - "dx_prior" (optional): [B, dx_num_classes]
        """
        if slide_emb.dim() != 2 or slide_emb.shape[1] != self.config.in_dim:
            raise ValueError(f"Expected [B, {self.config.in_dim}], got {tuple(slide_emb.shape)}")

        x = self.dropout(slide_emb)
        out: dict[str, Tensor] = {name: head(x) for name, head in self.categorical_heads.items()}
        if self.dx_prior_head is not None:
            out["dx_prior"] = self.dx_prior_head(x)
        return out

    def forward_tile(self, tile_features: Tensor) -> dict[str, Tensor]:
        """Compute tile-level head logits.

        Args:
            tile_features: [B, N, tile_in_dim] per-tile features from H-Optimus.
                N can be any positive integer:
                  - Mode A (WSI): N = number of tiles (~500-2000)
                  - Mode B (ROI): N = 1 (single ROI image)

        Returns:
            dict that may contain:
              - "tissue_severity": [B, N, num_classes] - only present if
                tissue_severity_head was built (i.e., config
                tissue_severity_num_classes is not None).
              - empty dict {} if no tile-level heads are enabled.

        Raises:
            ValueError: tile_features is not 3-D or last-dim mismatches tile_in_dim.
        """
        if tile_features.dim() != 3 or tile_features.shape[-1] != self.config.tile_in_dim:
            raise ValueError(f"Expected [B, N, {self.config.tile_in_dim}], got {tuple(tile_features.shape)}")
        if self.tissue_severity_head is None:
            return {}
        x = self.tile_dropout(tile_features)
        return {"tissue_severity": self.tissue_severity_head(x)}

    def _initialize_weights(self) -> None:
        """Xavier normal on Linear weights, zero on bias (NARWHAL convention).

        `self.modules()` walks all nn.Module descendants including the inner
        Linears of `tissue_severity_head` (nn.Sequential), so both layers of
        the 2-layer MLP get initialized.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
