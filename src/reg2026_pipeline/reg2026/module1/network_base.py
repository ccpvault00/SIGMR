"""Module 1 base network: multi-turn text self-attention + abstract WSI hook.

Architecture:
- 4-layer TransformerEncoder with Pre-LN, d_model=256, 4 heads, ffn=512, dropout=0.2
- Frozen embedding pools for q_emb / a_emb lookup - registered as buffers
- Type embedding (q vs a) + turn-idx positional embedding
- Full teacher forcing
- ABMIL aggregator + AuxHeads dependency-injected (constructor takes instances)
- WSI integration is abstract - variants override forward_wsi()
  - A: Q-Former 8 learnable queries × cross-attn to all tiles
  - B: top-K saliency-selected tiles → join text in self-attn
- Reserved regression head interface (closed by default)
- Mask convention: nn.TransformerEncoder expects True=mask out; we invert our
  DAGBatch.attn_mask (True=can attend) on the fly
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from reg2026.data.embedding_pools import EmbeddingPools
from reg2026.module1.dag_dataset import DAGBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleOneConfig:
    """Frozen config for ModuleOneBase + variants."""

    n_q_vocab: int = 92
    n_a_vocab: int = 405
    q_emb_dim: int = 768  # MedCPT
    a_emb_dim: int = 768  # biomedical embedder
    tile_emb_dim: int = 1536  # H-Optimus-1
    slide_emb_dim: int = 1024  # ABMIL output
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.2
    max_N: int = 30  # max collapsed turns per case (must fit turn_emb table)
    organ_aux_weight: float = 0.3
    regression_head_open: bool = False  # reserved interface, closed by default
    max_occ: int = 0  # if > 0, add additive occ_emb to Q+A tokens per occurrence (multi-dx ceiling work). 0 = no occ_emb (backward compat).


class ModuleOneBase(nn.Module):
    """Base class for Module 1. Variants override `forward_wsi`."""

    def __init__(
        self,
        config: ModuleOneConfig,
        pools: EmbeddingPools,
        abmil: nn.Module,
        aux_heads: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        # dependency-injected slide aggregator + aux heads
        self.abmil = abmil
        self.aux_heads = aux_heads

        # frozen embedding pools as buffers (move with .to(device))
        self.register_buffer("q_emb_pool", pools.Q_EMB.float().clone(), persistent=False)
        self.register_buffer("a_emb_pool", pools.A_EMB.float().clone(), persistent=False)

        # Projections to d_model
        self.q_proj = nn.Linear(config.q_emb_dim, config.d_model)
        self.a_proj = nn.Linear(config.a_emb_dim, config.d_model)

        # type + turn positional embeddings
        self.type_emb = nn.Embedding(2, config.d_model)  # 0=q, 1=a
        self.turn_emb = nn.Embedding(config.max_N, config.d_model)

        # per-occurrence embedding (multi-dx ceiling).
        # When max_occ > 0, each Q+A token gets an additive embedding based on
        # which occurrence (0/1/2/...) of that Q it represents in the trajectory.
        # Allows model to distinguish e.g., invasion=Yes (occ=0, neoplasm side)
        # vs invasion=No (occ=1, lesion side) for same slide.
        # max_occ=0 keeps backward compat (no occ_emb).
        self.occ_emb: nn.Embedding | None = nn.Embedding(config.max_occ, config.d_model) if config.max_occ > 0 else None

        # 4-layer Pre-LN TransformerEncoder, d_model=256, 4 heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # A predict MLP: [q_context, visual_context] → a_emb [768]
        self.a_predict_mlp = nn.Sequential(
            nn.Linear(config.d_model * 2, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.a_emb_dim),
        )

        # M1-cls: classification head over A_VOCAB for closed-set Qs. ADDITIVE - the cosine
        # a_predict_mlp stays (trained on all Qs, drives the history/scheduled-sampling a_emb);
        # cls_head adds discriminative closed-Q answers (balanced CE, masked per-Q to the legal
        # answer set at loss + inference). Replaces the cosine-collapse on fine-grained answers.
        self.cls_head = nn.Linear(config.d_model * 2, config.n_a_vocab)

        # regression head reserved interface; closed by default
        self.regression_head: nn.Linear | None = None
        if config.regression_head_open:
            self.regression_head = nn.Linear(config.d_model * 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward_wsi(
        self,
        tile_features: Tensor,
        tile_pad_mask: Tensor,
        q_context_seq: Tensor,
        qq_dag_mask: Tensor | None = None,
    ) -> Tensor:
        """Override in A / B.

        Args:
            tile_features:  [B, T_max_tiles, tile_emb_dim]
            tile_pad_mask:  [B, T_max_tiles] bool, True = real tile
            q_context_seq:  [B, N, d_model] - text-side q output per turn
            qq_dag_mask:    [B, N, N] bool - True if j is ancestor of i (incl self).
                            Optional; A ignores (Q-Former is N-immune by design).
                            B requires (DAG mask for wsi_transformer self-attn).

        Returns:
            visual_context_seq: [B, N, d_model]
        """
        raise NotImplementedError("Subclasses (e.g. ModuleOneBQCond) must implement forward_wsi.")

    def _build_text_sequence(self, q_indices: Tensor, a_indices_gt: Tensor, occ_indices: Tensor | None = None, a_emb_override: Tensor | None = None) -> Tensor:
        """Build [B, 2N, d_model] sequence: [q_0, a_0, q_1, a_1, ..., q_{N-1}, a_{N-1}].

        Type and turn embeddings added per token.
        occ_indices [B, N] long: per-position occurrence index for occ_emb (if enabled).
        a_emb_override [B, N, a_emb_dim]: scheduled-sampling support - if set, used in
            place of the A_pool lookup (mix of predicted/GT history A embeddings).
            Ported from module1-qformer for consolidation (no new params; ckpt-safe).
        """
        B, N = q_indices.shape
        # frozen lookups. clamp(-1 → 0) is safe because loss_mask hides those positions.
        q_emb = self.q_emb_pool[q_indices.clamp(min=0)]  # [B, N, q_emb_dim]
        a_emb = a_emb_override if a_emb_override is not None else self.a_emb_pool[a_indices_gt.clamp(min=0)]  # [B, N, a_emb_dim]
        q_tokens = self.q_proj(q_emb)  # [B, N, d_model]
        a_tokens = self.a_proj(a_emb)  # [B, N, d_model]

        # add type + turn embeddings
        turn_pos = self.turn_emb.weight[:N]  # [N, d_model]
        q_tokens = q_tokens + self.type_emb.weight[0] + turn_pos
        a_tokens = a_tokens + self.type_emb.weight[1] + turn_pos

        # per-occurrence additive embedding (Q and A tokens of same turn share occ_idx).
        if self.occ_emb is not None and occ_indices is not None:
            occ_clamped = occ_indices.clamp(min=0, max=self.config.max_occ - 1)  # safe for unset/-1
            occ_e = self.occ_emb(occ_clamped)  # [B, N, d_model]
            q_tokens = q_tokens + occ_e
            a_tokens = a_tokens + occ_e

        # Interleave [q_0, a_0, q_1, a_1, ...]
        seq = torch.stack([q_tokens, a_tokens], dim=2).reshape(B, 2 * N, -1)
        return seq

    def _expand_attn_mask(self, attn_mask: Tensor) -> Tensor:
        """Convert DAG mask (True=can attend) to PyTorch mask (True=mask out)
        and expand per-head for nn.TransformerEncoder.

        Input:  [B, 2N, 2N] bool - True means CAN attend
        Output: [B * n_heads, 2N, 2N] bool - True means MASK OUT (cannot attend)
        """
        pt_mask = ~attn_mask  # invert
        return pt_mask.repeat_interleave(self.config.n_heads, dim=0)

    def forward(self, batch: DAGBatch, a_emb_override: Tensor | None = None) -> dict[str, Tensor]:
        """Multi-turn parallel forward pass (teacher-forced).

        a_emb_override [B, N, a_emb_dim]: scheduled-sampling support (2-pass forward) -
            if set, replaces the A_pool lookup of history A embeddings.

        Returns:
            a_emb_pred:    [B, N, a_emb_dim]  - predicted A embedding per turn
            organ_logits:  [B, n_organ]       - slide-level organ aux prediction
            (optional) regression_pred: [B, N] - if regression_head_open
        """
        B, N = batch.q_indices.shape

        # 1. ABMIL aggregator → slide_emb (per-tile p_lesional optional; use uniform 1.0)
        # The exact ABMIL forward signature is implementation-dependent; we trust the
        # injected aggregator to consume (tile_features, tile_pad_mask) and produce
        # (slide_emb, attn_weights). Variant-specific ABMIL usage stays in subclasses.
        slide_emb, _ = self.abmil(batch.tile_features, mask=batch.tile_pad_mask)  # [B, slide_emb_dim], _

        # 2. Aux: organ prediction (slide-level)
        # AuxHeads.forward(slide_emb) returns dict of categorical head logits.
        aux_out = self.aux_heads(slide_emb)
        organ_logits = aux_out.get("organ")

        # 3. Build interleaved [q_0, a_0, q_1, a_1, ...] sequence
        occ_indices = getattr(batch, "occ_indices", None)
        seq = self._build_text_sequence(batch.q_indices, batch.a_indices_gt, occ_indices=occ_indices, a_emb_override=a_emb_override)  # [B, 2N, d_model]

        # 4. Text self-attn with DAG mask (invert + expand per-head)
        pt_mask = self._expand_attn_mask(batch.attn_mask)  # [B*H, 2N, 2N]
        text_repr = self.transformer(seq, mask=pt_mask)  # [B, 2N, d_model]

        # 5. Extract q positions (even indices) → q_context per turn
        q_context_seq = text_repr[:, 0::2, :]  # [B, N, d_model]

        # 6. WSI integration (variant-specific)
        # Extract q-q DAG mask: every other position (0, 2, 4, ...) of attn_mask
        # gives [B, N, N] showing which q positions can attend to which.
        qq_dag_mask = batch.attn_mask[:, 0::2, 0::2]  # [B, N, N]
        visual_context_seq = self.forward_wsi(batch.tile_features, batch.tile_pad_mask, q_context_seq, qq_dag_mask)  # [B, N, d_model]

        # 7. Predict a_emb from [q_context, visual_context]
        combined = torch.cat([q_context_seq, visual_context_seq], dim=-1)  # [B, N, 2*d_model]
        a_emb_pred = self.a_predict_mlp(combined)  # [B, N, a_emb_dim]

        out: dict[str, Tensor] = {
            "a_emb_pred": a_emb_pred,
            "cls_logits": self.cls_head(combined),  # [B, N, n_a_vocab] - M1-cls closed-Q classifier (raw; masked per-Q in loss/inference)
            "organ_logits": organ_logits,
        }
        if self.regression_head is not None:
            out["regression_pred"] = self.regression_head(combined).squeeze(-1)  # [B, N]
        return out
