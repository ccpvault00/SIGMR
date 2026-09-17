"""module1 per-organ MoE: LoRA wrappers for A/B.

Minimal LoRA implementation tailored to module1 architecture (avoids peft's
auto-wrapping complexity with nn.MultiheadAttention internals).

Approach:
- Wrap target nn.Linear modules with LoRALinear (base frozen + low-rank adapter)
- Keep AuxHeads + a_predict_mlp head fully trainable (small, organ-specific decoder)
- Freeze ABMIL + position embeddings + frozen embedding buffers

Trainable per organ (rank=16):
- LoRA on ~6-10 Linear modules ≈ 100-500 K params
- + AuxHeads + a_predict_mlp ≈ 200 K params
- Total ≈ 0.3-0.7 M (vs 4.76 M full fine-tune) → 10× less overfit risk
"""

from __future__ import annotations

import logging
import math

from torch import Tensor, nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Frozen base nn.Linear + trainable low-rank adapter (LoRA, Hu et al. 2021).

    out = base(x) + scale * lora_B(lora_A(x))
    """

    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # Properties so nn.TransformerEncoderLayer's attribute checks pass.
    # PyTorch's TransformerEncoderLayer accesses self.linear1.weight unconditionally
    # for fast-path dtype check, even when norm_first=True forces slow path.
    # Slow path goes through forward() and applies LoRA correctly.
    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.lora_B(self.dropout(self.lora_A(x)))


def _set_submodule(root: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace submodule at dotted path `name` with `new_module`."""
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


# Default LoRA target patterns. Match by suffix in module path.
# NOTE: skip out_proj/in_proj which are inside nn.MultiheadAttention - PyTorch's
# F.multi_head_attention_forward directly accesses .weight attribute, breaking
# LoRALinear wrapping. We LoRA-fine-tune FFN + standalone projections instead.
DEFAULT_TARGETS: tuple[str, ...] = (
    "q_proj",  # base projection 768->256 (top-level)
    "a_proj",  # base projection 768->256 (top-level)
    "tile_proj",  # WSI: 1536->256
    "linear1",  # FFN layer 1 in nn.TransformerEncoderLayer
    "linear2",  # FFN layer 2 in nn.TransformerEncoderLayer
    "saliency_proj",  # B saliency scorer
)


def add_lora_to_module1(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_suffixes: tuple[str, ...] = DEFAULT_TARGETS,
    keep_trainable_modules: tuple[str, ...] = ("aux_heads", "a_predict_mlp"),
) -> dict[str, int]:
    """Wrap target nn.Linear modules with LoRA + freeze everything else (except
    `keep_trainable_modules`).

    Returns: stats dict with trainable param counts.
    """
    # 1. Find target Linear modules
    to_wrap: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in target_suffixes):
            to_wrap.append((name, module))

    # 2. Replace with LoRALinear
    n_wrapped = 0
    for name, base in to_wrap:
        wrapped = LoRALinear(base, r=rank, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, wrapped)
        n_wrapped += 1

    # 3. Freeze everything except LoRA + keep_trainable_modules
    keep_set = set(keep_trainable_modules)
    for name, p in model.named_parameters():
        # LoRA params: param name contains "lora_A" or "lora_B"
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            continue
        # Keep_trainable_modules: any top-level module name in keep set
        top_level = name.split(".")[0]
        if top_level in keep_set:
            p.requires_grad = True
            continue
        # Otherwise freeze
        p.requires_grad = False

    # 4. Compute stats
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_total = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    lora_total = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n))
    head_total = trainable_total - lora_total
    stats = {
        "n_wrapped_linears": n_wrapped,
        "trainable_total": trainable_total,
        "frozen_total": frozen_total,
        "lora_params": lora_total,
        "head_params": head_total,
        "rank": rank,
        "alpha": alpha,
    }
    logger.info(
        "LoRA applied: %d Linear wrapped | LoRA params %d | head params %d | total trainable %d (%.2f%% of full)",
        n_wrapped,
        lora_total,
        head_total,
        trainable_total,
        100.0 * trainable_total / (trainable_total + frozen_total),
    )
    return stats
