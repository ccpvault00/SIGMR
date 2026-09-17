"""module1 trainer: loss, train/val loop, checkpointing.

Loss components (decided autonomously, alternatives recorded):
- A embedding cosine distance loss (per turn, masked by loss_mask)
  alt: MSE - but A_pool is on biomedical embedding manifold; cosine more natural
- Organ aux CE loss (slide-level), weight 0.3 from ModuleOneConfig
- Total = a_loss + organ_aux_weight * organ_loss

Metrics (val):
- A accuracy: nearest-neighbor in A_pool, exact match with GT a_idx
- Organ accuracy: argmax of organ_logits vs organ_idx
- Per-Q A accuracy: tracked for user's audit (organ / procedure / dx_count / other)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from reg2026.module1.dag_dataset import DAGBatch, DAGDataset, collate_dag
from reg2026.module1.network_base import ModuleOneBase

logger = logging.getLogger(__name__)


ORGAN_TO_IDX = {
    "bladder": 0,
    "breast": 1,
    "cervix": 2,
    "colon": 3,
    "lung": 4,
    "prostate": 5,
    "stomach": 6,
}


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 25
    batch_size: int = 8
    num_workers: int = 8
    grad_clip: float = 1.0
    log_every: int = 50
    save_every_epoch: bool = True
    organ_aux_weight: float = 0.3
    a_loss_weight: float = 1.0  # <1 down-weights cosine a_loss; 0 → pure-cls. Default 1.0 = unchanged.
    # Scheduled sampling (exposure bias fix). 2-pass forward when active:
    # (1) no-grad with GT a-emb to get predictions, (2) grad with prob-p mix
    # of predicted/GT a-emb. p linearly anneals 0 → sched_prob_max after warmup.
    # Ported from module1-qformer for consolidation (essential to SOTA M1).
    sched_sampling: bool = False
    sched_prob_max: float = 0.5
    sched_warmup_epochs: int = 2
    # M1-cls: additive classification head for closed-set Qs (balanced masked CE).
    # cls_loss_weight=0 → off (pure cosine, original behaviour). Tensors are device-resident.
    cls_loss_weight: float = 0.0
    m1v5_closed_by_q: Tensor | None = None  # [n_q_vocab] bool - Q is closed (→ classification)
    m1v5_legal_mask: Tensor | None = None  # [n_q_vocab, n_a_vocab] bool - per-Q legal answers
    m1v5_cls_weight: Tensor | None = None  # [n_q_vocab, n_a_vocab] float - per-(Q,a) balance weight
    m1v5_a_remap: Tensor | None = None  # [n_q_vocab, n_a_vocab] long - per-(Q) CE-target remap (pct: scattered % → nearest bucket; identity elsewhere)
    freeze_body_eval: bool = False  # run model in eval() during train (frozen-body cls-row fine-tune): BN/dropout fixed → body bit-identical to inference, zero buffer drift


def _sched_prob_at_epoch(epoch: int, num_epochs: int, prob_max: float, warmup: int) -> float:
    """Linear anneal 0 → prob_max after warmup epochs."""
    if epoch < warmup:
        return 0.0
    remaining = max(num_epochs - warmup - 1, 1)
    progress = min((epoch - warmup) / remaining, 1.0)
    return prob_max * progress


def cosine_a_loss(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """Cosine distance loss: 1 - cos(pred, gt), masked + averaged.

    Args:
        pred: [B, N, D] predicted A embeddings
        gt:   [B, N, D] GT A embeddings (already looked up from A_pool)
        mask: [B, N] bool - True = include in loss

    Returns:
        Scalar loss tensor.
    """
    pred_n = F.normalize(pred, dim=-1, eps=1e-8)
    gt_n = F.normalize(gt, dim=-1, eps=1e-8)
    cos = (pred_n * gt_n).sum(dim=-1)
    loss_per_turn = 1.0 - cos
    mask_f = mask.float()
    total = (loss_per_turn * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    return total


def organ_aux_loss_fn(logits: Tensor, organ_idx: Tensor) -> Tensor:
    """CE loss for organ aux head. Ignores organ_idx < 0 (unknown organ)."""
    valid = organ_idx >= 0
    if not valid.any():
        return torch.zeros((), device=logits.device, requires_grad=True)
    return F.cross_entropy(logits[valid], organ_idx[valid])


def compute_loss(out: dict[str, Tensor], batch: DAGBatch, a_emb_pool: Tensor, organ_aux_weight: float, cls: dict | None = None, a_loss_weight: float = 1.0) -> tuple[Tensor, dict[str, float]]:
    a_emb_gt = a_emb_pool[batch.a_indices_gt.clamp(min=0)]
    a_loss = cosine_a_loss(out["a_emb_pred"], a_emb_gt, batch.loss_mask)  # ALL nodes (drives history/SS a_emb)
    organ_loss = organ_aux_loss_fn(out["organ_logits"], batch.organ_idx)
    # a_loss_weight<1.0 down-weights the cosine objective; =0.0 → pure-cls. Default 1.0 = unchanged.
    total = a_loss_weight * a_loss + organ_aux_weight * organ_loss
    cls_loss_val = 0.0
    # M1-cls: balanced masked CE on CLOSED-set Q nodes only (additive to cosine).
    if cls is not None and cls["weight"] > 0 and "cls_logits" in out:
        qi = batch.q_indices.clamp(min=0)  # [B, N]
        closed = cls["closed_by_q"][qi] & batch.loss_mask  # [B, N]
        if closed.any():
            logits = out["cls_logits"].masked_fill(~cls["legal_mask"][qi], -1e4)  # [B, N, A]
            a_gt = batch.a_indices_gt.clamp(min=0)  # [B, N]
            if cls.get("a_remap") is not None:
                a_gt = cls["a_remap"][qi, a_gt]  # snap scattered % GT → its legal bucket (pct Qs only; identity elsewhere)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), a_gt.reshape(-1), reduction="none").reshape(a_gt.shape)
            sw = cls["cls_weight"][qi, a_gt]  # [B, N] per-(Q,a) balance weight
            cf = closed.float()
            cls_loss = (ce * sw * cf).sum() / cf.sum().clamp(min=1.0)
            total = total + cls["weight"] * cls_loss
            cls_loss_val = cls_loss.item()
    return total, {
        "a_loss": a_loss.item(),
        "organ_loss": organ_loss.item(),
        "cls_loss": cls_loss_val,
        "total": total.item(),
    }


@torch.no_grad()
def nn_search_batch(pred: Tensor, a_emb_pool: Tensor) -> Tensor:
    """Nearest-neighbor in A_pool by cosine similarity."""
    pred_n = F.normalize(pred, dim=-1, eps=1e-8)
    pool_n = F.normalize(a_emb_pool, dim=-1, eps=1e-8)
    sims = torch.einsum("bnd,kd->bnk", pred_n, pool_n)
    return sims.argmax(dim=-1)


def move_batch_to_device(batch: DAGBatch, device: str) -> DAGBatch:
    return DAGBatch(
        case_ids=batch.case_ids,
        tile_features=batch.tile_features.to(device, non_blocking=True),
        tile_pad_mask=batch.tile_pad_mask.to(device, non_blocking=True),
        q_indices=batch.q_indices.to(device, non_blocking=True),
        a_indices_gt=batch.a_indices_gt.to(device, non_blocking=True),
        attn_mask=batch.attn_mask.to(device, non_blocking=True),
        loss_mask=batch.loss_mask.to(device, non_blocking=True),
        organ_idx=batch.organ_idx.to(device, non_blocking=True),
        N_collapsed=batch.N_collapsed.to(device, non_blocking=True),
    )


def train_one_epoch(
    model: ModuleOneBase,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: str,
    epoch: int,
) -> dict[str, float]:
    # Frozen-body cls-row fine-tune: eval() so BN/dropout are fixed (zero buffer drift; the
    # cls_head learns to read the SAME h the engine sees at inference). Else normal train mode.
    model.eval() if config.freeze_body_eval else model.train(True)
    agg: dict[str, list[float]] = defaultdict(list)
    sched_p = _sched_prob_at_epoch(epoch, config.num_epochs, config.sched_prob_max, config.sched_warmup_epochs) if config.sched_sampling else 0.0
    _cls = None
    if config.cls_loss_weight > 0 and config.m1v5_closed_by_q is not None:
        _cls = {"weight": config.cls_loss_weight, "closed_by_q": config.m1v5_closed_by_q, "legal_mask": config.m1v5_legal_mask, "cls_weight": config.m1v5_cls_weight, "a_remap": config.m1v5_a_remap}
    for i, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if config.sched_sampling and sched_p > 0:
            # 2-pass: (1) no-grad GT-conditioned prediction, (2) grad with prob-p mix of pred/GT history A.
            with torch.no_grad():
                out1 = model(batch)
                a_pred_detached = out1["a_emb_pred"].detach()
            a_emb_gt = model.a_emb_pool[batch.a_indices_gt.clamp(min=0)]
            swap_mask = (torch.rand(a_emb_gt.shape[:2], device=device) < sched_p).unsqueeze(-1).expand_as(a_emb_gt)
            a_emb_input = torch.where(swap_mask, a_pred_detached, a_emb_gt)
            out = model(batch, a_emb_override=a_emb_input)
        else:
            out = model(batch)
        loss, metrics = compute_loss(out, batch, model.a_emb_pool, config.organ_aux_weight, cls=_cls, a_loss_weight=config.a_loss_weight)
        # Under --freeze-except-cls, batches with no closed-Q node have only frozen-param losses
        # (a_loss/organ_loss) → total has no grad_fn. Skip the step (nothing to learn this batch).
        if loss.requires_grad:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        metrics["sched_p"] = sched_p
        for k, v in metrics.items():
            agg[k].append(v)
        if (i + 1) % config.log_every == 0:
            recent = config.log_every
            logger.info(
                "epoch %d step %d/%d | a_loss=%.4f organ_loss=%.4f cls_loss=%.4f total=%.4f",
                epoch,
                i + 1,
                len(loader),
                sum(agg["a_loss"][-recent:]) / recent,
                sum(agg["organ_loss"][-recent:]) / recent,
                sum(agg.get("cls_loss", [0.0])[-recent:]) / recent,
                sum(agg["total"][-recent:]) / recent,
            )
    return {k: sum(v) / max(len(v), 1) for k, v in agg.items()}


@torch.no_grad()
def val_one_epoch(model: ModuleOneBase, loader: DataLoader, config: TrainConfig, device: str) -> dict[str, float]:
    model.train(False)
    a_correct = 0
    a_total = 0
    organ_correct = 0
    organ_total = 0
    val_a_loss_sum = 0.0
    val_a_loss_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        a_emb_gt = model.a_emb_pool[batch.a_indices_gt.clamp(min=0)]
        val_a_loss = cosine_a_loss(out["a_emb_pred"], a_emb_gt, batch.loss_mask)
        val_a_loss_sum += val_a_loss.item() * batch.loss_mask.sum().item()
        val_a_loss_count += batch.loss_mask.sum().item()

        nn_idx = nn_search_batch(out["a_emb_pred"], model.a_emb_pool)
        matches = (nn_idx == batch.a_indices_gt) & batch.loss_mask
        a_correct += matches.sum().item()
        a_total += batch.loss_mask.sum().item()

        valid_organ = batch.organ_idx >= 0
        if valid_organ.any():
            organ_pred = out["organ_logits"].argmax(dim=-1)
            organ_correct += ((organ_pred == batch.organ_idx) & valid_organ).sum().item()
            organ_total += valid_organ.sum().item()

    return {
        "val_a_loss": val_a_loss_sum / max(val_a_loss_count, 1),
        "val_a_acc": a_correct / max(a_total, 1),
        "val_organ_acc": organ_correct / max(organ_total, 1),
    }


def save_checkpoint(
    model: ModuleOneBase,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: Path,
    metrics: dict[str, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"epoch_{epoch:03d}.ckpt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)
    return path


def make_dataloader(dataset: DAGDataset, config: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_dag,
        pin_memory=False,
        persistent_workers=config.num_workers > 0,
    )
