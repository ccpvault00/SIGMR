"""Module 2 trainer.

Loss: BCEWithLogitsLoss with per-Q structural pos_weight.
- Default = 80
- dx_count Q × 5 = 400  (critical for multi-dx Edge-F1)
- 21 within-case fan-out Qs × 2 = 160  (force fan-out emission)
- Cap = 200
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from reg2026.data.embedding_pools import EmbeddingPools
from reg2026.module2.trajectory_model import TrajectoryTransformer
from reg2026.module2.m2_dataset import M2Dataset, collate_m2

logger = logging.getLogger(__name__)


# within-case fan-out Qs (21 total, max fan-out = 4), verified from the training fan-out distribution
FANOUT_QS_21: list[str] = [
    "Is there any neoplasm present?",
    "What is the grading system?",
    "What is the histologic type of neoplasm?",
    "Is there any abnormality present?",
    "What is the procedure?",
    "What is the organ?",
    "Is there any neuroendocrine morphology present?",
    "What is the Gleason score?",
    "Is there any additional finding present?",
    "What is the histologic type of lesion?",
    "Is there any invasion present?",
    "Is there any active inflammation present?",
    "Is there any proliferative lesion present?",
    "Is there any papillary lesion present?",
    "What is the histologic group of lesion?",
    "Is there any granuloma present?",
    "Is there any monotonous cell population present?",
    "Is there any inflammation present?",
    "Is there any associated epithelial lesion present?",
    "What is the #1 diagnosis?",
    "What is the number of diagnoses to includes?",
]

DX_COUNT_Q: str = "What is the number of diagnoses to includes?"


@dataclass(frozen=True)
class M2TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    grad_clip: float = 1.0
    log_every: int = 200
    # Per-Q pos_weight (structural boost + uniform base)
    base_pos_weight: float = 80.0
    fanout_boost: float = 2.0  # × base
    dx_count_boost: float = 5.0  # × base
    pos_weight_cap: float = 200.0
    fanout_qs: list[str] = field(default_factory=lambda: list(FANOUT_QS_21))
    dx_count_q: str = DX_COUNT_Q
    focal_gamma: float = 0.0  # 0 = BCE; >0 = focal loss (recommend 2.0)


def build_pos_weight(config: M2TrainConfig, pools: EmbeddingPools) -> Tensor:
    """Build [n_q_vocab] pos_weight tensor with structural boost (no frequency)."""
    n = len(pools.Q_VOCAB)
    pw = torch.full((n,), config.base_pos_weight)
    for q in config.fanout_qs:
        if q == config.dx_count_q:
            continue  # dx_count handled separately below for full ×5
        try:
            idx = pools.q_idx(q)
            pw[idx] = config.base_pos_weight * config.fanout_boost
        except KeyError:
            logger.warning("fan-out Q not in vocab: %r", q)
    # dx_count special boost (overrides fan-out boost)
    try:
        dxc_idx = pools.q_idx(config.dx_count_q)
        pw[dxc_idx] = config.base_pos_weight * config.dx_count_boost
    except KeyError:
        logger.warning("dx_count Q not in vocab: %r", config.dx_count_q)
    pw = pw.clamp(max=config.pos_weight_cap)
    logger.info(
        "pos_weight: base=%.0f, fanout_boost=%.1fx (n=%d), dx_count_boost=%.1fx, cap=%.0f",
        config.base_pos_weight,
        config.fanout_boost,
        sum(1 for q in config.fanout_qs if q != config.dx_count_q),
        config.dx_count_boost,
        config.pos_weight_cap,
    )
    return pw


def compute_loss(logits: Tensor, targets: Tensor, pos_weight: Tensor, focal_gamma: float = 0.0) -> Tensor:
    """BCE multilabel loss with per-Q pos_weight; optional focal modulation.

    focal_gamma=0: standard BCE (default).
    focal_gamma>0: focal loss - (1-pt)^gamma down-weights easy examples,
                   pushing gradient mass toward hard ones (TP miss + FP).
    """
    if focal_gamma <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    # Focal: per-element BCE × (1-pt)^gamma, then mean
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)  # p if y=1, 1-p if y=0
    modulator = (1 - pt).pow(focal_gamma)
    return (modulator * bce).mean()


@torch.no_grad()
def evaluate(model: TrajectoryTransformer, loader: DataLoader, pos_weight: Tensor, threshold: float, device: str) -> dict[str, float]:
    """Compute val loss + per-Q F1 macro + per-Q recall macro."""
    model.train(False)
    total_loss = 0.0
    total_samples = 0
    n_q = pos_weight.shape[0]
    tp = torch.zeros(n_q, device=device)
    fp = torch.zeros(n_q, device=device)
    fn = torch.zeros(n_q, device=device)
    for batch in loader:
        tokens = batch.tokens.to(device)
        attn = batch.attention_mask.to(device)
        targets = batch.targets.to(device)
        logits = model(tokens, attn)
        loss = compute_loss(logits, targets, pos_weight)
        bs = tokens.shape[0]
        total_loss += loss.item() * bs
        total_samples += bs

        preds = (torch.sigmoid(logits) > threshold).float()
        tp += (preds * targets).sum(dim=0)
        fp += (preds * (1 - targets)).sum(dim=0)
        fn += ((1 - preds) * targets).sum(dim=0)

    prec = tp / (tp + fp).clamp(min=1.0)
    rec = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_f1_macro": f1.mean().item(),
        "val_recall_macro": rec.mean().item(),
        "val_prec_macro": prec.mean().item(),
    }


def train_one_epoch(
    model: TrajectoryTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: Tensor,
    config: M2TrainConfig,
    device: str,
    epoch: int,
) -> dict[str, float]:
    model.train(True)
    losses: list[float] = []
    for i, batch in enumerate(loader):
        tokens = batch.tokens.to(device)
        attn = batch.attention_mask.to(device)
        targets = batch.targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, attn)
        loss = compute_loss(logits, targets, pos_weight, focal_gamma=config.focal_gamma)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        losses.append(loss.item())
        if (i + 1) % config.log_every == 0:
            logger.info("epoch %d step %d/%d | loss=%.4f", epoch, i + 1, len(loader), sum(losses[-config.log_every :]) / config.log_every)
    return {"train_loss": sum(losses) / max(len(losses), 1)}


def save_checkpoint(model: TrajectoryTransformer, optimizer: torch.optim.Optimizer, epoch: int, output_dir: Path, metrics: dict, model_cfg) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"epoch_{epoch:03d}.ckpt"
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": model_cfg,
        },
        path,
    )
    return path


def make_dataloader(
    dataset: M2Dataset,
    config: M2TrainConfig,
    n_q_vocab: int,
    shuffle: bool,
    multi_dx_oversample: float = 1.0,
    prostate_oversample: float = 1.0,
    rare_qa_oversample_scale: float = 0.0,
    rare_qa_oversample_cap: float = 32.0,
    multi_dx_organs: list[str] | None = None,
) -> DataLoader:
    """Build M2 DataLoader. WeightedRandomSampler combines multi-dx + Prostate +
    rare-(organ,Q,A) oversample (per-sample weight = max(multi_w, prostate_w, rare_qa_w, 1.0)).

    - multi_dx_oversample: 4.0 = current SOTA
    - prostate_oversample: 1.0 = off (Prostate handled by M1-vqa)
    - rare_qa_oversample_scale: 0 = off. >0 enables per-(organ,q,a) inverse-log-freq weight.
    """
    if shuffle and (multi_dx_oversample > 1.0 or prostate_oversample > 1.0 or rare_qa_oversample_scale > 0):
        from torch.utils.data import WeightedRandomSampler

        is_multi = dataset.get_multi_dx_mask() if multi_dx_oversample > 1.0 else [False] * len(dataset)
        # gate multi-dx oversample by organ (avoid contaminating Prostate routing).
        if multi_dx_organs and multi_dx_oversample > 1.0:
            organ_masks = [dataset.get_organ_mask(o) for o in multi_dx_organs]
            in_any_organ = [any(m[i] for m in organ_masks) for i in range(len(dataset))]
            is_multi = [m and o for m, o in zip(is_multi, in_any_organ, strict=True)]
        is_prostate = dataset.get_organ_mask("prostate") if prostate_oversample > 1.0 else [False] * len(dataset)
        rare_qa_w = dataset.get_rare_qa_weights(rare_qa_oversample_scale, rare_qa_oversample_cap) if rare_qa_oversample_scale > 0 else [1.0] * len(dataset)
        weights = []
        for m, p, rq in zip(is_multi, is_prostate, rare_qa_w, strict=True):
            w = 1.0
            if m:
                w = max(w, float(multi_dx_oversample))
            if p:
                w = max(w, float(prostate_oversample))
            w = max(w, rq)
            weights.append(w)
        n_multi = sum(is_multi)
        n_prostate = sum(is_prostate)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        logger.info(
            "M2 oversample active: multi-dx %.1fx (%d/%d), prostate %.1fx (%d/%d), rare_qa scale=%.2f cap=%.1f (max_w=%.2f, median=%.2f)",
            multi_dx_oversample,
            n_multi,
            len(is_multi),
            prostate_oversample,
            n_prostate,
            len(is_prostate),
            rare_qa_oversample_scale,
            rare_qa_oversample_cap,
            max(rare_qa_w),
            sorted(rare_qa_w)[len(rare_qa_w) // 2],
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=lambda samples: collate_m2(samples, n_q_vocab=n_q_vocab),
            pin_memory=False,
            persistent_workers=config.num_workers > 0,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=lambda samples: collate_m2(samples, n_q_vocab=n_q_vocab),
        pin_memory=False,
        persistent_workers=config.num_workers > 0,
    )
