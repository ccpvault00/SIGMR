"""Module 1 training entry point.

Usage:
    python scripts/train_module1.py \\
        --variant a \\
        --cot-json /mnt/data/reg2026/train_CoT.json \\
        --dag-masks-h5 /mnt/data/reg2026/checkpoints/m1_answer_space/dag_masks.h5 \\
        --features-dir /mnt/data/reg2026/features/full_run/20x_224px_0px_overlap/features_hoptimus1 \\
        --split-json /mnt/data/reg2026/checkpoints/phase_a_v3/split.json \\
        --output-dir /mnt/data/reg2026/checkpoints/phase_a_a \\
        --num-epochs 25 --batch-size 8 --num-workers 8

For smoke test: add --max-cases 100 --num-epochs 1.
For B: --variant b.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from reg2026.aggregate.abmil import ABMIL, ABMILConfig
from reg2026.aggregate.aux_heads import AuxHeads, AuxHeadsConfig
from reg2026.data.embedding_pools import EmbeddingPools
from reg2026.module1.dag_dataset import DAGDataset, DAGDatasetConfig
from reg2026.module1.network_b_qcond import ModuleOneBQCond
from reg2026.module1.network_base import ModuleOneConfig
from reg2026.module1.trainer import (
    ORGAN_TO_IDX,
    TrainConfig,
    make_dataloader,
    save_checkpoint,
    train_one_epoch,
    val_one_epoch,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["a", "b", "b-qcond"], required=True, help="A (Q-Former) | B (top-K saliency) | B-QCond (Q-conditional cross-attn)")
    p.add_argument("--cot-json", type=Path, required=True)
    p.add_argument("--dag-masks-h5", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-N", type=int, default=30, help="Max collapsed turns per case")
    p.add_argument("--max-cases", type=int, default=0, help="Limit train cases (0=all, debug)")
    # B specific
    p.add_argument("--n-wsi-layers", type=int, default=2, help="B: WSI self-attn depth")
    # Per-organ MoE
    p.add_argument("--target-organ", type=str, default=None, help="If set, filter training to this organ only")
    p.add_argument("--init-ckpt", type=Path, default=None, help="Init model state from this ckpt (per-organ fine-tune)")
    p.add_argument("--lora", action="store_true", help="Apply LoRA (freeze base, train rank-16 adapters + heads)")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # wandb
    p.add_argument("--wandb", action="store_true", help="Log to wandb (project=reg2026-module1)")
    p.add_argument("--wandb-project", type=str, default="reg2026-module1")
    p.add_argument("--wandb-entity", type=str, default=None, help="wandb entity (defaults to the logged-in account)")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Override run name (auto if None)")
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="Tags for grouping (e.g. m1a lora prostate)")
    # Scheduled sampling (exposure bias fix) - ported from module1-qformer; essential to SOTA M1.
    p.add_argument("--sched-sampling", action="store_true", help="2-pass scheduled sampling (history a_emb mixes predicted/GT, anneal 0->prob_max)")
    p.add_argument("--sched-prob-max", type=float, default=0.5)
    p.add_argument("--sched-warmup-epochs", type=int, default=2)
    # M1-cls: additive balanced classification head for closed-set Qs (dissolves cosine-collapse patches)
    p.add_argument("--per-q-answer-space", type=Path, default=None, help="per_q_answer_space.json (from build_per_q_answer_space.py); enables M1-cls closed-Q classification")
    p.add_argument("--cls-loss-weight", type=float, default=0.0, help="weight for closed-Q classification CE (0=off, pure cosine)")
    return p.parse_args()


def _auto_run_name(args) -> str:
    """Auto-generate run name like 'm1a_lora_prostate' or 'm1b_generic'."""
    parts = [f"module1{args.variant}"]
    if args.lora:
        parts.append("lora")
    if args.target_organ:
        parts.append(args.target_organ)
    if args.init_ckpt:
        # fine-tune (not from scratch)
        parts.append("ft")
    if not args.target_organ and not args.lora and not args.init_ckpt:
        parts.append("generic")
    return "_".join(parts)


def _auto_tags(args) -> list[str]:
    tags = [f"module1{args.variant}"]
    if args.lora:
        tags.append("lora")
    if args.target_organ:
        tags.append(f"organ-{args.target_organ}")
    if args.init_ckpt:
        tags.append("fine-tune")
    return tags


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # Save args for reproducibility
    (args.output_dir / "train_args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    # Optional wandb init - auto-captures stdout to Logs tab
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            run_name = args.wandb_run_name or _auto_run_name(args)
            tags = args.wandb_tags or _auto_tags(args)
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                tags=tags,
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                dir=str(args.output_dir),
            )
            logger.info("wandb run started: %s/%s/%s", args.wandb_entity, args.wandb_project, run_name)
        except Exception as e:
            logger.warning("wandb init failed: %s - continuing without wandb", e)

    # Load pools
    logger.info("Loading embedding pools")
    pools = EmbeddingPools()
    logger.info("Q_EMB %s  A_EMB %s", tuple(pools.Q_EMB.shape), tuple(pools.A_EMB.shape))

    # M1-cls: build per-Q classification tensors (closed flag + legal mask + per-(Q,a) balance weight)
    m1v5 = None
    if args.per_q_answer_space is not None and args.cls_loss_weight > 0:
        import math

        qas = json.loads(args.per_q_answer_space.read_text())
        nQ, nA = len(pools.Q_VOCAB), len(pools.A_VOCAB)
        closed_by_q = torch.zeros(nQ, dtype=torch.bool)
        legal_mask = torch.zeros(nQ, nA, dtype=torch.bool)
        cls_weight = torch.ones(nQ, nA, dtype=torch.float32)
        n_closed = 0
        for q, v in qas.items():
            if not v.get("closed"):
                continue
            try:
                qi = pools.q_idx(q)
            except KeyError:
                continue
            counts = {int(k): val for k, val in v["class_counts"].items()}
            legal = [ai for ai in v["legal_a_idx"]]
            if not legal:
                continue
            closed_by_q[qi] = True
            n_closed += 1
            ws = {ai: 1.0 / math.sqrt(max(counts.get(ai, 1), 1)) for ai in legal}
            mean_w = sum(ws.values()) / len(ws)
            for ai in legal:
                legal_mask[qi, ai] = True
                cls_weight[qi, ai] = ws[ai] / mean_w  # within-Q balanced, mean 1
        m1v5 = (closed_by_q.to(args.device), legal_mask.to(args.device), cls_weight.to(args.device))
        logger.info("M1-cls: %d closed Qs enabled (cls_loss_weight=%.2f)", n_closed, args.cls_loss_weight)

    # Build datasets (optionally filter to single target organ for per-organ MoE)
    train_cfg = DAGDatasetConfig(
        cot_json=args.cot_json,
        dag_masks_h5=args.dag_masks_h5,
        features_dir=args.features_dir,
        split_json=args.split_json,
        split="train",
        organ_to_idx=ORGAN_TO_IDX,
        target_organ=args.target_organ,
    )
    val_cfg = DAGDatasetConfig(
        cot_json=args.cot_json,
        dag_masks_h5=args.dag_masks_h5,
        features_dir=args.features_dir,
        split_json=args.split_json,
        split="val",
        organ_to_idx=ORGAN_TO_IDX,
        target_organ=args.target_organ,  # val also filtered for per-organ eval
    )

    logger.info("Loading train dataset")
    train_ds = DAGDataset(train_cfg, pools)
    if args.max_cases > 0:
        train_ds.samples = train_ds.samples[: args.max_cases]
        logger.info("Truncated train to %d cases (--max-cases)", args.max_cases)
    logger.info("Loading val dataset")
    val_ds = DAGDataset(val_cfg, pools)

    train_config = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sched_sampling=args.sched_sampling,
        sched_prob_max=args.sched_prob_max,
        sched_warmup_epochs=args.sched_warmup_epochs,
        cls_loss_weight=args.cls_loss_weight if m1v5 is not None else 0.0,
        m1v5_closed_by_q=m1v5[0] if m1v5 is not None else None,
        m1v5_legal_mask=m1v5[1] if m1v5 is not None else None,
        m1v5_cls_weight=m1v5[2] if m1v5 is not None else None,
    )
    train_loader = make_dataloader(train_ds, train_config, shuffle=True)
    val_loader = make_dataloader(val_ds, train_config, shuffle=False)

    # Build network components
    abmil = ABMIL(ABMILConfig(in_dim=1536, embed_dim=256, num_heads=4))
    aux_heads = AuxHeads(
        AuxHeadsConfig(
            in_dim=abmil.config.output_dim,
            categorical_heads={"organ": 7},
        )
    )

    module1_cfg = ModuleOneConfig(
        n_q_vocab=len(pools.Q_VOCAB),
        n_a_vocab=len(pools.A_VOCAB),
        tile_emb_dim=1536,
        slide_emb_dim=abmil.config.output_dim,
        max_N=args.max_N,
    )

    if args.variant != "b-qcond":
        raise ValueError(f"only the deployed variant 'b-qcond' is supported (got {args.variant!r})")
    model = ModuleOneBQCond(module1_cfg, pools, abmil, aux_heads, n_wsi_layers=args.n_wsi_layers)
    logger.info("Built ModuleOneBQCond (Q-conditional cross-attn) n_wsi_layers=%d", args.n_wsi_layers)

    # Init from existing ckpt (per-organ fine-tune)
    if args.init_ckpt is not None:
        init_state = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        # strict=False: M1-cls adds cls_head not present in older ckpts (warm-start body+cosine, cls_head stays random).
        missing, unexpected = model.load_state_dict(init_state["model_state"], strict=False)
        logger.info("Initialized from %s (epoch %d); missing=%s unexpected=%s", args.init_ckpt, init_state.get("epoch", -1), list(missing), list(unexpected))

    # Apply LoRA AFTER init (freeze base, train adapters only)
    if args.lora:
        from reg2026.module1.lora import add_lora_to_module1

        lora_stats = add_lora_to_module1(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
        logger.info("LoRA stats: %s", lora_stats)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d (%.2fM)", n_params, n_params / 1e6)
    model = model.to(args.device)

    # Only optimize trainable params (LoRA mode freezes base)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Training loop
    best_val_a_acc = -1.0
    history: list[dict] = []
    for epoch in range(args.num_epochs):
        logger.info("=== Epoch %d / %d ===", epoch + 1, args.num_epochs)
        train_metrics = train_one_epoch(model, train_loader, optimizer, train_config, args.device, epoch)
        val_metrics = val_one_epoch(model, val_loader, train_config, args.device)
        scheduler.step()
        epoch_record = {"epoch": epoch, **train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        logger.info(
            "Epoch %d done: train_a=%.4f train_organ=%.4f | val_a_loss=%.4f val_a_acc=%.4f val_organ_acc=%.4f",
            epoch + 1,
            train_metrics["a_loss"],
            train_metrics["organ_loss"],
            val_metrics["val_a_loss"],
            val_metrics["val_a_acc"],
            val_metrics["val_organ_acc"],
        )
        history.append(epoch_record)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

        # wandb live log (per-epoch metrics; stdout already auto-captured to Logs tab)
        if wandb_run is not None:
            wandb_run.log({**train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}, step=epoch)

        if train_config.save_every_epoch:
            save_checkpoint(model, optimizer, epoch, args.output_dir, epoch_record)
        if val_metrics["val_a_acc"] > best_val_a_acc:
            best_val_a_acc = val_metrics["val_a_acc"]
            save_checkpoint(model, optimizer, -1, args.output_dir, epoch_record).rename(args.output_dir / "best.ckpt")
            logger.info("New best val_a_acc=%.4f saved as best.ckpt", best_val_a_acc)

    logger.info("Training complete. Best val_a_acc=%.4f", best_val_a_acc)
    if wandb_run is not None:
        wandb_run.summary["best_val_a_acc"] = best_val_a_acc
        wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main() or 0)
