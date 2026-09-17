"""Module 2 training entry script.

Usage:
    python scripts/train_module2.py \\
        --cot-json /mnt/data/reg2026/train_CoT.json \\
        --split-json /mnt/data/reg2026/checkpoints/phase_a_v3/split.json \\
        --predictions-json /mnt/data/reg2026/checkpoints/phase_a_b/predictions_all.json \\
        --output-dir /mnt/data/reg2026/checkpoints/phase_a_module2 \\
        --num-epochs 50 --batch-size 32 --num-workers 4 --a-mix-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from reg2026.data.embedding_pools import EmbeddingPools
from reg2026.module2.trajectory_model import TrajectoryModelConfig, TrajectoryTransformer
from reg2026.module2.trajectory_tokenizer import TrajectoryTokenizer
from reg2026.module2.m2_dataset import M2Dataset, M2DatasetConfig
from reg2026.module2.m2_trainer import (
    M2TrainConfig,
    build_pos_weight,
    evaluate,
    make_dataloader,
    save_checkpoint,
    train_one_epoch,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cot-json", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--predictions-json", type=Path, required=False, default=None, help="B cached predictions JSON for M4 50/50 mix; omit = pure GT")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--a-mix-ratio", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5, help="Val threshold for F1 metric (per-Q calibration is separate step)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learn-q-threshold", action="store_true", help="Add per-Q learned threshold (subtracted from logits before BCE). Tackles M2 over-emission on low-precision Qs.")
    p.add_argument("--rel-pos", action="store_true", help="REVERSE position embedding (distance from current node) instead of absolute index - relative CoT distance, length-general.")
    p.add_argument("--attn-pool", action="store_true", help="Learned-query attention pool over the full history (added to last-position) instead of last-position-only bottleneck.")
    p.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma (0=BCE, 2=recommended). Down-weights easy examples to push gradient onto hard FP/FN.")
    p.add_argument("--multi-dx-oversample", type=float, default=1.0, help="WeightedRandomSampler weight for multi-dx (dx_count>=2) cases. 1.0=off, 4.0=recommended sweet spot.")
    p.add_argument("--prostate-oversample", type=float, default=1.0, help="WeightedRandomSampler weight for Prostate cases (fix Gleason chain routing). 1.0=off, 4.0=recommended.")
    p.add_argument("--rare-qa-oversample-scale", type=float, default=0.0, help="Scale for per-(organ,q,a) inverse-log-freq oversample. 0=off, 1.0=mild, 2.0=aggressive.")
    p.add_argument("--rare-qa-oversample-cap", type=float, default=32.0, help="Max sample weight for rare-QA oversample.")
    p.add_argument(
        "--multi-dx-organ-filter",
        type=str,
        default="",
        help="Comma-separated organ list (lowercase) to gate multi-dx oversample. Empty=all organs. Example: 'breast,bladder' (skip Prostate to avoid Gleason routing contamination).",
    )
    p.add_argument("--per-occ-collapse", action="store_true", help="Collapse by (Q, occ_idx) - non-consecutive same-Q turns = new logical node (cleaner multi-dx training).")
    p.add_argument(
        "--bfs-ancestor", action="store_true", help="Use BFS-simulated first-visit ancestor set (matches inference BFS) instead of static transitive closure. Fixes fan-in train/test mismatch."
    )
    p.add_argument(
        "--bfs-edge-drop-prob",
        type=float,
        default=0.0,
        help="For --bfs-ancestor: per-sample random edge dropout for schedule sampling on DAG structure (simulates inference noise from imperfect model predictions).",
    )
    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="reg2026-module1")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    (args.output_dir / "train_args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    logger.info("Loading pools")
    pools = EmbeddingPools()
    tokenizer = TrajectoryTokenizer(pools)
    logger.info("Vocab: Q=%d, A=%d, total tokens=%d", len(pools.Q_VOCAB), len(pools.A_VOCAB), tokenizer.vocab_size)

    # Datasets
    train_cfg = M2DatasetConfig(
        cot_json=args.cot_json,
        split_json=args.split_json,
        predictions_json=args.predictions_json,
        split="train",
        a_mix_ratio=args.a_mix_ratio,
        bfs_ancestor=args.bfs_ancestor,
        bfs_edge_drop_prob=args.bfs_edge_drop_prob,
        per_occ_collapse=args.per_occ_collapse,
    )
    val_cfg = M2DatasetConfig(
        cot_json=args.cot_json,
        split_json=args.split_json,
        predictions_json=args.predictions_json,
        split="val",
        a_mix_ratio=0.0,  # val always pure GT for stable comparison
        bfs_ancestor=args.bfs_ancestor,  # match training mode for consistent F1
        bfs_edge_drop_prob=0.0,  # val: no dropout for deterministic eval
        per_occ_collapse=args.per_occ_collapse,
    )
    train_ds = M2Dataset(train_cfg, pools, tokenizer)
    val_ds = M2Dataset(val_cfg, pools, tokenizer)

    train_config = M2TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        focal_gamma=args.focal_gamma,
    )
    multi_dx_organs = [o.strip() for o in args.multi_dx_organ_filter.split(",") if o.strip()] or None
    train_loader = make_dataloader(
        train_ds,
        train_config,
        n_q_vocab=len(pools.Q_VOCAB),
        shuffle=True,
        multi_dx_oversample=args.multi_dx_oversample,
        prostate_oversample=args.prostate_oversample,
        rare_qa_oversample_scale=args.rare_qa_oversample_scale,
        rare_qa_oversample_cap=args.rare_qa_oversample_cap,
        multi_dx_organs=multi_dx_organs,
    )
    val_loader = make_dataloader(val_ds, train_config, n_q_vocab=len(pools.Q_VOCAB), shuffle=False)

    # Model
    model_cfg = TrajectoryModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_q_vocab=len(pools.Q_VOCAB),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        learn_q_threshold=args.learn_q_threshold,
        rel_pos=args.rel_pos,
        attn_pool=args.attn_pool,
    )
    model = TrajectoryTransformer(model_cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d (%.2fM)", n_params, n_params / 1e6)

    # Pos weight
    pos_weight = build_pos_weight(train_config, pools).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Optional wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            run_name = args.wandb_run_name or f"module2_ep{args.num_epochs}_amix{args.a_mix_ratio}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                tags=args.wandb_tags or ["module2"],
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                dir=str(args.output_dir),
            )
            logger.info("wandb run started: %s/%s/%s", args.wandb_entity, args.wandb_project, run_name)
        except Exception as e:
            logger.warning("wandb init failed: %s - continuing without wandb", e)

    best_val_f1 = -1.0
    history = []
    for epoch in range(args.num_epochs):
        logger.info("=== Epoch %d / %d ===", epoch + 1, args.num_epochs)
        train_metrics = train_one_epoch(model, train_loader, optimizer, pos_weight, train_config, args.device, epoch)
        val_metrics = evaluate(model, val_loader, pos_weight, args.threshold, args.device)
        scheduler.step()
        rec = {"epoch": epoch, **train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        logger.info(
            "Epoch %d done: train_loss=%.4f | val_loss=%.4f f1=%.4f rec=%.4f prec=%.4f",
            epoch + 1,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            val_metrics["val_f1_macro"],
            val_metrics["val_recall_macro"],
            val_metrics["val_prec_macro"],
        )
        history.append(rec)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        if wandb_run is not None:
            wandb_run.log({**train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
        save_checkpoint(model, optimizer, epoch, args.output_dir, rec, model_cfg)
        if val_metrics["val_f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["val_f1_macro"]
            save_checkpoint(model, optimizer, -1, args.output_dir, rec, model_cfg).rename(args.output_dir / "best.ckpt")
            logger.info("New best val_f1_macro=%.4f saved as best.ckpt", best_val_f1)

    logger.info("Training complete. Best val_f1_macro=%.4f", best_val_f1)
    if wandb_run is not None:
        wandb_run.summary["best_val_f1_macro"] = best_val_f1
        wandb_run.finish()


if __name__ == "__main__":
    main()
