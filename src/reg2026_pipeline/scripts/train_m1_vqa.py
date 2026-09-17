"""M1-vqa training: reduced 3-anchor context Q-A learning.

Each training sample = 1 case + 1 target Q with 4-position chain:
  [organ_q+a, procedure_q+a, #1_dx_q+a, target_q+a_pred]

Trains M1 to predict target A given only 3 anchors (no cascade through full chain).
Hypothesis: cleaner context → less cascade error than full DAG history.

Usage:
    python scripts/train_m1_vqa.py \\
        --variant b-qcond \\
        --init-ckpt /mnt/data/reg2026/checkpoints/phase_a_b_qcond_ss/best.ckpt \\
        --cot-json /mnt/data/reg2026/train_CoT.json \\
        --features-dir /mnt/data/reg2026/features/full_run/20x_224px_0px_overlap/features_hoptimus1 \\
        --split-json /mnt/data/reg2026/checkpoints/phase_a_v3/split.json \\
        --output-dir /mnt/data/reg2026/checkpoints/phase_a_m1_vqa
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
from reg2026.module1.dag_dataset_vqa import M1VqaDataset, VqaDatasetConfig, collate_dag_vqa
from reg2026.module1.network_b_qcond import ModuleOneBQCond
from reg2026.module1.network_base import ModuleOneConfig
from reg2026.module1.trainer import (
    ORGAN_TO_IDX,
    TrainConfig,
    save_checkpoint,
    train_one_epoch,
    val_one_epoch,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["a", "b", "b-qcond", "b-qcond-max"], required=True)
    p.add_argument(
        "--yes-gate-oversample",
        type=float,
        default=1.0,
        help="WeightedRandomSampler weight for presence-gate samples (target_q='...present?' AND target_a='Yes'). 1.0=off, 4.0=recommended - fixes small-focus under-detection (additional finding / microcalc / invasion / fungal).",
    )
    p.add_argument(
        "--length-bucket",
        action="store_true",
        help="fp32 throughput: group batches by tile-count so collate pads less (cuts padding compute AND H2D transfer). Compatible with --yes-gate-oversample (via index duplication). Re-shuffles each epoch.",
    )
    p.add_argument("--cot-json", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-ckpt", type=Path, default=None, help="Init from existing M1 ckpt (e.g. B-qcond-SS).")
    p.add_argument("--num-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="Fine-tune LR (lowered from 2e-4 pretrain default)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-N", type=int, default=8)
    p.add_argument("--n-wsi-layers", type=int, default=2)
    p.add_argument("--organ-aux-weight", type=float, default=0.0, help="Disable organ aux for M1-vqa (organ already handled by M0)")
    p.add_argument("--schedsamp-rate", type=float, default=0.2, help="Probability of noisy primary_dx anchor (train only)")
    p.add_argument("--schedsamp-top2-json", type=Path, default=None, help="JSON with base-dx slot top-2 per case (required if schedsamp_rate>0)")
    p.add_argument("--drop-proc-anchor", action="store_true", help="Train the 2-anchor variant [organ, #1-dx, target] (drops the procedure anchor); this is the deployed m1_vqa_2anchor config")
    p.add_argument("--max-occ", type=int, default=0, help="Max occurrence for additive occ_emb (0=off, 4=enable per-occ training).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_loader(dataset, train_config, shuffle, sample_weights=None, batch_sampler=None):
    """fp32-compatible throughput: the GPU was starved (util ~30%, VRAM full) =
    I/O + H2D bound. We overlap I/O with compute WITHOUT changing feature precision:
    pin_memory=True (enables the non_blocking H2D already in move_batch_to_device),
    persistent_workers (no per-epoch respawn), prefetch_factor=4 (workers read N+1..N+4
    while GPU computes batch N). num_workers is the CLI --num-workers (recommend 12)."""
    from torch.utils.data import DataLoader, WeightedRandomSampler

    nw = train_config.num_workers
    kw = dict(
        num_workers=nw,
        collate_fn=collate_dag_vqa,
        pin_memory=True,  # fp32 features stay fp32; only the H2D path is pinned for async overlap
    )
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    if batch_sampler is not None:
        kw["batch_sampler"] = batch_sampler  # mutually exclusive with batch_size/shuffle/sampler
    else:
        kw["batch_size"] = train_config.batch_size
        if sample_weights is not None and shuffle:
            kw["sampler"] = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            kw["shuffle"] = shuffle
    return DataLoader(dataset, **kw)


def build_yes_gate_weights(dataset, pools, oversample: float) -> list[float] | None:
    """Weight presence-gate samples (target_q='...present?' AND target_a='Yes') up.

    target_q/target_a = position 3 of each VQA sample. Fixes small-focus
    under-detection on the launch gates (additional finding / microcalc / invasion / fungal).
    """
    if oversample <= 1.0:
        return None
    weights: list[float] = []
    n_yes = 0
    for s in dataset.samples:
        tq = int(s["q_indices"][3])
        ta = int(s["a_indices_gt"][3])
        q_text = pools.Q_VOCAB[tq] if 0 <= tq < len(pools.Q_VOCAB) else ""
        a_text = pools.A_VOCAB[ta] if 0 <= ta < len(pools.A_VOCAB) else ""
        is_presence_yes = ("present?" in q_text.lower()) and a_text.strip().lower().startswith("yes")
        weights.append(float(oversample) if is_presence_yes else 1.0)
        n_yes += is_presence_yes
    logger.info("yes-gate oversample %.1fx: %d/%d presence-yes samples", oversample, n_yes, len(weights))
    return weights


def _read_tile_counts(dataset) -> list[int]:
    """Per-sample tile count (HDF5 metadata only - no feature load), cached per case_id."""
    import h5py

    cache: dict[str, int] = {}
    counts: list[int] = []
    for s in dataset.samples:
        cid = s["case_id"]
        if cid not in cache:
            try:
                with h5py.File(dataset.config.features_dir / f"{cid}.h5", "r") as fh:
                    cache[cid] = int(fh["features"].shape[0])
            except Exception:
                cache[cid] = 0
        counts.append(cache[cid])
    return counts


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Yield batches of similar tile-count → collate pads less (cuts padding compute + H2D).

    Compatible with oversample: a sample with sample_weight w is included round(w) times in the
    pool. Each epoch: jitter tile-counts a little, sort, chunk into batches, shuffle batch order.
    """

    def __init__(self, tile_counts, batch_size, sample_weights=None, seed=42):
        self.tile_counts = tile_counts
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        # oversample via integer index duplication
        pool: list[int] = []
        for i in range(len(tile_counts)):
            reps = max(1, round(sample_weights[i])) if sample_weights is not None else 1
            pool.extend([i] * reps)
        self.pool = pool

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        import random as _r

        rng = _r.Random(self.seed + self.epoch)
        # jitter so equal-length items shuffle within their band (±5% of count)
        keyed = sorted(self.pool, key=lambda i: self.tile_counts[i] * (1.0 + rng.uniform(-0.05, 0.05)))
        batches = [keyed[b : b + self.batch_size] for b in range(0, len(keyed), self.batch_size)]
        rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return (len(self.pool) + self.batch_size - 1) // self.batch_size


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    (args.output_dir / "train_args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    pools = EmbeddingPools()
    logger.info("Q_EMB %s  A_EMB %s", tuple(pools.Q_EMB.shape), tuple(pools.A_EMB.shape))

    if args.schedsamp_rate > 0 and args.schedsamp_top2_json is None:
        raise ValueError("--schedsamp-rate > 0 requires --schedsamp-top2-json")

    train_cfg = VqaDatasetConfig(
        cot_json=args.cot_json,
        features_dir=args.features_dir,
        split_json=args.split_json,
        split="train",
        organ_to_idx=ORGAN_TO_IDX,
        schedsamp_rate=args.schedsamp_rate,
        schedsamp_top2_json=args.schedsamp_top2_json,
        drop_proc_anchor=args.drop_proc_anchor,
        seed=args.seed,
    )
    val_cfg = VqaDatasetConfig(
        cot_json=args.cot_json,
        features_dir=args.features_dir,
        split_json=args.split_json,
        split="val",
        organ_to_idx=ORGAN_TO_IDX,
        schedsamp_rate=0.0,  # no noise at eval
        schedsamp_top2_json=None,
        drop_proc_anchor=args.drop_proc_anchor,
        seed=args.seed,
    )
    train_ds = M1VqaDataset(train_cfg, pools)
    val_ds = M1VqaDataset(val_cfg, pools)

    train_config = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        organ_aux_weight=args.organ_aux_weight,
    )
    yes_gate_weights = build_yes_gate_weights(train_ds, pools, args.yes_gate_oversample)
    bucket_sampler = None
    if args.length_bucket:
        logger.info("length-bucket: reading tile counts (HDF5 metadata)...")
        _tc = _read_tile_counts(train_ds)
        bucket_sampler = LengthBucketBatchSampler(_tc, args.batch_size, sample_weights=yes_gate_weights, seed=args.seed)
        logger.info("length-bucket ON: %d batches/epoch (pad-minimized, oversample via dup)", len(bucket_sampler))
        train_loader = make_loader(train_ds, train_config, shuffle=True, batch_sampler=bucket_sampler)
    else:
        train_loader = make_loader(train_ds, train_config, shuffle=True, sample_weights=yes_gate_weights)
    val_loader = make_loader(val_ds, train_config, shuffle=False)

    abmil = ABMIL(ABMILConfig(in_dim=1536, embed_dim=256, num_heads=4))
    aux_heads = AuxHeads(AuxHeadsConfig(in_dim=abmil.config.output_dim, categorical_heads={"organ": 7}))
    module1_cfg = ModuleOneConfig(
        n_q_vocab=len(pools.Q_VOCAB),
        n_a_vocab=len(pools.A_VOCAB),
        tile_emb_dim=1536,
        slide_emb_dim=abmil.config.output_dim,
        max_N=args.max_N,
        max_occ=args.max_occ,
    )

    if args.variant != "b-qcond":
        raise ValueError(f"only the deployed variant 'b-qcond' is supported (got {args.variant!r})")
    model = ModuleOneBQCond(module1_cfg, pools, abmil, aux_heads, n_wsi_layers=args.n_wsi_layers)

    if args.init_ckpt is not None:
        init_state = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(init_state["model_state"])
        except RuntimeError as e:
            logger.warning("init_ckpt load_state_dict had issues (likely max_N mismatch): %s", e)
            # Strip positional emb if shape mismatch
            sd = init_state["model_state"]
            current_sd = model.state_dict()
            for k in list(sd.keys()):
                if k in current_sd and sd[k].shape != current_sd[k].shape:
                    logger.info("Skipping %s (shape mismatch: ckpt %s vs model %s)", k, sd[k].shape, current_sd[k].shape)
                    del sd[k]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info("Loaded init_ckpt: %d missing, %d unexpected", len(missing), len(unexpected))
        logger.info("Initialized from %s (epoch %d)", args.init_ckpt, init_state.get("epoch", -1))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable params: %d (%.2fM)", n_params, n_params / 1e6)
    model = model.to(args.device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    best_val_a_acc = -1.0
    history = []
    for epoch in range(args.num_epochs):
        logger.info("=== Epoch %d / %d ===", epoch + 1, args.num_epochs)
        if bucket_sampler is not None:
            bucket_sampler.set_epoch(epoch)  # re-bucket/shuffle each epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, train_config, args.device, epoch)
        val_metrics = val_one_epoch(model, val_loader, train_config, args.device)
        scheduler.step()
        rec = {"epoch": epoch, **train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(rec)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        logger.info(
            "Epoch %d done: train_a=%.4f | val_a_loss=%.4f val_a_acc=%.4f",
            epoch + 1,
            train_metrics["a_loss"],
            val_metrics["val_a_loss"],
            val_metrics["val_a_acc"],
        )
        if val_metrics["val_a_acc"] > best_val_a_acc:
            best_val_a_acc = val_metrics["val_a_acc"]
            save_checkpoint(model, optimizer, -1, args.output_dir, rec).rename(args.output_dir / "best.ckpt")
            logger.info("New best val_a_acc=%.4f saved", best_val_a_acc)

    logger.info("Training complete. Best val_a_acc=%.4f", best_val_a_acc)


if __name__ == "__main__":
    sys.exit(main() or 0)
