#!/usr/bin/env python3
"""Train the single-label ABMIL primary #1-diagnosis predictor (deployed `abmil_primary_dx.ckpt`).

The deployed pipeline routes this model's argmax to the #1-diagnosis node and the Final Report.
It is a plain single-label classifier: frozen H-Optimus tile features -> ABMIL slide embedding ->
softmax over the graded-diagnosis vocabulary, trained with cross-entropy on the primary (#1) dx.

The graded-dx vocabulary is derived here from the training split's #1..#4 diagnosis answers (no
external dependency), and is stored inside the checkpoint so the inference engine reads it back.

Usage:
    python scripts/train_abmil_primary.py \
        --cot-json     /mnt/data/reg2026/train_CoT.json \
        --features-dir /mnt/data/reg2026/features/full_run/20x_224px_0px_overlap/features_hoptimus1 \
        --split-json   /mnt/data/reg2026/checkpoints/phase_a_v3/split.json \
        --output-dir   /mnt/data/reg2026/checkpoints/abmil_primary_dx \
        --num-epochs 20 --batch-size 4 --num-workers 8 --max-tiles 8192

The best checkpoint (`<output-dir>/best.ckpt`) becomes `model/ckpts/abmil_primary_dx.ckpt`.
Add `--smoke` for a 2-step + val dry run.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from reg2026.aggregate.abmil import ABMIL, ABMILConfig
from reg2026.data.embedding_pools import EmbeddingPools

logger = logging.getLogger(__name__)

DX_Q_TEXTS = {f"What is the #{k} diagnosis?" for k in range(1, 5)}

# Grade / differentiation suffix patterns - strip to get the BASE diagnosis (for the base-top1 metric).
_GRADE_NUM_PAT = re.compile(r",?\s*grade\s+(I{1,3}V?|IV|\d+)\s*$", re.IGNORECASE)
_HL_GRADE_PAT = re.compile(r",?\s*(high|low|intermediate)\s+grade\s*$", re.IGNORECASE)
_DIFF_PAT = re.compile(r",?\s*(well|moderately|poorly|undifferentiated)\s+differentiated\s*$", re.IGNORECASE)


def strip_grade_diff(dx_text: str) -> str:
    """Strip the grade / differentiation suffix to get the base diagnosis."""
    for pat in (_GRADE_NUM_PAT, _HL_GRADE_PAT, _DIFF_PAT):
        dx_text = pat.sub("", dx_text).strip()
    return dx_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cot-json", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-mode", choices=["all", "single-only"], default="all",
                   help="all = train on every case's #1 dx (single + multi primary). single-only = drop multi-dx cases.")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--max-tiles", type=int, default=8192, help="Uniformly subsample tiles per case to cap monster-slide memory (0=all).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="1 epoch, a couple of steps, then a val pass; confirms the whole path runs.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Label + vocabulary extraction                                               #
# --------------------------------------------------------------------------- #


def build_case_dx_lists(cot_by_id: dict[str, dict]) -> dict[str, list[str]]:
    """case_id -> ORDERED list of graded dx texts (deduped, #1 first) from the #1..#4 diagnosis answers."""
    out: dict[str, list[str]] = {}
    for cid, c in cot_by_id.items():
        seen: list[str] = []
        for t in c["chain-of-thought"]:
            if t["question"] in DX_Q_TEXTS:
                a = t["answer"].strip()
                if a and a not in seen:
                    seen.append(a)
        out[cid] = seen
    return out


def build_dx_vocab(case_dx_graded: dict[str, list[str]], train_ids: list[str]) -> list[str]:
    """Graded-dx vocabulary = every distinct graded diagnosis appearing in the TRAIN split's
    #1..#4 diagnosis answers, sorted for determinism. These are the classes the primary head emits."""
    vocab: set[str] = set()
    for cid in train_ids:
        for dx in case_dx_graded.get(cid, []):
            vocab.add(dx)
    return sorted(vocab)


def build_dx_text_init(dx_vocab: list[str], pools: EmbeddingPools) -> torch.Tensor:
    """[n_dx, emb] init for the classifier text-prior from A_EMB lookup; small-random for unknown."""
    embs = []
    for dx in dx_vocab:
        a_idx = pools._a_to_idx.get(dx, -1)
        embs.append(pools.A_EMB[a_idx] if a_idx >= 0 else torch.randn(pools.A_EMB.shape[1]) * 0.02)
    return torch.stack(embs)  # [n_dx, emb_dim]


# --------------------------------------------------------------------------- #
# Model: ABMIL -> graded-dx softmax (single-label)                            #
# --------------------------------------------------------------------------- #


class SingleLabelPrimary(nn.Module):
    """ABMIL(tiles) -> slide_emb -> softmax classifier over the graded dx vocab.

    Trained with single-label cross-entropy on the #1 (primary) diagnosis - a pure single-label
    objective with no multi-label dilution or co-diagnosis competition.
    """

    def __init__(self, n_dx: int, tile_dim: int = 1536, embed_dim: int = 256, n_heads: int = 4, dropout: float = 0.25) -> None:
        super().__init__()
        self.n_dx = n_dx
        self.abmil = ABMIL(ABMILConfig(in_dim=tile_dim, embed_dim=embed_dim, num_heads=n_heads, dropout=dropout))
        slide_dim = self.abmil.config.output_dim  # n_heads * embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(slide_dim),
            nn.Linear(slide_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_dx),
        )

    def forward(self, tiles: torch.Tensor, tile_mask: torch.Tensor) -> torch.Tensor:
        """tiles [B, N, 1536], tile_mask [B, N] bool (True=valid) -> primary logits [B, n_dx]."""
        slide_emb, _ = self.abmil(tiles, mask=tile_mask)
        return self.classifier(slide_emb)


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #


class PrimaryDataset(Dataset):
    """Yields tiles + primary (graded) index + full GT graded-idx set + organ.

    `keep_multi=False` drops multi-dx cases entirely (single-only ablation).
    """

    def __init__(
        self,
        case_ids: list[str],
        features_dir: Path,
        case_dx_graded: dict[str, list[str]],
        dx_to_idx: dict[str, int],
        case_organ: dict[str, str],
        max_tiles: int = 0,
        keep_multi: bool = True,
    ) -> None:
        self.features_dir = features_dir
        self.dx_to_idx = dx_to_idx
        self.case_organ = case_organ
        self.max_tiles = max_tiles
        self.case_dx_graded = case_dx_graded
        self.case_ids: list[str] = []
        for cid in case_ids:
            dxs = case_dx_graded.get(cid)
            if not dxs:
                continue
            # The primary must be a class the model can emit (in-vocab).
            if dxs[0] not in dx_to_idx:
                continue
            if not keep_multi and len(dxs) >= 2:
                continue
            if not (features_dir / f"{cid}.h5").exists():
                continue
            self.case_ids.append(cid)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, i: int) -> dict:
        cid = self.case_ids[i]
        with h5py.File(self.features_dir / f"{cid}.h5", "r") as fh:
            tiles = torch.from_numpy(fh["features"][...]).float()  # [N, 1536]
        if self.max_tiles and tiles.shape[0] > self.max_tiles:
            idx = torch.linspace(0, tiles.shape[0] - 1, self.max_tiles).long()  # deterministic even subsample
            tiles = tiles[idx]
        dxs = self.case_dx_graded[cid]
        primary = self.dx_to_idx[dxs[0]]
        gt_set_idx = [self.dx_to_idx[d] for d in dxs if d in self.dx_to_idx]  # for lenient top1_in_gt
        return {
            "case_id": cid,
            "tiles": tiles,
            "primary": primary,
            "gt_set_idx": gt_set_idx,
            "n_dx_gt": len(dxs),
            "organ": self.case_organ.get(cid, "unknown"),
        }


def collate(samples: list[dict]) -> dict:
    B = len(samples)
    max_t = max(s["tiles"].shape[0] for s in samples)
    feat_dim = samples[0]["tiles"].shape[1]
    tiles = torch.zeros(B, max_t, feat_dim)
    mask = torch.zeros(B, max_t, dtype=torch.bool)
    for b, s in enumerate(samples):
        n = s["tiles"].shape[0]
        tiles[b, :n] = s["tiles"]
        mask[b, :n] = True
    return {
        "case_ids": [s["case_id"] for s in samples],
        "tiles": tiles,
        "mask": mask,
        "primary": torch.tensor([s["primary"] for s in samples], dtype=torch.long),
        "gt_set_idx": [s["gt_set_idx"] for s in samples],
        "n_dx_gt": torch.tensor([s["n_dx_gt"] for s in samples], dtype=torch.long),
        "organ": [s["organ"] for s in samples],
    }


# --------------------------------------------------------------------------- #
# Evaluation (graded / base top-1, per-organ)                                 #
# --------------------------------------------------------------------------- #

ORGANS = ["prostate", "breast", "colon", "stomach", "bladder", "lung", "cervix"]


def _empty_counters() -> dict[str, dict[str, int]]:
    """Per-organ + overall accumulators for the three metrics."""
    keys = ["graded_correct", "base_correct", "inset_correct", "total", "multi_total", "multi_graded_correct"]
    buckets = {o: {k: 0 for k in keys} for o in ORGANS}
    buckets["__all__"] = {k: 0 for k in keys}
    return buckets


def _accumulate(buckets: dict[str, dict[str, int]], organ: str, pred_idx: int, primary_idx: int,
                gt_set_idx: list[int], base_vocab: list[str], n_dx_gt: int) -> None:
    graded = int(pred_idx == primary_idx)
    base = int(base_vocab[pred_idx] == base_vocab[primary_idx])  # base(pred) == base(GT#1)
    inset = int(pred_idx in set(gt_set_idx))  # lenient: argmax anywhere in GT set
    for tgt in (buckets.get(organ), buckets["__all__"]):
        if tgt is None:
            continue
        tgt["graded_correct"] += graded
        tgt["base_correct"] += base
        tgt["inset_correct"] += inset
        tgt["total"] += 1
        if n_dx_gt >= 2:
            tgt["multi_total"] += 1
            tgt["multi_graded_correct"] += graded


def _finalize(buckets: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, c in buckets.items():
        tot = max(c["total"], 1)
        mt = max(c["multi_total"], 1)
        out[name] = {
            "graded_top1": c["graded_correct"] / tot,
            "base_top1": c["base_correct"] / tot,
            "top1_in_gtset": c["inset_correct"] / tot,
            "multi_graded_top1": c["multi_graded_correct"] / mt if c["multi_total"] else 0.0,
            "n": c["total"],
            "n_multi": c["multi_total"],
        }
    return out


@torch.inference_mode()
def evaluate_single_label(model: SingleLabelPrimary, loader: DataLoader, device: str, base_vocab: list[str]) -> dict:
    model.eval()
    buckets = _empty_counters()
    for batch in loader:
        tiles = batch["tiles"].to(device)
        mask = batch["mask"].to(device)
        logits = model(tiles, mask)  # [B, n_dx]
        pred = logits.argmax(dim=-1)  # [B]
        for b in range(tiles.shape[0]):
            _accumulate(buckets, batch["organ"][b], int(pred[b].item()), int(batch["primary"][b].item()),
                        batch["gt_set_idx"][b], base_vocab, int(batch["n_dx_gt"][b].item()))
    return _finalize(buckets)


def _fmt_table(name: str, m: dict[str, dict[str, float]]) -> str:
    lines = [f"=== {name} ==="]
    lines.append(f"  {'organ':<10} {'n':>5} {'graded':>8} {'base':>8} {'in_gt':>8} {'multi_n':>8} {'multi_g':>8}")
    allm = m["__all__"]
    lines.append(f"  {'ALL':<10} {allm['n']:>5} {allm['graded_top1']:>8.4f} {allm['base_top1']:>8.4f} {allm['top1_in_gtset']:>8.4f} {allm['n_multi']:>8} {allm['multi_graded_top1']:>8.4f}")
    for o in ORGANS:
        om = m[o]
        if om["n"] == 0:
            continue
        lines.append(f"  {o:<10} {om['n']:>5} {om['graded_top1']:>8.4f} {om['base_top1']:>8.4f} {om['top1_in_gtset']:>8.4f} {om['n_multi']:>8} {om['multi_graded_top1']:>8.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Train                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pools + cot + split")
    pools = EmbeddingPools()
    cot_raw = json.loads(args.cot_json.read_text())
    cot_by_id_full = cot_raw if isinstance(cot_raw, dict) else {c["id"]: c for c in cot_raw}
    cot_by_id = {k.removesuffix(".tiff"): v for k, v in cot_by_id_full.items()}
    case_organ = {cid: c.get("organ", "unknown") for cid, c in cot_by_id.items()}
    split = json.loads(args.split_json.read_text())

    case_dx_graded = build_case_dx_lists(cot_by_id)

    # Graded-dx vocabulary built from the TRAIN split (self-contained; stored in the checkpoint).
    dx_vocab = build_dx_vocab(case_dx_graded, split["train"])
    dx_to_idx = {dx: i for i, dx in enumerate(dx_vocab)}
    n_dx = len(dx_vocab)
    base_vocab = [strip_grade_diff(dx) for dx in dx_vocab]  # graded-idx -> base text, for base_top1
    logger.info("Graded dx vocab (from train split): %d classes | %d distinct base classes", n_dx, len(set(base_vocab)))
    (args.output_dir / "dx_vocab.json").write_text(json.dumps(dx_vocab, indent=2))

    keep_multi = args.data_mode == "all"
    train_ds = PrimaryDataset(split["train"], args.features_dir, case_dx_graded, dx_to_idx, case_organ, max_tiles=args.max_tiles, keep_multi=keep_multi)
    # Val ALWAYS includes every case (single + multi) regardless of data-mode - we
    # measure primary on the full held-out distribution.
    val_ds = PrimaryDataset(split["val"], args.features_dir, case_dx_graded, dx_to_idx, case_organ, max_tiles=args.max_tiles, keep_multi=True)
    logger.info("data-mode=%s | Train: %d cases (keep_multi=%s) | Val: %d cases", args.data_mode, len(train_ds), keep_multi, len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, drop_last=False, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=False)

    model = SingleLabelPrimary(n_dx, embed_dim=args.embed_dim, n_heads=args.n_heads, dropout=args.dropout).to(args.device)

    # Text-init the final classifier weight rows from A_EMB (graded dx text prior),
    # projected to slide_emb space - gives a sensible starting decision boundary.
    with torch.no_grad():
        dx_text = build_dx_text_init(dx_vocab, pools)  # [n_dx, 768]
        if dx_text.shape[1] == model.classifier[-1].in_features:
            model.classifier[-1].weight.copy_(dx_text)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("SingleLabelPrimary params: %d (%.2fM)", n_params, n_params / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    best_graded = -1.0
    history: list[dict] = []
    num_epochs = 1 if args.smoke else args.num_epochs
    for epoch in range(num_epochs):
        model.train()
        tot_loss = 0.0
        n_b = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"ep{epoch + 1}/{num_epochs}")):
            tiles = batch["tiles"].to(args.device)
            mask = batch["mask"].to(args.device)
            primary_gt = batch["primary"].to(args.device)
            logits = model(tiles, mask)
            loss = F.cross_entropy(logits, primary_gt)  # SINGLE-LABEL objective
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()
            n_b += 1
            if args.smoke and step >= 1:
                logger.info("SMOKE: 2 train steps OK (loss=%.4f)", loss.item())
                break
        scheduler.step()

        val_m = evaluate_single_label(model, val_loader, args.device, base_vocab)
        graded = val_m["__all__"]["graded_top1"]
        base = val_m["__all__"]["base_top1"]
        inset = val_m["__all__"]["top1_in_gtset"]
        rec = {"epoch": epoch, "train_loss": tot_loss / max(n_b, 1), "graded_top1": graded, "base_top1": base,
               "top1_in_gtset": inset, "lr": optimizer.param_groups[0]["lr"], "per_organ": val_m}
        history.append(rec)
        logger.info("Ep %2d/%d: loss=%.4f | graded_top1=%.4f base_top1=%.4f in_gtset=%.4f", epoch + 1, num_epochs, rec["train_loss"], graded, base, inset)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

        if args.smoke:
            logger.info("\n%s", _fmt_table("SMOKE single-label predictor", val_m))
            logger.info("SMOKE complete - model + data + train + eval paths all run.")
            return

        # Selection: parameter-free primary graded-top1 (nothing to overfit).
        if graded > best_graded:
            best_graded = graded
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_metrics": val_m, "dx_vocab": dx_vocab,
                 "base_vocab": base_vocab, "data_mode": args.data_mode, "args": vars(args)},
                args.output_dir / "best.ckpt",
            )
            (args.output_dir / "eval_best.json").write_text(json.dumps({"epoch": epoch, **val_m}, indent=2))
            logger.info("  -> New best (graded_top1=%.4f) saved", best_graded)

    best = json.loads((args.output_dir / "eval_best.json").read_text())
    best_m = {k: v for k, v in best.items() if k != "epoch"}
    logger.info("\n%s", _fmt_table(f"BEST single-label predictor (ep{best['epoch']}, data-mode={args.data_mode})", best_m))
    logger.info("Done. graded_top1=%.4f base=%.4f. Copy %s/best.ckpt -> model/ckpts/abmil_primary_dx.ckpt",
                best_m["__all__"]["graded_top1"], best_m["__all__"]["base_top1"], args.output_dir)


if __name__ == "__main__":
    main()
