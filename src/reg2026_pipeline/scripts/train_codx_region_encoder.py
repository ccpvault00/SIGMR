#!/usr/bin/env python3
"""Stage 1 of the region co-diagnosis integrator: a region-level diagnosis encoder.

Splits every single-diagnosis training slide into 5 mm grid regions and trains an ABMIL classifier
to predict the slide's diagnosis from each region's tiles. This teaches the ABMIL to encode a
*region* (not the whole slide) into a diagnosis-discriminative embedding. Its encoder warm-starts
the joint region co-dx integrator (Stage 2, train_codx_integrator.py).

Architecture matches the #1-dx primary (ABMIL 1536->1024 + classifier), warm-started from it.

Usage:
    python scripts/train_codx_region_encoder.py \
        --cot-json      /mnt/data/reg2026/train_CoT.json \
        --split-json    /mnt/data/reg2026/checkpoints/phase_a_v3/split.json \
        --features-dir  /mnt/data/reg2026/features/full_run/20x_224px_0px_overlap/features_hoptimus1 \
        --primary-ckpt  model/ckpts/abmil_primary_dx.ckpt \
        --output-dir    /mnt/data/reg2026/checkpoints/codx_region_encoder \
        --epochs 8

Output: <output-dir>/best.ckpt {model_state, dx_vocab} - fed to Stage 2 as --region-encoder-ckpt.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from reg2026.aggregate.abmil import ABMIL, ABMILConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGION_PX = 10080  # region edge in level-0 px (@0.5 mpp ~= 5 mm)
MIN_T = 40         # min tiles for a region to be kept
MAX_T = 1024       # max tiles per region (subsample the region's bag)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cot-json", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--primary-ckpt", type=Path, required=True, help="abmil_primary_dx.ckpt - warm-start + dx_vocab")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="a few steps + save; confirms the path runs")
    return p.parse_args()


def dx_list(case: dict) -> list[str]:
    out: list[str] = []
    for k in range(1, 6):
        for s in case["chain-of-thought"]:
            if s["question"] == f"What is the #{k} diagnosis?":
                out.append(s["answer"].strip())
                break
    return out


def regions_of(feat_dir: Path, cid: str) -> list[np.ndarray]:
    """Sorted tile-index arrays for each 5 mm grid cell holding >= MIN_T tiles."""
    with h5py.File(feat_dir / f"{cid}.h5", "r") as h:
        c = h["coords"][:]
    cx = ((c[:, 0] - c[:, 0].min()) // REGION_PX).astype(int)
    cy = ((c[:, 1] - c[:, 1].min()) // REGION_PX).astype(int)
    cell = cx * 10000 + cy
    return [np.sort(np.where(cell == u)[0]) for u in np.unique(cell) if int((cell == u).sum()) >= MIN_T]


class RegionDataset(Dataset):
    """One item = one region's tiles + the slide's (single) diagnosis index."""

    def __init__(self, index: list[tuple[str, np.ndarray, int]], feat_dir: Path) -> None:
        self.ix = index
        self.feat_dir = feat_dir

    def __len__(self) -> int:
        return len(self.ix)

    def __getitem__(self, i: int):
        cid, tiles, lab = self.ix[i]
        with h5py.File(self.feat_dir / f"{cid}.h5", "r") as h:
            f = h["features"][tiles]
        f = torch.from_numpy(f).float()
        if len(f) > MAX_T:
            f = f[torch.randperm(len(f))[:MAX_T]]
        return f, lab


def collate(b):
    fs, labs = zip(*b)
    m = max(len(f) for f in fs)
    out = torch.zeros(len(fs), m, 1536)
    msk = torch.zeros(len(fs), m, dtype=torch.bool)
    for i, f in enumerate(fs):
        out[i, : len(f)] = f
        msk[i, : len(f)] = True
    return out, msk, torch.tensor(labs)


class RegionClassifier(nn.Module):
    """ABMIL(1536->1024) + classifier - same architecture as the #1-dx primary."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.abmil = ABMIL(ABMILConfig())
        self.classifier = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.25), nn.Linear(256, n))

    def forward(self, x, msk):
        e, _ = self.abmil(x, msk)
        return self.classifier(e)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cot = {(c["id"][:-5] if c["id"].endswith(".tiff") else c["id"]): c for c in json.loads(args.cot_json.read_text())}
    split = json.loads(args.split_json.read_text())
    ck = torch.load(args.primary_ckpt, map_location="cpu", weights_only=False)
    dxv = ck["dx_vocab"]
    n = len(dxv)
    a2i = {d: i for i, d in enumerate(dxv)}

    # train index: each region of a single-dx slide, labelled with the slide's dx.
    index: list[tuple[str, np.ndarray, int]] = []
    for cid in split["train"]:
        c = cot.get(cid)
        if not c:
            continue
        dl = dx_list(c)
        if len(dl) != 1 or dl[0] not in a2i:
            continue
        if not (args.features_dir / f"{cid}.h5").exists():
            continue
        for tiles in regions_of(args.features_dir, cid):
            index.append((cid, tiles, a2i[dl[0]]))
    logger.info("train regions: %d from single-dx slides | %d dx classes", len(index), n)

    # inverse-sqrt-frequency class weights
    cnt = np.bincount([lab for _, _, lab in index], minlength=n).astype(float)
    w = 1.0 / np.sqrt(cnt + 1)
    w = w / w.mean()
    cw = torch.tensor(w, dtype=torch.float32, device=args.device)

    model = RegionClassifier(n)
    model.load_state_dict(ck["model_state"], strict=True)  # warm-start from the #1-dx primary
    model = model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    dl = DataLoader(RegionDataset(index, args.features_dir), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate, pin_memory=False)

    epochs = 1 if args.smoke else args.epochs
    for ep in range(epochs):
        model.train()
        tot = cor = 0
        run_loss = 0.0
        for i, (x, msk, y) in enumerate(dl):
            x, msk, y = x.to(args.device), msk.to(args.device), y.to(args.device)
            opt.zero_grad()
            lo = model(x, msk)
            loss = F.cross_entropy(lo, y, weight=cw)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            cor += int((lo.argmax(1) == y).sum().item())
            tot += len(y)
            if args.smoke and i >= 2:
                logger.info("SMOKE: 3 train steps OK (loss=%.4f)", loss.item())
                break
        logger.info("epoch %d: region train_acc=%.3f loss=%.3f", ep, cor / max(tot, 1), run_loss / max(i + 1, 1))
        torch.save({"model_state": model.state_dict(), "dx_vocab": dxv, "epoch": ep}, args.output_dir / "best.ckpt")
        if args.smoke:
            logger.info("SMOKE complete - model + data + train + save all run.")
            return
    logger.info("Done. Region encoder -> %s/best.ckpt (feed to Stage 2 as --region-encoder-ckpt).", args.output_dir)


if __name__ == "__main__":
    main()
