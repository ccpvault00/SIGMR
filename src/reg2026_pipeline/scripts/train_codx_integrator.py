#!/usr/bin/env python3
"""Stage 2 of the region co-diagnosis integrator (the deployed codx_region_integrator.ckpt).

Two-level MIL trained end to end: each 5 mm region is ABMIL-pooled into a region embedding by the
encoder (`enc`, warm-started from the Stage-1 region encoder); a learned per-base-diagnosis anchor
prototype (selected by the #1-diagnosis primary) cross-attends over the region embeddings and a head
predicts the reported co-diagnosis SET (multilabel over the genuine-diagnosis vocabulary). A
region-classification aux loss on single-diagnosis slides keeps the region embeddings discriminative.

The trained model + its base/genuine vocabularies are what interf1/codx_integrator.py loads
(`_RegionCoDxModel` + {state, bvocab, gvocab}).

Usage:
    python scripts/train_codx_integrator.py \
        --cot-json            /mnt/data/reg2026/train_CoT.json \
        --split-json          /mnt/data/reg2026/checkpoints/phase_a_v3/split.json \
        --features-dir        /mnt/data/reg2026/features/full_run/20x_224px_0px_overlap/features_hoptimus1 \
        --primary-ckpt        model/ckpts/abmil_primary_dx.ckpt \
        --region-encoder-ckpt /mnt/data/reg2026/checkpoints/codx_region_encoder/best.ckpt \
        --output-dir          /mnt/data/reg2026/checkpoints/codx_region_integrator \
        --epochs 6

Output: <output-dir>/best.ckpt {state, bvocab, gvocab} -> model/ckpts/cofinding/codx_region_integrator.ckpt.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
MIN_T = 40         # min tiles per region
MAX_T = 512        # max tiles per region bag
R_MAX = 16         # max regions per slide (subsample; caps cross-attention input)
PRIMARY_CAP = 2048  # tile cap for the #1-dx primary forward (matches inference)
NORMAL = {"no tumor present", "no evidence of tumor"}
# presence-Q / non-diagnostic findings excluded from the genuine co-dx target vocabulary
EXCLUDE_KEYS = ["microcalcification", "inflammation", "infection", "granulomatous", "fungal", "pneumocyte", "fibrosis"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cot-json", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--primary-ckpt", type=Path, required=True, help="abmil_primary_dx.ckpt (#1-dx anchor + dx_vocab)")
    p.add_argument("--region-encoder-ckpt", type=Path, required=True, help="Stage-1 codx_region_encoder best.ckpt (enc warm-start)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--aux-lambda", type=float, default=0.3, help="region-cls aux loss weight")
    p.add_argument("--accum", type=int, default=16, help="gradient accumulation steps (batch size is 1 slide)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="a few steps + save; confirms the path runs")
    return p.parse_args()


def _base(d: str) -> str:
    d = re.sub(r",?\s*grade\s+\S+\s*$", "", d, flags=re.I)
    d = re.sub(r",?\s*(well|moderately|poorly|undifferentiated)\s+differentiated\s*$", "", d, flags=re.I)
    return re.sub(r",?\s*(high|low|intermediate)[- ]grade\s*$", "", d, flags=re.I).strip().lower()


def _is_presence_q(entity: str) -> bool:
    return any(k in entity for k in EXCLUDE_KEYS)


def dx_list(case: dict) -> list[str]:
    out: list[str] = []
    for k in range(1, 6):
        for s in case["chain-of-thought"]:
            if s["question"] == f"What is the #{k} diagnosis?":
                out.append(s["answer"].strip())
                break
    return out


def regions_idx(feat_dir: Path, cid: str) -> list[np.ndarray]:
    with h5py.File(feat_dir / f"{cid}.h5", "r") as h:
        c = h["coords"][:]
    cx = ((c[:, 0] - c[:, 0].min()) // REGION_PX).astype(int)
    cy = ((c[:, 1] - c[:, 1].min()) // REGION_PX).astype(int)
    cell = cx * 10000 + cy
    return [np.sort(np.where(cell == u)[0]) for u in np.unique(cell) if int((cell == u).sum()) >= MIN_T]


class PrimaryClf(nn.Module):
    """The #1-dx primary (ABMIL 1536->1024 + classifier) - used only to pick the anchor prototype."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.abmil = ABMIL(ABMILConfig())
        self.classifier = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.25), nn.Linear(256, n))

    def forward(self, x, msk):
        e, _ = self.abmil(x, msk)
        return self.classifier(e)


class RegionCoDxModel(nn.Module):
    """Joint region encoder + integrator (identical to interf1/codx_integrator._RegionCoDxModel)."""

    def __init__(self, B: int, G: int, n: int, d: int = 256, h: int = 4) -> None:
        super().__init__()
        self.enc = ABMIL(ABMILConfig())            # 1536 -> 1024 region embedding
        self.proto = nn.Embedding(B, 1024)          # learned per-base-dx #1 anchor
        self.rp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d), nn.GELU())
        self.qp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d), nn.GELU())
        self.xa = nn.MultiheadAttention(d, h, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, G))
        self.rcls = nn.Linear(1024, n)              # region-cls aux

    def encode(self, bags, dev):
        embs = []
        for f in bags:
            x = f.to(dev).unsqueeze(0)
            e, _ = self.enc(x, torch.ones(1, x.shape[1], dtype=torch.bool, device=dev))
            embs.append(e[0])
        return torch.stack(embs)  # [R, 1024]

    def forward(self, bags, a1id, dev):
        reg = self.encode(bags, dev).unsqueeze(0)               # [1, R, 1024]
        proto = self.proto(torch.tensor([a1id], device=dev))    # [1, 1024]
        K = self.rp(reg)
        q = self.qp(proto).unsqueeze(1)
        att, _ = self.xa(q, K, K)
        setlog = self.head(torch.cat([self.qp(proto), att.squeeze(1)], -1))
        rcls = self.rcls(reg.squeeze(0))                        # [R, n]
        return setlog, rcls


class SlideDataset(Dataset):
    """One item = one slide's region bags + genuine-dx multihot target. The #1-dx anchor is resolved
    in the training loop (main process) because it needs the CUDA primary model, which cannot run in
    a forked DataLoader worker."""

    def __init__(self, ids, feat_dir, cot, g2i, G) -> None:
        self.ids = ids
        self.feat_dir = feat_dir
        self.cot = cot
        self.g2i = g2i
        self.G = G

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        cid = self.ids[i]
        with h5py.File(self.feat_dir / f"{cid}.h5", "r") as h:
            feats = h["features"][:]
        ri = regions_idx(self.feat_dir, cid)
        if len(ri) > R_MAX:
            ri = [ri[j] for j in np.random.default_rng(0).choice(len(ri), R_MAX, replace=False)]
        bags = []
        for t in ri:
            f = feats[t]
            if len(f) > MAX_T:
                f = f[np.random.default_rng(0).choice(len(f), MAX_T, replace=False)]
            bags.append(torch.from_numpy(f).float())
        gl = [_base(d) for d in dx_list(self.cot[cid])]
        y = np.zeros(self.G, dtype=np.float32)
        for g in gl:
            if g in self.g2i:
                y[self.g2i[g]] = 1
        return cid, bags, torch.from_numpy(y), gl


def _first(b):
    return b[0]  # batch size = 1 slide


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = args.device

    cot = {(c["id"][:-5] if c["id"].endswith(".tiff") else c["id"]): c for c in json.loads(args.cot_json.read_text())}
    split = json.loads(args.split_json.read_text())
    ck = torch.load(args.primary_ckpt, map_location="cpu", weights_only=False)
    dxv = ck["dx_vocab"]
    n = len(dxv)

    primary = PrimaryClf(n)
    primary.load_state_dict(ck["model_state"], strict=True)
    primary.eval().to(dev)

    # lazily compute + cache each slide's #1-dx (base), matching the deployed anchor selection.
    _ab1_cache: dict[str, str] = {}

    @torch.no_grad()
    def ab1(cid: str) -> str:
        if cid not in _ab1_cache:
            with h5py.File(args.features_dir / f"{cid}.h5", "r") as h:
                f = h["features"][:]
            if len(f) > PRIMARY_CAP:
                f = f[np.random.default_rng(0).choice(len(f), PRIMARY_CAP, replace=False)]
            x = torch.from_numpy(f).float().to(dev).unsqueeze(0)
            logit = primary(x, torch.ones(1, x.shape[1], dtype=torch.bool, device=dev))[0]
            _ab1_cache[cid] = _base(dxv[int(logit.argmax())])
        return _ab1_cache[cid]

    # vocabularies: genuine co-dx targets (base dxs seen in train+val, excl presence-Q + normal) + base-dx anchors
    gtset: set[str] = set()
    for cid in split["train"] + split["val"]:
        if cid in cot:
            for d in dx_list(cot[cid]):
                gtset.add(_base(d))
    gvocab = sorted(g for g in gtset if not _is_presence_q(g) and g not in NORMAL)
    g2i = {g: i for i, g in enumerate(gvocab)}
    G = len(gvocab)
    bvocab = sorted({_base(d) for d in dxv})
    b2i = {b: i for i, b in enumerate(bvocab)}
    B = len(bvocab)

    # train slides: >=1 genuine-dx GT + at least one region.
    def buildable(cid: str) -> bool:
        if cid not in cot or not (args.features_dir / f"{cid}.h5").exists():
            return False
        if not any(_base(d) in g2i for d in dx_list(cot[cid])):
            return False
        return bool(regions_idx(args.features_dir, cid))

    train = [cid for cid in split["train"] if buildable(cid)]
    logger.info("train slides=%d | G(genuine dx)=%d B(base dx)=%d", len(train), G, B)

    model = RegionCoDxModel(B, G, n).to(dev)
    # warm-start the region encoder from Stage 1
    enc_ck = torch.load(args.region_encoder_ckpt, map_location="cpu", weights_only=False)
    enc_sd = {k[len("abmil."):]: v for k, v in enc_ck["model_state"].items() if k.startswith("abmil.")}
    model.enc.load_state_dict(enc_sd)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    dl = DataLoader(SlideDataset(train, args.features_dir, cot, g2i, G), batch_size=1, shuffle=True,
                    num_workers=args.num_workers, collate_fn=_first)

    epochs = 1 if args.smoke else args.epochs
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        tl = 0.0
        nb = 0
        for step, (cid, bags, y, gl) in enumerate(dl):
            if not bags:
                continue
            a1id = b2i.get(ab1(cid), 0)  # resolve the #1-dx anchor here (main process; CUDA primary)
            setlog, rcls = model(bags, a1id, dev)
            loss = F.binary_cross_entropy_with_logits(setlog, y.to(dev).unsqueeze(0))
            # region-cls aux on single-dx slides (slide dx as every region's label)
            if len(set(gl)) == 1:
                slab = next((i for i, d in enumerate(dxv) if _base(d) == gl[0]), None)
                if slab is not None:
                    loss = loss + args.aux_lambda * F.cross_entropy(rcls, torch.full((rcls.shape[0],), slab, device=dev))
            (loss / args.accum).backward()
            tl += loss.item()
            nb += 1
            if (step + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
            if args.smoke and step >= 2:
                logger.info("SMOKE: 3 train steps OK (loss=%.4f)", loss.item())
                break
        opt.step()
        opt.zero_grad()
        logger.info("epoch %d: loss=%.4f", ep, tl / max(nb, 1))
        torch.save({"state": model.state_dict(), "bvocab": bvocab, "gvocab": gvocab}, args.output_dir / "best.ckpt")
        if args.smoke:
            logger.info("SMOKE complete - vocab + primary anchor + region warm-start + train + save all run.")
            return
    logger.info("Done. Integrator -> %s/best.ckpt. Copy to model/ckpts/cofinding/codx_region_integrator.ckpt.", args.output_dir)


if __name__ == "__main__":
    main()
