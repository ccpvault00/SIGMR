#!/usr/bin/env python3
"""Train the in-situ co-diagnosis detector heads (the additional-finding route co-findings).

Two deployed heads, both trained directly on frozen H-Optimus tile features (no MUSK):
  - DCIS @ breast  ("ductal carcinoma in situ")     -> LSE head:  per-tile s = w*feat + b, pooled by
                                                        LogSumExp_tau; presence = pooled > threshold.
  - urothelial-CIS @ bladder ("urothelial carcinoma in situ") -> TileHead: per-tile MLP, top-k mean
                                                        pooled (matches the engine's _TileHead).

Both fire at the "additional finding" gate when the primary is the invasive form and the in-situ
entity is detected, injecting it as the #2 diagnosis. Population per entity: same-organ slides, label
1 iff the entity appears as a #2+ diagnosis, 0 iff absent (slides with it only as #1 are excluded).
Thresholds are calibrated high-precision (in-situ co-dx is rare; false launches would regress the
majority single-dx cases).

Usage:
    python scripts/train_codx_detector_heads.py \
        --cot-json     /mnt/data/reg2026/train_CoT.json \
        --split-json   /mnt/data/reg2026/checkpoints/phase_a_v3/split.json \
        --features-dir /mnt/data/reg2026/features/full_run/20x_224px_0px_overlap/features_hoptimus1 \
        --output-dir   /mnt/data/reg2026/checkpoints/codx_detector_heads

Emits dcis_breast.pt (LSE format {w,b,tau,threshold,...}) + cis_bladder.pt (TileHead {state_dict,...})
-> model/ckpts/, wired into cofinding_heads.json as type "lse" / "tilehead".
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CAP = 1200
DXQ = re.compile(r"#(\d+)\s+diagnosis")


@dataclass
class CodxSpec:
    cof: str
    organ: str
    phrase: str          # lower-case substring identifying the in-situ entity in the diagnosis text
    head_type: str       # "lse" | "tilehead"
    agg: str = "top10"   # tilehead pooling
    target_spec: float = 0.90   # lse threshold: min train specificity
    target_prec: float = 0.90   # tilehead threshold: min train precision


SPECS: list[CodxSpec] = [
    CodxSpec("dcis_breast", "breast", "ductal carcinoma in situ", "lse"),
    CodxSpec("cis_bladder", "bladder", "urothelial carcinoma in situ", "tilehead", agg="top10"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cot-json", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lse-steps", type=int, default=400)
    p.add_argument("--tilehead-epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="+", default=None, help="restrict to these cof keys")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def dx_list(case: dict) -> list[str]:
    out: dict[int, str] = {}
    for t in case["chain-of-thought"]:
        m = DXQ.search((t.get("question") or "").lower())
        if m:
            out[int(m.group(1))] = _norm(t.get("answer"))
    return [out[k] for k in sorted(out)]


def build_population(spec: CodxSpec, cot: list, split: dict, feat_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Same-organ slides. label=1 iff the entity phrase is a #2+ co-diagnosis; 0 iff absent;
    slides with the phrase only as #1 are excluded (neither class)."""
    trainval = {s.replace(".tiff", "") for s in split["train"]} | {s.replace(".tiff", "") for s in split["val"]}
    sids: list[str] = []
    label: dict[str, int] = {}
    for c in cot:
        if (c.get("organ") or "").strip().lower() != spec.organ:
            continue
        cid = c["id"].replace(".tiff", "")
        if cid not in trainval or not (feat_dir / f"{cid}.h5").exists():
            continue
        dl = dx_list(c)
        is_codx = any(spec.phrase in d for d in dl[1:])
        is_any = any(spec.phrase in d for d in dl)
        if is_codx:
            label[cid] = 1
        elif not is_any:
            label[cid] = 0
        else:
            continue
        sids.append(cid)
    return sids, label


def load_hopt(feat_dir: Path, cid: str, device: str) -> torch.Tensor:
    with h5py.File(feat_dir / f"{cid}.h5", "r") as f:
        feats = np.array(f["features"], dtype=np.float32)
    if CAP and len(feats) > CAP:
        feats = feats[np.linspace(0, len(feats) - 1, CAP).astype(int)]
    return torch.from_numpy(feats).to(device)


def split_ids(sids, split):
    tr = {s.replace(".tiff", "") for s in split["train"]}
    va = {s.replace(".tiff", "") for s in split["val"]}
    return [s for s in sids if s in tr], [s for s in sids if s in va]


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos), np.asarray(neg)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


# ----------------------------- LSE head (linear + LogSumExp pooling) -----------------------------
def train_lse_head(spec, train, val, label, feats, args) -> dict:
    dev = args.device
    Xtr = [feats[c] for c in train]
    ytr = torch.tensor([float(label[c]) for c in train], device=dev)
    w = torch.zeros(1536, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    raw_tau = torch.tensor(2.0, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b, raw_tau], lr=0.02, weight_decay=1e-3)
    posw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=dev)

    def pooled(X):
        tau = F.softplus(raw_tau).clamp(1, 50)
        return torch.stack([(torch.logsumexp(tau * (x @ w + b), 0) - np.log(len(x))) / tau for x in X])

    for _ in range(args.lse_steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(pooled(Xtr), ytr, pos_weight=posw).backward()
        opt.step()

    with torch.no_grad():
        ptr = pooled(Xtr).cpu().numpy()
        pva = pooled([feats[c] for c in val]).cpu().numpy() if val else np.array([])
    ytr_np = ytr.cpu().numpy()
    yva = np.array([label[c] for c in val]) if val else np.array([])

    # threshold on TRAIN: highest-recall point with specificity >= target
    thr, best_rec = None, -1.0
    for t in np.unique(ptr):
        pred = ptr >= t
        spec_v = ((pred == 0) & (ytr_np == 0)).sum() / max((ytr_np == 0).sum(), 1)
        rec = ((pred == 1) & (ytr_np == 1)).sum() / max((ytr_np == 1).sum(), 1)
        if spec_v >= spec.target_spec and rec > best_rec:
            best_rec, thr = rec, float(t)
    if thr is None:
        thr = float(np.quantile(ptr, 0.95))
    v_auc = auc(pva[yva == 1].tolist(), pva[yva == 0].tolist()) if len(yva) else float("nan")
    v_rec = float(((pva >= thr)[yva == 1]).mean()) if (yva == 1).any() else 0.0
    v_spec = float(((pva < thr)[yva == 0]).mean()) if (yva == 0).any() else 0.0
    v_prec = float((yva[pva >= thr] == 1).mean()) if (pva >= thr).any() else 0.0
    return {
        "w": w.detach().cpu(), "b": b.detach().cpu(),
        "tau": F.softplus(raw_tau).clamp(1, 50).detach().cpu(), "threshold": thr,
        "cofinding": spec.phrase, "organ": spec.organ,
        "val_auc": v_auc, "val_recall": v_rec, "val_spec": v_spec, "val_precision": v_prec,
    }


# ----------------------------- TileHead (MLP + top-k pooling) -----------------------------
class TileHead(nn.Module):
    def __init__(self, d_in: int = 1536, d_hid: int = 256, p_drop: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _topk(logits: np.ndarray, k: int) -> float:
    s = np.sort(logits)[::-1]
    return float(s[: min(k, len(s))].mean())


def train_tilehead(spec, train, val, label, feats, args) -> dict:
    dev = args.device
    torch.manual_seed(args.seed)
    k = int(spec.agg[3:]) if spec.agg.startswith("top") else 10
    n_pos = sum(label[c] for c in train)
    pos_w = torch.tensor([max(len(train) - n_pos, 1) / max(n_pos, 1)], device=dev)
    head = TileHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    bags = [(feats[c], float(label[c])) for c in train]
    rng = np.random.default_rng(args.seed)
    head.train()
    accum = 16
    for _ in range(args.tilehead_epochs):
        opt.zero_grad()
        for j, bi in enumerate(rng.permutation(len(bags))):
            x, lab = bags[bi]
            slide_logit = torch.topk(head(x), min(k, x.shape[0])).values.mean().unsqueeze(0)
            (loss_fn(slide_logit, torch.tensor([lab], device=dev)) / accum).backward()
            if (j + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
    head.eval()

    @torch.no_grad()
    def score(ids):
        return np.array([_topk(head(feats[c]).cpu().numpy(), k) for c in ids])
    str_ = score(train)
    ytr = np.array([label[c] for c in train])
    # threshold on TRAIN: highest-recall point with precision >= target
    order = np.argsort(-str_)
    tp = fp = 0
    thr, best_rec = None, -1.0
    for i in order:
        tp += int(ytr[i] == 1)
        fp += int(ytr[i] == 0)
        prec = tp / (tp + fp)
        rec = tp / max((ytr == 1).sum(), 1)
        if prec >= spec.target_prec and rec > best_rec:
            best_rec, thr = rec, float(str_[i])
    if thr is None:
        thr = float(np.quantile(str_, 0.95))
    sva = score(val) if val else np.array([])
    yva = np.array([label[c] for c in val]) if val else np.array([])
    v_auc = auc(sva[yva == 1].tolist(), sva[yva == 0].tolist()) if len(yva) else float("nan")
    return {
        "type": "tilehead", "cof": spec.cof, "d_in": 1536, "d_hid": 256, "cap": CAP, "agg": spec.agg,
        "state_dict": head.state_dict(), "threshold": thr,
        "cofinding": spec.phrase, "organ": spec.organ, "val_auc": v_auc,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cot = json.loads(args.cot_json.read_text())
    cot = cot if isinstance(cot, list) else list(cot.values())
    split = json.loads(args.split_json.read_text())
    specs = {s.cof: s for s in SPECS}
    targets = list(specs) if not args.only else [k for k in specs if k in set(args.only)]

    for cof in targets:
        spec = specs[cof]
        sids, label = build_population(spec, cot, split, args.features_dir)
        train, val = split_ids(sids, split)
        feats = {c: load_hopt(args.features_dir, c, args.device) for c in sids}
        n_tr_pos = sum(label[c] for c in train)
        logger.info("[%s] organ=%s type=%s | train=%d (pos=%d) val=%d (pos=%d)", cof, spec.organ, spec.head_type,
                    len(train), n_tr_pos, len(val), sum(label[c] for c in val))
        if n_tr_pos == 0:
            logger.warning("[%s] no train positives -> skip", cof)
            continue
        payload = train_lse_head(spec, train, val, label, feats, args) if spec.head_type == "lse" \
            else train_tilehead(spec, train, val, label, feats, args)
        out = args.output_dir / f"{cof}.pt"
        torch.save(payload, out)
        logger.info("[%s] SAVED %s -> thr=%.4f val_auc=%.3f", cof, spec.head_type, payload["threshold"], payload["val_auc"])

    logger.info("Done. Copy <cof>.pt to model/ckpts/ and wire into cofinding_heads.json (type lse / tilehead).")


if __name__ == "__main__":
    main()
