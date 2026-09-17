#!/usr/bin/env python3
"""Train the deployable co-finding tile-head library (H-Optimus-only, no MUSK).

Each co-finding is a small per-tile MLP (LayerNorm -> Linear -> GELU -> Dropout -> Linear) trained
by top-k-pooled multiple-instance BCE directly on the frozen H-Optimus tile features. A head fires
at its organ-specific presence question (e.g. "is there any microcalcification present?") when the
top-k pooled slide score clears a threshold calibrated at train precision >= 0.80.

Produces the presence-Q co-finding heads used by model/configs/cofinding_heads.json
(microcalc_breast, granulomatous_fb_bladder, organizing_fibrosis_lung). The in-situ co-dx heads
(DCIS / urothelial-CIS, additional-finding route) use the same TileHead architecture but a co-dx
population + threshold; they are trained by the co-dx detector script.

Usage:
    python scripts/train_cofinding_heads.py \
        --cot-json     /mnt/data/reg2026/train_CoT.json \
        --split-json   /mnt/data/reg2026/checkpoints/phase_a_v3/split.json \
        --features-dir /mnt/data/reg2026/features/full_run/20x_224px_0px_overlap/features_hoptimus1 \
        --output-dir   /mnt/data/reg2026/checkpoints/cofinding_heads

Each head is saved as <output-dir>/<cof>.pt with a `state_dict` (loads via the engine's
_build_tilehead) plus its pooling / threshold / presence-Q metadata, and a library.json manifest.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CAP = 1200  # per-slide tile cap (deterministic linspace subsample), matches the deployed scorer
DXQ = re.compile(r"#(\d+)\s+diagnosis")


@dataclass
class CofindingSpec:
    """One organ-specific co-finding head. `phrase` is a lower-case substring matched against the
    #k-diagnosis answers; `presence_q` is the question the head answers YES at when it fires."""

    cof: str
    organ: str
    phrase: str
    presence_q: str


# The deployable presence-Q co-findings (n>=14 positives). Tiny-n candidates are excluded.
SPECS: list[CofindingSpec] = [
    CofindingSpec("microcalc_breast", "breast", "microcalcification", "is there any microcalcification present?"),
    CofindingSpec("granulomatous_fb_bladder", "bladder", "granulomatous inflammation with foreign body", "is there any inflammation present?"),
    CofindingSpec("organizing_fibrosis_lung", "lung", "organizing fibrosis", "is there any inflammation present?"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cot-json", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--microcalc-pooling", type=int, default=1, help="top-k pooling for microcalc (1=max, small-focus optimal)")
    p.add_argument("--only", nargs="+", default=None, help="restrict to these cof keys")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ----------------------------- labels / population -----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def dx_list(case: dict) -> list[str]:
    out: dict[int, str] = {}
    for t in case["chain-of-thought"]:
        m = DXQ.search((t.get("question") or "").lower())
        if m:
            out[int(m.group(1))] = _norm(t.get("answer"))
    return [out[k] for k in sorted(out)]


def build_population(spec: CofindingSpec, cot: list, split: dict, feat_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Organ train+val slides with H-Opt feats. label=1 iff phrase appears as a #2+ co-finding
    (substring); slides where phrase appears ONLY as #1 are dropped (neither pos nor neg)."""
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
        is_cof = any(spec.phrase in d for d in dl[1:])
        is_any = any(spec.phrase in d for d in dl)
        if is_cof:
            label[cid] = 1
        elif not is_any:
            label[cid] = 0
        else:
            continue  # phrase only as #1 dx: ambiguous, exclude from both classes
        sids.append(cid)
    return sids, label


def load_hopt_capped(feat_dir: Path, cid: str) -> np.ndarray | None:
    fh = feat_dir / f"{cid}.h5"
    if not fh.exists():
        return None
    with h5py.File(fh, "r") as f:
        feats = np.array(f["features"], dtype=np.float32)
    if CAP and len(feats) > CAP:
        feats = feats[np.linspace(0, len(feats) - 1, CAP).astype(int)]
    return feats


# ----------------------------- model + train (no-MUSK MIL) -----------------------------
class TileHead(nn.Module):
    """Per-tile MLP returning a logit per tile (matches the deployed engine's _TileHead)."""

    def __init__(self, d_in: int = 1536, d_hid: int = 256, p_drop: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_hid, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, d_in] -> [N]
        return self.net(x).squeeze(-1)


def topk_mean_logits(logits: np.ndarray, k: int) -> float:
    s = np.sort(logits)[::-1]
    return float(s[: min(k, len(s))].mean())


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos), np.asarray(neg)
    wins = (p[:, None] > n[None, :]).sum()
    ties = (p[:, None] == n[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(p) * len(n)))


def train_mil(train_sids, label, hopt, device, epochs, lr, wd, topk_pool, seed) -> TileHead:
    """Top-k-pooled MIL: per-tile logits, slide score = mean of top-k, BCE against the slide label."""
    torch.manual_seed(seed)
    n_pos = sum(label[s] for s in train_sids)
    n_neg = len(train_sids) - n_pos
    pos_w = torch.tensor([max(n_neg, 1) / max(n_pos, 1)], device=device)
    head = TileHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    bags = [(torch.from_numpy(hopt[s]).to(device), float(label[s])) for s in train_sids]
    rng = np.random.default_rng(seed)
    head.train()
    accum = 16
    for _ in range(epochs):
        order = rng.permutation(len(bags))
        opt.zero_grad()
        for j, bi in enumerate(order):
            feats, lab = bags[bi]
            logit = head(feats)
            k = min(topk_pool, logit.numel())
            slide_logit = torch.topk(logit, k).values.mean().unsqueeze(0)
            loss = loss_fn(slide_logit, torch.tensor([lab], device=device)) / accum
            loss.backward()
            if (j + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
    head.eval()
    return head


@torch.inference_mode()
def score_tilehead(head: TileHead, hopt, sids, device, topk) -> dict[str, float]:
    return {s: topk_mean_logits(head(torch.tensor(hopt[s], device=device)).cpu().numpy(), topk) for s in sids}


def thr_at_precision(scores: np.ndarray, labels: np.ndarray, min_prec: float = 0.80) -> dict:
    """Lowest-score threshold (max recall) achieving precision >= min_prec on the given set."""
    order = np.argsort(-scores)
    n_pos = int(labels.sum())
    tp = fp = 0
    best = {"thr": None, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0}
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / max(n_pos, 1)
        if prec >= min_prec and rec > best["recall"]:
            best = {"thr": float(scores[i]), "precision": round(prec, 4), "recall": round(rec, 4), "tp": tp, "fp": fp}
    return best


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cot = json.loads(args.cot_json.read_text())
    cot = cot if isinstance(cot, list) else list(cot.values())
    split = json.loads(args.split_json.read_text())
    tr_set = {s.replace(".tiff", "") for s in split["train"]}
    va_set = {s.replace(".tiff", "") for s in split["val"]}

    specs = {s.cof: s for s in SPECS}
    targets = list(specs) if not args.only else [k for k in specs if k in set(args.only)]
    library: dict[str, dict] = {}

    for cof in targets:
        spec = specs[cof]
        sids, label = build_population(spec, cot, split, args.features_dir)
        hopt = {s: load_hopt_capped(args.features_dir, s) for s in sids}
        hopt = {s: v for s, v in hopt.items() if v is not None}
        sids = [s for s in sids if s in hopt]
        train = [s for s in sids if s in tr_set]
        val = [s for s in sids if s in va_set]
        n_tr_pos = sum(label[s] for s in train)
        n_va_pos = sum(label[s] for s in val)
        logger.info("[%s] organ=%s | train=%d (pos=%d) val=%d (pos=%d)", cof, spec.organ, len(train), n_tr_pos, len(val), n_va_pos)
        if n_tr_pos == 0:
            logger.warning("[%s] no train positives -> skip", cof)
            continue

        # pick pooling: microcalc = small-focus (max); larger-region = max vs top10 by a quick TRAIN check.
        if cof == "microcalc_breast":
            pooling = args.microcalc_pooling
        else:
            probe = train_mil(train, label, hopt, args.device, args.epochs, args.lr, args.wd, 10, args.seed)
            s1 = score_tilehead(probe, hopt, train, args.device, 1)
            s10 = score_tilehead(probe, hopt, train, args.device, 10)
            a1 = auc([s1[s] for s in train if label[s] == 1], [s1[s] for s in train if label[s] == 0])
            a10 = auc([s10[s] for s in train if label[s] == 1], [s10[s] for s in train if label[s] == 0])
            pooling = 1 if a1 > a10 else 10
            logger.info("[%s] pooling probe TRAIN AUC max=%.4f top10=%.4f -> top%d", cof, a1, a10, pooling)
            del probe
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        head = train_mil(train, label, hopt, args.device, args.epochs, args.lr, args.wd, pooling, args.seed)
        s_tr = score_tilehead(head, hopt, train, args.device, pooling)
        s_va = score_tilehead(head, hopt, val, args.device, pooling) if val else {}
        tr_scores = np.array([s_tr[s] for s in train]); tr_labels = np.array([label[s] for s in train])
        thr = thr_at_precision(tr_scores, tr_labels, 0.80)
        train_auc = auc([s_tr[s] for s in train if label[s] == 1], [s_tr[s] for s in train if label[s] == 0])
        val_auc = auc([s_va[s] for s in val if label[s] == 1], [s_va[s] for s in val if label[s] == 0]) if n_va_pos > 0 else None

        ckpt = args.output_dir / f"{cof}.pt"
        torch.save({
            "state_dict": head.state_dict(),
            "cof": cof, "organ": spec.organ, "phrase": spec.phrase,
            "pooling_topk": pooling, "threshold": thr["thr"], "presence_q": spec.presence_q,
            "arch": "TileHead(1536->256->1) no-MUSK MIL top-k pool", "cap": CAP,
            "hparams": {"epochs": args.epochs, "lr": args.lr, "wd": args.wd, "seed": args.seed},
        }, ckpt)
        library[cof] = {
            "organ": spec.organ, "phrase": spec.phrase, "ckpt": str(ckpt), "pooling_topk": pooling,
            "threshold": thr["thr"], "presence_q": spec.presence_q,
            "n_train": len(train), "n_train_pos": n_tr_pos, "n_val": len(val), "n_val_pos": n_va_pos,
            "train_auc": round(train_auc, 4), "val_auc": (round(val_auc, 4) if val_auc is not None else None),
        }
        logger.info("[%s] SAVED pooling=top%d thr=%.4f (train P=%.2f R=%.2f) | train_auc=%.4f val_auc=%s",
                    cof, pooling, (thr["thr"] or float("nan")), thr["precision"], thr["recall"], train_auc,
                    (f"{val_auc:.4f}" if val_auc is not None else "n/a"))
        del head
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    (args.output_dir / "library.json").write_text(json.dumps(library, indent=2))
    logger.info("Done. %d heads -> %s. Copy each <cof>.pt to model/ckpts/ and wire it into cofinding_heads.json.", len(library), args.output_dir)


if __name__ == "__main__":
    main()
