#!/usr/bin/env python3
"""VL-LoRA fine-tune TITAN's slide encoder to lift zero-shot organ + #1-diagnosis on our distribution.

TITAN (slide encoder + text tower) is frozen; a LoRA adapter is injected on the slide transformer
(qkv + attn.proj) and the contrastive projection head is trained, with cross-entropy against fixed
TITAN text-prototype classifiers for organ (8-way, including the out-of-distribution "Uterine corpus")
and #1-diagnosis. This is the deployed out-of-distribution hybrid path: on slides the H-Optimus ABMIL
flags as uncertain, the LoRA-adapted TITAN supplies a zero-shot organ + coarse #1-dx.

Consumes CONCH v1.5 512px slide features (a separate extraction from the H-Optimus 224px features the
rest of the pipeline uses). Produces model/ckpts/titan_vl_lora_ce/lora_ep<N>.pt.

Usage:
    python scripts/train_titan_vl_lora.py \
        --cot-json           /mnt/data/reg2026/train_CoT.json \
        --split-json         model/configs/split.json \
        --conch-features-dir /mnt/data/reg2026/features/full_run/20x_512px_0px_overlap/features_conch_v15 \
        --titan-dir          model/ckpts/titan \
        --output-dir         model/ckpts/titan_vl_lora_ce \
        --epochs 8

The best epoch's lora_ep<N>.pt is the deployed adapter (loaded by interf1/titan_live.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("titan_lora")
DEV = "cuda"

TEMPLATES = ["CLASSNAME.", "a histopathology image of CLASSNAME.",
             "a whole slide image of CLASSNAME.", "histopathology showing CLASSNAME tissue."]
# 8-way organ set: the 7 in-distribution organs + the out-of-distribution "Uterine corpus".
ORGAN_CLASSES = [
    ("Prostate", ["prostate", "prostate gland"]), ("Breast", ["breast", "breast tissue"]),
    ("Colon", ["colon", "colonic mucosa", "large intestine"]), ("Stomach", ["stomach", "gastric mucosa"]),
    ("Urinary bladder", ["urinary bladder", "bladder urothelium"]), ("Lung", ["lung", "pulmonary tissue"]),
    ("Uterine cervix", ["uterine cervix", "cervix"]), ("Uterine corpus", ["uterine corpus", "endometrium", "myometrium"]),
]
ORG_CANON = {"gastric": "Stomach", "rectum": "Colon", "colorectal": "Colon", "anus": "Colon", "nipple": "Breast"}


def _register_titan_modules(titan_dir: Path) -> None:
    """TITAN's trust_remote_code modeling has a 2nd-level relative import (text_transformer ->
    conch_tokenizer) that transformers fails to resolve offline. Pre-seed the dynamic-module cache
    with ALL of TITAN's .py so every relative import resolves with no network access (mirrors
    interf1/titan_live.py)."""
    import os
    import shutil

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")
    tdir = Path(os.environ["HF_MODULES_CACHE"]) / "transformers_modules" / titan_dir.name
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "__init__.py").touch()
    for py in titan_dir.glob("*.py"):
        shutil.copy2(py, tdir / py.name)


def canon_org(s: str) -> str:
    s = (s or "").strip()
    return ORG_CANON.get(s.lower(), s)


def strip_dx(s: str) -> str:
    """Base diagnosis: drop grade/modifier/parenthetical/with-clauses for lenient matching."""
    s = (s or "").strip().lower()
    s = re.split(r"[,;(]| with | - |:", s)[0]
    s = re.sub(r"\b(grade|gleason|score|low|high|moderately|well|poorly|invasive|in situ)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def gt_maps(cot_json: Path):
    cot = json.loads(cot_json.read_text())
    org, dx = {}, {}
    for c in cot:
        sid = c["id"].replace(".tiff", "")
        for it in c["chain-of-thought"]:
            if it["question"] == "What is the organ?":
                org[sid] = canon_org(it["answer"])
            if it["question"] == "What is the #1 diagnosis?" and sid not in dx:
                dx[sid] = it["answer"].strip()
    return org, dx


def dx_vocab(cot_json: Path) -> list[str]:
    cot = json.loads(cot_json.read_text())
    cnt: Counter = Counter()
    for c in cot:
        for it in c["chain-of-thought"]:
            if re.match(r"what is the #\d+ diagnosis", it["question"].lower()):
                a = (it["answer"] or "").strip()
                if a:
                    cnt[a] += 1
    return sorted([k for k, v in cnt.items() if v >= 2])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cot-json", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--conch-features-dir", type=Path, required=True, help="CONCH v1.5 512px slide features")
    ap.add_argument("--titan-dir", type=Path, required=True, help="local TITAN HF snapshot (trust_remote_code)")
    ap.add_argument("--output-dir", type=Path, default=Path("model/ckpts/titan_vl_lora_ce"))
    ap.add_argument("--max-train-slides", type=int, default=0, help="0=all; else balanced per-organ cap for a fast recipe run")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--train-max-tiles", type=int, default=1024, help="cap tiles/slide during training (attention OOM guard)")
    ap.add_argument("--dx-weight", type=float, default=1.0)
    ap.add_argument("--n-val", type=int, default=478)
    return ap.parse_args()


def load_feats(h5: Path):
    with h5py.File(h5, "r") as f:
        return (torch.from_numpy(f["features"][:]), torch.from_numpy(f["coords"][:]),
                int(f["coords"].attrs.get("patch_size_level0", 512)))


def slide_emb(titan, feats, coords, ps, max_tiles=None, rng=None):
    """Normalized TITAN slide embedding [768]. Training caps tiles (O(N^2) attention memory); eval uses all."""
    if max_tiles and feats.shape[0] > max_tiles:
        idx = torch.from_numpy(rng.permutation(feats.shape[0])[:max_tiles]) if rng is not None else torch.randperm(feats.shape[0])[:max_tiles]
        feats, coords = feats[idx], coords[idx]
    e = titan.encode_slide_from_patch_features(feats.unsqueeze(0).to(DEV), coords.unsqueeze(0).to(DEV), ps)
    return F.normalize(e @ titan.vision_encoder.proj, dim=-1).squeeze(0)


@torch.inference_mode()
def evaluate(titan, val_slides, gt_org, gt_dx, onames, oclf, dxv, dclf):
    titan.eval()
    o_hit = d_hit = n = 0
    for sid, h5 in val_slides:
        emb = slide_emb(titan, *load_feats(h5))
        po = onames[int((emb @ oclf).argmax())]
        pd = dxv[int((emb @ dclf).argmax())]
        n += 1
        o_hit += (po == gt_org[sid])
        sg, sp = strip_dx(gt_dx.get(sid, "")), strip_dx(pd)
        d_hit += int(bool(sg) and bool(sp) and (sp == sg or sp in sg or sg in sp))
    return o_hit / max(1, n), d_hit / max(1, n)


def main() -> None:
    args = parse_args()
    _register_titan_modules(args.titan_dir)
    titan = AutoModel.from_pretrained(str(args.titan_dir), trust_remote_code=True).to(DEV)
    for p in titan.parameters():
        p.requires_grad = False
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=["qkv", "attn.proj"], bias="none")
    inject_adapter_in_model(lcfg, titan.vision_encoder)
    titan.vision_encoder.proj.requires_grad_(True)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), device=DEV))
    trainable = [p for p in titan.parameters() if p.requires_grad] + [logit_scale]
    log.info("trainable params: %.2fM", sum(p.numel() for p in trainable) / 1e6)

    split = json.loads(args.split_json.read_text())
    train_ids = {s.replace(".tiff", "") for s in split["train"]}
    val_ids = {s.replace(".tiff", "") for s in split["val"]}
    gt_org, gt_dx = gt_maps(args.cot_json)
    all_conch = {p.stem: p for p in args.conch_features_dir.glob("*.h5")}
    # train pool: in-distribution organs only (corpus is OOD, held for zero-shot generalization)
    pool = [(s, p) for s, p in all_conch.items() if s in train_ids and s in gt_org and gt_org[s] != "Uterine corpus"]
    rng0 = np.random.default_rng(0)
    by_org: dict[str, list] = {}
    for s, p in pool:
        by_org.setdefault(canon_org(gt_org[s]), []).append((s, p))
    if args.max_train_slides:
        cap = max(1, args.max_train_slides // max(1, len(by_org)))
        train = [lst[i] for lst in by_org.values() for i in rng0.permutation(len(lst))[:cap]]
    else:
        train = pool
    log.info("train organ dist: %s", dict(Counter(canon_org(gt_org[s]) for s, _ in train)))
    val = [(s, p) for s, p in all_conch.items() if s in val_ids and s in gt_org][: args.n_val]
    log.info("train slides %d | val slides %d", len(train), len(val))
    if len(train) < 50:
        log.warning("too few train CONCH features (%d) - aborting.", len(train))
        return

    onames = [n for n, _ in ORGAN_CLASSES]
    with torch.no_grad():  # frozen classifiers must remain autograd operands in the CE loss
        oclf = titan.zero_shot_classifier([s for _, s in ORGAN_CLASSES], TEMPLATES, device=DEV)
        dxv = dx_vocab(args.cot_json)
        dclf = titan.zero_shot_classifier([[d] for d in dxv], TEMPLATES, device=DEV)

    o0, d0 = evaluate(titan, val, gt_org, gt_dx, onames, oclf, dxv, dclf)
    log.info("BASELINE (LoRA=0): organ %.3f | dx %.3f", o0, d0)

    o_idx_of = {sid: onames.index(canon_org(gt_org[sid])) for sid, _ in train if canon_org(gt_org[sid]) in onames}
    dx_pos = {d: i for i, d in enumerate(dxv)}
    d_idx_of = {sid: dx_pos.get(gt_dx.get(sid, ""), -1) for sid, _ in train}

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        titan.train()
        order = rng.permutation(len(train))
        tot = 0.0
        for bi in range(0, len(order), args.batch_size):
            idx = order[bi:bi + args.batch_size]
            S = torch.stack([slide_emb(titan, *load_feats(train[j][1]), max_tiles=args.train_max_tiles, rng=rng) for j in idx])
            ls = logit_scale.exp()
            o_lbl = torch.tensor([o_idx_of[train[j][0]] for j in idx], device=DEV)
            loss = F.cross_entropy(ls * S @ oclf, o_lbl)
            d_lbl = torch.tensor([d_idx_of[train[j][0]] for j in idx], device=DEV)
            m = d_lbl >= 0
            if m.any():
                loss = loss + args.dx_weight * F.cross_entropy((ls * S @ dclf)[m], d_lbl[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        o, d = evaluate(titan, val, gt_org, gt_dx, onames, oclf, dxv, dclf)
        log.info("ep%d loss %.4f | val organ %.3f | dx %.3f", ep + 1, tot / max(1, len(order) // args.batch_size), o, d)
        # per-epoch save (dx plateaus early, organ keeps improving - pick the best-val epoch to deploy)
        sd = {k: v.detach().cpu() for k, v in titan.vision_encoder.state_dict().items() if "lora" in k.lower() or k.endswith("proj")}
        torch.save({"lora_state": sd, "logit_scale": logit_scale.detach().cpu(), "epoch": ep + 1,
                    "val": {"organ": o, "dx": d}}, args.output_dir / f"lora_ep{ep + 1}.pt")
    log.info("Done. Adapters -> %s/lora_ep<N>.pt; deploy the best-val epoch.", args.output_dir)


if __name__ == "__main__":
    main()
