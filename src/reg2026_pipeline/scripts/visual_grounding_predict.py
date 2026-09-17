"""Visual Grounding inference pipeline.

Per Metric B spec (2026-06-01):
  Input: per-ROI Q&A - 5x magnification, 256x256 patches, anonymous IDs.
  Output: per-ROI answer string.

Pipeline:
  Tier 0 - HEST tissue detection (TRIDENT MahmoodLab seg) on ROI:
    if tissue area < threshold -> "background, not assessable" magic string
  Tier 1 - tissue ROI:
    upsample 5x ROI 4x -> 1024x1024 -> tile into 16 x 256x256 patches
    -> H-Optimus encode each patch -> [16, 1536] tile features
    -> B-qcond forward (single Q, no history, 16 tiles)
    -> cosine NN over A_VOCAB -> output text

Usage:
    python scripts/visual_grounding_predict.py \\
        --roi-dir /path/to/rois/ \\
        --roi-pairs-json /path/to/roi_question_pairs.json \\
        --module1-ckpt /path/phase_a_b_qcond_ss/best.ckpt \\
        --hopt-cache-dir /mnt/data/.huggingface \\
        --output /path/to/predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BG_MAGIC = "This region is background. No tissue is present; the question is not assessable."

TISSUE_PRESENT_ANSWER = "Yes, tissue is visible."


def is_tissue_presence_q(question: str) -> bool:
    """True for tissue-presence questions (vs dominant-content / morphology questions)."""
    q = (question or "").lower()
    return "tissue" in q and any(w in q for w in ("visible", "present", "contain", "is there", "analyzable", "analysable"))


def set_eval_mode(model):
    """Put model into inference mode."""
    model.train(False)
    return model


# Tier 0 - HEST tissue detection


def load_hest_segmenter(device: str):
    """Load HEST tissue segmentation model from TRIDENT."""
    from trident.segmentation_models.load import HESTSegmenter

    seg = HESTSegmenter(confidence_thresh=0.5)
    seg = set_eval_mode(seg.to(device))
    logger.info("HEST tissue segmenter loaded (input %dx%d, target %dx)", seg.input_size, seg.input_size, seg.target_mag)
    return seg


@torch.no_grad()
def hest_tissue_fraction(roi_img: Image.Image, hest_model, device: str) -> float:
    """Return tissue area fraction (0..1) for a single ROI image."""
    img = roi_img.convert("RGB").resize((hest_model.input_size, hest_model.input_size), Image.BICUBIC)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    x = transform(img).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
        mask = hest_model(x)[0]
    m = np.squeeze(mask.float().cpu().numpy())
    if m.ndim != 2:
        return float(mask.float().mean().item())
    rgb = np.asarray(img.resize((m.shape[1], m.shape[0]), Image.NEAREST))
    nonblack = rgb.max(axis=2) >= 15
    return float(m[nonblack].mean()) if nonblack.any() else float(m.mean())


def is_background(roi_img: Image.Image, hest_model, device: str, threshold: float = 0.10) -> tuple[bool, float]:
    """Return (is_bg, tissue_pct)."""
    pct = hest_tissue_fraction(roi_img, hest_model, device)
    return pct < threshold, pct


# Tier 1 - H-Optimus feature extraction (upsample + tile)


def load_h_optimus(device: str, cache_dir: str | None = None):
    """Load H-Optimus-1 from HuggingFace (cached)."""
    import timm

    if cache_dir:
        import os

        os.environ["HF_HOME"] = cache_dir
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model = set_eval_mode(model.to(device))
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    logger.info("H-Optimus-1 loaded; expects input %s normalized %s", cfg["input_size"], cfg["mean"])
    return model, transform


@torch.no_grad()
def extract_roi_features(roi_img: Image.Image, h_optimus, h_transform, device: str) -> torch.Tensor:
    """Upsample 5x ROI 4x -> tile 16 x 256x256 patches -> H-Optimus encode each.

    Returns: [16, 1536] tensor on CPU.
    """
    arr = np.array(roi_img.convert("RGB"))
    if arr.shape[:2] != (256, 256):
        arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_CUBIC)
    up = cv2.resize(arr, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    tile_tensors = []
    for y in range(0, 1024, 256):
        for x in range(0, 1024, 256):
            patch_np = up[y : y + 256, x : x + 256]
            patch_pil = Image.fromarray(patch_np)
            tile_tensors.append(h_transform(patch_pil))
    batch = torch.stack(tile_tensors).to(device)
    feats = h_optimus(batch)
    return feats.cpu()


# Tier 1 - B-qcond single-Q forward


def load_module1_qcond(args, pools, device: str):
    from reg2026.aggregate.abmil import ABMIL, ABMILConfig
    from reg2026.aggregate.aux_heads import AuxHeads, AuxHeadsConfig
    from reg2026.module1.network_b_qcond import ModuleOneBQCond
    from reg2026.module1.network_base import ModuleOneConfig

    abmil = ABMIL(ABMILConfig(in_dim=1536, embed_dim=256, num_heads=4))
    aux_heads = AuxHeads(AuxHeadsConfig(in_dim=abmil.config.output_dim, categorical_heads={"organ": 7}))
    cfg = ModuleOneConfig(
        n_q_vocab=len(pools.Q_VOCAB),
        n_a_vocab=len(pools.A_VOCAB),
        tile_emb_dim=1536,
        slide_emb_dim=abmil.config.output_dim,
        max_N=30,
    )
    model = ModuleOneBQCond(cfg, pools, abmil, aux_heads, n_wsi_layers=2)
    ckpt = torch.load(args.module1_ckpt, map_location=device, weights_only=False)
    # strict=False: base qcond_ss predates the M1-cls head (VG uses cosine, not cls).
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        logger.info("module1 load (strict=False): missing=%s unexpected=%s", list(missing), list(unexpected))
    return set_eval_mode(model.to(device))


_MEDCPT_CACHE = {"model": None, "tokenizer": None, "device": None}


def load_medcpt(device: str):
    """Load MedCPT-Query-Encoder (matches the encoder used to build Q_EMB pool)."""
    if _MEDCPT_CACHE["model"] is not None and _MEDCPT_CACHE["device"] == device:
        return _MEDCPT_CACHE["model"], _MEDCPT_CACHE["tokenizer"]
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    model = set_eval_mode(model.to(device))
    _MEDCPT_CACHE.update({"model": model, "tokenizer": tokenizer, "device": device})
    logger.info("MedCPT-Query-Encoder loaded (for OOD Q semantic NN)")
    return model, tokenizer


@torch.no_grad()
def nearest_q_idx(question: str, pools, device: str = "cuda", restrict_to: list[int] | None = None) -> int:
    """Map free-form Q text -> nearest Q_VOCAB index by MedCPT cosine similarity.

    Exact match short-circuits; otherwise encode Q with MedCPT-Query-Encoder
    (same encoder used to build Q_EMB pool) and cosine NN against pools.Q_EMB.

    restrict_to: optional allow-list of Q_VOCAB indices. For VG content/morphology questions
    we pass the CONTENT Q whitelist so a free-form "dominant content"/"morphology" question
    routes to a histologic-type/pattern Q instead of mis-mapping to a Yes/No gate or a count Q.
    """
    if question in pools._q_to_idx and (restrict_to is None or pools._q_to_idx[question] in set(restrict_to)):
        return pools._q_to_idx[question]
    model, tokenizer = load_medcpt(device)
    inputs = tokenizer([question], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    out = model(**inputs)
    q_emb = out.last_hidden_state[:, 0, :].float().cpu()  # [1, 768]
    pool = pools.Q_EMB.float()  # [n_q, 768]
    sims = F.cosine_similarity(q_emb, pool, dim=-1)  # [n_q]
    if restrict_to:
        masked = torch.full_like(sims, float("-inf"))
        keep = torch.tensor(sorted(set(restrict_to)), dtype=torch.long)
        masked[keep] = sims[keep]
        sims = masked
    return int(sims.argmax().item())


@torch.no_grad()
def predict_tissue_answer(tile_feats: torch.Tensor, question: str, module1_model, pools, device: str) -> str:
    from reg2026.module1.dag_dataset import DAGBatch

    q_idx = nearest_q_idx(question, pools, device=device)
    N = 1
    batch = DAGBatch(
        case_ids=["roi"],
        tile_features=tile_feats.unsqueeze(0).to(device),
        tile_pad_mask=torch.ones(1, tile_feats.shape[0], dtype=torch.bool, device=device),
        q_indices=torch.tensor([[q_idx]], dtype=torch.long, device=device),
        a_indices_gt=torch.zeros(1, N, dtype=torch.long, device=device),
        attn_mask=torch.tensor([[[True, False], [True, True]]], dtype=torch.bool, device=device),
        loss_mask=torch.ones(1, N, dtype=torch.bool, device=device),
        organ_idx=torch.zeros(1, dtype=torch.long, device=device),
        N_collapsed=torch.tensor([N], dtype=torch.long, device=device),
    )
    out = module1_model(batch)
    a_emb_pred = out["a_emb_pred"][0, 0]
    a_pool = module1_model.a_emb_pool.to(device)
    sims = F.cosine_similarity(a_emb_pred.unsqueeze(0), a_pool, dim=-1)
    nn_idx = int(sims.argmax().item())
    return pools.A_VOCAB[nn_idx]


def load_m1_vqa(ckpt_path, pools, device: str):
    """Load the EXACT interf1 M1-vqa (3-anchor) model, detecting max_N from the ckpt's
    turn_emb so it is NOT silently dropped (strict=False) -> random anchors."""
    from reg2026.aggregate.abmil import ABMIL, ABMILConfig
    from reg2026.aggregate.aux_heads import AuxHeads, AuxHeadsConfig
    from reg2026.module1.network_b_qcond import ModuleOneBQCond
    from reg2026.module1.network_base import ModuleOneConfig

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    max_N = state["turn_emb.weight"].shape[0] if "turn_emb.weight" in state else 8
    abmil = ABMIL(ABMILConfig(in_dim=1536, embed_dim=256, num_heads=4))
    aux_heads = AuxHeads(AuxHeadsConfig(in_dim=abmil.config.output_dim, categorical_heads={"organ": 7}))
    cfg = ModuleOneConfig(
        n_q_vocab=len(pools.Q_VOCAB), n_a_vocab=len(pools.A_VOCAB),
        tile_emb_dim=1536, slide_emb_dim=abmil.config.output_dim, max_N=max_N,
    )
    model = ModuleOneBQCond(cfg, pools, abmil, aux_heads, n_wsi_layers=2)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if "turn_emb.weight" in set(missing):
        raise RuntimeError(f"M1-vqa turn_emb dropped (max_N mismatch {max_N}) - anchors would be random")
    logger.info("M1-vqa loaded (max_N=%d, strict=False: missing=%d unexpected=%d)", max_N, len(missing), len(unexpected))
    return set_eval_mode(model.to(device))


def _build_vqa_batch(tile_feats, q_seq, a_seq, anc, device):
    """1-case 4-position DAGBatch (same causal anchor mask as engine build_module1_batch)."""
    from reg2026.module1.dag_dataset import DAGBatch

    N = len(q_seq)
    attn = torch.zeros(2 * N, 2 * N, dtype=torch.bool)
    for i, s in enumerate(anc):
        for j in s:
            attn[2 * i, 2 * j] = attn[2 * i, 2 * j + 1] = attn[2 * i + 1, 2 * j] = attn[2 * i + 1, 2 * j + 1] = True
    for i in range(N):
        attn[2 * i, 2 * i + 1] = False
    T = tile_feats.shape[0]
    return DAGBatch(
        case_ids=[""], tile_features=tile_feats.unsqueeze(0).to(device),
        tile_pad_mask=torch.ones(1, T, dtype=torch.bool, device=device),
        q_indices=torch.tensor([q_seq], dtype=torch.long, device=device),
        a_indices_gt=torch.tensor([a_seq], dtype=torch.long, device=device),
        attn_mask=attn.unsqueeze(0).to(device),
        loss_mask=torch.ones(1, N, dtype=torch.bool, device=device),
        organ_idx=torch.zeros(1, dtype=torch.long, device=device),
        N_collapsed=torch.tensor([N], dtype=torch.long, device=device),
    )


@torch.no_grad()
def predict_tissue_answer_m1vqa(tile_feats: torch.Tensor, question: str, m1vqa, pools, device: str,
                                fixed_proc: str | None = "Biopsy", drop_proc: bool = False,
                                content_q_restrict: list[int] | None = None) -> str:
    """Autoregressive M1-vqa on a VG ROI (the EXACT interf1 model + anchor protocol).

    drop_proc=True (2-anchor ckpt, deployed): [organ -> #1-dx -> content]. The ROI has no
    visible procedure, so the 2-anchor model omits that anchor entirely - no "Biopsy" kludge.

    drop_proc=False (3-anchor ckpt): [organ -> procedure -> #1-dx -> content]; procedure is
    clinical metadata NOT visible in a single ROI -> fixed to the most-common in-distribution
    value (`fixed_proc`, default "Biopsy") rather than predicted (noise) or null (OOD).

    organ + #1-dx are ROI-visible -> predicted autoregressively from the ROI tiles either way.
    """
    organ_q = pools.q_idx("What is the organ?")
    proc_q = pools.q_idx("What is the procedure?")
    dx1_q = pools.q_idx("What is the #1 diagnosis?")
    content_q = nearest_q_idx(question, pools, device=device, restrict_to=content_q_restrict)
    if drop_proc:
        q_seq = [organ_q, dx1_q, content_q]
        anc = [{0}, {0, 1}, {0, 1, 2}]
        a_seq = [0, 0, 0]
        decode_positions = (0, 1, 2)  # predict organ, #1-dx, content (no procedure anchor)
    else:
        q_seq = [organ_q, proc_q, dx1_q, content_q]
        anc = [{0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}]
        a_seq = [0, 0, 0, 0]
        if fixed_proc is not None and fixed_proc in pools.A_VOCAB:
            a_seq[1] = pools.A_VOCAB.index(fixed_proc)  # fix procedure anchor (in-distribution)
            decode_positions = (0, 2, 3)  # predict organ, #1-dx, content; procedure stays fixed
        else:
            decode_positions = (0, 1, 2, 3)
    a_pool = m1vqa.a_emb_pool.to(device)
    occ_aware = getattr(m1vqa.config, "max_occ", 0) > 0
    for pos in decode_positions:
        batch = _build_vqa_batch(tile_feats, q_seq, a_seq, anc, device)
        if occ_aware:
            batch.occ_indices = torch.zeros(1, len(q_seq), dtype=torch.long, device=device)
        out = m1vqa(batch)
        a_emb = out["a_emb_pred"][0, pos]
        a_seq[pos] = int(F.cosine_similarity(a_emb.unsqueeze(0), a_pool, dim=-1).argmax().item())
    return pools.A_VOCAB[a_seq[-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi-dir", type=Path, required=True, help="Directory containing roi_*.jpg files")
    ap.add_argument("--roi-pairs-json", type=Path, required=True, help="roi_question_pairs.json [{id, image, question}]")
    ap.add_argument("--module1-ckpt", type=Path, required=True, help="B-qcond ckpt for tissue ROI answers")
    ap.add_argument("--output", type=Path, required=True, help="Output JSON: [{id, answer}]")
    ap.add_argument("--hopt-cache-dir", type=str, default=None, help="HF_HOME for H-Optimus weights cache")
    ap.add_argument("--bg-threshold", type=float, default=0.10, help="Tissue area threshold for background")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dump-diagnostics", type=Path, default=None, help="Optional: per-ROI tissue pct + decision")
    args = ap.parse_args()

    from reg2026.data.embedding_pools import EmbeddingPools

    pools = EmbeddingPools()
    logger.info("Pools loaded: Q_VOCAB=%d A_VOCAB=%d", len(pools.Q_VOCAB), len(pools.A_VOCAB))

    hest = load_hest_segmenter(args.device)
    h_opt, h_tf = load_h_optimus(args.device, args.hopt_cache_dir)
    module1 = load_module1_qcond(args, pools, args.device)

    pairs = json.loads(args.roi_pairs_json.read_text())
    logger.info("Loaded %d ROI-Q pairs", len(pairs))

    diagnostics = []
    results = []
    for i, pair in enumerate(pairs):
        roi_id = pair["id"]
        roi_path = args.roi_dir / pair["image"]
        question = pair["question"]
        roi_img = Image.open(roi_path)

        bg, tissue_pct = is_background(roi_img, hest, args.device, threshold=args.bg_threshold)
        matched_q = None
        if bg:
            answer = BG_MAGIC
            decision = "background"
        elif is_tissue_presence_q(question):
            answer = TISSUE_PRESENT_ANSWER
            decision = "tissue-presence"
        else:
            tile_feats = extract_roi_features(roi_img, h_opt, h_tf, args.device)
            matched_q_idx = nearest_q_idx(question, pools, device=args.device)
            matched_q = pools.Q_VOCAB[matched_q_idx]
            answer = predict_tissue_answer(tile_feats, question, module1, pools, args.device)
            decision = "tissue"

        results.append({"id": roi_id, "answer": answer})
        diagnostics.append({"id": roi_id, "image": pair["image"], "tissue_pct": tissue_pct, "decision": decision, "question": question, "matched_q_vocab": matched_q, "answer": answer})

        if (i + 1) % 10 == 0:
            logger.info("Progress %d/%d", i + 1, len(pairs))

    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Wrote %d predictions to %s", len(results), args.output)

    if args.dump_diagnostics:
        args.dump_diagnostics.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False))
        logger.info("Wrote diagnostics to %s", args.dump_diagnostics)


if __name__ == "__main__":
    main()
