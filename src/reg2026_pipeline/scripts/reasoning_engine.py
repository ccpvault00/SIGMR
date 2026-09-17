"""Reasoning engine: builds the chain-of-thought reasoning trajectory for the REG2026 pipeline.

For one case it produces the Q&A trajectory:
- Module 1 (M1) answers each question, conditioned on the ancestor Q&A context in the breadth-first-discovered
  question DAG - a closed-set answerer (M1-cls) or an autoregressive VQA answerer (M1-vqa).
- Module 2 (M2), a TrajectoryTransformer, predicts the next question(s) from the (Q, A) history.
- breadth-first search over the question DAG, seeded from organ + procedure; the diagnosis chain (dx_count -> #1 -> #2 …)
  is deterministic.

Used two ways: as the inference engine (imported by the interf1 container - setup_inference /
predict_case) and as a standalone eval/inspect CLI (see main(), which also emits the Edge-F1 / BPV /
MESS / FinalReport scoring tables).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from reg2026.aggregate.abmil import ABMIL, ABMILConfig
from reg2026.aggregate.aux_heads import AuxHeads, AuxHeadsConfig
from reg2026.data.embedding_pools import EmbeddingPools, canonicalize_q
from reg2026.module2.trajectory_model import TrajectoryModelConfig, TrajectoryTransformer
from reg2026.module2.trajectory_tokenizer import TrajectoryTokenizer
from reg2026.module1.dag_dataset import DAGBatch
from reg2026.module1.network_b_qcond import ModuleOneBQCond
from reg2026.module1.network_base import ModuleOneConfig
from reg2026.module1.trainer import nn_search_batch

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["b-qcond"], default="b-qcond")
    p.add_argument("--module1-ckpt", type=Path, required=True)
    p.add_argument("--module2-ckpt", type=Path, required=True)
    p.add_argument("--cot-json", type=Path, default=Path("/mnt/data/reg2026/train_CoT.json"))
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--m1-tile-emb-dim", type=int, default=1536, help="tile embedding dim of the M1 model build (ABMIL in_dim + tile_emb_dim); the H-Optimus-1 feature width.")
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5, help="M2 next-question emission threshold: emit a candidate next-Q when its sigmoid prob exceeds this (per-Q thresholds override).")
    p.add_argument(
        "--candidate-mask-json",
        type=Path,
        default=None,
        help="{source_q_text: [allowed next_q_text]}; restrict M2 emission to per-source-Q candidate set (removes illegal-FP edges).",
    )
    p.add_argument("--max-cases", type=int, default=0, help="If >0, truncate val to first N cases (smoke test).")
    p.add_argument(
        "--candidate-escape-thr",
        type=float,
        default=1.0,
        help="OOD safety: with --candidate-mask-json, keep a NON-candidate next-Q if M2 prob >= this (so novel Phase-2 schema edges aren't hard-blocked). 1.0 = pure hard R4 (no escape); e.g. 0.85 = let through high-confidence novel edges.",
    )
    p.add_argument("--max-edges", type=int, default=60, help="hard cap on the number of edges (Q → next-Q) emitted in one trajectory.")
    p.add_argument("--max-dx", type=int, default=4, help="max number of diagnoses (#1..#k) the trajectory can emit.")
    p.add_argument("--max-N", type=int, default=30, help="max BFS positions = the M1 forward's DAG-batch length / padding limit.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-wsi-layers", type=int, default=2, help="number of WSI-transformer layers in the M1 (Module-1) model build.")
    p.add_argument(
        "--codx-integrator-json",
        type=Path,
        default=None,
        help="per-case {case_id:[A_VOCAB idx]} region-integrator co-dx (agreement-gated). Injected into the dx chain like a cofinding head (+ independent subtree). Static precomputed; off by default.",
    )
    # when a normalize-glitches-trained M2 is used, the fanout soft-cap (--m2-top-k-from-train)
    # must be computed on the SAME normalized chains, else the raw-train cap (bladder extent fanout=1)
    # suppresses the M2's learned extent->histtype edge (~32 val bladder cases dropped).
    p.add_argument(
        "--normalize-glitches",
        action="store_true",
        help="apply reg2026.data.cot_normalize to train_CoT before building the --m2-top-k-from-train fanout cap, so a normalize-glitches-trained M2's learned edges (lung subtype, bladder extent->histtype) are not capped away.",
    )
    # selective gate-context for M2 (learns fan-in conditions the ancestor-only
    # context cannot, e.g. colon serration's behavior->primary->serration gate).
    p.add_argument(
        "--m2-gate-context",
        action="store_true",
        help=(
            "inject diagnostic GATE answers (abnormality/neoplasm/behavior) into M2's "
            "ancestor-only history. behavior is a SIBLING of primary (not an ancestor) but is the "
            "deciding fan-in signal for colon serration (primary->serration iff behavior=Benign); "
            "ancestor-only M2 cannot see it. Gates are early-backbone so train (CoT node-index order) "
            "and inference (breadth-first search pos order) MATCH. Requires an M2 trained with the same gate context "
            "(an M2-gate ckpt). Avoids full-history's leaf-order OOD. Default-OFF."
        ),
    )
    p.add_argument(
        "--m1-vqa-ckpt",
        type=Path,
        default=None,
        help="M1-vqa best.ckpt - cascade-decoupled M1 (3 anchors + target Q). When set, route ALL non-excluded Qs to M1-vqa instead of B-qcond. Anchors: organ/procedure from breadth-first search a_seq, primary_dx from dx_ranked[0]. Excluded Qs (use cascade M1): organ, procedure, dx_count, dx#1-4, final_report, computed_qs (Hook C).",
    )
    p.add_argument(
        "--m1-vqa-drop-proc-anchor",
        action="store_true",
        help="2-anchor M1-vqa: the ckpt was trained with drop_proc_anchor (anchors [organ, #1-dx] only). "
        "Build the VQA batch as [organ, #1-dx, target] (3 positions) instead of [organ, proc, #1-dx, target]. "
        "Use with a 2-anchor ckpt (m1_vqa_2anchor).",
    )
    p.add_argument(
        "--m1cls-answer-space",
        type=Path,
        default=None,
        help="M1-cls: per_q_answer_space.json -> answer all CLOSED Qs via the model cls_head (masked argmax) instead of cosine NN. Requires an M1-cls ckpt (--module1-ckpt with cls_head).",
    )
    p.add_argument(
        "--gleason-from-patterns",
        action="store_true",
        help="FRP: derive 'What is the Gleason score?' (and grade group) from the predominant+secondary pattern answers in history (Z (X+Y) / ISUP grade group) instead of the M1-cls score cls head. Enforces score/pattern/grade-group consistency.",
    )
    p.add_argument(
        "--per-node-anchor",
        action="store_true",
        help="Co-dx subtree re-anchoring: answer Qs INSIDE a co-finding's additional-finding subtree "
        "using the subtree's own dx (the fired co-finding entity, e.g. DCIS/CIS) as the M1-vqa "
        "primary-dx anchor (vqa_a_seq[2]) instead of the global #1 dx. Anchor propagates parent->child "
        "at enqueue; the additional-finding gate that fires (cf_force_addfind) re-anchors its children to "
        "the co-finding's a_idx. OFF by default → baseline byte-identical.",
    )
    p.add_argument(
        "--abmil-primary-ckpt",
        type=Path,
        default=None,
        help="Single-label ABMIL primary: the #1 diagnosis is this ABMIL's argmax over the 139 graded dx (tiles capped to 2048, seeded rng0). Cofinding heads add #2 on top.",
    )
    p.add_argument(
        "--m2-top-k-from-train",
        action="store_true",
        help="Cap M2 emission per source-Q to top-K predictions where K = max train fanout. Suppresses M2 over-emission (training collapse-fan-in artifact).",
    )
    p.add_argument(
        "--cofinding-heads-json",
        type=Path,
        default=None,
        help="JSON list of dedicated co-finding detection heads "
        "[{ckpt, a_text, organ, primary_substr, threshold}]. When a head fires (LSE max-pool "
        "score>=threshold, conditioned on predicted organ + primary-dx substring), inject its "
        "co-finding A_VOCAB idx into dx_ranked + bump n_dx_emit → launches the #k dx "
        "chain (fixes BPV+EdgeF1+MESS+FR, not just report), using "
        "learned high-precision visual detectors (train_cofinding_head.py).",
    )
    p.add_argument(
        "--dump-routing",
        type=Path,
        default=None,
        help="dump the FAITHFUL per-Q answer-source map (which mechanism answered "
        "each Q, recorded live from the priority cascade) to this JSON - the machine-derived single "
        "source of truth for routing. No behavior change.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch seed for deterministic inference (fixes M1-vqa's random cls_head init; default 0).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="kill cuDNN/cublas non-determinism (use_deterministic_algorithms + cudnn.deterministic) so a "
        "code/config change (e.g. adding or removing a model head) cannot perturb other modules' borderline predictions. "
        "Removes the run-to-run noise band that contaminates small WFR comparisons.",
    )
    p.add_argument(
        "--dx-subspace-guard",
        action="store_true",
        help="restrict the #k-diagnosis cosine fall-through to the diagnosis subspace (all answers to "
        "'What is the #k diagnosis?' in train CoT, incl. 'No tumor present'). Keeps gate sentences out of "
        "the dx slot (case 00244). Parameter-free / structural; worst case falls to the nearest valid dx.",
    )
    p.add_argument(
        "--routing-config",
        type=Path,
        default=None,
        help="the EDITABLE single-source-of-truth routing config "
        "(configs/q_routing.json, from build_routing_config.py). When set, the per-Q-static knobs are "
        "READ from it instead of the inline hardcodes: m1cls_closed_qis <- {q: decoder=='cls'} and "
        "m1_vqa_excluded_qis <- {q: context=='context-aware'}. Behaviour-identical to the inline sets "
        "when the file is unedited; edit 'decoder'/'context' to re-route a Q (e.g. add grades to cls).",
    )
    p.add_argument(
        "--codx-independent-subtree",
        action="store_true",
        help="when a co-finding head fires (DCIS/CIS), GENERATE the 2nd-dx reasoning subtree ",
    )
    p.add_argument(
        "--q-conditioned-mask-json",
        type=Path,
        default=None,
        help="per-Q candidate mask JSON ({q_idx: [a_idx, ...]}) from build_per_q_a_mask.py. Restricts M1 cosine NN to per-Q answer set for closed-A Qs (Yes/No, 1/2/3/4, ...).",
    )
    p.add_argument(
        "--bfs-allow-revisit",
        action="store_true",
        help="breadth-first search allows Q to be revisited if M2 emits from a new parent. Each revisit = new q_seq position. Revisit positions routed to cascade M1 (B-qcond) instead of M1-vqa for cascade-history differentiation.",
    )
    p.add_argument("--bfs-max-revisit", type=int, default=3, help="Max number of revisits per Q (default 3).")
    p.add_argument(
        "--dump-official-pred-json",
        type=Path,
        default=None,
        help="dump per-case predicted trajectory in OFFICIAL format [{id, chain-of-thought:[{question,answer,next_question}]}] from res['edges'] (q,a,next_q triples). Feed run_official_eval.py to score with the official evaluate_metrics.py (true alignment; lexical MESS + terminal edges).",
    )
    return p.parse_args()


ORGANS_LIST = ["bladder", "breast", "cervix", "colon", "lung", "prostate", "stomach"]


def parse_organ_from_trajectory(trajectory: list[tuple[str, str]]) -> str | None:
    """Read the organ answer off a decoded Q&A trajectory (lowercased), or None if not asked.

    Inlined from the removed reg2026.generate.report_generator (its last surviving symbol).
    """
    for q, a in trajectory:
        if q == "What is the organ?":
            return a.strip().lower()
    return None


def load_abmil_primary(ckpt_path: Path, pools: EmbeddingPools, device: str) -> dict:
    """Single-label ABMIL primary #1-diagnosis predictor.

    Reconstructs the standard reg2026 4-head ABMIL (1536→256, enc_layers 3 → 1024) + a
    1024→256→139 classifier, loads best.ckpt, and precomputes dx_vocab→A_VOCAB index map.
    Returns {model, dx_to_a_idx}; the model's argmax over the 139 graded dx becomes the #1 diagnosis.
    """
    import torch.nn as nn

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dx_vocab = ck["dx_vocab"]
    n_cls = len(dx_vocab)

    class _AbmilPrimary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.abmil = ABMIL(ABMILConfig())  # defaults: 1536→256, 4 heads, enc_layers 3 → 1024
            self.classifier = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.25), nn.Linear(256, n_cls))

    model = _AbmilPrimary()
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval().to(device)
    a_lc = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}
    dx_to_a_idx = [a_lc.get(dx.strip().lower(), -1) for dx in dx_vocab]
    n_mapped = sum(1 for x in dx_to_a_idx if x >= 0)
    logger.info("ABMIL primary loaded: %d dx classes, %d/%d mapped to A_VOCAB (epoch=%s)", n_cls, n_mapped, n_cls, ck.get("epoch"))
    return {"model": model, "dx_to_a_idx": dx_to_a_idx}


class _TileHead(torch.nn.Module):
    """Co-finding per-tile MLP head (returns one logit per tile).

    Architecture matches train_cofinding_heads.py: LayerNorm -> Linear -> GELU -> Dropout -> Linear.
    """

    def __init__(self, d_in: int = 1536, d_hid: int = 256, p_drop: float = 0.25) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(d_in),
            torch.nn.Linear(d_in, d_hid),
            torch.nn.GELU(),
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(d_hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, d_in] -> [N]
        return self.net(x).squeeze(-1)


def _build_tilehead(ckpt: dict, device: str) -> torch.nn.Module:
    """Instantiate + load a TileHead from its saved ``state_dict``.

    Same LayerNorm->Linear->GELU->Dropout->Linear arch as train_cofinding_heads.py.
    """
    model = _TileHead(int(ckpt.get("d_in", 1536)), int(ckpt.get("d_hid", 256)))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).train(False)


def _tilehead_score(model: torch.nn.Module, feats: torch.Tensor, agg: str, cap: int) -> float:
    """Pooled co-finding score: top-k (or max/mean) of per-tile logits, matching the trainer.

    feats: [N, d_in] on any device. cap subsamples via linspace (mirrors training subsample).
    """
    if cap and feats.shape[0] > cap:
        # truncation (.long()) matches train_cofinding_heads.py's np.linspace(...).astype(int) CAP
        # subsample, so the threshold (calibrated on the trainer's tile subset) transfers exactly.
        idx = torch.linspace(0, feats.shape[0] - 1, cap).long()
        feats = feats[idx]
    with torch.inference_mode():
        logits = model(feats.to(next(model.parameters()).device))
    s, _ = torch.sort(logits, descending=True)
    if agg == "max":
        return float(s[0].item())
    if agg == "mean":
        return float(s.mean().item())
    k = int(agg[3:]) if agg.startswith("top") else 10
    k = max(1, min(k, s.shape[0]))
    return float(s[:k].mean().item())


def build_module1(args, pools):
    _m1_dim = getattr(args, "m1_tile_emb_dim", 1536)
    abmil = ABMIL(ABMILConfig(in_dim=_m1_dim, embed_dim=256, num_heads=4))
    aux_heads = AuxHeads(AuxHeadsConfig(in_dim=abmil.config.output_dim, categorical_heads={"organ": 7}))
    module1_cfg = ModuleOneConfig(
        n_q_vocab=len(pools.Q_VOCAB),
        n_a_vocab=len(pools.A_VOCAB),
        tile_emb_dim=_m1_dim,
        slide_emb_dim=abmil.config.output_dim,
        max_N=args.max_N,
        max_occ=getattr(args, "max_occ", 0),
    )
    if args.variant != "b-qcond":
        raise ValueError(f"only the deployed variant 'b-qcond' is supported (got {args.variant!r})")
    model = ModuleOneBQCond(module1_cfg, pools, abmil, aux_heads, n_wsi_layers=args.n_wsi_layers)
    ckpt = torch.load(args.module1_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model.to(args.device).train(False)


def load_module2(args, pools, device):
    ckpt = torch.load(args.module2_ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", TrajectoryModelConfig())
    module2 = TrajectoryTransformer(cfg)
    module2.load_state_dict(ckpt["model"])
    return module2.to(device).train(False), TrajectoryTokenizer(pools), cfg.max_seq_len


def _restricted_nn(
    a_emb,
    a_emb_pool,
    q_text,
    pools,
    per_q_a_mask: dict[int, list[int]] | None = None,
    q_idx: int | None = None,
    restrict_to: list[int] | None = None,
) -> int:
    """Cosine NN, optionally restricted to per-Q candidates.

    Restriction sources (first non-empty wins; intersected when both present):
      - restrict_to (dx-subspace guard): explicit a_idx allow-list for THIS
        call (e.g. the diagnosis subspace for a #k-diagnosis Q), so the cosine NN
        can never return a gate sentence into the dx slot.
      - per_q_a_mask: {q_idx: [a_idx, ...]} built from train CoT,
        guaranteed A_VOCAB-resolved. Safe for closed-A Qs (Yes/No, 1/2/3/4, ...).
    """
    cand_idx = None
    if per_q_a_mask is not None and q_idx is not None:
        cand_idx = per_q_a_mask.get(q_idx)
    if restrict_to:
        if not cand_idx:
            cand_idx = restrict_to
        else:
            _inter = [i for i in cand_idx if i in set(restrict_to)]
            cand_idx = _inter if _inter else restrict_to
    if cand_idx:
        cand_t = torch.tensor(cand_idx, device=a_emb.device)
        cand_emb = a_emb_pool[cand_t]
        sims = F.cosine_similarity(a_emb.view(1, -1), cand_emb, dim=-1)
        return cand_idx[int(sims.argmax().item())]
    return nn_search_batch(a_emb.view(1, 1, -1), a_emb_pool)[0, 0].item()


def compute_ancestors_dynamic(node_parents: dict[int, set[int]], q_idx: int) -> set[int]:
    """Transitive ancestor set of q_idx (including q_idx itself), walking up node_parents.

    Stack-based (LIFO) upward walk - traversal order is irrelevant since the result is a set. The
    ``seen`` guard makes it cycle-safe, and the set is bounded by the distinct question nodes
    (node_parents keys are q_idx values), so there is no unbounded growth even if a back-edge ever
    slipped into node_parents.
    """
    seen = {q_idx}
    stack = list(node_parents.get(q_idx, set()))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(node_parents.get(p, set()))
    return seen


def build_module1_batch(
    tile_features: torch.Tensor,
    q_idx_ordered: list[int],
    a_idx_ordered: list[int],
    ancestor_set_per_pos: list[set[int]],
    device: str,
    max_N: int,
) -> DAGBatch:
    """Build a 1-case DAGBatch for the M1 forward.

    Args:
        tile_features: [T, 1536]
        q_idx_ordered: breadth-first search visit order of Q indices (length N)
        a_idx_ordered: A indices predicted so far (length N) - last is dummy for current
        ancestor_set_per_pos: for position k, the set of POSITIONS (not Q indices) visible
        max_N: padding limit
    """
    N = len(q_idx_ordered)
    assert max_N >= N

    q_t = torch.tensor([q_idx_ordered], dtype=torch.long, device=device)
    a_t = torch.tensor([a_idx_ordered], dtype=torch.long, device=device)

    attn = torch.zeros(2 * N, 2 * N, dtype=torch.bool)
    for i, anc_set in enumerate(ancestor_set_per_pos):
        for j in anc_set:
            attn[2 * i, 2 * j] = True
            attn[2 * i, 2 * j + 1] = True
            attn[2 * i + 1, 2 * j] = True
            attn[2 * i + 1, 2 * j + 1] = True
    for i in range(N):
        attn[2 * i, 2 * i + 1] = False  # q can't see own a

    T = tile_features.shape[0]
    # loss_mask and organ_idx are training-only fields (per-position loss weighting; organ-aux CE target).
    return DAGBatch(
        case_ids=[""],
        tile_features=tile_features.unsqueeze(0).to(device),
        tile_pad_mask=torch.ones(1, T, dtype=torch.bool, device=device),
        q_indices=q_t,
        a_indices_gt=a_t,
        attn_mask=attn.unsqueeze(0).to(device),
        loss_mask=torch.ones(1, N, dtype=torch.bool, device=device),  # training-only (see note above)
        organ_idx=torch.zeros(1, dtype=torch.long, device=device),  # training-only aux-head label; unused by forward
        N_collapsed=torch.tensor([N], dtype=torch.long, device=device),
    )


@torch.no_grad()
def module2_predict_next_qs(
    module2,
    tokenizer,
    history_text: list[tuple[str, str]],
    threshold,  # float (uniform) OR list[float] per-Q OR Tensor[n_q]
    max_seq_len: int,
    device: str,
    soft_cap: int | None = None,  # if set, after threshold filter, cap to top-K by prob
    cand_set: set[int] | None = None,  # restrict to candidate next-Qs BEFORE threshold/cap
    cand_escape_thr: float = 1.0, 
) -> list[int]:
    tokens = tokenizer.encode_trajectory(history_text, add_bos=True, add_eos=False)
    if len(tokens) > max_seq_len:
        tokens = [tokens[0]] + tokens[-(max_seq_len - 1) :]
    t = torch.tensor([tokens], dtype=torch.long, device=device)
    attn = torch.ones_like(t, dtype=torch.bool)
    logits = module2(t, attn)
    probs = torch.sigmoid(logits[0]).cpu().tolist()
    n_q = tokenizer.n_q_vocab
    if cand_set is not None:  # mask non-candidates to 0 BEFORE threshold+soft_cap
        for i in range(n_q):
            if i not in cand_set and probs[i] < cand_escape_thr:
                probs[i] = 0.0
    if isinstance(threshold, (int, float)):
        passed = [(i, probs[i]) for i in range(n_q) if probs[i] > threshold]
    else:
        passed = [(i, probs[i]) for i in range(n_q) if probs[i] > float(threshold[i])]
    if soft_cap is not None and len(passed) > soft_cap:
        # Over-emit detected: keep only top-soft_cap by prob (preserves M2's top-1 when above thr)
        passed.sort(key=lambda x: -x[1])
        passed = passed[:soft_cap]
    return [i for i, _ in passed]


def _vqa_anchor_seqs(organ_q, proc_q, dx1_q, target_q, organ_a, proc_a, dx_a, drop_proc):
    """M1-vqa anchor batch sequences. drop_proc=True (a 2-anchor ckpt trained with
    drop_proc_anchor) omits the procedure position -> [organ, #1-dx, target]; else the
    standard 4-anchor [organ, proc, #1-dx, target]. Target Q is always the last position."""
    if drop_proc:
        q_seq = [organ_q, dx1_q, target_q]
        a_seq = [organ_a, dx_a, 0]
        anc = [{0}, {0, 1}, {0, 1, 2}]
    else:
        q_seq = [organ_q, proc_q, dx1_q, target_q]
        a_seq = [organ_a, proc_a, dx_a, 0]
        anc = [{0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}]
    return q_seq, a_seq, anc


def predict_case(
    case_id: str,
    features_path: Path,
    module1_model,
    module2,
    tokenizer,
    max_seq_len_m2: int,
    pools: EmbeddingPools,
    cfg,
    device: str,
):
    # ===========================================================================================
    # Runs the reasoning trajectory for ONE case - the deployed interf1 path
    # (interf1/model.py::_inference_argv). A couple of rule hooks stay off by default ([off]).
    #
    #   Phase 1  load H-Opt features
    #   Phase 2  dx sources:
    #              abmil-primary      #1 dx = the primary ABMIL + computes abmil_entropy -> TITAN gate
    #              cofinding heads    inject #2 co-finding dx (DCIS/CIS/microcalcification)
    #              region integrator  live region co-dx (net-neutral hedge)
    #   Phase 3  breadth-first search: M2 next-Q (gate-context) + M1 answerer routing (M1-vqa / cls)
    #   Phase 4  co-dx independent subtree (mini breadth-first search on the co-finding dx)
    #   Phase 5  return {edges, abmil_entropy, ...};  model.py builds the report from edges
    # ===========================================================================================
    with h5py.File(features_path, "r") as fh:
        tile_features = torch.from_numpy(fh["features"][...]).float()

    organ_q = pools.q_idx("What is the organ?")
    procedure_q = pools.q_idx("What is the procedure?")
    dx_count_q = pools.q_idx("What is the number of diagnoses to includes?")
    final_q = pools.q_idx("What is the final pathology report?")
    diagnosis_qs = {k: pools.q_idx(f"What is the #{k} diagnosis?") for k in range(1, cfg.max_dx + 1)}

    # Ranked diagnoses (A_VOCAB indices, #1 first) - set by the ABMIL primary + cofinding heads below.
    dx_ranked: list[int] = []
    n_dx_emit: int | None = None
    # #1 diagnosis comes from the ABMIL primary below; dx_ranked / n_dx_emit stay at these defaults until then.

    # Live #1-diagnosis: argmax of the single-label ABMIL primary. Tiles capped to 2048
    _abmil_entropy = -1.0  # dx-OOD gate signal
    _abmil_p = getattr(cfg, "abmil_primary", None)
    if _abmil_p is not None:
        _feats = tile_features
        if _feats.shape[0] > 2048:
            _sub = np.random.default_rng(0).choice(_feats.shape[0], 2048, replace=False)
            _feats = _feats[_sub]
        with torch.no_grad():
            _b = _feats.unsqueeze(0).to(device)
            _m = torch.ones(1, _b.shape[1], dtype=torch.bool, device=device)
            _emb, _ = _abmil_p["model"].abmil(_b, _m)
            _alogits = _abmil_p["model"].classifier(_emb)[0]
            _top1 = int(_alogits.argmax().item())
            _ap = F.softmax(_alogits, -1)
            _abmil_entropy = float(-(_ap * (_ap + 1e-9).log()).sum())  # HYBRID dx-OOD gate (high = OOD)
        _aidx = _abmil_p["dx_to_a_idx"][_top1]
        if _aidx >= 0:
            dx_ranked = [_aidx] + dx_ranked[1:]

    # Co-finding head gate-override. Independent visual detectors (LSE max-pool)
    cf_heads = getattr(cfg, "cofinding_heads", None)
    cf_force_addfind = False 
    cf_presence_yes: dict[str, int] = {} 
    cf_subtree_dx_idx: int | None = None 
    cf_subtree_dx_idxs: list[int] = []  # ALL fired additional-finding co-dx idxs (one independent subtree each)
    if cf_heads and dx_ranked:
        prim_text = pools.a_text(dx_ranked[0]).lower()
        cf_pred_organ = None
        for h in cf_heads:
            n = n_dx_emit if n_dx_emit is not None else len(dx_ranked)
            if h["a_idx"] in dx_ranked[:n]:
                continue
            if h["primary_substr"] is not None and h["primary_substr"] not in prim_text:
                continue
            if h["organ"] is not None:
                if cf_pred_organ is None:
                    with torch.no_grad():
                        _td = tile_features.unsqueeze(0).to(device)
                        _se, _ = module1_model.abmil(_td, mask=torch.zeros(1, _td.shape[1], dtype=torch.bool, device=device))
                        cf_pred_organ = ORGANS_LIST[int(module1_model.aux_heads(_se)["organ"].argmax(-1).item())]
                if cf_pred_organ != h["organ"]:
                    continue
            if h.get("type") == "tilehead":
                score = _tilehead_score(h["model"], tile_features, h["agg"], h.get("cap", 0))
            else:
                with torch.no_grad():
                    tf = tile_features
                    _w = h["w"].to(tf.device)
                    _b = h["b"].to(tf.device)
                    _tau = h["tau"].to(tf.device)
                    score = float((torch.logsumexp(_tau * (tf @ _w + _b), 0) - np.log(len(tf))) / _tau)
            if score >= h["threshold"]:
                if h["a_idx"] in dx_ranked:
                    dx_ranked.remove(h["a_idx"])  # drop the later (non-emitted) occurrence
                dx_ranked.insert(min(n, len(dx_ranked)), h["a_idx"])  # emit right after current chain
                n_dx_emit = min(cfg.max_dx, n + 1)
                if h.get("route") == "presence_q" and h.get("presence_q"):
                    cf_presence_yes[h["presence_q"]] = h["presence_yes_a_idx"]
                else:
                    cf_force_addfind = True 
                    if cf_subtree_dx_idx is None:
                        cf_subtree_dx_idx = h["a_idx"]
                    if h["a_idx"] not in cf_subtree_dx_idxs:
                        cf_subtree_dx_idxs.append(h["a_idx"])

    _integ = getattr(cfg, "codx_integrator", None)
    if _integ is not None and dx_ranked:
        for _aidx in _integ.get(case_id, []):
            n = n_dx_emit if n_dx_emit is not None else len(dx_ranked)
            if _aidx in dx_ranked[:n]:
                continue
            if _aidx in dx_ranked:
                dx_ranked.remove(_aidx)
            dx_ranked.insert(min(n, len(dx_ranked)), _aidx)
            n_dx_emit = min(cfg.max_dx, n + 1)
            cf_force_addfind = True
            if cf_subtree_dx_idx is None:
                cf_subtree_dx_idx = _aidx
            if _aidx not in cf_subtree_dx_idxs:
                cf_subtree_dx_idxs.append(_aidx)

    # breadth-first search state - frontier seeded with organ + procedure.
    init_q_idx = [organ_q, procedure_q]
    frontier: list[int] = list(init_q_idx)
    visited: set[int] = set(init_q_idx)
    node_parents: dict[int, set[int]] = {q: set() for q in init_q_idx}

    # per-node-anchor (gated): per q_idx primary-dx a_idx used as the M1-vqa primary anchor.
    # dx_ranked is fully resolved above (ABMIL primary + cofinding heads + region integrator), so the
    # init Qs anchor to the global #1 dx; co-dx subtree children re-anchor at enqueue (below).
    _global_dx1 = dx_ranked[0] if dx_ranked else -1
    node_anchor_dx: dict[int, int] = {q: _global_dx1 for q in init_q_idx} if getattr(cfg, "per_node_anchor", False) else {}

    q_seq: list[int] = []  # breadth-first search order
    a_seq: list[int] = []  # predicted a_idx per position
    pos_of_q: dict[int, int] = {}
    history_text: list[tuple[str, str]] = []
    edges: list[tuple[str, str, str]] = []
    routing_trace: list[tuple[str, str]] = []  #(q_text, answer-source) per answered Q 
    # breadth-first search revisit state (multi-dx ceiling lever).
    # q_visit_count[q] = number of times Q has been added to frontier (each = own q_seq position).
    # position_revisit_parent[pos] = if this pos is a revisit, its sole direct parent position (for restricted ancestor walk).
    q_visit_count: dict[int, int] = {q: 1 for q in init_q_idx}
    position_revisit_parent: dict[int, int] = {}  # position -> parent Q (only set for revisit positions)
    _pending_revisit_q: dict[int, list[int]] = {}  # nq -> list of parent q_idx pending revisit pop (FIFO)

    dx_queue: list[int] = []
    dx_resolved = False
    n_dx = 0

    while frontier and len(edges) < cfg.max_edges and len(q_seq) < cfg.max_N:
        q_idx = frontier.pop(0)
        new_pos = len(q_seq)
        # Detect revisit: if Q already has a position (was visited before), this is a revisit pop.
        is_revisit = q_idx in pos_of_q
        if is_revisit and _pending_revisit_q.get(q_idx):
            position_revisit_parent[new_pos] = _pending_revisit_q[q_idx].pop(0)
        pos_of_q[q_idx] = new_pos 
        q_seq.append(q_idx)
        a_seq.append(0)  # placeholder; module1 forward will produce real a_emb at last pos

        # Build ancestor set per position (in terms of position indices).
        # For revisit positions: restrict ancestors to the path through the new parent only.
        # For first-visit positions: use full DAG ancestry.
        ancestor_sets_pos: list[set[int]] = []
        for _pos, q in enumerate(q_seq):
            if _pos in position_revisit_parent:
                # Revisit: ancestors = {self, the new parent's position + its ancestors}
                # Build via causal walk back from parent.
                parent_q = position_revisit_parent[_pos]
                parent_pos = pos_of_q.get(parent_q)
                # Reuse parent's already-computed ancestors + self if parent exists and is causal
                anc_set = set(ancestor_sets_pos[parent_pos]) | {_pos} if parent_pos is not None and parent_pos < _pos else {_pos}
                ancestor_sets_pos.append(anc_set)
            else:
                anc_q = compute_ancestors_dynamic(node_parents, q)
                ancestor_sets_pos.append({pos_of_q[a] for a in anc_q if a in pos_of_q and pos_of_q[a] <= _pos})

        batch = build_module1_batch(tile_features, q_seq, a_seq, ancestor_sets_pos, device, cfg.max_N)
        active_m1_for_case = module1_model
        out = active_m1_for_case(batch)
        active_model = active_m1_for_case
        a_emb = out["a_emb_pred"][0, -1]  # current Q at last position
        m1cls_idx = None
        _m5mask = getattr(cfg, "m1cls_legal_mask", None)
        if _m5mask is not None and q_idx in cfg.m1cls_closed_qis:
            # M1-cls closed-Q answer: masked argmax over the model's cls_logits.
            if "cls_logits" in out:
                m1cls_idx = int(out["cls_logits"][0, -1].masked_fill(~_m5mask[q_idx], -1e4).argmax())

        # M1-vqa override for non-excluded Qs (cascade-decoupled forward)
        # Revisits: the deployed M1-vqa is occ-unaware (max_occ=0) so they route to cascade
        # M1 (B-qcond); an occ-aware M1-vqa (max_occ>0) would instead re-answer it with target_occ.
        _vqa_fired = False
        _is_revisit_pos = new_pos in position_revisit_parent
        _vqa_is_occ_aware = cfg.m1_vqa is not None and getattr(cfg.m1_vqa.config, "max_occ", 0) > 0
        # suffix mode: suffix Qs (e.g., "invasion?-2") get copied q_emb from base,
        # so M1-vqa (anchor-only, no cascade) cannot differentiate. Route them to B-qcond
        # (cascade-conditioned) which sees different history pattern at suffix positions.
        _is_suffix_q = pools.Q_VOCAB[q_idx].endswith(("-2", "-3", "-4")) if q_idx < len(pools.Q_VOCAB) else False
        _vqa_route = cfg.m1_vqa is not None and q_idx not in cfg.m1_vqa_excluded_qis and (not _is_revisit_pos or _vqa_is_occ_aware) and not _is_suffix_q
        if _vqa_route:
            organ_pos_vqa = pos_of_q.get(organ_q)
            proc_pos_vqa = pos_of_q.get(procedure_q)
            dx1_qi = pools.q_idx("What is the #1 diagnosis?")
            _drop_proc = getattr(cfg, "m1_vqa_drop_proc", False)  # 2-anchor M1-vqa: no procedure anchor
            _proc_ok = _drop_proc or (proc_pos_vqa is not None and a_seq[proc_pos_vqa] >= 0)
            if organ_pos_vqa is not None and a_seq[organ_pos_vqa] >= 0 and _proc_ok and dx_ranked and dx_ranked[0] >= 0:
                vqa_organ_q = pools.q_idx("What is the organ?")
                vqa_proc_q = pools.q_idx("What is the procedure?")
                # per-node-anchor: a Q inside a co-dx subtree anchors to its subtree dx, not global #1.
                _anchor_dx = dx_ranked[0]
                if getattr(cfg, "per_node_anchor", False):
                    _na = node_anchor_dx.get(q_idx, dx_ranked[0])
                    if _na is not None and _na >= 0:
                        _anchor_dx = _na
                _proc_anchor = a_seq[proc_pos_vqa] if (proc_pos_vqa is not None and a_seq[proc_pos_vqa] >= 0) else 0
                vqa_q_seq, vqa_a_seq, vqa_anc = _vqa_anchor_seqs(vqa_organ_q, vqa_proc_q, dx1_qi, q_idx, a_seq[organ_pos_vqa], _proc_anchor, _anchor_dx, _drop_proc)
                vqa_batch = build_module1_batch(tile_features, vqa_q_seq, vqa_a_seq, vqa_anc, device, max_N=8)
                # pass occ_indices for occ-aware M1-vqa (max_occ > 0).
                # Target Q (last position) gets occ = q_visit_count[q_idx] - 1 (0=first, 1=revisit, ...).
                if getattr(cfg.m1_vqa.config, "max_occ", 0) > 0:
                    target_occ = max(0, q_visit_count.get(q_idx, 1) - 1)
                    _occ = [0] * len(vqa_q_seq)
                    _occ[-1] = target_occ
                    vqa_batch.occ_indices = torch.tensor([_occ], dtype=torch.long, device=device)
                vqa_out = cfg.m1_vqa(vqa_batch)
                a_emb = vqa_out["a_emb_pred"][0, -1]
                active_model = cfg.m1_vqa
                _vqa_fired = True
        # Diagnosis Qs: answer #k from the ranked dx list (dx_ranked, set above)
        sb_dx_override = None
        if dx_ranked:
            for k_dx, dx_q_idx in diagnosis_qs.items():
                if q_idx == dx_q_idx and k_dx <= len(dx_ranked):
                    sb_dx_override = dx_ranked[k_dx - 1]
                    break
        q_text = pools.Q_VOCAB[q_idx]
        # derive Gleason score / grade group from predominant+secondary in history
        gleason_pat_idx = None
        if getattr(cfg, "gleason_from_patterns", False) and q_text in (_Q_GLEASON_SCORE, _Q_GRADE_GROUP):
            gleason_pat_idx = gleason_from_patterns_idx(q_text, history_text, pools)
        cf_addfind_idx = None
        if cf_force_addfind and q_text == "Is there any additional finding present?":
            cf_addfind_idx = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}.get("yes, there is an additional finding.", -1)
            cf_addfind_idx = cf_addfind_idx if cf_addfind_idx >= 0 else None
        # Route-B presence_q head fired → force its own organ-specific presence Q to YES
        # (presence-Q forced-YES pattern). q_text is matched case-insensitively against the fired set.
        cf_presence_idx = None
        if cf_presence_yes:
            cf_presence_idx = cf_presence_yes.get(q_text) or cf_presence_yes.get(q_text.strip().lower())
        _dx_restrict = None
        if getattr(cfg, "dx_subspace_guard", False) and getattr(cfg, "dx_answer_subspace", None) and q_idx in diagnosis_qs.values():
            _dx_restrict = cfg.dx_answer_subspace
        # Restrict the cosine fall-through to legal_a_idx so a confused slide can't emit an out-of-vocab token.
        if _dx_restrict is None:
            _dx_restrict = getattr(cfg, "closed_q_legal", {}).get(q_idx)
        _ans_src = None 
        if cf_presence_idx is not None:
            a_idx = cf_presence_idx
            _ans_src = "cofinding_presence_q[max-tile]"
        elif cf_addfind_idx is not None:
            a_idx = cf_addfind_idx
            _ans_src = "cofinding_addfind[max-tile]"
        elif gleason_pat_idx is not None:
            a_idx = gleason_pat_idx
            _ans_src = "rule:gleason_from_patterns"
        elif m1cls_idx is not None:
            a_idx = m1cls_idx
            _ans_src = "cls:M1-cls[context-aware]"
        elif sb_dx_override is not None:
            a_idx = sb_dx_override
            _ans_src = "dx-slot[ranked]"
        else:
            # cosine NN over the answer pool, restricted to the per-Q candidate set (per_q_a_mask / dx-subspace).
            a_idx = _restricted_nn(a_emb, active_model.a_emb_pool, q_text, pools, per_q_a_mask=getattr(cfg, "per_q_a_mask", None), q_idx=q_idx, restrict_to=_dx_restrict)
            _ans_src = "cosine:M1-vqa[standalone]" if _vqa_fired else "cosine:cascade[context-aware]"
        if getattr(cfg, "dump_routing", False):
            routing_trace.append((strip_q_suffix(q_text), _ans_src))
        a_seq[-1] = a_idx  # persist for future turns
        a_text = pools.a_text(a_idx)
        history_text.append((q_text, a_text))

        # dx_count handling
        if q_idx == dx_count_q and not dx_resolved:
            from reg2026.data.embedding_pools import DX_COUNT_ANCHORS

            if n_dx_emit is not None:
                n_dx = n_dx_emit
                dx_anchor_idx = pools.dx_count_anchor_indices()
                anchor_pos = max(0, min(len(dx_anchor_idx) - 1, n_dx - 1))
                a_idx = dx_anchor_idx[anchor_pos]
                a_seq[-1] = a_idx
                a_text = pools.a_text(a_idx)
                history_text[-1] = (q_text, a_text)
            else:
                dx_anchor_idx = pools.dx_count_anchor_indices()
                sims = F.cosine_similarity(a_emb.view(1, -1), module1_model.a_emb_pool[dx_anchor_idx], dim=-1)
                best = sims.argmax().item()
                try:
                    n_dx = int(DX_COUNT_ANCHORS[best])
                except ValueError:
                    n_dx = 1
            n_dx = max(1, min(cfg.max_dx, n_dx))
            dx_queue = [diagnosis_qs[k] for k in range(1, n_dx + 1)]
            dx_resolved = True

        # final report terminator
        if q_idx == final_q:
            edges.append((q_text, a_text, ""))
            continue

        is_dx_chain_source = (q_idx == dx_count_q) or (q_idx in diagnosis_qs.values())
        if dx_queue and is_dx_chain_source:
            nq = dx_queue.pop(0)
            nq_text = pools.Q_VOCAB[nq]
            edges.append((q_text, a_text, nq_text))
            node_parents.setdefault(nq, set()).add(q_idx)
            if getattr(cfg, "per_node_anchor", False) and nq not in node_anchor_dx:
                node_anchor_dx[nq] = node_anchor_dx.get(q_idx, _global_dx1)
            if nq not in visited:
                frontier.append(nq)
                visited.add(nq)
            continue
        if dx_resolved and not dx_queue and q_idx in diagnosis_qs.values():
            edges.append((q_text, a_text, pools.Q_VOCAB[final_q]))
            if final_q not in visited:
                frontier.append(final_q)
                visited.add(final_q)
                node_parents.setdefault(final_q, set()).add(q_idx)
                if getattr(cfg, "per_node_anchor", False) and final_q not in node_anchor_dx:
                    node_anchor_dx[final_q] = node_anchor_dx.get(q_idx, _global_dx1)
            continue

        # Module 2: predict next Qs
        ancestors_of_current = compute_ancestors_dynamic(node_parents, q_idx)
        ancestors_ordered = sorted(ancestors_of_current, key=lambda x: pos_of_q.get(x, 1 << 30))
        organ_pos_for_filter = pos_of_q.get(organ_q)
        organ_text_for_filter = pools.a_text(a_seq[organ_pos_for_filter]) if organ_pos_for_filter is not None else None
        is_prostate = organ_text_for_filter == "Prostate"
        ctx_qs = list(ancestors_ordered)
        if getattr(cfg, "m2_gate_context", False):
            _cur_pos = pos_of_q.get(q_idx, 1 << 30)
            for _gt in (
                "Is there any abnormality present?",
                "Is there any neoplasm present?",
                "What is the behavior of neoplasm?",
            ):
                try:
                    _gq = pools.q_idx(_gt)
                except Exception:
                    continue
                if _gq in pos_of_q and pos_of_q[_gq] < _cur_pos and _gq not in ctx_qs:
                    ctx_qs.append(_gq)
            ctx_qs = sorted(ctx_qs, key=lambda x: pos_of_q.get(x, 1 << 30))
        m2_history = []
        for anc_q in ctx_qs:
            if anc_q not in pos_of_q:
                continue
            if is_prostate and anc_q == procedure_q and anc_q != q_idx:
                continue
            anc_pos = pos_of_q[anc_q]
            anc_q_text = pools.Q_VOCAB[q_seq[anc_pos]]
            anc_a_idx = a_seq[anc_pos]
            anc_a_text = pools.a_text(anc_a_idx) if anc_a_idx >= 0 else ""
            m2_history.append((anc_q_text, anc_a_text))
        # GT is a true DAG with fan-out (e.g., abnormality emits both microcalc AND
        # proliferative_lesion), so M2 must be allowed to fan out.
        # Soft-cap: keep all M2 > threshold emissions, but cap at train max fanout if over-emitting
        soft_cap = cfg.q_max_fanout.get(q_idx) if cfg.q_max_fanout else None
        next_qs = module2_predict_next_qs(
            module2,
            tokenizer,
            m2_history,
            cfg.threshold,
            max_seq_len_m2,
            device,
            soft_cap=soft_cap,
            cand_set=(cfg.candidate_filter.get(q_idx) if getattr(cfg, "candidate_filter", None) is not None else None),
            cand_escape_thr=getattr(cfg, "candidate_escape_thr", 1.0),
        )
        if getattr(cfg, "per_node_anchor", False):
            _child_anchor = node_anchor_dx.get(q_idx, _global_dx1)
            _is_addfind_gate_fired = _ans_src == "cofinding_addfind[max-tile]"
            if _is_addfind_gate_fired and cf_subtree_dx_idx is not None and cf_subtree_dx_idx >= 0:
                _child_anchor = cf_subtree_dx_idx
        emitted_any = False
        for nq in next_qs:
            if nq >= tokenizer.n_q_vocab:
                continue
            nq_text = pools.Q_VOCAB[nq]
            edges.append((q_text, a_text, nq_text))
            emitted_any = True
            if getattr(cfg, "per_node_anchor", False) and nq not in node_anchor_dx:
                node_anchor_dx[nq] = _child_anchor
            prior_parents = node_parents.setdefault(nq, set())
            is_new_parent = q_idx not in prior_parents
            prior_parents.add(q_idx)
            if nq not in visited:
                frontier.append(nq)
                visited.add(nq)
                q_visit_count[nq] = q_visit_count.get(nq, 0) + 1
            elif (
                getattr(cfg, "bfs_allow_revisit", False)
                and is_new_parent
                and q_visit_count.get(nq, 0) < 1 + cfg.bfs_max_revisit
                # Only allow revisit for Qs that go through M1-vqa (lesion subtree, etc.).
                # Excluded Qs (anchors, dx, dx_count, report, computed grades) keep single-visit semantics.
                and nq not in cfg.m1_vqa_excluded_qis
                and nq in pos_of_q
            ):
                frontier.append(nq)
                q_visit_count[nq] = q_visit_count.get(nq, 0) + 1
                _pending_revisit_q.setdefault(nq, []).append(q_idx)
            if len(edges) >= cfg.max_edges:
                break
        if not emitted_any:
            edges.append((q_text, a_text, ""))

    # Build summary
    from reg2026.generate.summary_report import build_summary_report

    organ_display = history_text[0][1].strip() if history_text else ""
    organ_key = parse_organ_from_trajectory(history_text)
    summary = build_summary_report(organ_display, organ_key, history_text)

    if getattr(cfg, "codx_independent_subtree", False) and cf_subtree_dx_idxs:
        _organ_pos = pos_of_q.get(organ_q)
        _proc_pos = pos_of_q.get(procedure_q)
        _organ_a = a_seq[_organ_pos] if _organ_pos is not None else -1
        _proc_a = a_seq[_proc_pos] if _proc_pos is not None else -1
        for _sub_dx in cf_subtree_dx_idxs:
            _sub_edges = _run_codx_subtree(
                _sub_dx,
                _organ_a,
                _proc_a,
                module1_model,
                module2,
                tokenizer,
                max_seq_len_m2,
                tile_features,
                cfg,
                pools,
                device,
            )
            if _sub_edges:
                edges = _merge_codx_independent_subtree(edges, _sub_edges)

    return {
        "case_id": case_id,
        "edges": edges,
        "history": history_text,
        "summary": summary,
        "dx_count": n_dx if n_dx > 0 else 1,
        "routing_trace": routing_trace,
        "abmil_entropy": _abmil_entropy,
    }


_SUFFIX_RE = re.compile(r"\?-\d+$")


def strip_q_suffix(q: str) -> str:
    """Strip suffix `-N` from suffixed Q text (multi-dx workup).

    'Is there any invasion present?-2' -> 'Is there any invasion present?'
    """
    return _SUFFIX_RE.sub("?", q)


_AF_CANON = None

# co-dx independent subtree constants.
_AF_Q = "Is there any additional finding present?"
_AF_YES_A = "Yes, there is an additional finding."
_AF_NO_A = "No, there is no additional finding."
_CODX_TERMINAL_QS = (
    "What is the histologic type of lesion?",
    "What is the histologic type of neoplasm?",
)
_CODX_STOP_QS = (
    "What is the number of diagnoses to includes?",
    "What is the final pathology report?",
    "What is the #1 diagnosis?",
    "What is the #2 diagnosis?",
    "What is the #3 diagnosis?",
    "What is the #4 diagnosis?",
)


def _run_codx_subtree(
    subtree_dx_aidx: int,
    organ_a_idx: int,
    proc_a_idx: int,
    module1_model,
    module2,
    tokenizer,
    max_seq_len_m2: int,
    tile_features: torch.Tensor,
    cfg,
    pools: EmbeddingPools,
    device: str,
    max_depth: int = 8,
):
    """Generate a co-dx 2nd-dx reasoning subtree live via a fresh-state mini breadth-first search.

    The main breadth-first search cannot re-walk this subtree: it reuses Qs+edges already present in the main
    tree (e.g. proliferative->invasion), and the global ``is_new_parent`` revisit gate is
    q_idx-keyed / position-blind, so it blocks the re-emission and the subtree dies at ~1 node.
    Running the subtree as an INDEPENDENT instance (own visited/node_parents/frontier) sidesteps
    the global state entirely.

    The M2 history is seeded with an injected ``[organ, procedure, #1diagnosis=subtree_dx]``
    prefix (NOT emitted as edges) so M2 sees an in-distribution trajectory in which the co-dx is
    the primary diagnosis; M1-vqa (cosine) answers the visual gate Qs anchored on ``subtree_dx``, the
    in-situ entailment gate answers invasion/papillary deterministically, and the terminal
    histologic-type node is the co-dx itself.

    Args:
        subtree_dx_aidx: A_VOCAB idx of the co-finding dx (DCIS/CIS) - anchors + terminal answer.
        organ_a_idx / proc_a_idx: the main tree's organ / procedure A indices (injected anchors).
        max_depth: hard cap on subtree Q count (GT subtrees are ~5-6 workup Qs).

    Returns:
        list[(q_text, a_text, nq_text)] = workup edges only (launch AF=Yes -> ... -> terminal dx),
        with NO organ/proc/#1-dx anchor edges and NO dx-count/report. Empty if generation failed.
    """
    organ_a_text = pools.a_text(organ_a_idx) if organ_a_idx >= 0 else ""
    proc_a_text = pools.a_text(proc_a_idx) if proc_a_idx >= 0 else ""
    subtree_dx_text = pools.a_text(subtree_dx_aidx)

    inj_prefix: list[tuple[str, str]] = [
        ("What is the organ?", organ_a_text),
        ("What is the procedure?", proc_a_text),
        ("What is the #1 diagnosis?", subtree_dx_text),
    ]

    def _m2_next(history_workup: list[tuple[str, str]]) -> list[int]:
        m2_hist = inj_prefix + [(_AF_Q, _AF_YES_A)] + history_workup
        soft_cap = 1  # subtree spine is a chain; keep M2's single best continuation
        nqs = module2_predict_next_qs(
            module2,
            tokenizer,
            m2_hist,
            cfg.threshold,
            max_seq_len_m2,
            device,
            soft_cap=soft_cap,
            cand_set=None,
            cand_escape_thr=getattr(cfg, "candidate_escape_thr", 1.0),
        )
        return nqs

    _a_to_idx_ci = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}

    from reg2026.protocols.cofindings import insitu_gate_entailment as _insitu_entail

    edges_out: list[tuple[str, str, str]] = []
    history_workup: list[tuple[str, str]] = []  # (q_text, a_text) for M2 context (workup only)
    visited_q: set[int] = set()

    root_nqs = _m2_next([])
    cur_q_idx = next((nq for nq in root_nqs if nq < tokenizer.n_q_vocab), None)
    if cur_q_idx is None:
        return []

    prev_q_text = _AF_Q
    prev_a_text = _AF_YES_A

    for _depth in range(max_depth):
        if cur_q_idx is None or cur_q_idx in visited_q:
            break
        q_text = pools.Q_VOCAB[cur_q_idx]
        # Guard: a subtree must never re-emit dx-count / report / dx Qs (main-tree-owned).
        if q_text in _CODX_STOP_QS:
            break
        visited_q.add(cur_q_idx)

        # ---- Answer the current subtree Q ----
        is_terminal = q_text in _CODX_TERMINAL_QS
        # entailment mode: in-situ co-dx ⟹ invasion/papillary = No (Hook-C clinical derivation, computed_a).
        _insitu_gate_a = _insitu_entail(subtree_dx_text, q_text)
        if is_terminal:
            a_idx = subtree_dx_aidx
        elif _insitu_gate_a is not None and _a_to_idx_ci.get(_insitu_gate_a.strip().lower(), -1) >= 0:
            a_idx = _a_to_idx_ci[_insitu_gate_a.strip().lower()]
        else:
            # M1-vqa answers the co-dx gate Q on the whole slide.
            a_idx = _answer_codx_q(
                cur_q_idx,
                q_text,
                subtree_dx_aidx,
                organ_a_idx,
                proc_a_idx,
                module1_model,
                tile_features,
                cfg,
                pools,
                device,
            )
        a_text = pools.a_text(a_idx)

        # ---- Emit the edge from the previous node to this one ----
        edges_out.append((prev_q_text, prev_a_text, q_text))
        history_workup.append((q_text, a_text))

        if is_terminal:
            _dxcount_q = "What is the number of diagnoses to includes?"
            edges_out.append((q_text, a_text, _AF_Q))
            edges_out.append((_AF_Q, _AF_NO_A, _dxcount_q))
            break

        # ---- Route to the next subtree Q via M2 ----
        nqs = _m2_next(history_workup)
        # Stop when M2 closes the subtree (additional-finding gate) or emits nothing.
        next_q_idx = None
        for nq in nqs:
            if nq >= tokenizer.n_q_vocab:
                continue
            nq_text = pools.Q_VOCAB[nq]
            if nq_text == _AF_Q or nq_text in _CODX_STOP_QS:
                continue
            if nq in visited_q:
                continue
            next_q_idx = nq
            break
        prev_q_text, prev_a_text = q_text, a_text
        cur_q_idx = next_q_idx
        if cur_q_idx is None:
            break

    _has_terminal = any(q in _CODX_TERMINAL_QS for q, _a, _nq in edges_out) or any(nq in _CODX_TERMINAL_QS for _q, _a, nq in edges_out)
    if not _has_terminal and len(edges_out) < 2:
        return []
    return edges_out


def _answer_codx_q(
    q_idx: int,
    q_text: str,
    subtree_dx_aidx: int,
    organ_a_idx: int,
    proc_a_idx: int,
    module1_model,
    tile_features: torch.Tensor,
    cfg,
    pools: EmbeddingPools,
    device: str,
) -> int:
    """Answer one (non-terminal) subtree Q, anchored on the subtree dx.

    Answers via M1-vqa cosine (anchor batch: organ, [proc,] subtree_dx, target_q) restricted to the
    per-Q answer mask. Closed whitelist Qs are NOT cls-answered here: the only one that reaches a
    subtree ("invasion") is handled upstream by the in-situ entailment gate, and M1-vqa's cls_head
    is untrained - so this path stays cosine-only.
    """
    organ_q = pools.q_idx("What is the organ?")
    procedure_q = pools.q_idx("What is the procedure?")
    dx1_q = pools.q_idx("What is the #1 diagnosis?")

    # VQA batch anchored on the subtree dx (not global #1). 2-anchor ckpts drop the procedure pos
    # -> [organ, subtree_dx, target_q]; else [organ, proc, subtree_dx, target_q].
    _drop_proc = getattr(cfg, "m1_vqa_drop_proc", False)
    vqa_q_seq, vqa_a_seq, vqa_anc = _vqa_anchor_seqs(organ_q, procedure_q, dx1_q, q_idx, organ_a_idx, proc_a_idx, subtree_dx_aidx, _drop_proc)

    active_model = cfg.m1_vqa if cfg.m1_vqa is not None else module1_model
    feats_for_model = tile_features
    vqa_batch = build_module1_batch(feats_for_model, vqa_q_seq, vqa_a_seq, vqa_anc, device, max_N=8)
    if cfg.m1_vqa is not None and getattr(cfg.m1_vqa.config, "max_occ", 0) > 0:
        vqa_batch.occ_indices = torch.tensor([[0] * len(vqa_q_seq)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = active_model(vqa_batch)
    a_emb = out["a_emb_pred"][0, -1]

    return _restricted_nn(
        a_emb,
        active_model.a_emb_pool,
        q_text,
        pools,
        per_q_a_mask=getattr(cfg, "per_q_a_mask", None),
        q_idx=q_idx,
    )


def _merge_codx_independent_subtree(edges, subtree_edges):
    """Merge a live-generated co-dx subtree into the main trajectory at the AF=Yes launch.

    Merge contract: the additional-finding node's out-edges are
    REPLACED by the subtree (which provides both the AF=Yes launch and, implicitly via M2's close,
    the workup), every subtree edge wins over a colliding main edge, and we de-dupe. The subtree's
    terminal ``histologic type of lesion = subtree_dx`` is the main tree's #2 diagnosis (already in
    the dx-chain from the cofinding head; the subtree only supplies the workup structure).
    """
    from reg2026.data.embedding_pools import canonicalize_q

    global _AF_CANON
    if _AF_CANON is None:
        _AF_CANON = canonicalize_q("Is there any additional finding present?")

    def cq(s):
        return canonicalize_q(strip_q_suffix((s or "").strip()))

    if not subtree_edges:
        return edges

    sub_keys = {(cq(q), cq(nq)) for q, _a, nq in subtree_edges}
    kept = []
    for q, a, nq in edges:
        if cq(q) == _AF_CANON:
            if (a or "").strip() == _AF_NO_A:
                kept.append((q, a, nq))
            continue
        if (cq(q), cq(nq)) in sub_keys:
            continue  # subtree provides this edge live
        kept.append((q, a, nq))

    out, seen = [], set()
    for t in kept + list(subtree_edges):
        k = (cq(t[0]), (t[1] or "").strip(), cq(t[2]))
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


_GLEASON_PAT_TO_INT = {"gleason pattern 3": 3, "gleason pattern 4": 4, "gleason pattern 5": 5}
_GLEASON_ISUP_GG = {(3, 3): 1, (3, 4): 2, (4, 3): 3, (4, 4): 4, (3, 5): 4, (5, 3): 4, (4, 5): 5, (5, 4): 5, (5, 5): 5}
_Q_GLEASON_SCORE = "What is the Gleason score?"
_Q_GRADE_GROUP = "What is the grade group?"
_Q_PRIDOMINANT = "What is the pridominant pattern?"
_Q_SECONDARY = "What is the secondary pattern constituting more than 5% of tumor?"


def gleason_from_patterns_idx(q_text, history_text, pools):
    """FRP: derive Gleason score / grade group A-index from the predominant+secondary
    pattern answers already in history. Returns a_idx, or None if patterns unavailable
    or the derived answer is not in A_VOCAB (caller falls back to existing routing).

    Pattern-derived is slightly more accurate than the direct score cls AND makes
    score/pattern/grade-group mutually consistent (eliminates cases where the
    independent score cls contradicts the predominant/secondary Qs).
    """
    pri = sec = None
    for q, a in history_text:
        if q == _Q_PRIDOMINANT:
            pri = _GLEASON_PAT_TO_INT.get(a.strip().lower())
        elif q == _Q_SECONDARY:
            sec = _GLEASON_PAT_TO_INT.get(a.strip().lower())
    if pri is None:
        return None
    if sec is None:
        sec = pri  # pure pattern → secondary == predominant
    if q_text == _Q_GLEASON_SCORE:
        target = f"{pri + sec} ({pri}+{sec})"
    else:  # grade group
        gg = _GLEASON_ISUP_GG.get((pri, sec))
        if gg is None:
            return None
        target = f"Grade group {gg}"
    idx = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}.get(target.strip().lower(), -1)
    return idx if idx >= 0 else None


def setup_inference(args):
    """Load all models + build the inference cfg. Shared by main() (val eval) and the
    submission container (interf1). GT-free except the optional --use-gt-a oracle path."""
    torch.manual_seed(getattr(args, "seed", 0))
    if getattr(args, "deterministic", False):
        import os as _os

        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Deterministic mode ON (cuDNN deterministic + use_deterministic_algorithms warn_only)")

    logger.info("Loading pools")
    pools = EmbeddingPools()

    logger.info("Loading module1-%s model from %s", args.variant.upper(), args.module1_ckpt)
    module1_model = build_module1(args, pools)

    _routing_cfg = None
    if getattr(args, "routing_config", None) is not None:
        _routing_cfg = json.loads(args.routing_config.read_text()).get("routing", {})
        logger.info("Routing config loaded from %s (%d Qs) - per-Q decoder/context override inline hardcodes", args.routing_config, len(_routing_cfg))

    m1_vqa_model = None
    m1_vqa_excluded_qis: set[int] = set()
    if args.m1_vqa_ckpt is not None:
        logger.info("Loading M1-vqa from %s", args.m1_vqa_ckpt)
        import copy as _copy

        _vqa_args = _copy.copy(args)
        _vqa_args.module1_ckpt = args.m1_vqa_ckpt
        _vqa_ckpt_peek = torch.load(args.m1_vqa_ckpt, map_location="cpu", weights_only=False)
        _vqa_ckpt_state = _vqa_ckpt_peek.get("model_state", {})
        _vqa_max_occ = _vqa_ckpt_state["occ_emb.weight"].shape[0] if "occ_emb.weight" in _vqa_ckpt_state else 0
        _vqa_args.max_occ = _vqa_max_occ
        _vqa_args.max_N = _vqa_ckpt_state["turn_emb.weight"].shape[0] if "turn_emb.weight" in _vqa_ckpt_state else 8
        logger.info("M1-vqa max_N detected from ckpt turn_emb: %d", _vqa_args.max_N)
        del _vqa_ckpt_peek, _vqa_ckpt_state
        logger.info("M1-vqa max_occ detected from ckpt: %d", _vqa_max_occ)
        m1_vqa_model = build_module1(_vqa_args, pools)
        if _routing_cfg is not None:
            _excluded = [q for q, v in _routing_cfg.items() if v.get("context") == "context-aware"]
        else:
            _excluded = [
                "What is the organ?",
                "What is the procedure?",
                "What is the number of diagnoses to includes?",
                "What is the final pathology report?",
                "What is the #1 diagnosis?",
                "What is the #2 diagnosis?",
                "What is the #3 diagnosis?",
                "What is the #4 diagnosis?",
                "What is the Gleason score?",
                "What is the grade group?",
                "What is the overall score?",
                "What is the grading system?",
                "What is the grading system of neoplasm?",
                "What is the grading system of dysplasia?",
                "What is the grading system of atypia?",
            ]
        import contextlib

        for qt in _excluded:
            with contextlib.suppress(KeyError):
                m1_vqa_excluded_qis.add(pools.q_idx(qt))
        logger.info("M1-vqa excluded Q indices (use cascade M1 instead): %d", len(m1_vqa_excluded_qis))

    m1cls_closed_qis: set[int] = set()
    m1cls_legal_mask = None
    closed_q_legal: dict[int, list[int]] = {}  # ALL closed Qs -> legal a_idx (cosine restrict_to guard)
    if getattr(args, "m1cls_answer_space", None) is not None:
        _qas = json.loads(args.m1cls_answer_space.read_text())
        if _routing_cfg is not None:
            _M1CLS_INCLUDE = {q for q, v in _routing_cfg.items() if v.get("decoder") == "cls"}
        else:
            _M1CLS_INCLUDE = {
                # prior 6 minus histologic-group (re-confirmed cls>vqa)
                "Is there any invasion present?",
                "What is the Gleason score?",  # cls > Hook-C gleason_sum
                "What is the worst grade pattern?",
                "Is there any mucinous feature present?",
                "Is there any inflammation present?",
                # additions - m1 visual Qs, cls>vqa teacher-forced
                "What is the score for nuclear pleomorphism?",  # Nottingham sub-score
                "What is the score for tubular differentiation?",
                "What is the score for mitotic rate?",
                "What is the grade of dysplasia?",
                "Is there any dysplasia present?",
                "Is there any necrosis present?",
                "Is there any morphological squamous cell pattern present?",
                "What is the pridominant pattern?",  # prostate Gleason
                "What is the secondary pattern constituting more than 5% of tumor?",
            }
        m1cls_legal_mask = torch.zeros(len(pools.Q_VOCAB), len(pools.A_VOCAB), dtype=torch.bool, device=args.device)
        for _q, _v in _qas.items():
            if _q not in _M1CLS_INCLUDE or not _v.get("closed") or not _v["legal_a_idx"]:
                continue
            try:
                _qi = pools.q_idx(_q)
            except KeyError:
                continue
            m1cls_closed_qis.add(_qi)
            for _ai in _v["legal_a_idx"]:
                m1cls_legal_mask[_qi, _ai] = True
        for _q, _v in _qas.items():
            if not _v.get("closed") or not _v.get("legal_a_idx"):
                continue
            try:
                closed_q_legal[pools.q_idx(_q)] = list(_v["legal_a_idx"])
            except KeyError:
                continue
        logger.info("M1-cls inference: %d whitelisted Qs -> cls_head (strict cls>cosine, no SOTA rule); %d closed Qs legal-guarded", len(m1cls_closed_qis), len(closed_q_legal))

    logger.info("Loading Module 2 from %s", args.module2_ckpt)
    module2, tokenizer, max_seq_len_m2 = load_module2(args, pools, args.device)

    logger.info("Loading CoT GT + val split")
    cot = json.loads(args.cot_json.read_text())
    cot_by_id = {c["id"].replace(".tiff", ""): c for c in cot}
    split = json.loads(args.split_json.read_text())
    val_ids = sorted(s.replace(".tiff", "") for s in split["val"])
    if getattr(args, "max_cases", 0):
        val_ids = val_ids[: args.max_cases]

    threshold = args.threshold

    codx_integrator = None
    if getattr(args, "codx_integrator_json", None) is not None:
        codx_integrator = json.loads(args.codx_integrator_json.read_text())
        logger.info("Loaded integrator co-dx from %s (%d cases, %d co-dx)", args.codx_integrator_json, len(codx_integrator), sum(len(v) for v in codx_integrator.values()))

    per_q_a_mask_loaded: dict[int, list[int]] | None = None
    if args.q_conditioned_mask_json is not None:
        raw_mask = json.loads(args.q_conditioned_mask_json.read_text())
        per_q_a_mask_loaded = {int(k): list(v) for k, v in raw_mask.items()}
        logger.info("Loaded per-Q candidate mask: %d Qs masked from %s", len(per_q_a_mask_loaded), args.q_conditioned_mask_json)

    abmil_primary = None
    if args.abmil_primary_ckpt is not None:
        logger.info("Loading ABMIL primary from %s", args.abmil_primary_ckpt)
        abmil_primary = load_abmil_primary(args.abmil_primary_ckpt, pools, args.device)

    dx_answer_subspace = None
    if getattr(args, "dx_subspace_guard", False):
        a_lc = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}
        cot_entries = json.loads(args.cot_json.read_text())
        _dxset: set[int] = set()
        for e in cot_entries:
            for t in e.get("chain-of-thought", []):
                q = (t.get("question") or "").strip()
                if q.startswith("What is the #") and q.endswith("diagnosis?"):
                    ai = a_lc.get((t.get("answer") or "").strip().lower(), -1)
                    if ai >= 0:
                        _dxset.add(ai)
        dx_answer_subspace = sorted(_dxset)
        logger.info("dx-subspace-guard ON: %d valid diagnosis answers in subspace", len(dx_answer_subspace))

    # Build per-source-Q max train fanout for top-K decoding
    q_max_fanout: dict[int, int] = {}
    if args.m2_top_k_from_train:
        from collections import defaultdict as _dd

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from precompute_dag_masks import collapse_chain

        cot_raw = json.loads(args.cot_json.read_text())
        _norm_cap = getattr(args, "normalize_glitches", False)
        if _norm_cap:
            from reg2026.data.cot_normalize import normalize_cot_chain as _ncc
        nq_count_by_q: dict[int, int] = _dd(int)
        for c in cot_raw:
            try:
                _chain = _ncc(c["chain-of-thought"], c.get("organ") or "") if _norm_cap else c["chain-of-thought"]
                nodes, _ = collapse_chain(_chain)
            except Exception:
                continue
            for n in nodes:
                try:
                    q_idx_t = pools.q_idx(n.q_text)
                except KeyError:
                    continue
                if len(n.next_qs) > nq_count_by_q[q_idx_t]:
                    nq_count_by_q[q_idx_t] = len(n.next_qs)
        q_max_fanout = dict(nq_count_by_q)
        logger.info(
            "M2 top-K cap from train: %d source Qs (max fanout range %d-%d)",
            len(q_max_fanout),
            min(q_max_fanout.values()) if q_max_fanout else 0,
            max(q_max_fanout.values()) if q_max_fanout else 0,
        )

    # per-source-Q candidate filter {q_idx: set(allowed next q_idx)} - SHIPPABLE
    candidate_filter = None
    if args.candidate_mask_json is not None and args.candidate_mask_json.exists():
        _cm = json.loads(args.candidate_mask_json.read_text())
        candidate_filter = {}
        for _sq, _allowed in _cm.items():
            try:
                _si = pools.q_idx(_sq)
            except (KeyError, ValueError):
                continue
            _aset = set()
            for _tq in _allowed:
                try:
                    _aset.add(pools.q_idx(_tq))
                except (KeyError, ValueError):
                    continue
            candidate_filter[_si] = _aset
        logger.info("candidate filter: %d source Qs (e2e emission restricted to legal candidates)", len(candidate_filter))

    cofinding_heads = None
    if getattr(args, "cofinding_heads_json", None) is not None:
        _a_lc = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}
        cofinding_heads = []
        for spec in json.loads(args.cofinding_heads_json.read_text()):
            a_idx = _a_lc.get(spec["a_text"].strip().lower(), -1)
            if a_idx < 0:
                logger.warning("cofinding head a_text %r not in A_VOCAB; skipping", spec["a_text"])
                continue
            _ht = spec.get("type", "lse")
            _hd = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
            _route = (spec.get("route") or "additional_finding").strip().lower()
            _pq = (spec.get("presence_q") or "").strip() or None
            _pq_yes_idx = None
            if _route == "presence_q":
                if _pq is None:
                    logger.warning("cofinding head %r route=presence_q but no presence_q; skipping", spec["a_text"])
                    continue
                _pq_yes_txt = (spec.get("presence_yes_a_text") or "").strip()
                _pq_yes_idx = _a_lc.get(_pq_yes_txt.lower(), -1)
                if _pq_yes_idx < 0:
                    logger.warning("cofinding head %r presence_yes_a_text %r not in A_VOCAB; skipping", spec["a_text"], _pq_yes_txt)
                    continue
            _common = {
                "a_idx": a_idx,
                "organ": spec.get("organ"),
                "primary_substr": ((spec.get("primary_substr") or "").strip().lower() or None),
                "route": _route,
                "presence_q": _pq,
                "presence_yes_a_idx": _pq_yes_idx,
            }
            if _ht == "tilehead":
                model = _build_tilehead(_hd, device=args.device)
                _agg = spec.get("agg", _hd.get("best_student_agg", "top10"))
                thr = float(spec.get("threshold", _hd.get("threshold", 0.0)))
                _cap = int(spec.get("cap", _hd.get("cap", 0)))  # heads cap H-Opt to 1200-tile linspace
                cofinding_heads.append({**_common, "type": "tilehead", "model": model, "agg": _agg, "cap": _cap, "threshold": thr})
                logger.info(
                    "Co-finding TileHead loaded: %r organ=%s route=%s prim=%s agg=%s thr=%.4f cap=%s presence_q=%r (auc=%.3f)",
                    spec["a_text"],
                    spec.get("organ"),
                    _route,
                    spec.get("primary_substr"),
                    _agg,
                    thr,
                    _cap,
                    _pq,
                    _hd.get("val_auc", _hd.get("best_student_auc", 0.0)),
                )
            else:
                thr = float(spec.get("threshold", _hd["threshold"]))
                cofinding_heads.append({**_common, "type": "lse", "w": _hd["w"].float(), "b": _hd["b"].float(), "tau": _hd["tau"].float(), "threshold": thr})
                logger.info("Co-finding head loaded: %r organ=%s prim=%s thr=%.3f (ckpt val_auc=%.3f)", spec["a_text"], spec.get("organ"), spec.get("primary_substr"), thr, _hd.get("val_auc", 0.0))

    cfg = type(
        "Cfg",
        (),
        {
            "threshold": threshold,
            "candidate_filter": candidate_filter,
            "max_edges": args.max_edges,
            "max_dx": args.max_dx,
            "max_N": args.max_N,
            "abmil_primary": abmil_primary,
            "dx_subspace_guard": getattr(args, "dx_subspace_guard", False),
            "dx_answer_subspace": dx_answer_subspace,
            "codx_integrator": codx_integrator,
            "q_max_fanout": q_max_fanout,
            "candidate_escape_thr": args.candidate_escape_thr,
            "m1_vqa": m1_vqa_model,
            "m1_vqa_excluded_qis": m1_vqa_excluded_qis,
            "m1_vqa_drop_proc": getattr(args, "m1_vqa_drop_proc_anchor", False),
            "bfs_allow_revisit": getattr(args, "bfs_allow_revisit", False),
            "bfs_max_revisit": getattr(args, "bfs_max_revisit", 3),
            "per_q_a_mask": per_q_a_mask_loaded,
            "m1cls_closed_qis": m1cls_closed_qis,
            "m1cls_legal_mask": m1cls_legal_mask,
            "closed_q_legal": closed_q_legal,
            "gleason_from_patterns": getattr(args, "gleason_from_patterns", False),
            "per_node_anchor": getattr(args, "per_node_anchor", False),
            "cofinding_heads": cofinding_heads,
            "m2_gate_context": getattr(args, "m2_gate_context", False),
            "codx_independent_subtree": getattr(args, "codx_independent_subtree", False),
            "dump_routing": getattr(args, "dump_routing", None) is not None,
        },
    )()

    return module1_model, module2, tokenizer, max_seq_len_m2, pools, cfg, cot_by_id, val_ids


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    module1_model, module2, tokenizer, max_seq_len_m2, pools, cfg, cot_by_id, val_ids = setup_inference(args)

    # Run the BFS trajectory per case and dump the official-format predictions. Scoring is delegated
    # to the official scorer (run_official_eval.py + evaluate_metrics.py); this entrypoint no longer
    # computes a local WFR estimate - it only produces the predicted trajectories that scorer
    # consumes (plus an optional per-Q routing dump for mechanism transparency).
    official_preds: dict[str, list[dict]] = {}  # per-case predicted trajectory in official format
    routing_q: dict[str, Counter] = defaultdict(Counter)  # q_text -> handler counts (faithful)
    routing_qo: dict[str, Counter] = defaultdict(Counter)  # "q_text @ organ" -> handler counts

    n_done = 0
    for sid in val_ids:
        if sid not in cot_by_id:
            continue
        feat_path = args.features_dir / f"{sid}.h5"
        if not feat_path.exists():
            continue

        try:
            res = predict_case(sid, feat_path, module1_model, module2, tokenizer, max_seq_len_m2, pools, cfg, args.device)
        except Exception as e:
            logger.warning("Skip %s: %s", sid, e)
            continue

        # aggregate the faithful per-Q answer-source map (single source of truth).
        if args.dump_routing is not None:
            _org = (cot_by_id[sid].get("organ") or "?").strip().lower()
            for _q, _src in res.get("routing_trace", []):
                routing_q[_q][_src] += 1
                routing_qo[f"{_q} @ {_org}"][_src] += 1

        # official-format trajectory (q,a,next_q triples) for the official scorer.
        if args.dump_official_pred_json is not None:
            official_preds[sid] = [{"question": (q or "").strip(), "answer": (a or "").strip(), "next_question": (nq or "").strip()} for q, a, nq in res["edges"]]

        n_done += 1
        if n_done % 200 == 0:
            logger.info("Done %d val cases", n_done)

    # Minimal run summary. The OFFICIAL WFR (incl. 0.40×FinalReport) is computed by run_official_eval.py
    # on the --dump-official-pred-json output; this entrypoint no longer estimates a local score.
    summary = [
        f"# module1-{args.variant.upper()} predicted-trajectory dump",
        "",
        f"Val cases processed: {n_done}",
        f"Module 1: {args.module1_ckpt.name} | Module 2: {args.module2_ckpt.name}",
        "",
        "Official scoring: feed the --dump-official-pred-json output to run_official_eval.py",
        "(WFR = 0.05×BPV + 0.30×Edge-F1 + 0.25×MESS + 0.40×FinalReport).",
    ]
    args.output.write_text("\n".join(summary) + "\n")
    logger.info("Wrote %s", args.output)
    if args.dump_official_pred_json is not None:
        args.dump_official_pred_json.write_text(json.dumps(official_preds, indent=2))
        logger.info("Wrote %d official-format predicted trajectories to %s (feed run_official_eval.py)", len(official_preds), args.dump_official_pred_json)
    if args.dump_routing is not None:
        # single source of truth: per-Q handler (+ per-Q-per-organ where it differs), faithful from the live cascade.
        routing_out = {}
        for q in sorted(routing_q):
            handlers = dict(routing_q[q].most_common())
            organs = {qo.split(" @ ")[-1]: dict(routing_qo[qo].most_common()) for qo in routing_qo if qo.startswith(q + " @ ")}
            per_organ_differs = len({next(iter(h)) for h in organs.values()}) > 1
            routing_out[q] = {"handlers": handlers, "primary_handler": next(iter(handlers)), **({"per_organ": organs} if per_organ_differs else {})}
        args.dump_routing.write_text(json.dumps(routing_out, ensure_ascii=False, indent=2))
        logger.info("routing dump: %d Qs → %s (machine-derived single source of truth)", len(routing_out), args.dump_routing)
    print("\n".join(summary))


if __name__ == "__main__":
    sys.exit(main() or 0)
