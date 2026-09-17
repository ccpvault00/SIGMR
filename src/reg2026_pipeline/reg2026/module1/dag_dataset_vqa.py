"""M1-vqa dataset: 3-anchor reduced-context Q-A samples.

For each case in CoT, for each non-excluded target Q, generates one sample with:
  - q_indices = [organ_q, procedure_q, #1_dx_q, target_q]
  - a_indices_gt = [organ_a, procedure_a, primary_dx_a (BASE), target_a]
  - attn_mask = causal (each Q attends to itself + all prior)
  - loss_mask (via collate_dag_vqa override) = only TRUE for target position (3)

Critical design (v2 fix, 2026-05-29):
- Anchor positions (0,1,2) keep REAL a_indices_gt values (organ, procedure, primary_dx)
  → model sees correct anchor A embedding via teacher forcing
- Loss is masked to target position only via custom collate (collate_dag_vqa)
  → anchor positions don't contribute to loss but model still conditions on them
- primary_dx anchor (position 2) is stripped of grade/diff modifiers to match
  base-dx slot bank inference output (vocab 133)
- 1076 Breast "IDC of NST" cases (base not in A_VOCAB) proxy to grade-II variant

Anchor scheduled sampling (training only):
- With probability schedsamp_rate (default 0.2), replace primary_dx anchor
  with base-dx slot top-1 (if wrong) or top-2 (if top-1 == GT).
- Forces M1-vqa to learn visual override when anchor contradicts visual evidence.
- Critical for downstream review step that reconciles anchor mistakes.

Excluded Qs (handled by other modules, M2 routing, or rule_dag computed_a):
  - organ, procedure                            -> M0 / existing M1 standalone
  - dx_count                                    -> M1-multi head
  - #1/#2/#3/#4 diagnosis                       -> slot bank
  - Gleason score, grade group, overall score   -> Hook C computed_a
  - 4 grading_system variants                   -> Hook C computed_a (#76)
  - final pathology report                      -> M1-report template
NOTE: "grade of neoplasm" is INTENTIONALLY NOT excluded - Hook C only covers
Breast Nottingham; Lung/HCC "Moderately differentiated" etc. need M1-vqa fallback.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from reg2026.data.embedding_pools import EmbeddingPools
from reg2026.module1.dag_dataset import DAGBatch, DAGSample, collate_dag

logger = logging.getLogger(__name__)


ORGAN_Q = "What is the organ?"
PROCEDURE_Q = "What is the procedure?"
DX_COUNT_Q = "What is the number of diagnoses to includes?"
DX_QS = {f"What is the #{k} diagnosis?" for k in range(1, 5)}
PRIMARY_DX_Q = "What is the #1 diagnosis?"
FINAL_REPORT_Q = "What is the final pathology report?"

# Qs handled by Hook C computed_a (deterministic from earlier As; M1-vqa skips).
# NOTE: "What is the grade of neoplasm?" intentionally NOT in this set - Hook C's
# nottingham_grade only covers Breast (returns None for Lung/HCC "Moderately
# differentiated" etc.); M1-vqa must predict the Lung-style answers.
COMPUTED_QS = {
    "What is the Gleason score?",
    "What is the grade group?",
    "What is the overall score?",
    "What is the grading system?",
    "What is the grading system of neoplasm?",
    "What is the grading system of dysplasia?",
    "What is the grading system of atypia?",
}

EXCLUDED_TARGET_QS = {ORGAN_Q, PROCEDURE_Q, DX_COUNT_Q, FINAL_REPORT_Q} | DX_QS | COMPUTED_QS

# --- Modifier stripping (MUST match scripts/the slot-bank trainer:strip_grade_diff) ---
# Why: M1-vqa anchor must agree with base-dx slot bank vocab (vocab 133, modifier-stripped).
_GRADE_NUM_PAT = re.compile(r",?\s*grade\s+(I{1,3}V?|IV|\d+)\s*$", re.IGNORECASE)
_HL_GRADE_PAT = re.compile(r",?\s*(high|low|intermediate)\s+grade\s*$", re.IGNORECASE)
_DIFF_PAT = re.compile(r",?\s*(well|moderately|poorly|undifferentiated)\s+differentiated\s*$", re.IGNORECASE)


def strip_grade_diff(dx_text: str) -> str:
    for pat in (_GRADE_NUM_PAT, _HL_GRADE_PAT, _DIFF_PAT):
        dx_text = pat.sub("", dx_text).strip()
    return dx_text


# Proxy aidx for base-dx texts whose base form is not in A_VOCAB.
# Currently only 1 case: "Invasive carcinoma of no special type" (1076 Breast IDC).
# Use grade-II variant as proxy (most common; Breast's M1-vqa targets are visual
# Nottingham subscores so the grade-II bias doesn't leak useful info to target Qs).
BASE_DX_AIDX_PROXY: dict[str, str] = {
    "invasive carcinoma of no special type": "Invasive carcinoma of no special type, grade II",
}


@dataclass(frozen=True)
class VqaDatasetConfig:
    cot_json: Path
    features_dir: Path
    split_json: Path
    split: str = "train"
    organ_to_idx: dict[str, int] | None = None
    schedsamp_rate: float = 0.0
    schedsamp_top2_json: Path | None = None
    drop_proc_anchor: bool = False  # deployed m1_vqa_2anchor: drop the procedure anchor -> 2-anchor batch [organ, #1-dx, target]
    seed: int = 42
    # percentage Qs as 11-class buckets. {q_text: [bucket pct ints]} → GT a_idx
    # for these Qs is snapped to the nearest bucket so the M1-cls cls CE has a legal target.
    pct_bucket_qs: dict[str, list[int]] | None = None


class M1VqaDataset(Dataset[DAGSample]):
    """One sample per (case, target_q) tuple - 4 positions (3 anchors + target), or 3 positions
    (2 anchors + target) when config.drop_proc_anchor is set (the deployed m1_vqa_2anchor variant)."""

    def __init__(self, config: VqaDatasetConfig, pools: EmbeddingPools) -> None:
        self.config = config
        self.pools = pools
        self.a_text_to_idx = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}
        self.rng = random.Random(config.seed)

        # per-Q {raw pct a_idx → nearest-bucket a_idx} snap maps (gated by config.pct_bucket_qs).
        self._pct_snap: dict[str, dict[int, int]] = {}
        for q_text, buckets in (config.pct_bucket_qs or {}).items():
            bmap: dict[int, int] = {}
            for i, a in enumerate(pools.A_VOCAB):
                m = re.match(r"\s*(\d+)\s*%", a)
                if m is None:
                    continue
                nearest = min(buckets, key=lambda b: abs(b - int(m.group(1))))
                bmap[i] = self.a_text_to_idx.get(f"{nearest}%", -1)
            self._pct_snap[q_text] = bmap

        cot = json.loads(config.cot_json.read_text())
        cot_by_id = {c["id"].replace(".tiff", ""): c for c in cot}

        split_data = json.loads(config.split_json.read_text())
        split_ids = {s.replace(".tiff", "") for s in split_data[config.split]}

        # Load top-2 dict for scheduled sampling (TRAIN split only).
        self.case_top2_aidx: dict[str, list[int]] = {}
        if config.split == "train" and config.schedsamp_rate > 0 and config.schedsamp_top2_json is not None:
            top2_raw = json.loads(config.schedsamp_top2_json.read_text())
            for cid, dx_list in top2_raw.items():
                aidxs: list[int] = []
                for dx in dx_list:
                    aidx = self._resolve_base_dx_aidx(dx)
                    if aidx >= 0:
                        aidxs.append(aidx)
                if aidxs:
                    self.case_top2_aidx[cid] = aidxs
            logger.info("Loaded scheduled sampling top-2 for %d cases (rate=%.2f)", len(self.case_top2_aidx), config.schedsamp_rate)

        self.samples: list[dict] = []
        n_no_features = 0
        n_no_anchors = 0
        n_unknown_q = 0
        n_missing_base_aidx = 0

        organ_qi = pools.q_idx(ORGAN_Q)
        procedure_qi = pools.q_idx(PROCEDURE_Q)
        dx1_qi = pools.q_idx(PRIMARY_DX_Q)

        for case_id in sorted(split_ids):
            if case_id not in cot_by_id:
                continue
            if not (config.features_dir / f"{case_id}.h5").exists():
                n_no_features += 1
                continue
            chain = cot_by_id[case_id]["chain-of-thought"]

            organ_a = next((t["answer"].strip() for t in chain if t["question"] == ORGAN_Q), None)
            procedure_a = next((t["answer"].strip() for t in chain if t["question"] == PROCEDURE_Q), None)
            primary_dx_a_full = next((t["answer"].strip() for t in chain if t["question"] == PRIMARY_DX_Q), None)
            if not (organ_a and procedure_a and primary_dx_a_full):
                n_no_anchors += 1
                continue

            # Strip grade/diff modifiers from primary_dx - must match base-dx slot bank output.
            primary_dx_a_base = strip_grade_diff(primary_dx_a_full)

            organ_aidx = self.a_text_to_idx.get(organ_a.lower(), -1)
            procedure_aidx = self.a_text_to_idx.get(procedure_a.lower(), -1)
            primary_dx_aidx = self._resolve_base_dx_aidx(primary_dx_a_base)
            if primary_dx_aidx < 0:
                n_missing_base_aidx += 1

            organ_idx = config.organ_to_idx.get(organ_a, -1) if config.organ_to_idx is not None else -1

            # Per-occ enumeration with SUFFIX MODE (multi-dx ceiling).
            # Same Q in non-consecutive chain positions = new occurrence → suffixed target_q (e.g., "X-2").
            # Suffixed Qs must be in pools.Q_VOCAB (use embedding_pools_occ).
            per_q_occ_count: dict[str, int] = {}
            sample_seen: set[tuple[str, int]] = set()
            prev_q_in_chain: str | None = None

            def _suffixed(q, occ):
                return q if occ == 0 else f"{q}-{occ + 1}"

            for t in chain:
                tq = t["question"]
                if tq in EXCLUDED_TARGET_QS:
                    prev_q_in_chain = tq
                    continue
                ta_text = t["answer"].strip()
                is_consecutive = tq == prev_q_in_chain
                if not is_consecutive:
                    per_q_occ_count[tq] = per_q_occ_count.get(tq, -1) + 1
                occ_idx = per_q_occ_count[tq]
                prev_q_in_chain = tq
                key = (tq, occ_idx)
                if key in sample_seen:
                    continue
                sample_seen.add(key)
                # Suffix target Q if occ>0 (new visit). Pools must have the suffixed entry.
                target_q_text = _suffixed(tq, occ_idx)
                try:
                    target_qi = pools.q_idx(target_q_text)
                except KeyError:
                    n_unknown_q += 1
                    continue
                target_aidx = self.a_text_to_idx.get(ta_text.lower(), -1)
                # snap percentage GT to nearest bucket (base Q text, pre-suffix).
                snap = self._pct_snap.get(tq)
                if snap is not None and target_aidx in snap:
                    target_aidx = snap[target_aidx]
                if target_aidx < 0:
                    continue
                if config.drop_proc_anchor:
                    q_arr = np.array([organ_qi, dx1_qi, target_qi], dtype=np.int64)
                    a_arr = np.array([organ_aidx, primary_dx_aidx, target_aidx], dtype=np.int64)
                else:
                    q_arr = np.array([organ_qi, procedure_qi, dx1_qi, target_qi], dtype=np.int64)
                    a_arr = np.array([organ_aidx, procedure_aidx, primary_dx_aidx, target_aidx], dtype=np.int64)
                self.samples.append(
                    {
                        "case_id": case_id,
                        "q_indices": q_arr,
                        "a_indices_gt": a_arr,
                        "organ_idx": organ_idx,
                    }
                )

        logger.info(
            "M1VqaDataset(%s): %d samples · skipped: %d no_features, %d no_anchors, %d unknown_q · missing_base_aidx: %d (Breast IDC of NST proxy used)",
            config.split,
            len(self.samples),
            n_no_features,
            n_no_anchors,
            n_unknown_q,
            n_missing_base_aidx,
        )

    def _resolve_base_dx_aidx(self, dx_text: str) -> int:
        """Look up A_VOCAB idx for base dx, with proxy fallback for known missing bases."""
        idx = self.a_text_to_idx.get(dx_text.lower(), -1)
        if idx >= 0:
            return idx
        proxy = BASE_DX_AIDX_PROXY.get(dx_text.lower())
        if proxy is not None:
            return self.a_text_to_idx.get(proxy.lower(), -1)
        return -1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DAGSample:
        s = self.samples[idx]
        case_id = s["case_id"]
        with h5py.File(self.config.features_dir / f"{case_id}.h5", "r") as fh:
            tile_features = torch.from_numpy(fh["features"][...]).float()

        N = int(s["q_indices"].shape[0])  # 4 [organ, procedure, #1-dx, target], or 3 if drop_proc_anchor
        attn_mask = torch.zeros(2 * N, 2 * N, dtype=torch.bool)
        for i in range(2 * N):
            attn_mask[i, : i + 1] = True

        # Real anchor a_indices_gt (…, primary_dx, target).
        # collate_dag_vqa will mask loss to target position only.
        a_indices = s["a_indices_gt"].copy()

        # Anchor scheduled sampling: noise the #1-dx anchor (second-to-last position) only in train.
        primary_pos = N - 2
        if self.config.split == "train" and self.config.schedsamp_rate > 0 and case_id in self.case_top2_aidx and self.rng.random() < self.config.schedsamp_rate:
            gt_aidx = int(a_indices[primary_pos])
            alts = self.case_top2_aidx[case_id]
            noisy_aidx = -1
            for a in alts:
                if a != gt_aidx:
                    noisy_aidx = a
                    break
            if noisy_aidx >= 0:
                a_indices[primary_pos] = noisy_aidx

        occ_t = torch.from_numpy(s["occ_indices"]) if "occ_indices" in s else None
        return DAGSample(
            case_id=case_id,
            tile_features=tile_features,
            q_indices=torch.from_numpy(s["q_indices"]),
            a_indices_gt=torch.from_numpy(a_indices),
            attn_mask=attn_mask,
            organ_idx=int(s["organ_idx"]),
            N_collapsed=N,
            occ_indices=occ_t,
        )


def collate_dag_vqa(samples: list[DAGSample]) -> DAGBatch:
    """M1-vqa custom collate - masks loss to target position (index 3) only.

    Differs from collate_dag in dag_dataset.py only by overriding loss_mask for
    anchor positions. Anchor a_indices remain in batch.a_indices_gt so the model
    receives correct anchor A embeddings via teacher forcing.
    """
    batch = collate_dag(samples)
    # Only the target (the last position) contributes to loss; the anchors are input-only.
    # n_anchor = N-1: 3 for the full [organ, procedure, #1-dx, target] batch, 2 for the drop-proc batch.
    n_anchor = samples[0].N_collapsed - 1
    batch.loss_mask[:, :n_anchor] = False
    return batch
