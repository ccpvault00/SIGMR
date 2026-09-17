"""DAG-aware multi-turn dataset for Module 1 training.

Pairs three sources per case:
- WSI tile features (HDF5, lazy-loaded)
- CoT trajectory (collapsed - same Q text = one logical node)
- Pre-computed 2N × 2N DAG attention mask (HDF5, from scripts/precompute_dag_masks.py)

Yields DAGSample → collated into DAGBatch for batched multi-turn forward.

Key design:
- A's not in A_pool (e.g., final pathology report free-text) → loss_mask=False
- padded positions in batch → attn_mask=False (no attend / no be attended)
- h5py lazy load + DataLoader num_workers parallelism
- tile_features padding pattern (proven in production)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from reg2026.data.embedding_pools import EmbeddingPools

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DAGDatasetConfig:
    """Frozen config for DAGDataset."""

    cot_json: Path
    dag_masks_h5: Path
    features_dir: Path
    split_json: Path
    split: str = "train"  # "train" or "val"
    organ_to_idx: dict[str, int] | None = None  # if None, no organ aux
    target_organ: str | None = None  # if set, filter to cases with this organ


@dataclass
class DAGSample:
    """One case's training sample (variable-length per-turn sequence)."""

    case_id: str
    tile_features: Tensor  # [N_tiles, 1536]
    q_indices: Tensor  # [N_collapsed] long
    a_indices_gt: Tensor  # [N_collapsed] long; -1 = not in A_pool (skip in loss)
    attn_mask: Tensor  # [2N_collapsed, 2N_collapsed] bool
    organ_idx: int  # -1 if unknown
    N_collapsed: int
    occ_indices: Tensor | None = None  # [N_collapsed] long; per-position occurrence index for occ_emb


@dataclass
class DAGBatch:
    """Batched samples for the multi-turn forward.

    The training collate_fn pads every case to max_N, giving the shapes annotated below. The
    single-case inference builder (build_module1_batch) reuses this container UNPADDED - B=1 and the
    max_N dimensions carry the case's real N (≤ max_N), e.g. attn_mask is [1, 2*N, 2*N] - which the
    forward handles identically.
    """

    case_ids: list[str]
    tile_features: Tensor  # [B, N_max_tiles, 1536]
    tile_pad_mask: Tensor  # [B, N_max_tiles] bool, True = real tile
    q_indices: Tensor  # [B, max_N] long; -1 padding
    a_indices_gt: Tensor  # [B, max_N] long; -1 padding or unknown
    attn_mask: Tensor  # [B, 2*max_N, 2*max_N] bool
    loss_mask: Tensor  # [B, max_N] bool; True for positions to compute A-prediction loss
    organ_idx: Tensor  # [B] long
    N_collapsed: Tensor  # [B] long
    occ_indices: Tensor | None = None  # [B, max_N] long; per-position occurrence index for occ_emb. None = no occ data.


class DAGDataset(Dataset[DAGSample]):
    """REG² 2026 module1 dataset: WSI features + collapsed DAG trajectory + attention mask."""

    def __init__(self, config: DAGDatasetConfig, pools: EmbeddingPools) -> None:
        self.config = config
        self.pools = pools
        self.a_text_to_idx = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}

        # Load CoT JSON
        cot = json.loads(config.cot_json.read_text())
        cot_by_id = {c["id"].replace(".tiff", ""): c for c in cot}

        # Load split
        split_data = json.loads(config.split_json.read_text())
        if config.split not in split_data:
            raise KeyError(f"split={config.split!r} not in split.json keys {list(split_data)}")
        split_ids = {s.replace(".tiff", "") for s in split_data[config.split]}

        # Pre-process: cache (q_indices, a_indices_gt, organ_idx) per case
        # (Mask + tile_features remain on disk; lazy-loaded in __getitem__)
        self.samples: list[dict] = []
        n_skipped_no_mask = 0
        n_skipped_no_features = 0
        n_skipped_unknown_q = 0
        n_a_not_in_pool = 0

        n_skipped_wrong_organ = 0
        with h5py.File(config.dag_masks_h5, "r") as fh_mask:
            for case_id in sorted(split_ids):
                if case_id not in cot_by_id:
                    continue
                # Per-organ filter (if target_organ set)
                if config.target_organ is not None and cot_by_id[case_id].get("organ") != config.target_organ:
                    n_skipped_wrong_organ += 1
                    continue
                if case_id not in fh_mask:
                    n_skipped_no_mask += 1
                    continue
                if not (config.features_dir / f"{case_id}.h5").exists():
                    n_skipped_no_features += 1
                    continue

                # Pull q_order / a_order from mask h5 group
                grp = fh_mask[case_id]
                q_order = [s.decode("utf-8") if isinstance(s, bytes) else s for s in grp["q_order"][:]]
                a_order = [s.decode("utf-8") if isinstance(s, bytes) else s for s in grp["a_order"][:]]

                # Map to indices
                try:
                    q_indices = [pools.q_idx(q) for q in q_order]
                except KeyError as e:
                    n_skipped_unknown_q += 1
                    logger.debug("Skip %s: %s", case_id, e)
                    continue

                a_indices_gt = []
                for a in a_order:
                    idx = self.a_text_to_idx.get(a.strip().lower(), -1)
                    if idx < 0:
                        n_a_not_in_pool += 1
                    a_indices_gt.append(idx)

                organ = cot_by_id[case_id].get("organ", "")
                organ_idx = config.organ_to_idx.get(organ, -1) if config.organ_to_idx is not None else -1

                self.samples.append(
                    {
                        "case_id": case_id,
                        "q_indices": np.array(q_indices, dtype=np.int64),
                        "a_indices_gt": np.array(a_indices_gt, dtype=np.int64),
                        "organ_idx": organ_idx,
                        "N_collapsed": len(q_indices),
                    }
                )

        logger.info(
            "DAGDataset(%s): %d kept · skipped: %d no_mask, %d no_features, %d unknown_q · free-text A: %d / %d total Qs (skip in loss)",
            config.split,
            len(self.samples),
            n_skipped_no_mask,
            n_skipped_no_features,
            n_skipped_unknown_q,
            n_a_not_in_pool,
            sum(s["N_collapsed"] for s in self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DAGSample:
        s = self.samples[idx]
        case_id = s["case_id"]

        # Lazy load tile features (per-case, h5py reopens - OS page cache handles hot pages)
        with h5py.File(self.config.features_dir / f"{case_id}.h5", "r") as fh:
            tile_features = torch.from_numpy(fh["features"][...]).float()

        # Lazy load DAG mask
        with h5py.File(self.config.dag_masks_h5, "r") as fh:
            mask = torch.from_numpy(fh[f"{case_id}/mask"][...]).bool()

        return DAGSample(
            case_id=case_id,
            tile_features=tile_features,
            q_indices=torch.from_numpy(s["q_indices"]),
            a_indices_gt=torch.from_numpy(s["a_indices_gt"]),
            attn_mask=mask,
            organ_idx=int(s["organ_idx"]),
            N_collapsed=int(s["N_collapsed"]),
        )


def collate_dag(samples: list[DAGSample]) -> DAGBatch:
    """Pad variable-length samples into a single batch.

    Padding rules:
    - tile_features: pad to max N_tiles in batch, pad value 0, tile_pad_mask tracks real
    - q_indices / a_indices_gt: pad to max_N with -1 sentinel
    - attn_mask: pad to (2*max_N, 2*max_N) with False (padded positions can't attend / be attended)
    - loss_mask: True only for positions where (a_indices_gt >= 0) AND (position < N_collapsed)
    """
    B = len(samples)
    max_N = max(s.N_collapsed for s in samples)
    max_T_tiles = max(s.tile_features.shape[0] for s in samples)
    feat_dim = samples[0].tile_features.shape[1]

    tile_features = torch.zeros(B, max_T_tiles, feat_dim)
    tile_pad_mask = torch.zeros(B, max_T_tiles, dtype=torch.bool)
    q_indices = torch.full((B, max_N), -1, dtype=torch.long)
    a_indices_gt = torch.full((B, max_N), -1, dtype=torch.long)
    attn_mask = torch.zeros(B, 2 * max_N, 2 * max_N, dtype=torch.bool)
    loss_mask = torch.zeros(B, max_N, dtype=torch.bool)
    organ_idx = torch.tensor([s.organ_idx for s in samples], dtype=torch.long)
    N_collapsed = torch.tensor([s.N_collapsed for s in samples], dtype=torch.long)
    case_ids = [s.case_id for s in samples]
    has_occ = any(s.occ_indices is not None for s in samples)
    occ_indices_t = torch.zeros(B, max_N, dtype=torch.long) if has_occ else None

    for b, s in enumerate(samples):
        N = s.N_collapsed
        N_tiles = s.tile_features.shape[0]
        tile_features[b, :N_tiles] = s.tile_features
        tile_pad_mask[b, :N_tiles] = True
        q_indices[b, :N] = s.q_indices
        a_indices_gt[b, :N] = s.a_indices_gt
        attn_mask[b, : 2 * N, : 2 * N] = s.attn_mask
        loss_mask[b, :N] = s.a_indices_gt >= 0  # skip free-text A's
        if occ_indices_t is not None and s.occ_indices is not None:
            occ_indices_t[b, :N] = s.occ_indices

        # NaN guard for padded sequence positions: set self-loop on diagonal so
        # padded q/a have at least one True column. Without this, an all-False
        # attention row → softmax(-inf, -inf, ...) → NaN, which corrupts the
        # padded position's output (real positions are still safe because the
        # DAG mask blocks them from attending to padded, but PyTorch
        # may still emit NaN warnings or backprop noise).
        for i in range(2 * N, 2 * max_N):
            attn_mask[b, i, i] = True

    return DAGBatch(
        case_ids=case_ids,
        tile_features=tile_features,
        tile_pad_mask=tile_pad_mask,
        q_indices=q_indices,
        a_indices_gt=a_indices_gt,
        attn_mask=attn_mask,
        loss_mask=loss_mask,
        organ_idx=organ_idx,
        N_collapsed=N_collapsed,
        occ_indices=occ_indices_t,
    )
