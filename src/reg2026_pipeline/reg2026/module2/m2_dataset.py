"""Module 2: DAG-aware multilabel next-Q prediction dataset.

For each collapsed node in each case, one training sample:
- Input: tokenized text history of ancestors (BFS order) + current node (Q, A).
- Target: multi-hot [n_q_vocab] of next_qs (from collapsed.next_qs set).

A source mix (50/50):
- Per-ancestor random choice between GT A text and B predicted A text.
- Inference distribution: all ancestor A's are from Module 1 predictions.

Reuses TrajectoryTokenizer from reg2026.module2.
Reuses collapse_chain + build_parents + compute_ancestors from precompute_dag_masks.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from reg2026.data.embedding_pools import EmbeddingPools
from reg2026.module2.trajectory_tokenizer import PAD_ID, TrajectoryTokenizer

# Reuse DAG utilities from preprocessing script
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from precompute_dag_masks import build_parents, build_parents_occ, collapse_chain, collapse_chain_occ, compute_ancestors  # noqa: E402

logger = logging.getLogger(__name__)


def _bfs_simulated_ancestors(nodes, parents, q_to_node):
    """Compute BFS-time ancestor set per node (matches inference BFS distribution).

    For each node, ancestors = set of nodes visited BEFORE this node in BFS order
    (NOT transitive parent closure). Matches what e2e BFS inference sees when
    first visiting this node.

    Returns dict[node_idx → set[node_idx]] (includes self at end).
    """
    n = len(nodes)
    # Build children map
    from collections import defaultdict, deque

    children = defaultdict(set)
    for child in range(n):
        for p in parents.get(child, set()):
            children[p].add(child)
    # Roots = nodes with no parents
    roots = [i for i in range(n) if not parents.get(i)]
    visited = set()
    visited_at_time: dict[int, set[int]] = {}
    queue = deque(sorted(roots))  # deterministic order
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited_at_time[node] = set(visited)  # snapshot BEFORE adding self
        visited.add(node)
        visited_at_time[node].add(node)  # include self (like compute_ancestors does)
        for c in sorted(children[node]):
            if c not in visited:
                queue.append(c)
    # Fill any unreachable nodes with self only (defensive)
    for i in range(n):
        if i not in visited_at_time:
            visited_at_time[i] = {i}
    return visited_at_time


@dataclass(frozen=True)
class M2DatasetConfig:
    cot_json: Path
    split_json: Path
    predictions_json: Path | None  # B predictions cache; None = pure GT
    split: str = "train"
    a_mix_ratio: float = 0.5  # per-ancestor prob of using predicted A instead of GT
    bfs_ancestor: bool = False  # if True, use BFS-simulated first-visit ancestor set (matches inference) instead of static transitive closure
    bfs_edge_drop_prob: float = 0.0  # for bfs_ancestor: per-sample random edge dropout to simulate inference noise (schedule sampling for DAG structure)
    per_occ_collapse: bool = False  # per-occurrence collapse - non-consecutive same-Q turns = new logical node (cleaner multi-dx training)


@dataclass
class M2Sample:
    case_id: str
    node_idx: int
    context_tokens: list[int]  # tokenized [BOS, Q_0, A_0, ..., Q_X, A_X]
    target_next_qs: list[int]  # list of q_indices that are next_qs of this node


@dataclass
class M2Batch:
    case_ids: list[str]
    tokens: Tensor  # [B, max_L]
    attention_mask: Tensor  # [B, max_L] bool, True = valid
    targets: Tensor  # [B, n_q_vocab] float, multi-hot


class M2Dataset(Dataset[M2Sample]):
    """Module 2 dataset. One training sample per (case, collapsed node)."""

    def __init__(self, config: M2DatasetConfig, pools: EmbeddingPools, tokenizer: TrajectoryTokenizer) -> None:
        self.config = config
        self.pools = pools
        self.tokenizer = tokenizer

        # Load CoT JSON
        cot = json.loads(config.cot_json.read_text())
        cot_by_id = {c["id"].replace(".tiff", ""): c for c in cot}

        split_data = json.loads(config.split_json.read_text())
        if config.split not in split_data:
            raise KeyError(f"split={config.split!r} not in split.json")
        split_ids = {s.replace(".tiff", "") for s in split_data[config.split]}

        # Load B predictions if provided
        self.predictions: dict[str, list[int]] = {}
        if config.predictions_json is not None and config.predictions_json.exists():
            self.predictions = json.loads(config.predictions_json.read_text())
            logger.info("Loaded %d cached predictions from %s", len(self.predictions), config.predictions_json)

        # Pre-enumerate (case_id, node_idx) pairs and compute per-case structure
        # Stored: case_id -> {"q_order": [q_idx], "gt_a_order": [a_idx or -1],
        #                     "pred_a_order": [a_idx] (optional),
        #                     "ancestors_per_node": [set of ancestor node indices],
        #                     "next_qs_per_node": [set of q_idx]}
        self.case_data: dict[str, dict] = {}
        self.samples: list[tuple[str, int]] = []  # (case_id, node_idx)

        n_skipped = 0
        n_unknown_q = 0

        for case_id in sorted(split_ids):
            if case_id not in cot_by_id:
                n_skipped += 1
                continue
            chain = cot_by_id[case_id]["chain-of-thought"]
            try:
                if config.per_occ_collapse:
                    # apply_suffix=False (was True) - suffixed Q text ("invasion?-2") is not in
                    # Q_VOCAB so q_idx() below KeyError'd → the WHOLE repeat case was dropped (n_unknown_q),
                    # making the per-occ M2 degenerate (never saw a repeat). Base Q text keeps the
                    # case; occ1/occ2 share q_idx and the repeat is expressed as a re-ask edge from the
                    # occ2-precursor state (chain-order resolution below), routed at inference via
                    # --bfs-allow-revisit. Matches the M1-raw (dag_masks_occ_nosuffix) representation.
                    nodes, _q_occ_to_node = collapse_chain_occ(chain, apply_suffix=False)
                    parents = build_parents_occ(nodes, chain)
                else:
                    nodes, q_to_node = collapse_chain(chain)
                    parents = build_parents(nodes, q_to_node)
                # If bfs_ancestor with dropout, defer to __getitem__ (per-sample random). Else precompute.
                if config.bfs_ancestor and config.bfs_edge_drop_prob > 0:
                    ancestors = None  # placeholder; will compute per __getitem__ call
                elif config.bfs_ancestor:
                    ancestors = _bfs_simulated_ancestors(nodes, parents, None)
                else:
                    ancestors = compute_ancestors(parents)
            except Exception:
                n_skipped += 1
                continue

            # Convert Q text to q_idx; A text to a_idx via exact A_VOCAB match
            a_text_to_idx = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}
            try:
                q_indices = [pools.q_idx(n.q_text) for n in nodes]
            except KeyError:
                n_unknown_q += 1
                continue
            a_indices_gt = [a_text_to_idx.get(n.a_text.strip().lower(), -1) for n in nodes]

            # next_qs per node as q_idx set.
            # Per-occ mode: resolve each next_q to the OCC-indexed target node (next turn after src with matching Q).
            # Default: text-based resolution against q_to_node (single occurrence).
            next_qs_per_node: list[set[int]] = []
            if config.per_occ_collapse:
                # Build src_node -> {tgt_node_idx} via chain-order resolution (consistent with build_parents_occ).
                from reg2026.data.embedding_pools import canonicalize_q

                turn_to_node = {ti: ni for ni, n in enumerate(nodes) for ti in n.turn_idxs}
                edges_per_src: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
                for ti, turn in enumerate(chain):
                    nq_raw = turn.get("next_question") or ""
                    if not nq_raw:
                        continue
                    nq_can = canonicalize_q(nq_raw)
                    for tj in range(ti + 1, len(chain)):
                        if canonicalize_q(chain[tj]["question"]) == nq_can:
                            src_n = turn_to_node[ti]
                            tgt_n = turn_to_node[tj]
                            if src_n != tgt_n:
                                edges_per_src[src_n].add(tgt_n)
                            break
                for i in range(len(nodes)):
                    target = set()
                    for tgt_n in edges_per_src[i]:
                        target.add(pools.q_idx(nodes[tgt_n].q_text))
                    next_qs_per_node.append(target)
            else:
                for n in nodes:
                    target = set()
                    for nq_text in n.next_qs:
                        if nq_text in q_to_node:
                            nq_idx = pools.q_idx(nodes[q_to_node[nq_text]].q_text)
                            target.add(nq_idx)
                    next_qs_per_node.append(target)

            # Predicted a from cache (optional)
            pred_a = self.predictions.get(case_id, [])

            # Capture organ (for per-organ oversample). Top-level field or chain.
            organ = (cot_by_id[case_id].get("organ") or "").lower()
            if not organ:
                organ = next((t["answer"].strip().lower() for t in chain if t["question"] == "What is the organ?"), "")

            self.case_data[case_id] = {
                "q_idx": q_indices,
                "gt_a_idx": a_indices_gt,
                "pred_a_idx": pred_a,
                "ancestors": [sorted(ancestors[i]) for i in range(len(nodes))] if ancestors is not None else None,
                "next_qs": next_qs_per_node,
                "parents": parents,  # kept for runtime BFS-sim with dropout
                "nodes_len": len(nodes),
                "organ": organ,
            }
            # Add one sample per node
            for node_idx in range(len(nodes)):
                self.samples.append((case_id, node_idx))

        logger.info(
            "M2Dataset(%s): %d cases (%d skipped) · %d total samples · %d unknown_q skips",
            config.split,
            len(self.case_data),
            n_skipped,
            len(self.samples),
            n_unknown_q,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def get_organ_mask(self, organ_name: str) -> list[bool]:
        """Per-sample bool: True if case's organ matches `organ_name` (case-insensitive).

        Used by WeightedRandomSampler for per-organ oversampling.
        """
        target = organ_name.strip().lower()
        return [self.case_data[cid]["organ"] == target for cid, _ in self.samples]

    def get_rare_qa_weights(self, scale: float = 1.0, cap: float = 32.0) -> list[float]:
        """Per-sample weight by rarity of (organ, q_idx, a_idx) tuple in train set.

        Why: M2 routing on rare (Q, A) pairs (e.g., (breast, additional_finding, Yes)
        in multi-dx context) is undersampled vs single-dx patterns. Boost samples
        proportional to inverse log frequency.

        weight = clip(scale * log(max_freq / (freq + 1)) + 1, 1.0, cap)

        Returns per-sample list (one weight per (case_id, node_idx) tuple).
        """
        import math

        # Build (organ, q_idx, a_idx) -> count map across all samples
        freq: dict[tuple[str, int, int], int] = {}
        for cid, ni in self.samples:
            cd = self.case_data[cid]
            q_idx = cd["q_idx"][ni]
            a_idx = cd["gt_a_idx"][ni]
            organ = cd["organ"]
            if a_idx < 0:
                continue
            key = (organ, q_idx, a_idx)
            freq[key] = freq.get(key, 0) + 1
        max_freq = max(freq.values()) if freq else 1
        weights: list[float] = []
        for cid, ni in self.samples:
            cd = self.case_data[cid]
            q_idx = cd["q_idx"][ni]
            a_idx = cd["gt_a_idx"][ni]
            organ = cd["organ"]
            if a_idx < 0:
                weights.append(1.0)
                continue
            f = freq.get((organ, q_idx, a_idx), 1)
            w = scale * math.log(max_freq / (f + 1)) + 1.0
            weights.append(max(1.0, min(cap, w)))
        return weights

    def get_multi_dx_mask(self) -> list[bool]:
        """Per-sample boolean: True if case has dx_count >= 2 (multi-dx).

        Used by WeightedRandomSampler for multi-dx oversampling (Bug 2 fix).
        Multi-dx cases ~6% of train; M2 trained on them undersamples → poor multi-dx
        routing (additional_finding=Yes → proliferative re-exam branch).
        """
        dx_count_qi = self.pools.q_idx("What is the number of diagnoses to includes?")
        per_case_is_multi: dict[str, bool] = {}
        for case_id, cd in self.case_data.items():
            multi = False
            for q, a in zip(cd["q_idx"], cd["gt_a_idx"], strict=True):
                if q == dx_count_qi and a >= 0:
                    try:
                        n = int(self.pools.A_VOCAB[a].strip())
                        if n >= 2:
                            multi = True
                    except (ValueError, IndexError):
                        pass
                    break
            per_case_is_multi[case_id] = multi
        return [per_case_is_multi.get(cid, False) for cid, _ in self.samples]

    def __getitem__(self, idx: int) -> M2Sample:
        case_id, node_idx = self.samples[idx]
        cd = self.case_data[case_id]
        # Per-sample BFS sim with edge dropout (schedule sampling for DAG structure)
        if self.config.bfs_ancestor and self.config.bfs_edge_drop_prob > 0:
            n = cd["nodes_len"]
            perturbed_parents: dict[int, set[int]] = {}
            for child, par_set in cd["parents"].items():
                if not par_set:
                    perturbed_parents[child] = set()
                    continue
                kept = {p for p in par_set if random.random() > self.config.bfs_edge_drop_prob}
                # If all dropped and node had parents, keep one random (else node becomes orphan unreachable)
                if not kept:
                    kept = {next(iter(par_set))}
                perturbed_parents[child] = kept
            for i in range(n):
                perturbed_parents.setdefault(i, set())
            anc_dict = _bfs_simulated_ancestors([None] * n, perturbed_parents, None)
            ancestor_positions = sorted(anc_dict[node_idx])
        else:
            ancestor_positions = cd["ancestors"][node_idx]  # precomputed

        # Build (q_text, a_text) sequence
        history: list[tuple[str, str]] = []
        for pos in ancestor_positions:
            q_text = self.pools.Q_VOCAB[cd["q_idx"][pos]]
            # 50/50 mix: per-ancestor random
            use_pred = self.config.a_mix_ratio > 0 and pos < len(cd["pred_a_idx"]) and random.random() < self.config.a_mix_ratio
            a_idx = cd["pred_a_idx"][pos] if use_pred else cd["gt_a_idx"][pos]
            a_text = self.pools.A_VOCAB[a_idx] if a_idx >= 0 else ""  # "" for free-text A
            history.append((q_text, a_text))

        # Encode tokens (BOS + each (Q, A) pair as 2 tokens)
        tokens = self.tokenizer.encode_trajectory(history, add_bos=True, add_eos=False)

        # Targets: multi-hot of next_qs
        targets = list(cd["next_qs"][node_idx])

        return M2Sample(
            case_id=case_id,
            node_idx=node_idx,
            context_tokens=tokens,
            target_next_qs=targets,
        )


def collate_m2(samples: list[M2Sample], n_q_vocab: int) -> M2Batch:
    """Pad token sequences + build multi-hot targets."""
    B = len(samples)
    max_L = max(len(s.context_tokens) for s in samples)
    tokens = torch.full((B, max_L), PAD_ID, dtype=torch.long)
    attn = torch.zeros(B, max_L, dtype=torch.bool)
    targets = torch.zeros(B, n_q_vocab, dtype=torch.float)
    for b, s in enumerate(samples):
        L = len(s.context_tokens)
        tokens[b, :L] = torch.tensor(s.context_tokens, dtype=torch.long)
        attn[b, :L] = True
        for q_idx in s.target_next_qs:
            targets[b, q_idx] = 1.0
    return M2Batch(
        case_ids=[s.case_id for s in samples],
        tokens=tokens,
        attention_mask=attn,
        targets=targets,
    )
