"""precompute per-case DAG attention masks for Module 1 training.

Builds a 2N × 2N boolean attention mask per case where:
- N = number of UNIQUE Q nodes in the chain (collapse: same Q text → same logical node)
- Sequence layout: [q_0, a_0, q_1, a_1, ..., q_{N-1}, a_{N-1}]
- mask[i, j] = True iff position j is visible to position i during self-attention

Rules:
- Collapse - repeated Q in chain encodes fan-out, treated as SAME logical node.
  All instances' next_q union as outgoing edges; A value taken from first instance
  (sanity-checked to be equal across instances).
- Full DAG mask - both q and a positions attend per node ancestor rule
  (standard transformer behavior; a positions get cross-layer refinement).
- Strict DAG - organ and procedure are independent roots; sibling Qs see no
  cross-sibling info (e.g., Gleason pattern 3/4/5 mutually invisible).

Single exception to symmetric mask: q_i is masked OUT from seeing its own a_i
(prevents label leak - a_i is what q_i predicts).

Usage:
    python scripts/precompute_dag_masks.py \\
        --cot-json /mnt/data/reg2026/train_CoT.json \\
        --output /mnt/data/reg2026/checkpoints/m1_answer_space/dag_masks.h5

Output: HDF5 file with two datasets per case_id:
    - "<case_id>/mask": bool [2N, 2N]
    - "<case_id>/q_order": str[N] - Q text in node-order (for dataset to reconstruct sequence)
    - "<case_id>/a_order": str[N] - A text in node-order
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CollapsedNode:
    """One logical DAG node (collapsed across same-Q turns)."""

    q_text: str
    a_text: str  # First A seen for this Q (sanity-checked)
    next_qs: set[str]  # Union of all instances' next_questions
    turn_idxs: list[int]  # Original turn positions where this Q appeared
    occ_idx: int = 0  # occurrence index - when (Q,occ)-collapse, second Q occurrence = new node with occ_idx=1


def collapse_chain(chain: list[dict]) -> tuple[list[CollapsedNode], dict[str, int]]:
    """Collapse same-Q turns to one logical node.

    Order: nodes listed by first appearance in chain. A taken from first instance.
    """
    from reg2026.data.embedding_pools import canonicalize_q

    q_to_node: dict[str, int] = {}
    nodes: list[CollapsedNode] = []
    for i, turn in enumerate(chain):
        q = canonicalize_q(turn["question"])
        a = turn["answer"]
        nq_raw = turn.get("next_question") or ""
        nq = canonicalize_q(nq_raw) if nq_raw else ""
        if q not in q_to_node:
            q_to_node[q] = len(nodes)
            nodes.append(CollapsedNode(q_text=q, a_text=a, next_qs=set(), turn_idxs=[]))
        node = nodes[q_to_node[q]]
        node.turn_idxs.append(i)
        if nq:
            node.next_qs.add(nq)
        # Sanity: warn if A diverges across instances (shouldn't in clean GT)
        if node.a_text != a:
            logger.debug("A divergence for Q=%r: first=%r vs turn %d=%r", q, node.a_text, i, a)
    return nodes, q_to_node


def suffixed_q(base_q: str, occ_idx: int) -> str:
    """Suffix Q text for non-first occurrence (multi-dx ceiling).

    occ_idx=0 → base_q (no suffix)
    occ_idx=N (N>=1) → "{base_q}-{N+1}"
    """
    return base_q if occ_idx == 0 else f"{base_q}-{occ_idx + 1}"


def collapse_chain_occ(chain: list[dict], apply_suffix: bool = False) -> tuple[list[CollapsedNode], dict[tuple[str, int], int]]:
    """Per-occurrence collapse - same Q in non-consecutive turns = different node.

    Rule: a turn EXTENDS the previous turn's node iff (chain[i-1].question == chain[i].question)
    - i.e., consecutive same-Q turns are intra-visit fan-out and collapse together.
    A non-consecutive same-Q turn (intervening different Q) starts a new occurrence node.

    Returns:
        nodes: list of CollapsedNode (each has occ_idx tracking which occurrence)
        q_occ_to_node: dict (q_text, occ_idx) -> node_idx
    """
    from reg2026.data.embedding_pools import canonicalize_q

    q_occ_to_node: dict[tuple[str, int], int] = {}
    nodes: list[CollapsedNode] = []
    q_max_occ: dict[str, int] = {}  # Q text -> current max occ_idx
    prev_q: str | None = None
    turn_to_node: dict[int, int] = {}  # turn idx -> node idx (for next_q resolution)

    for i, turn in enumerate(chain):
        q = canonicalize_q(turn["question"])
        a = turn["answer"]
        nq_raw = turn.get("next_question") or ""
        nq = canonicalize_q(nq_raw) if nq_raw else ""

        if q == prev_q and nodes:
            # Consecutive same Q = intra-visit fan-out, extend last node
            node = nodes[-1]
            node.turn_idxs.append(i)
            if nq:
                node.next_qs.add(nq)
            if node.a_text != a:
                logger.debug("A divergence in consecutive Q=%r at turn %d (prev A=%r, this A=%r)", q, i, node.a_text, a)
            turn_to_node[i] = len(nodes) - 1
        else:
            # New visit (Q changed from previous OR first turn)
            occ_idx = q_max_occ.get(q, -1) + 1
            q_max_occ[q] = occ_idx
            node_idx = len(nodes)
            stored_q = suffixed_q(q, occ_idx) if apply_suffix else q
            nodes.append(CollapsedNode(q_text=stored_q, a_text=a, next_qs={nq} if nq else set(), turn_idxs=[i], occ_idx=occ_idx))
            q_occ_to_node[(q, occ_idx)] = node_idx
            turn_to_node[i] = node_idx

        prev_q = q

    # When apply_suffix: resolve each node's next_qs (currently base text) to suffixed text
    # via chain-order resolution. For each node's emitting turns, find next turn with matching Q.
    if apply_suffix:
        for src_node in nodes:
            new_next_qs: set[str] = set()
            for src_turn in src_node.turn_idxs:
                src_nq_raw = chain[src_turn].get("next_question") or ""
                if not src_nq_raw:
                    continue
                src_nq = canonicalize_q(src_nq_raw)
                # Find next turn j > src_turn with question == src_nq
                for j in range(src_turn + 1, len(chain)):
                    if canonicalize_q(chain[j]["question"]) == src_nq:
                        tgt_node_idx = turn_to_node[j]
                        new_next_qs.add(nodes[tgt_node_idx].q_text)  # suffixed
                        break
                else:
                    # next_q points to nothing - keep base for graceful degradation
                    new_next_qs.add(src_nq)
            src_node.next_qs = new_next_qs

    return nodes, q_occ_to_node


def build_parents_occ(nodes: list[CollapsedNode], chain: list[dict]) -> dict[int, set[int]]:
    """Build parent map for per-occurrence collapsed nodes via chain-order resolution.

    For each turn i with next_question = X, find the next turn j > i with question == X
    (skipping intermediates). Edge: node_of(turn_i) -> node_of(turn_j). Skip self-edges.
    """
    from reg2026.data.embedding_pools import canonicalize_q

    n = len(nodes)
    parents: dict[int, set[int]] = {i: set() for i in range(n)}
    turn_to_node = {ti: ni for ni, node in enumerate(nodes) for ti in node.turn_idxs}
    for i, turn in enumerate(chain):
        nq_raw = turn.get("next_question") or ""
        if not nq_raw:
            continue
        nq = canonicalize_q(nq_raw)
        # Find next turn j > i with question == nq
        for j in range(i + 1, len(chain)):
            if canonicalize_q(chain[j]["question"]) == nq:
                src = turn_to_node[i]
                tgt = turn_to_node[j]
                if src != tgt:
                    parents[tgt].add(src)
                break
    return parents


def build_parents(nodes: list[CollapsedNode], q_to_node: dict[str, int]) -> dict[int, set[int]]:
    """For each collapsed node, find parents via incoming edges.

    Edge: src node has X in its `next_qs` AND X is some other node's Q → src is parent of that other node.
    Self-loops (src.next_qs contains src.q_text) are skipped.
    Orphan next_qs (text not matching any node) are silently dropped.
    """
    n = len(nodes)
    parents: dict[int, set[int]] = {i: set() for i in range(n)}
    for src_idx, src_node in enumerate(nodes):
        for tgt_q in src_node.next_qs:
            if tgt_q not in q_to_node:
                continue
            tgt_idx = q_to_node[tgt_q]
            if tgt_idx != src_idx:
                parents[tgt_idx].add(src_idx)
    return parents


def compute_ancestors(parents: dict[int, set[int]]) -> dict[int, set[int]]:
    """Transitive closure of parents, including self (BFS up parent edges)."""
    n = len(parents)
    ancestors: dict[int, set[int]] = {}
    for i in range(n):
        seen: set[int] = {i}
        frontier: deque[int] = deque(parents[i])
        while frontier:
            p = frontier.popleft()
            if p in seen:
                continue
            seen.add(p)
            frontier.extend(parents[p])
        ancestors[i] = seen
    return ancestors


def build_attention_mask(N: int, ancestors: dict[int, set[int]]) -> np.ndarray:
    """Build 2N × 2N boolean attention mask.

    Sequence: [q_0, a_0, q_1, a_1, ..., q_{N-1}, a_{N-1}]
    mask[query, key] = True iff query can attend to key.

    Rule: for each node i and each ancestor j ∈ ancestors[i] (including self):
        mask[2i,   2j]   = True  # q_i sees q_j
        mask[2i,   2j+1] = True  # q_i sees a_j
        mask[2i+1, 2j]   = True  # a_i sees q_j
        mask[2i+1, 2j+1] = True  # a_i sees a_j

    Single exception: mask[2i, 2i+1] = False (q_i must NOT see its own a_i; a_i
    is what q_i predicts via the MLP head).
    """
    mask = np.zeros((2 * N, 2 * N), dtype=bool)
    for i in range(N):
        for j in ancestors[i]:
            mask[2 * i, 2 * j] = True
            mask[2 * i, 2 * j + 1] = True
            mask[2 * i + 1, 2 * j] = True
            mask[2 * i + 1, 2 * j + 1] = True
    for i in range(N):
        mask[2 * i, 2 * i + 1] = False
    return mask


def process_case(case: dict, per_occ: bool = False, occ_suffix: bool = True) -> tuple[np.ndarray, list[str], list[str], dict[str, float]]:
    """Process one case → (mask, q_order, a_order, stats).

    occ_suffix (per_occ only): when True (default) the non-first occurrence of a Q is
    stored as suffixed text ("Q-2"); these suffixed strings are NOT in Q_VOCAB so the dataset
    drops the whole case (unknown_q). When False (raw test) repeats keep the BASE Q text →
    same q_idx → case is kept → occ1/occ2 differ ONLY in ancestor history (build_parents_occ is
    text-independent, so the ancestor mask is identical either way). This isolates "can the
    Q-conditional cross-attn use history to answer repeated Qs differently?".
    """
    chain = case["chain-of-thought"]
    if per_occ:
        nodes, _ = collapse_chain_occ(chain, apply_suffix=occ_suffix)
        parents = build_parents_occ(nodes, chain)
    else:
        nodes, q_to_node = collapse_chain(chain)
        parents = build_parents(nodes, q_to_node)
    ancestors = compute_ancestors(parents)
    mask = build_attention_mask(len(nodes), ancestors)

    q_order = [n.q_text for n in nodes]
    a_order = [n.a_text for n in nodes]

    stats = {
        "T_raw": len(chain),
        "N_collapsed": len(nodes),
        "n_roots": float(sum(1 for i in range(len(nodes)) if not parents[i])),
        "n_terminals": float(sum(1 for n in nodes if not n.next_qs)),
        "avg_ancestors": float(np.mean([len(ancestors[i]) for i in range(len(nodes))])),
        "max_ancestors": float(max(len(ancestors[i]) for i in range(len(nodes)))),
        "collapse_savings": float(len(chain) - len(nodes)),
    }
    return mask, q_order, a_order, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cot-json", type=Path, required=True, help="Path to train_CoT.json")
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 path")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N cases (0 = all, debug)")
    parser.add_argument("--per-occ-collapse", action="store_true", help="Per-occurrence collapse (non-consecutive same-Q = new node).")
    parser.add_argument(
        "--occ-no-suffix",
        action="store_true",
        help="Raw test (per-occ only): keep BASE Q text for repeated occurrences (no '-2'/'-3' suffix). "
        "Repeats then share the same q_idx so the case is KEPT (suffixed text is unknown_q → whole case dropped). "
        "occ1/occ2 differ only in ancestor history → tests whether history alone can answer repeats differently.",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Build masks WITHOUT typo canonicalization (empties Q_ALIAS_MAP). "
        "Required to reproduce SOTA M1, which trained on non-canonicalized masks "
        "(Prostate procedure orphaned by the missing-'?' typo) + handles the typo "
        "inference-side via Plan C. Canonicalizing training masks regressed WFR.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.no_canonicalize:
        # Empty the alias map so canonicalize_q becomes identity everywhere (it reads this dict).
        from reg2026.data import embedding_pools as _ep

        _ep.Q_ALIAS_MAP.clear()
        logger.info("--no-canonicalize: Q_ALIAS_MAP cleared → masks preserve organizer typos (SOTA-faithful)")

    logger.info("Loading %s", args.cot_json)
    cot = json.loads(args.cot_json.read_text())
    if args.limit > 0:
        cot = cot[: args.limit]
    logger.info("Processing %d cases", len(cot))

    agg_stats: dict[str, list[float]] = defaultdict(list)
    n_written = 0

    with h5py.File(args.output, "w") as fh:
        for case in cot:
            case_id = case["id"]
            try:
                mask, q_order, a_order, stats = process_case(case, per_occ=args.per_occ_collapse, occ_suffix=not args.occ_no_suffix)
            except Exception as e:
                logger.warning("Skip %s: %s", case_id, e)
                continue
            key = case_id.replace(".tiff", "")
            grp = fh.create_group(key)
            grp.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
            grp.create_dataset("q_order", data=np.array(q_order, dtype=h5py.string_dtype()))
            grp.create_dataset("a_order", data=np.array(a_order, dtype=h5py.string_dtype()))
            for k, v in stats.items():
                agg_stats[k].append(v)
            n_written += 1
            if n_written % 1000 == 0:
                logger.info("Processed %d / %d", n_written, len(cot))

    logger.info("Wrote %s (%d cases)", args.output, n_written)
    logger.info("=== Aggregate stats ===")
    for k, vals in agg_stats.items():
        arr = np.array(vals)
        logger.info("  %s: mean=%.2f median=%.2f min=%d max=%d", k, arr.mean(), np.median(arr), arr.min(), arr.max())


if __name__ == "__main__":
    main()
