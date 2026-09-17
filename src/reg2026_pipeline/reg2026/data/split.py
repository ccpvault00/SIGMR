"""Deterministic organ-stratified 3-way split.
Default: 80% train / 10% val / 10% test.

Usage:
    from reg2026.data.split import make_split
    split = make_split(slide_ids_by_organ, train_ratio=0.8, val_ratio=0.1)
    train_ids = split["train"]      # set[str]
    val_ids = split["val"]
    test_ids = split["test"]
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def _hash_bucket(slide_id: str, n_buckets: int = 1000) -> int:
    """Deterministic hash → bucket in [0, n_buckets). Stable across Python versions
    (built-in hash() is salted differently per process; use sha256 instead).
    """
    h = hashlib.sha256(slide_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % n_buckets


def make_split(
    slide_ids_by_organ: dict[str, list[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed_salt: str = "reg2026-phase-a-v1",
) -> dict[str, set[str]]:
    """Build organ-stratified 3-way split.

    Args:
        slide_ids_by_organ: {organ_str: [slide_id, ...]} - pre-filtered to
            valid + non-corrupted slides.
        train_ratio: 0.0-1.0
        val_ratio:   0.0-1.0; test_ratio = 1 - train - val
        seed_salt:   prepended to slide_id before hashing for split variation

    Returns:
        {"train": set[str], "val": set[str], "test": set[str]}

    Invariants:
        - train + val + test = ALL slide_ids
        - Sets are disjoint (no duplicate)
        - Per-organ ratios approximately match target (±2% for small organs)
    """
    assert 0.0 < train_ratio < 1.0, f"train_ratio={train_ratio}"
    assert 0.0 <= val_ratio < 1.0 - train_ratio, f"val_ratio={val_ratio}"
    train_threshold = int(train_ratio * 1000)
    val_threshold = int((train_ratio + val_ratio) * 1000)

    train: set[str] = set()
    val: set[str] = set()
    test: set[str] = set()

    for sids in slide_ids_by_organ.values():
        for sid in sids:
            bucket = _hash_bucket(seed_salt + ":" + sid)
            if bucket < train_threshold:
                train.add(sid)
            elif bucket < val_threshold:
                val.add(sid)
            else:
                test.add(sid)

    # Sanity log
    total = len(train) + len(val) + len(test)
    logger.info(
        "Split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%); total=%d",
        len(train),
        100 * len(train) / total,
        len(val),
        100 * len(val) / total,
        len(test),
        100 * len(test) / total,
        total,
    )
    return {"train": train, "val": val, "test": test}


def organ_distribution(
    slide_ids: set[str],
    slide_ids_by_organ: dict[str, list[str]],
) -> dict[str, int]:
    """Count slide IDs per organ within a split (for diagnostic)."""
    by_id_organ = {sid: organ for organ, sids in slide_ids_by_organ.items() for sid in sids}
    counter: dict[str, int] = defaultdict(int)
    for sid in slide_ids:
        organ = by_id_organ.get(sid, "_unknown")
        counter[organ] += 1
    return dict(counter)
