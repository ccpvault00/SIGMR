"""Regenerate / verify the deterministic organ-stratified train/val/test split.

The deployed ``model/configs/split.json`` was produced by
``reg2026.data.split.make_split`` (SHA256-bucketed, organ-stratified, salt
``reg2026-phase-a-v1``) over the training slides. make_split is per-slide
deterministic - a slide's bucket depends only on its id + the salt - so the
split is fully reproducible and provably not hand-picked.

This script:
  * ``--verify <split.json>``  - assert every id in a shipped split lands in its
    make_split bucket (the audit proof: deterministic + organ-stratified, no
    cherry-picking). Prints the per-organ distribution.
  * ``--out <split.json>``     - regenerate the split from train_CoT.json.

Note on the shipped split's id set: split.json froze the 11 192 slides available
at generation time (28 of the 11 220 train_CoT slides had no extracted features
yet and were skipped). The id SET is therefore frozen in split.json; this script
reproduces the make_split ASSIGNMENT - which bucket each id belongs to - which is
exactly what "deterministic split" means. A plain ``--out`` regenerates over all
train_CoT slides (a superset of the shipped ids, same buckets).

Usage:
    python scripts/make_split.py --cot-json train_CoT.json --verify model/configs/split.json
    python scripts/make_split.py --cot-json train_CoT.json --out /tmp/split.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from reg2026.data.split import make_split, organ_distribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _strip(sid: str) -> str:
    return sid.replace(".tiff", "").replace(".tif", "")


def _organs_by_id(cot_path: Path) -> dict[str, list[str]]:
    """Group every train_CoT slide id under its organ. Ids are extension-stripped to
    match the form make_split originally hashed (the CoTDataset stripped the '.tiff')."""
    cot = json.loads(cot_path.read_text())
    by_organ: dict[str, list[str]] = {}
    for c in cot:
        by_organ.setdefault(c["organ"], []).append(_strip(c["id"]))
    return by_organ


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate / verify the deterministic organ-stratified split.")
    ap.add_argument("--cot-json", type=Path, default=Path("/mnt/data/reg2026/train_CoT.json"))
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed-salt", type=str, default="reg2026-phase-a-v1", help="must match the shipped split's salt")
    ap.add_argument("--out", type=Path, default=None, help="write the regenerated split (over all train_CoT slides) here")
    ap.add_argument("--verify", type=Path, default=None, help="assert every id in this split.json lands in its make_split bucket")
    args = ap.parse_args()

    by_organ = _organs_by_id(args.cot_json)
    split = make_split(by_organ, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed_salt=args.seed_salt)

    if args.out is not None:
        args.out.write_text(json.dumps({k: sorted(v) for k, v in split.items()}, indent=2))
        log.info("Regenerated split → %s (%s)", args.out, {k: len(v) for k, v in split.items()})

    if args.verify is not None:
        shipped = json.loads(args.verify.read_text())
        # make_split assignment for each id (strip ext for a stable id key both sides).
        assigned = {_strip(sid): name for name, ids in split.items() for sid in ids}
        mism = [(name, sid) for name, ids in shipped.items() for sid in ids if assigned.get(_strip(sid)) != name]
        n = sum(len(v) for v in shipped.values())
        if mism:
            log.error("%d / %d shipped ids are NOT in their make_split bucket (split is NOT reproducible!):", len(mism), n)
            for name, sid in mism[:20]:
                log.error("  %s claimed %s, make_split → %s", sid, name, assigned.get(_strip(sid)))
            sys.exit(1)
        log.info("VERIFIED: all %d shipped ids land in their make_split bucket (deterministic, organ-stratified).", n)
        for name, ids in shipped.items():
            dist = organ_distribution({_strip(s) for s in ids}, {o: [_strip(s) for s in v] for o, v in by_organ.items()})
            log.info("  %-5s n=%d  per-organ=%s", name, len(ids), sorted(dist.items()))


if __name__ == "__main__":
    main()
