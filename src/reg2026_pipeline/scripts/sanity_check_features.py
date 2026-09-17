"""Sanity-check H-Optimus features extracted by TRIDENT.

Snapshots the currently-extracted features (h5 files), validates each, and
reports aggregate statistics including per-organ patch count distribution.
Safe to run while TRIDENT is still extracting more features (only reads
the snapshot taken at start, ignores files added after).

USAGE:
    python scripts/sanity_check_features.py
    python scripts/sanity_check_features.py --feat_dir <path> --cot_json <path>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

EXPECTED_FEATURE_DIM = 1536  # H-Optimus-1 output dim


def load_organ_map(cot_path: Path) -> dict[str, str]:
    """Build {wsi_basename_no_ext: organ} from train_CoT.json."""
    with open(cot_path) as f:
        records = json.load(f)
    out: dict[str, str] = {}
    for r in records:
        # id is e.g. "<wsi_id>.tiff" - strip extension
        stem = Path(r["id"]).stem
        out[stem] = r.get("organ", "Unknown")
    return out


def check_one(h5_path: Path) -> dict:
    """Validate one h5; return per-WSI stats dict."""
    out = {
        "name": h5_path.stem,
        "ok": False,
        "n_patches": 0,
        "feat_mean": None,
        "feat_std": None,
        "has_nan": False,
        "has_inf": False,
        "feat_shape": None,
        "coord_shape": None,
        "error": None,
    }
    try:
        with h5py.File(h5_path, "r") as f:
            if "features" not in f or "coords" not in f:
                out["error"] = f"missing dataset (have: {list(f.keys())})"
                return out
            feats = f["features"][...]
            coords = f["coords"][...]
        out["feat_shape"] = tuple(feats.shape)
        out["coord_shape"] = tuple(coords.shape)
        out["n_patches"] = feats.shape[0]
        if feats.ndim != 2 or feats.shape[1] != EXPECTED_FEATURE_DIM:
            out["error"] = f"feature shape wrong: {feats.shape}"
            return out
        if coords.ndim != 2 or coords.shape[0] != feats.shape[0] or coords.shape[1] != 2:
            out["error"] = f"coord shape wrong: {coords.shape} vs feat {feats.shape}"
            return out
        out["has_nan"] = bool(np.isnan(feats).any())
        out["has_inf"] = bool(np.isinf(feats).any())
        out["feat_mean"] = float(feats.mean())
        out["feat_std"] = float(feats.std())
        out["ok"] = not (out["has_nan"] or out["has_inf"])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", type=Path, default=Path("/mnt/data/reg2026/features/full_run/20x_224px_0px_overlap/features_hoptimus1"))
    ap.add_argument("--cot_json", type=Path, default=Path("/mnt/data/reg2026/train_CoT.json"))
    args = ap.parse_args()

    # Snapshot file list NOW so TRIDENT additions during run are ignored
    snapshot = sorted(args.feat_dir.glob("*.h5"))
    print(f"Snapshot: {len(snapshot)} h5 files at {args.feat_dir}")

    organ_map = load_organ_map(args.cot_json)
    print(f"Loaded organ map: {len(organ_map)} WSIs from {args.cot_json}")

    results = []
    bad = []
    for i, p in enumerate(snapshot):
        r = check_one(p)
        if r["ok"]:
            results.append(r)
        else:
            bad.append(r)
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(snapshot)} checked")

    print(f"\n=== SUMMARY ({len(results)} ok, {len(bad)} bad) ===")
    if bad:
        print("\nBAD FILES:")
        for r in bad[:20]:
            print(f"  ✗ {r['name']:30s} {r['error']}")
        if len(bad) > 20:
            print(f"  ... +{len(bad) - 20} more")

    if not results:
        print("No valid files to summarize.")
        return

    n_patches = np.array([r["n_patches"] for r in results])
    means = np.array([r["feat_mean"] for r in results])
    stds = np.array([r["feat_std"] for r in results])

    print("\n── Patch-count distribution (all WSIs) ──")
    print(f"  min/p5/median/p95/max:  {n_patches.min()} / {np.percentile(n_patches, 5):.0f} / {int(np.median(n_patches))} / {np.percentile(n_patches, 95):.0f} / {n_patches.max()}")
    print(f"  mean ± std:             {n_patches.mean():.0f} ± {n_patches.std():.0f}")
    print(f"  total patches:          {n_patches.sum():,}")

    print("\n── Feature value health ──")
    print(f"  mean of per-WSI means:  {means.mean():.4f} (std {means.std():.4f})")
    print(f"  mean of per-WSI stds:   {stds.mean():.4f} (std {stds.std():.4f})")
    n_nan = sum(1 for r in results if r["has_nan"])
    n_inf = sum(1 for r in results if r["has_inf"])
    print(f"  WSIs with NaN: {n_nan}, with Inf: {n_inf}")

    print("\n── Patch counts by organ ──")
    by_organ: dict[str, list[int]] = defaultdict(list)
    by_organ_unknown = 0
    for r in results:
        organ = organ_map.get(r["name"])
        if organ is None:
            by_organ_unknown += 1
            continue
        by_organ[organ].append(r["n_patches"])
    print(f"  {'organ':<14s} {'n_wsi':>6s}  {'min':>5s} {'p25':>6s} {'med':>6s} {'p75':>6s} {'max':>7s}  {'mean':>7s}")
    for organ in sorted(by_organ.keys()):
        arr = np.array(by_organ[organ])
        print(f"  {organ:<14s} {len(arr):>6d}  {arr.min():>5d} {int(np.percentile(arr, 25)):>6d} {int(np.median(arr)):>6d} {int(np.percentile(arr, 75)):>6d} {arr.max():>7d}  {arr.mean():>7.0f}")
    if by_organ_unknown:
        print(f"  (unknown organ: {by_organ_unknown})")

    # Outliers worth flagging
    print("\n── Outliers (≤10 patches; possible tissue mask issue) ──")
    low = sorted([r for r in results if r["n_patches"] <= 10], key=lambda r: r["n_patches"])
    if low:
        for r in low[:20]:
            organ = organ_map.get(r["name"], "?")
            print(f"  ⚠ {r['name']:30s} n={r['n_patches']:>5d}  organ={organ}")
        if len(low) > 20:
            print(f"  ... +{len(low) - 20} more")
    else:
        print("  none ✓")


if __name__ == "__main__":
    main()
