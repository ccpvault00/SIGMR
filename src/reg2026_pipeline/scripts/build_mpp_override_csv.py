"""Build MPP override CSV for TRIDENT batch feature extraction.

Scans a WSI directory and produces a CSV with one row per WSI, force-setting
MPP for cases where the embedded metadata is broken (mpp=1000 or missing).

Why needed:
    A large fraction of training WSIs have broken or missing MPP metadata.
    TRIDENT raises ValueError on these ("Identified mpp is very low:
    mpp=1000.0"). REG²2026 spec mandates 20× single magnification, so we
    force mpp=0.5 (= 20×) for all broken cases.

CSV format expected by TRIDENT (from trident/Processor.py L160-180):
    wsi,mpp
    <wsi_id>.tiff,0.5018
    <wsi_id>.tiff,0.5
    ...

USAGE:
    python scripts/build_mpp_override_csv.py \\
        --wsi_dir /mnt/data/reg2026/train \\
        --output_csv /mnt/data/reg2026/wsi_mpp_override.csv

    # Dry-run (scan + summary, no CSV written):
    python scripts/build_mpp_override_csv.py \\
        --wsi_dir /mnt/data/reg2026/train \\
        --output_csv /tmp/dummy.csv \\
        --dry_run

EXPECTED OUTPUT (from the MPP survey):
    Kept 0.5 (~20×):     ~4871  (43.4%)
    Kept 0.23 (~40×):     ~882  ( 7.9%)
    Override broken:    ~4477  (39.9%)  ← 1000, 353
    Override missing:    ~981  ( 8.7%)  ← unreadable
    Override extreme:      ~9  ( 0.1%)  ← 0.061, 0.028
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import openslide
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# REG²2026 official spec: WSIs are at 20× single magnification.
# 20× corresponds to MPP ≈ 0.5 microns/pixel.
REG2026_DEFAULT_MPP: float = 0.5

# Categories used in summary stats.
CATEGORIES = (
    "kept_0.5",  # case 1: real MPP ~0.5, no override
    "kept_0.23",  # case 2: real MPP ~0.23 (40×), TRIDENT downsamples
    "override_broken",  # case 3: MPP > 10 (broken metadata like 1000, 353)
    "override_missing",  # case 4: unreadable file or missing MPP property
    "override_extreme",  # case 5: MPP < 0.2 (very high mag, possibly broken)
)


def detect_mpp(wsi_path: Path) -> float | None:
    """Open WSI with openslide and return its MPP-X value (or None on failure).

    Args:
        wsi_path: path to .tiff WSI file.

    Returns:
        MPP value as float if readable, else None.
    """
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception:
        return None

    with slide:
        mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        if mpp is None:
            mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        if mpp is None:
            return None

        try:
            return float(mpp)
        except (ValueError, TypeError):
            return None


def decide_mpp(detected: float | None) -> tuple[float, str]:
    """Apply our 5-case decision logic; return (final_mpp, category).

    Args:
        detected: output of `detect_mpp` - None if unreadable, else float.

    Returns:
        (final_mpp, category):
          final_mpp: value to write into CSV (used by TRIDENT)
          category: one of CATEGORIES (for summary stats)

    Decision tree:
      detected is None         → (0.5,      "override_missing")
      detected > 10            → (0.5,      "override_broken")
      0.4 <= detected <= 0.6   → (detected, "kept_0.5")
      0.2 <= detected <= 0.3   → (detected, "kept_0.23")
      0 < detected < 0.2       → (0.5,      "override_extreme")
      fallback (rare)          → (0.5,      "override_missing")
    """

    if detected is None:
        return REG2026_DEFAULT_MPP, "override_missing"

    if detected > 10:
        return REG2026_DEFAULT_MPP, "override_broken"

    if 0.4 <= detected <= 0.6:
        return detected, "kept_0.5"

    if 0.2 <= detected <= 0.3:
        return detected, "kept_0.23"

    if 0 < detected < 0.2:
        return REG2026_DEFAULT_MPP, "override_extreme"

    return REG2026_DEFAULT_MPP, "override_missing"


def build_csv(wsi_dir: Path, output_csv: Path, dry_run: bool = False) -> Counter:
    """Scan wsi_dir, build per-WSI MPP CSV, return summary counters.

    Args:
        wsi_dir: directory containing .tiff WSIs (no nested search).
        output_csv: where to write the CSV.
        dry_run: if True, scan + count but skip writing CSV.

    Returns:
        Counter mapping category name → count (for printing summary).
    """
    wsi_list = sorted(wsi_dir.glob("*.tiff"))

    rows = []
    counters = Counter()

    for wsi_path in tqdm(wsi_list, desc="Scanning WSIs"):
        detected = detect_mpp(wsi_path)
        final_mpp, category = decide_mpp(detected)
        rows.append({"wsi": wsi_path.name, "mpp": final_mpp})
        counters[category] += 1

    if not dry_run:
        df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"CSV written: {output_csv} ({len(df)} rows)")
    return counters


def print_summary(counters: Counter, total: int) -> None:
    """Pretty-print category counts with percentages.

    Args:
        counters: from build_csv()
        total: total WSIs scanned (for percentage)
    """
    print(f"\n=== MPP override CSV summary ({total} WSIs) ===")
    for cat in CATEGORIES:
        count = counters.get(cat, 0)
        pct = 100.0 * count / total if total else 0.0
        print(f"  {cat:<17}: {count:>5} ({pct:>4.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MPP override CSV for TRIDENT batch feature extraction.")
    parser.add_argument(
        "--wsi_dir",
        type=Path,
        required=True,
        help="Directory containing .tiff WSIs (no nested search).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to write the resulting CSV.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan + print summary but do not write CSV.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.wsi_dir.is_dir():
        raise FileNotFoundError(f"WSI directory not found: {args.wsi_dir}")

    logger.info(f"Scanning WSIs in {args.wsi_dir}")
    counters = build_csv(args.wsi_dir, args.output_csv, dry_run=args.dry_run)

    total = sum(counters.values())
    print_summary(counters, total)

    if args.dry_run:
        logger.info("Dry-run: no CSV written.")
    else:
        logger.info(f"Done. Pass to TRIDENT via --custom_list_of_wsis {args.output_csv}")


if __name__ == "__main__":
    main()
