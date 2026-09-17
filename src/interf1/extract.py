"""
Single-WSI tile + H-Optimus feature extraction (REG2026 submission, interf1).

Replicates the training feature pipeline for ONE whole-slide image using
TRIDENT, so the produced [N, 1536] H-Optimus features + coords match exactly what
the downstream heads were trained on.
Segmentation + patching + encoding use:
  --segmenter hest --patch_encoder hoptimus1 --mag 20 --patch_size 224 --overlap 0.

Platform WSI = single-level 20x with stripped/bogus MPP -> we force mpp=0.5 (20x)
via the Processor's per-WSI mpp override CSV.

All models load OFFLINE (caller sets HF_HOME=/opt/ml/model/hf, HF_HUB_OFFLINE=1,
and registers the HEST ckpt in trident's local_ckpts.json).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

# TRIDENT's seg DataLoaders run with their default (multiprocess) workers: the platform provides
# ample /dev/shm (50% of system memory, GB-scale), sufficient for DataLoader worker shared tensors.

TARGET_MAG = 20
PATCH_SIZE = 224
OVERLAP = 0
SEG_CONF = 0.5
PLATFORM_MPP = 0.5  # WSIs are delivered at 20x
COORDS_DIR = f"{TARGET_MAG}x_{PATCH_SIZE}px_{OVERLAP}px_overlap"
MAX_TILES = 8000  # chosen to stay within the per-case budget on multi-GB slides. 
N_READ_THREADS = 8 
FEAT_BATCH = 256
SEG_MAG = 2.5  
GATE_PIXELS = 0  # >0 px -> fast path. i.e. EVERY openslide-READABLE slide uses the bounded grid+Q80 fast path
# (32-41s in-container, NOT subject to the in-container vips+seg slowdown). Validated dx-equivalent to the cached
# trident-seg path: fast-path(Q80) == cached on 20/20 <=0.8GB + 6/6 1-4GB + 4/4 borderline. Only openslide-
# UNREADABLE slides (_W*_H=0, ~5% of test, all <=1.65GP) fall to the vips normal path (+seg@2.5 -> 58-72s, fine).
# WAS 0.8GP: that routed the ~half of test that is <=0.8GP onto the in-container-slow normal path (seg@10 176s)
# = the timeout. Lowering the gate moves them to the fast path. (RAISING the gate is strictly wrong: more normal path.)


def register_hest_ckpt(model_path: Path) -> None:
    """Point trident's seg registry at the bundled HEST ckpt (offline, no download)."""
    import json

    import trident

    seg_json = Path(trident.__file__).parent / "segmentation_models" / "local_ckpts.json"
    try:
        data = json.loads(seg_json.read_text())
        data["hest"] = str(model_path / "trident" / "deeplabv3_seg_v4.ckpt")
        seg_json.write_text(json.dumps(data))
    except Exception as e:  # noqa: BLE001
        print(f"[interf1/extract] WARN could not register HEST ckpt: {e}")


def _patch_deeplabv3_offline() -> None:
    """TRIDENT's HESTSegmenter calls deeplabv3_resnet50(weights=None), which still
    downloads the ImageNet ResNet50 backbone (fails offline). Force weights_backbone=None
    - the full HEST checkpoint overwrites the backbone anyway."""
    import torchvision.models.segmentation as tvseg

    if getattr(tvseg.deeplabv3_resnet50, "_offline_patched", False):
        return
    _orig = tvseg.deeplabv3_resnet50

    def _patched(*a, **k):
        k.setdefault("weights_backbone", None)
        return _orig(*a, **k)

    _patched._offline_patched = True
    tvseg.deeplabv3_resnet50 = _patched


def _patch_hoptimus_offline() -> None:
    """trident loads hoptimus1 via timm `hf-hub:bioptimus/H-optimus-1` (registry entry is
    empty, so it takes the auto-download else-branch). Offline that branch first calls
    ensure_has_internet(), which pings and aborts. The weights ARE bundled in the HF cache
    (HF_HOME/hub), so: (1) tell trident the internet check passed, (2) force huggingface_hub
    offline → timm serves the cached safetensors with no network. This is the SAME load path
    training-time feature extraction used → bit-identical features."""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as _hfc

        _hfc.HF_HUB_OFFLINE = True
    except Exception:  # noqa: BLE001
        pass

    from trident.patch_encoder_models.load import BasePatchEncoder

    BasePatchEncoder._has_internet = True


def _to_pyramid(wsi_path: Path, out_dir: Path) -> Path:
    """vips tiled-pyramid conversion. Platform WSIs are single-level 20x, so any low-magnification
    read (seg) decodes the whole slide and trident's seg/patch/feat each re-read it - tens of seconds
    of fixed I/O. One fast vips pass produces a tiled pyramid whose low-mag levels read instantly.
    """
    out = Path(out_dir) / wsi_path.name
    try:
        subprocess.run(
            ["vips", "tiffsave", str(wsi_path), str(out), "--tile", "--pyramid", "--compression", "jpeg", "--Q", "80"],
            check=True, capture_output=True, timeout=600,
        )
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[interf1/extract] WARN vips pyramid failed ({e}); using original WSI", file=sys.stderr)
        return wsi_path


def _cap_coords_h5(h5_path: Path, max_tiles: int) -> int:
    """Random-subsample tissue coords to <= max_tiles so feat-extraction time is bounded on huge
    slides. ABMIL is attention pooling → a representative subsample preserves the slide embedding.
    Preserves the trident coords attrs (patch_size_level0, magnifications, ...) feat reads."""
    with h5py.File(h5_path, "r") as f:
        coords = np.array(f["coords"])
        attrs = dict(f["coords"].attrs)
    if len(coords) <= max_tiles:
        return len(coords)
    rng = np.random.default_rng(0)  # fixed seed → reproducible
    keep = np.sort(rng.choice(len(coords), size=max_tiles, replace=False))
    coords = coords[keep]
    with h5py.File(h5_path, "w") as f:
        d = f.create_dataset("coords", data=coords)
        for k, v in attrs.items():
            d.attrs[k] = v
    print(f"[interf1/extract] capped tiles to {max_tiles} (slide had more)")
    return len(coords)


def _diversity_sample(coords: np.ndarray, max_tiles: int) -> np.ndarray:
    """Spatially-stratified subsample to <= max_tiles: bucket tiles into a ~sqrt(max) grid over the
    tissue bbox, round-robin across buckets -> even spatial coverage (preserves small foci better than
    uniform random). Returns indices into coords. Deterministic (fixed seed)."""
    n = len(coords)
    if n <= max_tiles:
        return np.arange(n)
    rng = np.random.default_rng(0)
    xs, ys = coords[:, 0].astype(np.float64), coords[:, 1].astype(np.float64)
    g = max(1, int(np.sqrt(max_tiles)))
    bx = np.clip(((xs - xs.min()) / (np.ptp(xs) + 1) * g).astype(int), 0, g - 1)
    by = np.clip(((ys - ys.min()) / (np.ptp(ys) + 1) * g).astype(int), 0, g - 1)
    buckets: dict[int, list[int]] = {}
    for i, k in enumerate(bx * g + by):
        buckets.setdefault(int(k), []).append(i)
    for v in buckets.values():
        rng.shuffle(v)
    order = list(buckets.values())
    keep: list[int] = []
    while len(keep) < max_tiles and any(order):
        for v in order:
            if v:
                keep.append(v.pop())
                if len(keep) >= max_tiles:
                    break
    return np.array(sorted(keep))


def _q80_roundtrip(img):
    """JPEG Q80 round-trip to match the TRAINING pyramid quality. ALL heads were trained on Q80 features
    (pyvips tiffsave compression=jpeg Q=80). The fast path reads the ORIGINAL tiff (native quality != Q80)
    -> per-tile Q80 re-encode aligns it to the training distribution (it recovers the correct primary
    diagnosis on borderline slides). The NORMAL path reads the Q80 pyramid already -> NOT re-encoded
    (avoids a double-Q80 that would re-introduce drift)."""
    import io

    from PIL import Image

    b = io.BytesIO()
    img.save(b, format="JPEG", quality=80)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _threaded_read_and_encode(wsi_path, coords, encoder, device: str, read_px: int, white_filter: bool = False,
                              max_tiles: int | None = None, q80_reencode: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Read tiles in PARALLEL threads (one openslide handle per thread) + H-Optimus forward.
    PARITY-VALIDATED (cos 1.00000 vs trident's feat job), and unlike that job it supports the custom
    fast-path/diversity sampling below (white_filter, max_tiles early-stop, Q80 re-encode). MEASURED on
    the deployment GPU: feat is GPU-bound at ~67 t/s (H-Optimus ViT-g forward), so threaded reads give no
    speedup over trident (67 vs 64 t/s) but enable the gate/diversity path. white_filter drops background
    (mean>=235) tiles for fast-path grid probing; max_tiles early-stops once enough tissue tiles are collected."""
    import openslide
    import torch

    wsi_path = str(wsi_path)
    tf = encoder.eval_transforms
    model = encoder.model.to(device).half().eval()
    _tl = threading.local()

    def _read(xy):
        if not hasattr(_tl, "sl"):
            _tl.sl = openslide.OpenSlide(wsi_path)  # one handle per thread (parallel decode)
        img = _tl.sl.read_region((int(xy[0]), int(xy[1])), 0, (read_px, read_px)).convert("RGB")
        if white_filter and np.asarray(img).mean() >= 235:
            return None
        if q80_reencode:  # fast path reads original (not the Q80 pyramid) -> align to training Q80 quality
            img = _q80_roundtrip(img)
        return tf(img), [int(xy[0]), int(xy[1])]

    kept: list[list[int]] = []
    feats: list[np.ndarray] = []
    buf_img: list = []
    buf_xy: list = []

    def _flush():
        if not buf_img:
            return
        x = torch.stack(buf_img).to(device).half()
        with torch.inference_mode():
            feats.append(model(x).float().cpu().numpy())
        kept.extend(buf_xy)
        buf_img.clear()
        buf_xy.clear()

    with ThreadPoolExecutor(max_workers=N_READ_THREADS) as ex:
        for r in ex.map(_read, coords):
            if r is None:
                continue
            buf_img.append(r[0])
            buf_xy.append(r[1])
            if max_tiles is not None and len(kept) + len(buf_xy) >= max_tiles:
                _flush()
                break
            if len(buf_img) >= FEAT_BATCH:
                _flush()
        _flush()

    dim = feats[0].shape[1] if feats else 1536
    features = np.concatenate(feats, 0).astype(np.float32) if feats else np.zeros((0, dim), np.float32)
    coords_arr = np.array(kept, dtype=np.int64) if kept else np.zeros((0, 2), np.int64)
    return coords_arr, features


def _fast_extract(wsi_path: Path, encoder, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Bounded, seg-free extraction. With GATE_PIXELS=0 this is the DEFAULT path for every openslide-readable
    slide (~95%): it avoids the vips-pyramid + HEST-seg cost that scales with slide area (in-container that
    overran the per-case limit even on mid-size slides, not just monsters). The platform TIFF is single-level
    but TILED (fast random access): grid-sample patch positions at 20x, threaded read + white-filter background,
    stop at MAX_TILES tissue tiles, run the SAME H-Optimus encoder (parity 1.0). Coarser tissue selection than
    HEST (grid+white-filter vs learned seg) but validated dx-equivalent; only openslide-UNREADABLE slides fall
    back to the vips/HEST normal path."""
    import openslide

    sl = openslide.OpenSlide(str(wsi_path))
    W, H = sl.level_dimensions[0]
    step = PATCH_SIZE
    nx, ny = W // step, H // step
    total = max(1, nx * ny)
    target_probes = min(total, MAX_TILES * 2)  # bound probe reads (white-filter early-stops at MAX_TILES)
    stride = max(1, int(np.ceil((total / target_probes) ** 0.5)))
    positions = [(ix * step, iy * step) for iy in range(0, ny, stride) for ix in range(0, nx, stride)]
    np.random.default_rng(0).shuffle(positions)  # so max_tiles early-stop -> spatially-uniform sample (not top-biased)

    coords_arr, features = _threaded_read_and_encode(
        wsi_path, positions, encoder, device, step, white_filter=True, max_tiles=MAX_TILES, q80_reencode=True
    )
    print(f"[interf1/extract] {wsi_path.name}: FAST PATH {len(coords_arr)} tissue tiles (grid+threaded), features {features.shape}")
    return coords_arr, features


_ENCODER = None


def _get_encoder():
    """
    Load the H-Optimus-1 encoder ONCE and cache it. 
    """
    global _ENCODER
    if _ENCODER is None:
        from trident.patch_encoder_models.load import encoder_factory

        _patch_hoptimus_offline()  # serve bundled H-Optimus-1 weights from HF cache, no network
        _ENCODER = encoder_factory("hoptimus1")
    return _ENCODER


def extract_wsi_features(wsi_path: Path, device: str, mpp: float = PLATFORM_MPP, job_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tile a single WSI and extract H-Optimus features. Returns (coords [N,2] int, features [N,1536] f32)."""
    _patch_deeplabv3_offline()
    from trident import Processor
    from trident.segmentation_models.load import segmentation_model_factory

    job_dir = job_dir or tempfile.mkdtemp(prefix="trident_job_")
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    wsi_path = Path(wsi_path)

    import openslide

    try:
        _W, _H = openslide.OpenSlide(str(wsi_path)).level_dimensions[0]
    except Exception:  # noqa: BLE001
        _W = _H = 0
    if _W * _H > GATE_PIXELS:
        print(f"[interf1/extract] {wsi_path.name}: {_W}x{_H} ({_W * _H / 1e9:.1f} GP), openslide-readable -> fast path")
        return _fast_extract(wsi_path, _get_encoder(), device)

    slide_dir = Path(job_dir) / "slide"
    slide_dir.mkdir(parents=True, exist_ok=True)
    wsi = _to_pyramid(wsi_path, slide_dir)

    # per-WSI mpp override CSV (pyramid carries no mpp tags -> force 20x interpretation)
    csv_path = Path(job_dir) / "wsis.csv"
    csv_path.write_text(f"wsi,mpp\n{wsi.name},{mpp}\n")

    proc = Processor(
        job_dir=job_dir,
        wsi_source=str(wsi.parent),
        custom_list_of_wsis=str(csv_path),
        wsi_ext=[wsi.suffix],
        reader_type="openslide",
        skip_errors=False,
        max_workers=1,  # Processor slide-collection ThreadPoolExecutor (one slide per case)
    )

    seg_model = segmentation_model_factory("hest", confidence_thresh=SEG_CONF)
    proc.run_segmentation_job(seg_model, seg_mag=SEG_MAG, holes_are_tissue=True, device=device)
    proc.run_patching_job(target_magnification=TARGET_MAG, patch_size=PATCH_SIZE, overlap=OVERLAP)

    coords_h5 = Path(job_dir) / COORDS_DIR / "patches" / f"{wsi.stem}_patches.h5"
    with h5py.File(coords_h5, "r") as fh:
        coords = np.array(fh["coords"])
        read_px = int(fh["coords"].attrs.get("patch_size_level0", PATCH_SIZE))
    coords = coords[_diversity_sample(coords, MAX_TILES)]

    encoder = _get_encoder()
    coords, features = _threaded_read_and_encode(wsi, coords, encoder, device, read_px, white_filter=False)
    print(f"[interf1/extract] {wsi_path.name}: {len(coords)} tiles, features {features.shape}")
    return coords, features
