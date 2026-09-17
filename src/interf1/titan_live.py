"""Live TITAN VL-LoRA (CE ep4) slide-level organ + #1-dx predictor for the HYBRID OOD path.

All weights load OFFLINE from MODEL_PATH (/opt/ml/model): MODEL_PATH/titan (HF snapshot, trust_remote_code),
MODEL_PATH/titan_vl_lora_ce/lora_ep4.pt (LoRA state).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import openslide
import torch
import torch.nn.functional as F
from PIL import Image

PS = 512          # 512px @ 20x level0 (TITAN/CONCH v1.5 spec)
TILE_CAP = 160  # CONCH-512 tiles for the slide-level pass
TEMPLATES = ["a histopathology slide showing CLASSNAME.", "histopathology image of CLASSNAME.",
             "pathology tissue showing CLASSNAME.", "presence of CLASSNAME tissue on image."]
# The 7 in-distribution challenge organs.
IN_DIST_ORGANS = ["Prostate", "Breast", "Colon", "Stomach", "Urinary bladder", "Lung", "Uterine cervix"]
DISTINCT_OOD_ORGANS = [
    ("Uterine corpus", ["uterine corpus", "endometrium", "myometrium"]),
]
# PROVENANCE: TITAN's pretraining organ set (Ding et al., Nat Med 2025, Fig 1: Mass-340K spans 20 organs)
# Cross reference with the Data Description of REG2026 challenge on Grand Challenge website.
# Dropped TITAN organs: Skin, Head & neck, Pleura, Eye, Thyroid, Kidney
# The official 7 challenge organs, with the one principled subdivision: the organizer's organ list writes 
# "uterine" un-subdivided, but uterine cervix and uterine corpus (endometrium/myometrium) 
# So the only OOD organ is Uterine corpus → 8 organ options total. 
# Broad out-of-scope OOD organs (Liver/Brain/...) are deliberately NOT added.
ORGAN_CLASSES = [
    ("Prostate", ["prostate", "prostate gland"]), ("Breast", ["breast", "breast tissue"]),
    ("Colon", ["colon", "colonic mucosa", "large intestine"]), ("Stomach", ["stomach", "gastric mucosa"]),
    ("Urinary bladder", ["urinary bladder", "bladder urothelium"]), ("Lung", ["lung", "pulmonary tissue"]),
    ("Uterine cervix", ["uterine cervix", "cervix"]),
] + DISTINCT_OOD_ORGANS
# OOD dx (∉ the 139 in-dist vocab) - SCOPED to the 8 organs
# PROVENANCE: Uterine carcinosarcoma = TCGA-UCS; the sarcomas = TCGA-SARC (TITAN-covered, Ding et al. Nat Med 2025)
# benign uterine Leiomyoma/polyp/hyperplasia = WHO Classification of Tumours, 5th edition
OOD_DX = [
    "Leiomyoma", "Endometrial polyp", "Endometrial hyperplasia", "Uterine carcinosarcoma",  # Uterine corpus
    "Schwannoma", "Neurofibroma", "Hemangioma", "Lipoma",
    "Leiomyosarcoma", "Angiosarcoma", "Liposarcoma", "Synovial sarcoma",                # sarcomas in the 8 organs
    "Rhabdomyosarcoma", "Undifferentiated pleomorphic sarcoma",                         # (TCGA-SARC)
]
# The 3 canonical menstrual-cycle phases (standard histology / general pathology knowledge).
CORPUS_BENIGN = ["Proliferative endometrium", "Secretory endometrium", "Atrophic endometrium"]

class TitanLive:
    def __init__(self, model_path: Path, dx_vocab: list[str], device: str = "cuda") -> None:
        import shutil
        from transformers import AutoModel
        self.device = device
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")
        # trust_remote_code on a LOCAL dir copies only the 1st level of relative imports into the
        # dynamic-module cache, but TITAN's modeling has a 2nd level (text_transformer -> conch_tokenizer)
        # that transformers fails to resolve -> FileNotFoundError offline. Pre-seed the module dir with
        # ALL of TITAN's .py so every relative import (any depth) resolves with no network/hub access.
        _tdir = Path(os.environ["HF_MODULES_CACHE"]) / "transformers_modules" / (model_path / "titan").name
        _tdir.mkdir(parents=True, exist_ok=True)
        (_tdir / "__init__.py").touch()
        for _py in (model_path / "titan").glob("*.py"):
            shutil.copy2(_py, _tdir / _py.name)
        from peft import LoraConfig, inject_adapter_in_model
        titan = AutoModel.from_pretrained(str(model_path / "titan"), trust_remote_code=True).to(device).eval()
        for p in titan.parameters():
            p.requires_grad = False
        inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                target_modules=["qkv", "attn.proj"], bias="none"), titan.vision_encoder)
        ck = torch.load(model_path / "titan_vl_lora_ce" / "lora_ep4.pt", map_location=device, weights_only=True)
        sd = titan.vision_encoder.state_dict()
        for k, v in ck["lora_state"].items():
            if k in sd:
                sd[k] = v.to(device)
        titan.vision_encoder.load_state_dict(sd, strict=False)
        self.titan = titan
        # build_conch() (inside return_conch) hardcodes hf_hub_download("MahmoodLab/TITAN",
        # "conch_v1_5_pytorch_model.bin"). The container is offline, so redirect that single fetch to
        # the bundled flat file. build_conch does `from huggingface_hub import hf_hub_download` at call
        # time, so patching the module attribute before return_conch() is sufficient.
        import huggingface_hub as _hf
        _conch_bin = model_path / "titan" / "conch_v1_5_pytorch_model.bin"
        _orig_dl = _hf.hf_hub_download
        def _offline_dl(*a, **k):
            fn = k.get("filename") or (a[1] if len(a) > 1 else "")
            if fn == "conch_v1_5_pytorch_model.bin" and _conch_bin.exists():
                return str(_conch_bin)
            return _orig_dl(*a, **k)
        _hf.hf_hub_download = _offline_dl
        self.conch, self.transform = titan.return_conch()
        self.conch = self.conch.to(device).eval()
        self.organ_names = [n for n, _ in ORGAN_CLASSES]
        self._indist_organs = set(IN_DIST_ORGANS)
        self.n_indist_dx = len(dx_vocab)          # 139; argmax < this = in-dist dx
        self._n_ood_dx = len(OOD_DX)              # next block = OOD dx; the tail = benign discriminators
        self.dxv = list(dx_vocab) + OOD_DX + CORPUS_BENIGN
        self._benign_set = set(CORPUS_BENIGN)
        with torch.no_grad():
            self.oclf = titan.zero_shot_classifier([s for _, s in ORGAN_CLASSES], TEMPLATES, device=device)
            self.dclf = titan.zero_shot_classifier([[d] for d in self.dxv], TEMPLATES, device=device)

    @torch.inference_mode()
    def _conch_feats(self, wsi_path: Path, read_deadline: float | None = None):
        # threaded reads (the deployment GPU read wall): probe up to TILE_CAP*2 candidates in parallel, keep tissue tiles.
        # read_deadline (abs time.time()): stop probing once reached - the WALL-CLOCK guard so TITAN never
        # pushes a large/slow slide past the 300s per-case limit (reads are the variable cost; the CONCH
        # forward is bounded by the collected-tile count). Fewer tiles on a slow slide = lower-res but safe.
        import time as _time
        from concurrent.futures import ThreadPoolExecutor
        try:
            sl = openslide.OpenSlide(str(wsi_path))
        except Exception:  # noqa: BLE001 - striped/unreadable TIFF: SKIP TITAN (return None -> H-Opt fallback).
            return None, None
        W, H = sl.level_dimensions[0]
        cand = [(x, y) for x in range(0, W - PS, PS) for y in range(0, H - PS, PS)]
        np.random.default_rng(0).shuffle(cand)
        cand = cand[: TILE_CAP * 2]  # bound the read budget

        def _read(xy):
            a = np.asarray(sl.read_region((int(xy[0]), int(xy[1])), 0, (PS, PS)).convert("RGB"))
            return (xy, a) if a.mean() < 220 else None  # tissue filter

        tiles, coords = [], []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for i in range(0, len(cand), 64):  # chunked so we can honour read_deadline between chunks
                if read_deadline is not None and _time.time() > read_deadline:
                    break
                for r in ex.map(_read, cand[i:i + 64]):
                    if r is not None:
                        xy, a = r
                        tiles.append(self.transform(Image.fromarray(a)))
                        coords.append(xy)
                if len(tiles) >= TILE_CAP:
                    break
        sl.close()
        if len(tiles) < 16:  # too few tissue tiles for a reliable slide-level call
            return None, None
        tiles = tiles[:TILE_CAP]; coords = coords[:TILE_CAP]
        feats = []
        for i in range(0, len(tiles), 64):
            b = torch.stack(tiles[i:i + 64]).to(self.device, dtype=torch.float32)
            with torch.autocast("cuda", torch.float16):
                feats.append(self.conch(b).float())
        return torch.cat(feats), torch.tensor(coords, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def predict(self, wsi_path: Path, time_budget_s: float | None = None) -> dict | None:
        import time as _time
        _deadline = (_time.time() + 0.55 * time_budget_s) if time_budget_s else None
        feats, coords = self._conch_feats(Path(wsi_path), read_deadline=_deadline)
        if feats is None:
            return None
        with torch.autocast("cuda", torch.float16):
            emb = self.titan.encode_slide_from_patch_features(feats.unsqueeze(0).to(self.device), coords.unsqueeze(0), PS)
        # emb leaves autocast as fp16; proj + oclf/dclf are fp32 -> cast up before the projection matmul.
        e = F.normalize(emb.float() @ self.titan.vision_encoder.proj, dim=-1).squeeze(0)
        organ = self.organ_names[int((e @ self.oclf).argmax())]
        dx_idx = int((e @ self.dclf).argmax())
        dx = self.dxv[dx_idx]
        organ_is_ood = organ not in self._indist_organs
        dx_is_ood = self.n_indist_dx <= dx_idx < self.n_indist_dx + self._n_ood_dx
        if dx in self._benign_set:  # benign cycle-phase discriminators -> in-dist benign convention
            dx = "No tumor present"
        return {"organ": organ, "dx": dx, "is_ood": bool(organ_is_ood or dx_is_ood)}
