"""
Interface 1 - Chain of thought reasoning.

Runs the full pipeline on ONE raw WSI:
  Step 1  tile + H-Optimus feature extraction (extract.py, TRIDENT, mpp=0.5 for 20x)
  Step 2  setup_inference() loads M1 / M2 / cfg from MODEL_PATH;
          predict_case() runs the BFS trajectory on the live features
  Step 3  build_report() (literal-\n) fills the final pathology report step
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from core import MODEL_PATH

# --- Offline model resolution (before torch/trident/transformers import) ---
os.environ.setdefault("HF_HOME", str(MODEL_PATH / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# TITAN loads via trust_remote_code -> transformers copies the modeling .py into HF_MODULES_CACHE
# and imports from there. The platform mounts MODEL_PATH read-only, so this MUST point at a
# writable path (/tmp is the only guaranteed-writable mount). Set before any transformers import.
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")
os.environ.setdefault("TRIDENT_HOME", str(MODEL_PATH / "trident"))
os.environ.setdefault("REG_POOL_DIR", str(MODEL_PATH / "pools"))

_PIPE = Path(__file__).resolve().parent.parent / "reg2026_pipeline"
sys.path.insert(0, str(_PIPE))            # the reg2026 package (reg2026_pipeline/reg2026)
sys.path.insert(0, str(_PIPE / "scripts"))

CKPTS = MODEL_PATH / "ckpts"
CONFIGS = MODEL_PATH / "configs"

_CTX = None  # (m1, m2, tok, max_sequence_length, pools, cfg) - loaded once per container
_TITAN = None  # TITAN VL-LoRA hybrid OOD predictor (loaded once); None = H-Opt-only fallback
TITAN_GATE_ENTROPY = 0.05  # HYBRID gate: run TITAN only when the ABMIL softmax entropy > this computed gate 
# WALL-CLOCK guard (replaces the fixed elapsed budget). TITAN gets whatever time is left under the per-case limit
HARD_LIMIT_S = 300.0     
TITAN_LOAD_S = 40.0     
TITAN_RESERVE_S = 30.0  # relaxed 90->30: fp16 extract + predict self-limits via wall-clock -> run TITAN up to elapsed<185s (finish <270s, 30s margin)
TITAN_MIN_S = 45.0    

# TITAN supplies only the COARSE/base #1-dx. So strip TITAN's grade suffix and let the grade/detail come from 
# M1-vqa via build_report's _format_dx (the separate grade Qs: Gleason grade-group, Nottingham Tubule/Nuclear/Mitoses,
# dysplasia grade - all M1-vqa-answered).
_GRADE_SUFFIX = re.compile(
    r",?\s*(grade\s+\S+|(?:well|moderately|poorly|un)\w*[\s-]*differentiated|(?:high|low|intermediate)[- ]grade)\s*$",
    re.I,
)


def _titan_base_dx(dx: str) -> str:
    """Strip the (weak) graded suffix from TITAN's #1-dx → base; grade comes from M1-vqa downstream."""
    prev = None
    while prev != dx:
        prev = dx
        dx = _GRADE_SUFFIX.sub("", dx).strip()
    return dx


class ChainOfThoughtStep(TypedDict):
    question: str
    answer: str
    next_question: str


def _inference_argv() -> list[str]:
    """Inference config for the deployed Workflow-Reasoning pipeline: the primary single-label
    ABMIL for the #1 diagnosis, a closed-set answerer (M1-cls) and an autoregressive VQA answerer
    (M1-vqa) for the trajectory answers, the next-question generator (M2), the co-finding tile
    heads, and the in-situ co-diagnosis subtree. Deterministic.
    See protocols/PROTOCOLS.md for the per-organ reporting SOP."""
    d = str(CKPTS)
    # argparse is order-independent for optional flags; grouped here by role for readability.
    return [
        "interf1",

        # --- Core: variant, primary #1 diagnosis, determinism ---
        "--variant", "b-qcond",  # answerer architecture: the query-conditioned B variant
        "--abmil-primary-ckpt", f"{d}/abmil_primary_dx.ckpt",  # #1 diagnosis: dedicated single-label ABMIL 
        "--deterministic",  # disable cuDNN / cuBLAS non-determinism so the trajectory + report are reproducible

        # --- M1 answerers: closed-set (M1-cls) + autoregressive VQA (M1-vqa, shared with Visual Grounding) ---
        "--module1-ckpt", f"{d}/m1_cls/epoch_024.ckpt",  # M1-cls: closed-set classification head for the structured questions
        "--m1cls-answer-space", f"{d}/m1_answer_space/per_q_answer_space.json",
        "--m1-vqa-ckpt", f"{d}/m1_vqa_2anchor/best.ckpt",  # M1-vqa: 2-anchor [organ, #1-dx] autoregressive answerer
        "--m1-vqa-drop-proc-anchor",  # drop the procedure anchor (not ROI-visible; adds noise)

        # --- M2 next-question generator + trajectory traversal ---
        "--module2-ckpt", f"{d}/module2_next_q/best.ckpt",  # M2: predicts the next clinically-appropriate question(s)
        "--m2-gate-context",  # feed M2 the diagnostic gate answers (abnormality/neoplasm/behavior) so it learns sibling-conditioned edges
        "--normalize-glitches",  # build the fan-out cap on glitch-normalized chains (matches how M2 was trained; avoids capping away learned edges)
        "--m2-top-k-from-train",  # cap M2's next-question fan-out per question to the max seen in training
        "--bfs-allow-revisit",  # let the trajectory re-ask a question under a new parent (multi-diagnosis workup)

        # --- Co-finding / co-diagnosis: secondary + multi-lesion findings ---
        "--cofinding-heads-json", f"{CONFIGS}/cofinding_heads.json",  # visual tile-head gates: DCIS, CIS, microcalcification, granulomatous inflammation, organizing fibrosis
        "--codx-independent-subtree",  # in-situ co-diagnosis subtree (DCIS / CIS) - see protocols.cofindings

        # --- Additional routing / answer guards / report derivations ---
        "--routing-config", f"{CONFIGS}/q_routing.json",  # per-question routing table (which answerer handles each question)
        "--dx-subspace-guard",  # restrict the #k-diagnosis answer to the diagnosis vocabulary
        "--gleason-from-patterns",  # prostate: derive the Gleason score / grade group from the predominant + secondary patterns

        # --- Data / I/O paths ---
        "--split-json", f"{CONFIGS}/split.json",
        "--cot-json", f"{CONFIGS}/train_CoT.json",
        "--features-dir", "/tmp",  # unused: we pass the temp h5 path directly to predict_case
        "--output", "/tmp/interf1_inspect.md",
    ]


def _get_ctx():
    """Load all models + cfg once (heavy); reused if the container runs >1 inference."""
    global _CTX
    if _CTX is None:
        import reasoning_engine as engine

        # register bundled HEST ckpt for trident (offline)
        from .extract import register_hest_ckpt

        register_hest_ckpt(MODEL_PATH)

        argv_bak = sys.argv
        try:
            sys.argv = _inference_argv()
            args = engine.parse_args()
        finally:
            sys.argv = argv_bak
        m1, m2, tok, msl, pools, cfg, _, _ = engine.setup_inference(args)
        dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        # region co-dx integrator (live #2 co-diagnosis). Optional: if its checkpoints are absent
        # or loading fails, codx_integ=None and the engine falls back to the base co-finding heads.
        codx_integ = None
        try:
            from .codx_integrator import load_codx_integrator

            codx_integ = load_codx_integrator(CKPTS, dev)
            print("[interf1] region co-dx integrator loaded")
        except Exception:  # noqa: BLE001
            import traceback

            print("[interf1] codx integrator unavailable - base co-finding only:")
            traceback.print_exc()
        # HYBRID OOD path (TITAN VL-LoRA) is LAZY-loaded via _get_titan() - only on the first uncertain
        # case that still has time budget, so confident cases never pay the ~15-30s TITAN+CONCH load.
        _CTX = (m1, m2, tok, msl, pools, cfg, codx_integ)
        print("[interf1] inference pipeline loaded")
    return _CTX


def _get_titan():
    """Lazy-load the TITAN VL-LoRA OOD predictor ONCE. Returns it, or None if absent/load-failed
    (→ H-Opt-only). Called only from inside the time-budget+entropy gate so the ~15-30s TITAN+CONCH
    load is paid at most once, and never on confident or already-slow cases."""
    global _TITAN
    if _TITAN is None:  # not yet attempted
        try:
            import torch as _t
            from .titan_live import TitanLive
            _dxv = _t.load(CKPTS / "abmil_primary_dx.ckpt", map_location="cpu", weights_only=False)["dx_vocab"]
            _TITAN = TitanLive(CKPTS, dx_vocab=_dxv, device="cuda" if _t.cuda.is_available() else "cpu")
            print("[interf1] TITAN VL-LoRA hybrid OOD path loaded (lazy)")
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("[interf1] TITAN unavailable - H-Opt-only (no OOD path)")
            _TITAN = False  # sentinel: attempted + failed, do not retry
    return _TITAN or None


# Minimal schema-valid fallback so a single bad case scores low instead of crashing the
# whole evaluation ("algorithm failed on one or more cases" zeroes the entire submission).
_FALLBACK_COT: list[ChainOfThoughtStep] = [
    {"question": "What is the organ?", "answer": "", "next_question": "What is the final pathology report?"},
    {"question": "What is the final pathology report?", "answer": "", "next_question": ""},
]


def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    """Workflow Reasoning on one WSI. Never raises: on any per-case failure, log the full
    traceback to stderr and return a valid fallback CoT so the rest of the submission still
    scores. (Model-load happens inside; it was validated, so a failure here
    is a per-slide problem - e.g. 0-tissue / unreadable WSI - not a config error.)"""
    try:
        return _predict_chain_of_thought_impl(wsi_path=wsi_path)
    except Exception:  # noqa: BLE001
        import sys
        import traceback

        print(f"[interf1] ERROR on {Path(wsi_path).name} - returning fallback CoT:", file=sys.stderr)
        traceback.print_exc()
        return [dict(s) for s in _FALLBACK_COT]


def _predict_chain_of_thought_impl(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    """Run Workflow Reasoning on one WSI; return the predicted chain-of-thought."""
    import time
    import h5py
    import torch

    _t0 = time.time()  # per-case timer for the TITAN time-budget guard (300s platform limit)

    import reasoning_engine as engine
    from build_final_report import build_report
    from .extract import extract_wsi_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()  # defensive: no-op in a per-case container, frees reserve if persistent
    m1, m2, tok, msl, pools, cfg, codx_integ = _get_ctx()

    # G1: live tile + H-Optimus features -> temp h5 (predict_case reads features_path)
    _job = tempfile.mkdtemp(prefix="reg_job_")
    coords, feats = extract_wsi_features(Path(wsi_path), device, mpp=0.5, job_dir=_job)
    # SHARE the one vips pass: for a striped/openslide-unreadable slide, extract already converted it to a
    # tiled pyramid at <job>/slide/<name>. Hand THAT to TITAN so CONCH reads it directly - no 2nd vips (the
    # double conversion was the budget overrun) AND striped OOD slides still get a TITAN pass. Readable slides
    # make no pyramid -> TITAN reads the original WSI.
    _pyr = Path(_job) / "slide" / Path(wsi_path).name
    _titan_src = _pyr if _pyr.exists() else Path(wsi_path)
    tmp_h5 = Path(tempfile.mkdtemp()) / "feat.h5"
    with h5py.File(tmp_h5, "w") as f:
        f.create_dataset("features", data=feats)
        f.create_dataset("coords", data=coords)

    # #1 diagnosis from the single-label ABMIL primary. The region integrator computes a live
    # agreement-gated co-diagnosis (#2) that the engine injects like a co-finding head. Returns []
    # on any failure -> base co-finding heads only.
    sid = Path(wsi_path).stem
    cfg.codx_integrator = {sid: codx_integ.compute(coords, feats)} if codx_integ is not None else None
    res = engine.predict_case(sid, tmp_h5, m1, m2, tok, msl, pools, cfg, device)

    qa = {q.strip(): (a or "").strip() for q, a, _ in res["edges"] if q}
    # HYBRID path (OOD-gated): a two-signal route. 
    # (a) compute gate - the H-Opt ABMIL's softmax entropy > τ decides WHEN to run TITAN.
    # (b) override filter (in titan_live.predict's is_ood) - TITAN's organ + base #1-dx are taken ONLY when 
    #     it names an OOD entity (organ ∉ the in-dist set OR dx ∉ the 139-vocab)
    _ORG_Q, _DX1_Q = "What is the organ?", "What is the #1 diagnosis?"
    _titan_ovr = None
    _elapsed = time.time() - _t0
    # Gate the TITAN LOAD (not just the predict) on remaining time.
    if (res.get("abmil_entropy", -1.0) > TITAN_GATE_ENTROPY
            and HARD_LIMIT_S - _elapsed > TITAN_LOAD_S + TITAN_RESERVE_S + TITAN_MIN_S
            and _get_titan() is not None):
        _budget = HARD_LIMIT_S - (time.time() - _t0) - TITAN_RESERVE_S  # recompute after the load
        print(f"[interf1] dx-OOD gate: abmil_entropy={res.get('abmil_entropy', -1.0):.3f} (thr {TITAN_GATE_ENTROPY}) "
              f"elapsed={time.time() - _t0:.0f}s titan_budget={_budget:.0f}s", flush=True)
        if _budget >= TITAN_MIN_S:
            try:
                _tp = _get_titan().predict(_titan_src, time_budget_s=_budget)
            except Exception:  # noqa: BLE001 - TITAN path must never crash the case; fall back to H-Opt
                _tp = None
            if _tp is not None and _tp.get("is_ood"):
                _tp_base = _titan_base_dx(_tp["dx"])  # TITAN = organ + BASE dx; grade from M1-vqa (build_report)
                if not re.search(r"[A-Za-z]", _tp_base):
                    _tp_base = qa.get(_DX1_Q, _tp_base)
                _titan_ovr = {"organ": _tp["organ"], "dx": _tp_base}  # the APPLIED override (guarded)
                qa[_ORG_Q] = _tp["organ"]
                qa[_DX1_Q] = _tp_base
                print(f"[interf1] HYBRID OOD override (entropy {res['abmil_entropy']:.2f}): organ={_tp['organ']} dx={_tp_base!r} (TITAN raw {_tp['dx']!r}; grade←M1-vqa)")
            elif _tp is not None:
                print(f"[interf1] TITAN ran but in-dist (organ={_tp['organ']} dx={_tp['dx']}) - keep H-Opt")
    elif res.get("abmil_entropy", -1.0) > TITAN_GATE_ENTROPY:
        print(f"[interf1] TITAN skipped - only {HARD_LIMIT_S - _elapsed:.0f}s left (elapsed {_elapsed:.0f}s)", flush=True)
    report = build_report(qa)
    cot: list[ChainOfThoughtStep] = []
    for q, a, nq in res["edges"]:
        step = {"question": (q or "").strip(), "answer": (a or "").strip(), "next_question": (nq or "").strip()}
        if "final pathology report" in step["question"].lower():
            step["answer"] = report
        elif _titan_ovr is not None and step["question"] == _ORG_Q:
            step["answer"] = _titan_ovr["organ"]
        elif _titan_ovr is not None and step["question"] == _DX1_Q:
            step["answer"] = _titan_ovr["dx"]  # the guarded base dx; grade/detail via M1-vqa in the report
        cot.append(step)
    print(f"[interf1] chain-of-thought: {len(cot)} steps")
    import shutil
    shutil.rmtree(_job, ignore_errors=True)            # reclaim the per-case trident/pyramid scratch
    shutil.rmtree(tmp_h5.parent, ignore_errors=True)   # and the per-case feature h5 (avoid /tmp growth)
    return cot
