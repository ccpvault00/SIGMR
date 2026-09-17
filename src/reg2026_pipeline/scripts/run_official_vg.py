"""Score our VG answers with the OFFICIAL evaluate_metrics VG metric (B1/B2/B3 + Qwen3-8B judge)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, default=Path("/tmp/vg_answers_official.json"))
    ap.add_argument("--mapping", type=Path, default=Path("reference/REG26-Visual Grounding Examples/anonymous_rois_mapping.txt"))
    ap.add_argument("--official-repo", type=Path, default=Path.home() / "src/REG2026/submission_evaluation_code")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--voting", type=int, default=1)
    args = ap.parse_args()
    sys.path.insert(0, str(args.official_repo))
    import evaluate_metrics as em

    judge = em.LocalQwenJudgeLLM(model_path=args.judge_model, device=args.device, max_new_tokens=args.max_new_tokens)
    res = em.run_visual_dataset_from_answer_json(args.answers, args.mapping, judge, 0.30, 0.30, 0.40, args.voting)
    print("\n=== OFFICIAL Visual Grounding (B1/B2/B3 + Qwen3-8B judge) ===")
    print(f"B1 (background rejection)   = {res['average_B1_background']:.4f}")
    print(f"B2 (input sensitivity)      = {res['average_B2_sensitivity']:.4f}")
    print(f"B3 (cross-region consist.)  = {res['average_B3_cross_region']:.4f}")
    print(f"VG final = 0.30*B1 + 0.30*B2 + 0.40*B3 = {res['final_visual_score']:.4f}")


if __name__ == "__main__":
    main()
