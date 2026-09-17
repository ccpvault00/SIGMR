"""Score our e2e predictions with the official evaluate_metrics.py.

Pipeline:
  1. load e2e --dump-official-pred-json  ({sid: [{question, answer, next_question}]}).
  2. inject the Final Report (build_report on the predicted Q->A) as the
     answer to the "What is the final pathology report?" step.
  3. build val-split GT in official case format (from train_CoT).
  4. run official evaluate_metrics.evaluate_workflow_dataset
     (semantic backend = lexical, FR embedding = NeuML/pubmedbert-base-embeddings).
  5. print final_ranking_score (= official WFR) + single/multi breakdown.

Usage:
  PYTHONPATH=$PWD/src python scripts/run_official_eval.py \
      --official-pred-json /tmp/repro_v3_official.json \
      --official-repo ~/src/REG2026/submission_evaluation_code
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FINAL_Q = "What is the final pathology report?"


def dxc(case: dict) -> int:
    for t in case.get("chain-of-thought", []):
        if "number of diagnoses" in t.get("question", "").lower():
            m = re.search(r"\d+", t.get("answer", "") or "")
            return int(m.group()) if m else 1
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-pred-json", type=Path, required=True, help="e2e --dump-official-pred-json output")
    ap.add_argument("--cot-json", type=Path, default=Path("/mnt/data/reg2026/train_CoT.json"))
    ap.add_argument("--split-json", type=Path, default=Path("/mnt/data/reg2026/checkpoints/phase_a_v3/split.json"))
    ap.add_argument("--official-repo", type=Path, default=Path.home() / "src/REG2026/submission_evaluation_code")
    ap.add_argument("--embedding-model", default="NeuML/pubmedbert-base-embeddings")
    ap.add_argument("--literal-nl", action="store_true", help="emit literal backslash-n in the report to match GT format (GT uses '\\\\n', not real newlines)")
    ap.add_argument("--out-json", type=Path, default=Path("/tmp/official_eval_result.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.official_repo))
    import evaluate_metrics as em  # noqa: E402
    from build_final_report import build_report  # noqa: E402

    preds = json.loads(args.official_pred_json.read_text())
    cot = {c["id"].replace(".tiff", ""): c for c in json.loads(args.cot_json.read_text())}
    val = {s.replace(".tiff", "") for s in json.loads(args.split_json.read_text())["val"]}

    # GT: val cases in official format (already {id, chain-of-thought, organ}).
    gt_cases = [cot[sid] for sid in val if sid in cot]

    # PRED: inject SOTA final report, build official case dicts.
    pred_cases = []
    for sid, steps in preds.items():
        sid0 = sid.replace(".tiff", "")
        if sid0 not in val:
            continue
        steps = [dict(s) for s in steps]
        qa = {s["question"]: s["answer"] for s in steps if s.get("question")}
        report = build_report(qa)
        if args.literal_nl:
            report = report.replace("\n", "\\n")
        # override existing final-report step's answer, else append one
        found = False
        for s in steps:
            if "final pathology report" in (s.get("question", "") or "").lower():
                s["answer"] = report
                found = True
        if not found:
            steps.append({"question": FINAL_Q, "answer": report, "next_question": ""})
        pred_cases.append({"id": sid0, "chain-of-thought": steps})

    print(f"[official-eval] GT val cases={len(gt_cases)}  pred cases={len(pred_cases)}", flush=True)

    scorer = em.SemanticScorer(backend="lexical")
    fr_eval = em.REG25FinalReportEvaluator(embedding_model=args.embedding_model)
    summary = em.evaluate_workflow_dataset(gt_cases, pred_cases, scorer, fr_eval, strict_missing_predictions=False)

    # single/multi breakdown from per_case + GT dx count
    dx_by = {cot[sid]["id"].replace(".tiff", ""): dxc(cot[sid]) for sid in val if sid in cot}
    grp = defaultdict(list)
    for cs in summary["per_case"]:
        cid = str(cs["case_id"]).replace(".tiff", "")
        g = "multi" if dx_by.get(cid, 1) >= 2 else "single"
        grp[g].append(cs)

    def avg(rows, f):
        return sum(r[f] for r in rows) / len(rows) if rows else 0.0

    print("\n=== OFFICIAL evaluate_metrics.py (lexical MESS, NeuML PubMedBERT FR) ===")
    print(f"{'group':10} {'n':>5} {'BPV':>6} {'EF1':>6} {'MESS':>6} {'FinalRep':>8} {'WFR':>7}")
    for g in ("single", "multi"):
        rows = grp[g]
        if not rows:
            continue
        print(
            f"{g:10} {len(rows):>5} {avg(rows, 'binary_path_validity'):>6.3f} {avg(rows, 'edge_f1'):>6.3f} "
            f"{avg(rows, 'mess_nonfinal'):>6.3f} {avg(rows, 'final_report_score'):>8.3f} {avg(rows, 'ranking_score'):>7.3f}"
        )
    print(f"\nOFFICIAL WFR (final_ranking_score) = {summary['final_ranking_score']:.4f}")
    print(
        f"  avg BPV={summary['average_binary_path_validity']:.4f}  EF1={summary['average_edge_f1']:.4f}  "
        f"MESS={summary['average_mess_nonfinal']:.4f}  FinalRep={summary['average_final_report_score']:.4f}"
    )
    print(f"  missing pred cases={summary['num_missing_prediction_cases']}")

    args.out_json.write_text(json.dumps({k: v for k, v in summary.items() if k != "per_case"}, indent=2))
    print(f"saved -> {args.out_json}")


if __name__ == "__main__":
    main()
