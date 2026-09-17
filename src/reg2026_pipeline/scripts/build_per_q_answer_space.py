"""M1-cls answer-space builder: auto-derive each Q's answer space from train CoT.

The M1-cls thesis: M1's collapse on fine-grained answers is its cosine-regress-to-MedCPT-vocab
objective (small answer sets like grades/gates embed close → cosine can't separate → majority
collapse), while open large sets (dx names) embed far apart → cosine fine. So the closed-vs-open
split - which decides classification-head vs cosine - is DERIVABLE FROM DATA (answer-set size per
Q), not a manual per-Q routing decision.

This script: for each canonical Q, collect its train answer set, map answers to A_VOCAB indices,
flag closed (small categorical set, all in A_VOCAB) vs open (large / free-text), and save the
per-Q answer space (legal A_VOCAB mask + class counts) used by the M1-cls trainer + inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict

from reg2026.data.embedding_pools import EmbeddingPools, canonicalize_q

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

COT = "/mnt/data/reg2026/train_CoT.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-json", default=COT)
    ap.add_argument("--out", default="/mnt/data/reg2026/checkpoints/m1_answer_space/per_q_answer_space.json")
    ap.add_argument("--closed-max", type=int, default=15, help="answer-set size <= this AND all in A_VOCAB => closed (classification)")
    args = ap.parse_args()

    pools = EmbeddingPools()
    a2i = {a.strip().lower(): i for i, a in enumerate(pools.A_VOCAB)}
    with open(args.cot_json) as f:
        cot = json.load(f)

    ans_by_q: dict[str, Counter] = defaultdict(Counter)
    for c in cot:
        for t in c.get("chain-of-thought", []):
            q = canonicalize_q(t.get("question", "") or "")
            a = (t.get("answer", "") or "").strip()
            if q and a:
                ans_by_q[q][a] += 1

    out = {}
    n_closed = 0
    closed_qs, open_qs = [], []
    for q, ac in ans_by_q.items():
        n_distinct = len(ac)
        in_vocab = sum(1 for a in ac if a.strip().lower() in a2i)
        frac_in = in_vocab / n_distinct
        # closed: small categorical set fully resolvable to A_VOCAB
        closed = n_distinct <= args.closed_max and frac_in >= 0.95
        legal = sorted({a2i[a.strip().lower()] for a in ac if a.strip().lower() in a2i})
        counts = {a2i[a.strip().lower()]: n for a, n in ac.items() if a.strip().lower() in a2i}
        out[q] = {
            "n_distinct": n_distinct,
            "frac_in_vocab": round(frac_in, 3),
            "closed": closed,
            "legal_a_idx": legal,
            "class_counts": counts,
            "total": sum(ac.values()),
        }
        if closed:
            n_closed += 1
            closed_qs.append((q, n_distinct, sum(ac.values())))
        else:
            open_qs.append((q, n_distinct, frac_in))

    log.info("Qs total=%d  closed=%d (classification)  open=%d (cosine)", len(ans_by_q), n_closed, len(open_qs))
    log.info("answer-set-size histogram: %s", dict(sorted(Counter(min(v["n_distinct"], 30) for v in out.values()).items())))
    log.info("\n=== CLOSED Qs (-> classification head, balanced CE) [n_distinct, n_train] ===")
    for q, nd, tot in sorted(closed_qs, key=lambda x: -x[2]):
        log.info("  [%2d ans, %5d] %s", nd, tot, q[:62])
    log.info("\n=== OPEN Qs (-> keep cosine) [n_distinct, frac_in_vocab] ===")
    for q, nd, fr in sorted(open_qs, key=lambda x: -x[1]):
        log.info("  [%4d ans, in_vocab %.2f] %s", nd, fr, q[:60])

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info("saved per-Q answer space -> %s", args.out)


if __name__ == "__main__":
    main()
