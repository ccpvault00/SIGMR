"""Normalize known train_CoT.json annotation glitches that the (clean) test GT is not expected to have.

Two glitches were found by orphan-node analysis; both make a diagnostic node unreachable by
an edge-following BFS router, so without this fix M2 cannot learn the edge and local eval scores the
clean prediction unfairly against a malformed GT.

(1) LUNG - histologic-subtype node question string blanked.
    The named histtype turn already carries next_question="What is the histologic subtype of
    neoplasm?" (the edge EXISTS), but the subtype turn's own question field is the empty string,
    so it canonicalizes to "" and never matches the edge target -> orphan. 141/146 lung empty-Q
    turns are this (answer = "Non-small cell carcinoma, favor ..."). Restore the named question.

(2) BLADDER - invasive-UC histtype node has no parent edge at all.
    455/455 invasive-urothelial-carcinoma histtype occurrences are orphan (nothing's next_question
    points to histtype). The sequential predecessor is "What is the extent of invasion?" in 367/367
    single-occurrence cases. Inject a fan-out edge extent-of-invasion -> histtype so the detail
    branch (histtype -> subtype -> additional finding) is reachable. bladder-only, 0% in all other organs.
"""

from __future__ import annotations

from typing import Any

_LUNG_SUBTYPE_Q = "What is the histologic subtype of neoplasm?"
_BLADDER_EXTENT_Q = "What is the extent of invasion?"
_HISTTYPE_Q = "What is the histologic type of neoplasm?"


def _canon(s: str | None) -> str:
    return (s or "").strip().lower().rstrip("?").strip()


_HT_C = _canon(_HISTTYPE_Q)
_EXT_C = _canon(_BLADDER_EXTENT_Q)


def normalize_cot_chain(chain: list[dict[str, Any]], organ: str | None) -> list[dict[str, Any]]:
    """Return a normalized copy of a CoT chain (list of {question, answer, next_question} turns).

    Pure / order-preserving. No-op for organs/turns without the known glitches.
    """
    org = (organ or "").lower()
    out = [dict(t) for t in chain]

    if org == "lung":
        for t in out:
            if not (t.get("question") or "").strip() and "carcinoma" in (t.get("answer") or "").lower():
                t["question"] = _LUNG_SUBTYPE_Q
        return out

    if org == "bladder":
        ext_ans = next((t.get("answer", "") for t in out if _canon(t.get("question")) == _EXT_C), None)
        if ext_ans is None:
            return out
        # invasive-UC histtype is always orphan -> inject extent->histtype fan-out before each.
        new: list[dict[str, Any]] = []
        for t in out:
            if _canon(t.get("question")) == _HT_C and "invasive urothelial" in (t.get("answer") or "").lower():
                new.append({"question": _BLADDER_EXTENT_Q, "answer": ext_ans, "next_question": _HISTTYPE_Q})
            new.append(t)
        return new

    return out
