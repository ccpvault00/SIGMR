"""Protocols - co-finding (#2 diagnosis) derivations.

Co-findings are reportable diagnoses beyond the primary (#1). Two mechanisms:
  1. in-situ entailment - a dx-keyed clinical derivation (below).
  2. visual cofinding heads (DCIS / urothelial CIS tile-heads) and the region
     integrator are wired in the engine via configs/cofinding_*.json and the
     co-dx integrator; their per-organ usage is documented in protocols/sop.py.

In-situ entailment: an in-situ carcinoma (DCIS, urothelial CIS) is BY DEFINITION
non-invasive (basement membrane intact) and, for bladder CIS, flat/non-papillary.
So within a co-dx subtree anchored on an in-situ dx, the "invasion?" / "papillary
lesion?" gates are ENTAILED No - a property of the dx LABEL, not the slide. The
whole-slide visual answerer mis-answers these in ~20-40% of co-dx cases (slide is
dominated by the co-existing invasive primary); on the in-situ lesion in isolation
it is 100% correct. Per pathology definition - same family as gleason_sum.
"""

from __future__ import annotations

INSITU_DX_MARKERS: tuple[str, ...] = ("carcinoma in situ", "in situ")

_INSITU_GATE_ENTAILED_NO: dict[str, str] = {
    "Is there any invasion present?": "No, there is no invasion.",
    "Is there any papillary lesion present?": "No, there is no papillary lesion.",
}


def is_insitu_dx(dx_text: str) -> bool:
    """True if the diagnosis text denotes an in-situ carcinoma (DCIS / CIS)."""
    d = (dx_text or "").strip().lower()
    return any(m in d for m in INSITU_DX_MARKERS)


def insitu_gate_entailment(dx_text: str, gate_q_text: str) -> str | None:
    """Entailed gate answer for an in-situ diagnosis, or None if the model should answer.

    Args:
        dx_text: the (subtree) diagnosis anchor, e.g. "Ductal carcinoma in situ".
        gate_q_text: the gate question being answered.
    Returns:
        "No, there is no invasion." / "No, there is no papillary lesion." when dx
        is in-situ and the gate is invasion/papillary; else None.
    """
    if not is_insitu_dx(dx_text):
        return None
    return _INSITU_GATE_ENTAILED_NO.get(gate_q_text.strip())
