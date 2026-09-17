"""Protocols - the diagnostic decision policy for REG²2026, in one readable place.

This package consolidates every per-organ / per-Q diagnostic rule the model follows
into a single auditable "instruction manual" (the submission is manually reviewed):

  grading.py    deterministic clinical derivations (Gleason/ISUP/Nottingham, grading systems)
  cofindings.py co-finding (#2 dx) derivations: in-situ entailment + visual tile-head gates
  reroutes.py   named answer reroutes (gleason-from-patterns, microcalc co-finding)
  sop.py        per-organ SOP - the human-readable summary
"""

from __future__ import annotations

from reg2026.protocols import cofindings, grading, reroutes, sop
from reg2026.protocols.sop import PROTOCOLS

__all__ = ["grading", "cofindings", "reroutes", "sop", "PROTOCOLS", "render_manual"]


def render_manual() -> str:
    """Render the full per-organ SOP as a single text manual (for PROTOCOLS.md / review)."""
    header = (sop.__doc__ or "").strip()
    organs = "\n".join(PROTOCOLS[o] for o in ["prostate", "breast", "colon", "stomach", "bladder", "lung", "cervix"])
    return f"{header}\n\n{'=' * 72}\nPER-ORGAN PROTOCOLS\n{'=' * 72}\n\n{organs}"
