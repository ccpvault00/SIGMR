"""Protocols - named answer reroutes (engine-flag triggered, documented here).

Two reroutes override the default visual/cls answerer for specific Qs because a
deterministic or more-accurate source exists. The engine applies them via flags
(--gleason-from-patterns, --microcalc-bump); this module names the affected Qs and
their target so the policy is auditable in one place.
"""

from __future__ import annotations

# --gleason-from-patterns : prostate grade-group + Gleason score are DERIVED from the
# predominant + secondary patterns (Per pathology definition: ISUP 2014 / X+Y=Z) rather
# than read from a cls head. More accurate and removes score-vs-pattern contradictions.
GLEASON_FROM_PATTERNS: dict[str, str] = {
    "What is the grade group?": "grading.isup_grade_group_2014",
    "What is the Gleason score?": "grading.gleason_sum",
}

# --microcalc-bump : breast microcalcification is a focal co-finding; detected by the
# visual tile-head with top-k pooling (mean-pooling collapses point findings). Wired via
# configs/cofinding_*.json (handler 'cofinding_microcalc'). See protocols.sop BREAST.
MICROCALC_COFINDING_Q = "Is there any microcalcification present?"
