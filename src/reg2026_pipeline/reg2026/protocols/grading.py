"""Protocols - computed grading derivations (Hook-C family).

Each formula takes the case history (list of (Q, A) tuples) and returns either a
computed A text, or None if a prerequisite Q is missing. These are deterministic
clinical derivations (Per pathology definition). All diagnostic decision logic
lives under one readable `protocols/` package.
"""

from __future__ import annotations

import re

# ISUP 2014 grade group from Gleason score (Epstein et al., Mod Pathol 2014/2016).
ISUP_GRADE_GROUP_2014: dict[tuple[int, int], str] = {
    (3, 3): "Grade group 1",
    (3, 4): "Grade group 2",
    (4, 3): "Grade group 3",
    (4, 4): "Grade group 4",
    (3, 5): "Grade group 4",
    (5, 3): "Grade group 4",
    (4, 5): "Grade group 5",
    (5, 4): "Grade group 5",
    (5, 5): "Grade group 5",
}


def _find_a(history: list[tuple[str, str]], q_text: str) -> str | None:
    """Return last A text for Q in history, or None."""
    for q, a in reversed(history):
        if q == q_text:
            return a.strip()
    return None


def _parse_int(a: str | None) -> int | None:
    """Extract leading integer from A text. '3' -> 3, 'Score 5' -> 5."""
    if a is None:
        return None
    m = re.search(r"\d+", a)
    return int(m.group()) if m else None


def gleason_sum(history: list[tuple[str, str]]) -> str | None:
    """Gleason score = predominant + secondary, formatted 'X+Y=Z'. Per pathology definition."""
    primary = _parse_int(_find_a(history, "What is the pridominant pattern?"))
    secondary = _parse_int(_find_a(history, "What is the secondary pattern constituting more than 5% of tumor?"))
    if primary is None or secondary is None:
        return None
    return f"{primary}+{secondary}={primary + secondary}"


def isup_grade_group_2014(history: list[tuple[str, str]]) -> str | None:
    """ISUP 2014 grade group from Gleason patterns. Per pathology definition."""
    primary = _parse_int(_find_a(history, "What is the pridominant pattern?"))
    secondary = _parse_int(_find_a(history, "What is the secondary pattern constituting more than 5% of tumor?"))
    if primary is None or secondary is None:
        return None
    return ISUP_GRADE_GROUP_2014.get((primary, secondary))


def nottingham_sum(history: list[tuple[str, str]]) -> str | None:
    """Nottingham overall score = tubular + nuclear + mitotic. Per pathology definition."""
    tubular = _parse_int(_find_a(history, "What is the score for tubular differentiation?"))
    nuclear = _parse_int(_find_a(history, "What is the score for nuclear pleomorphism?"))
    mitotic = _parse_int(_find_a(history, "What is the score for mitotic rate?"))
    if tubular is None or nuclear is None or mitotic is None:
        return None
    return str(tubular + nuclear + mitotic)


def nottingham_grade(history: list[tuple[str, str]]) -> str | None:
    """Nottingham grade from overall score. 3-5=I, 6-7=II, 8-9=III. Per pathology definition."""
    overall = _parse_int(_find_a(history, "What is the overall score?"))
    if overall is None:
        overall_str = nottingham_sum(history)
        if overall_str is None:
            return None
        overall = int(overall_str)
    if 3 <= overall <= 5:
        return "Grade I"
    if 6 <= overall <= 7:
        return "Grade II"
    if 8 <= overall <= 9:
        return "Grade III"
    return None


# Grading-system Qs are deterministic 1:1 from histologic_type / dx (audit 2026-05-28).
GRADING_SYSTEM_LOOKUP: dict[str, str] = {
    "Acinar adenocarcinoma": "Gleason grading system",
    "Invasive carcinoma of no special type, grade I": "Nottingham combined histologic grade",
    "Invasive carcinoma of no special type, grade II": "Nottingham combined histologic grade",
    "Invasive carcinoma of no special type, grade III": "Nottingham combined histologic grade",
    "Invasive carcinoma of no special type": "Nottingham combined histologic grade",
    "Ductal carcinoma in situ": "Nottingham combined histologic grade",
    "Microcalcification": "Nottingham combined histologic grade",
    "Fibroadenoma": "Nottingham combined histologic grade",
    "Atypical lobular hyperplasia": "Nottingham combined histologic grade",
    "Intraductal papilloma": "Nottingham combined histologic grade",
    "Fibroadenomatoid change": "Nottingham combined histologic grade",
    "Tubulovillous adenoma": "2-tier grading system",
    "Adenocarcinoma, moderately differentiated": "3-tier grading system",
}

GRADING_SYSTEM_NEOPLASM_LOOKUP: dict[str, str] = {
    "Adenocarcinoma": "3-tier grading system",
    "Adenocarcinoma, moderately differentiated": "3-tier grading system",
    "Adenocarcinoma, well differentiated": "3-tier grading system",
    "Adenocarcinoma, poorly differentiated": "3-tier grading system",
    "Neuroendocrine tumor, grade 1": "WHO grading system of neuroendocrine neoplasms",
    "Neuroendocrine tumor": "WHO grading system of neuroendocrine neoplasms",
}


def _lookup_from_dx_or_histtype(history: list[tuple[str, str]], lookup: dict[str, str]) -> str | None:
    """Find dx in lookup via #1 dx text first, fallback to histologic_type_of_neoplasm (grade/diff-stripped)."""
    for q_text in ["What is the #1 diagnosis?", "What is the histologic type of neoplasm?"]:
        a = _find_a(history, q_text)
        if a is None:
            continue
        if a in lookup:
            return lookup[a]
        a_stripped = re.sub(r",?\s*grade\s+\S+\s*$", "", a, flags=re.IGNORECASE)
        a_stripped = re.sub(r",?\s*(well|moderately|poorly|undifferentiated)\s+differentiated\s*$", "", a_stripped, flags=re.IGNORECASE)
        a_stripped = re.sub(r",?\s*(high|low|intermediate)\s+grade\s*$", "", a_stripped, flags=re.IGNORECASE)
        a_stripped = a_stripped.strip()
        if a_stripped in lookup:
            return lookup[a_stripped]
    return None


def grading_system_from_histologic_type(history: list[tuple[str, str]]) -> str | None:
    """'What is the grading system?' - deterministic 1:1 from histologic_type/dx."""
    return _lookup_from_dx_or_histtype(history, GRADING_SYSTEM_LOOKUP)


def grading_system_neoplasm_from_histologic_type(history: list[tuple[str, str]]) -> str | None:
    """'What is the grading system of neoplasm?' - 1:1 from Adenocarcinoma variants + Neuroendocrine."""
    return _lookup_from_dx_or_histtype(history, GRADING_SYSTEM_NEOPLASM_LOOKUP)


def const_dysplasia_grading(history: list[tuple[str, str]]) -> str | None:
    """'What is the grading system of dysplasia?' - ALWAYS 2-tier (audit 1513/1513)."""
    return "2-tier grading system"


def const_atypia_grading(history: list[tuple[str, str]]) -> str | None:
    """'What is the grading system of atypia?' - ALWAYS 3-tier (audit 686/686)."""
    return "3-tier grading system"


# Registry: formula name -> callable (referenced by name in the DAG's computed_a).
FORMULAS: dict[str, callable] = {
    "gleason_sum": gleason_sum,
    "isup_grade_group_2014": isup_grade_group_2014,
    "nottingham_sum": nottingham_sum,
    "nottingham_grade": nottingham_grade,
    "grading_system_from_histologic_type": grading_system_from_histologic_type,
    "grading_system_neoplasm_from_histologic_type": grading_system_neoplasm_from_histologic_type,
    "const_dysplasia_grading": const_dysplasia_grading,
    "const_atypia_grading": const_atypia_grading,
}
