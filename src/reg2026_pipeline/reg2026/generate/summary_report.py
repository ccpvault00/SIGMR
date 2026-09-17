"""Deterministic summary report builder.

Trajectory text (organ, procedure, dx_count, #k dx, intermediate
grading/scoring Qs) is templated directly into the canonical CAP-style
report format.

Format observed in train_CoT.json final-report Q answers:

    {Organ display}, {procedure lowercase};\\n
      {dx_1}                              # if dx_count == 1, no numbering
    OR
      1. {dx_1 [+ enrichment]}\\n
      2. {dx_2}\\n
      ...
    [organ-specific suffix, e.g. bladder muscularis Note]

Newlines stored as literal `\\n` (two-char) - matching training data.
"""

from __future__ import annotations

ORGAN_Q = "What is the organ?"
PROCEDURE_Q = "What is the procedure?"
DX_COUNT_Q = "What is the number of diagnoses to includes?"
DX_Q_TEMPLATE = "What is the #{} diagnosis?"
NL = "\\n"  # literal backslash-n (matches training data encoding)


def _find_a(trajectory: list[tuple[str, str]], question: str) -> str | None:
    """Return the answer to the first occurrence of `question` (exact match)."""
    for q, a in trajectory:
        if q == question:
            return a.strip() if a else None
    return None


def _parse_dx_count(trajectory: list[tuple[str, str]]) -> int:
    """Decode dx_count Q answer (clamped 1..4). Default 1 if missing/malformed."""
    a = _find_a(trajectory, DX_COUNT_Q)
    if not a:
        return 1
    try:
        n = int(a.strip())
        return max(1, min(4, n))
    except (ValueError, AttributeError):
        return 1


def _parse_dx_answers(trajectory: list[tuple[str, str]], dx_count: int) -> list[str]:
    """Collect #1..#dx_count diagnosis answers from trajectory."""
    answers: list[str] = []
    for k in range(1, dx_count + 1):
        a = _find_a(trajectory, DX_Q_TEMPLATE.format(k))
        answers.append(a or "")
    return answers


def _enrich_breast(dx: str, trajectory: list[tuple[str, str]]) -> str:
    """Add Nottingham scoring detail to invasive carcinoma dx."""
    if "invasive carcinoma" not in dx.lower():
        return dx
    tubule = _find_a(trajectory, "What is the score for tubular differentiation?")
    nuclear = _find_a(trajectory, "What is the score for nuclear pleomorphism?")
    mitoses = _find_a(trajectory, "What is the score for mitotic rate?")
    if tubule and nuclear and mitoses:
        return f"{dx} (Tubule formation: {tubule}, Nuclear grade: {nuclear}, Mitoses: {mitoses})"
    return dx


def _enrich_prostate(dx: str, trajectory: list[tuple[str, str]]) -> str:
    """Add Gleason score, grade group, tumor volume to acinar adenocarcinoma dx."""
    if "adenocarcinoma" not in dx.lower():
        return dx
    gleason = _find_a(trajectory, "What is the Gleason score?")
    grade_group = _find_a(trajectory, "What is the grade group?")
    volume = _find_a(trajectory, "What is the tumor volume?")
    parts = [dx]
    if gleason:
        # Training format: "Gleason's score 6 (3+3)" - model emits "7 (3+4)"
        parts.append(f"Gleason's score {gleason}")
    if grade_group:
        # Training format: "grade group 1" - model emits "Grade group 1"
        gg = grade_group.lower().replace("grade group", "").strip() or grade_group
        parts.append(f"grade group {gg}")
    if volume:
        parts.append(f"tumor volume: {volume}")
    return ", ".join(parts)


def _enrich_dx(organ_key: str, dx: str, trajectory: list[tuple[str, str]]) -> str:
    """Dispatch enrichment by organ. organ_key = aliased coarse key."""
    if not dx:
        return ""
    if organ_key == "breast":
        return _enrich_breast(dx, trajectory)
    if organ_key == "prostate":
        return _enrich_prostate(dx, trajectory)
    return dx


def _build_organ_suffix(organ_key: str, trajectory: list[tuple[str, str]]) -> str:
    """Per-organ trailing note (e.g. bladder muscularis status)."""
    if organ_key == "bladder":
        m = _find_a(trajectory, "Is there any muscularis propria present?")
        if not m:
            return ""
        m_lower = m.lower()
        if m_lower.startswith("yes"):
            return NL + NL + "Note) The specimen includes muscle proper."
        if m_lower.startswith("no"):
            return NL + NL + "Note) The specimen does not include muscle proper."
    return ""


def build_summary_report(
    organ_display: str,
    organ_key: str,
    trajectory: list[tuple[str, str]],
) -> str:
    """Construct final pathology report from trajectory text (Stage 2).

    Args:
        organ_display: raw organ answer (e.g. "Uterine cervix", "Breast") -
            used in header. Preserves training capitalization.
        organ_key: aliased coarse pool key (e.g. "cervix", "breast") - used
            for enrichment dispatch + suffix logic.
        trajectory: full Q-A list from `decode_cot_trajectory`.

    Returns:
        Final report string with literal `\\n` separators (matches training
        format in train_CoT.json).
    """
    procedure = _find_a(trajectory, PROCEDURE_Q)
    procedure_text = procedure.lower() if procedure else "?"
    organ_text = organ_display or "?"

    header = f"{organ_text}, {procedure_text};"

    dx_count = _parse_dx_count(trajectory)
    dx_answers = _parse_dx_answers(trajectory, dx_count)

    if dx_count == 1:
        body = NL + f"  {_enrich_dx(organ_key, dx_answers[0], trajectory)}"
    else:
        body_lines = [f"  {k}. {_enrich_dx(organ_key, dx, trajectory)}" for k, dx in enumerate(dx_answers, 1)]
        body = NL + NL.join(body_lines)

    suffix = _build_organ_suffix(organ_key, trajectory)

    return header + body + suffix
