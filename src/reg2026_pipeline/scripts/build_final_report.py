"""Rule-based Final Report generator from cascade Q&A.

Given a case's cascade trajectory, produce a CAP-protocol-style pathology report

Strategy: GT reports are highly structured.
Cascade Q&A already contains all the keywords (organ, procedure, dx, grade,
Gleason scores, etc.). A per-organ template fills slots from cascade.

Usage as library:
    from build_final_report import build_report
    rep = build_report(cascade_qa, organ='Breast', procedure='Core needle biopsy', dxs=['Invasive carcinoma of no special type, grade III'])
"""

from __future__ import annotations

import re

# Cascade Q identifiers (canonical text from train_CoT.json)
Q_ORGAN = "What is the organ?"
Q_PROCEDURE = "What is the procedure?"
Q_DX_COUNT = "What is the number of diagnoses to includes?"
Q_DX_N = [f"What is the #{k} diagnosis?" for k in range(1, 5)]

# Breast invasive carcinoma scores
Q_TUBULE_SCORE = "What is the score for tubular differentiation?"
Q_NUCLEAR_SCORE = "What is the score for nuclear pleomorphism?"
Q_MITOSES_SCORE = "What is the score for mitotic rate?"

# Breast DCIS sub-fields
Q_ARCH_PATTERN = "What is the architectural pattern of lesion?"
Q_NUCLEAR_GRADE = "What is the nuclear grade of lesion?"
Q_NECROSIS_TYPE = "What is the type of necrosis?"

# Prostate cancer fields
Q_GLEASON_SCORE = "What is the Gleason score?"
Q_GRADE_GROUP = "What is the grade group?"
Q_PCT_PATTERN_4 = "What is the percentage of Gleason pattern 4?"
Q_TUMOR_VOLUME = "What is the tumor volume?"


def _normalize_procedure(proc: str) -> str:
    """Lowercase first letter to match GT format (e.g., 'Core needle biopsy' → 'core needle biopsy')."""
    if not proc:
        return ""
    return proc[0].lower() + proc[1:] if proc else proc


def _is_invasive_breast_ca(dx: str) -> bool:
    return "Invasive carcinoma of no special type" in dx or "invasive breast carcinoma" in dx.lower()


def _is_breast_dcis(dx: str) -> bool:
    return dx.strip().startswith("Ductal carcinoma in situ")


def _is_prostate_adenoca(dx: str) -> bool:
    return "adenocarcinoma" in dx.lower() and ("acinar" in dx.lower() or "prostat" in dx.lower())


def _format_breast_invasive_ca(dx: str, qa: dict[str, str]) -> str:
    """Append (Tubule formation: N, Nuclear grade: N, Mitoses: N) if all scores present.

    When dx carries NO grade word (a TITAN base dx - TITAN supplies only the coarse dx, grade comes from
    M1-vqa per the hybrid design), DERIVE the Nottingham grade from the 3 M1-vqa scores (sum 3-5→I, 6-7→II,
    8-9→III) and insert ', grade N' so the grade summary survives. Never touches a dx that already has a
    grade (H-Opt graded breast dx unchanged) → safe for the non-TITAN path.
    """
    t = qa.get(Q_TUBULE_SCORE, "").strip()
    n = qa.get(Q_NUCLEAR_SCORE, "").strip()
    m = qa.get(Q_MITOSES_SCORE, "").strip()
    if "grade" not in dx.lower():
        digs = [re.search(r"[1-3]", s) for s in (t, n, m)]
        if all(digs):
            total = sum(int(d.group()) for d in digs)  # type: ignore[union-attr]
            g = 1 if total <= 5 else (2 if total <= 7 else 3)
            dx = f"{dx}, grade {('I', 'II', 'III')[g - 1]}"
    if t and n and m:
        return f"{dx} (Tubule formation: {t}, Nuclear grade: {n}, Mitoses: {m})"
    return dx


def _format_breast_dcis(dx: str, qa: dict[str, str]) -> str:
    """Append bullet list - Type, - Nuclear grade, - Necrosis when fields present."""
    arch = qa.get(Q_ARCH_PATTERN, "").strip()
    ng = qa.get(Q_NUCLEAR_GRADE, "").strip()
    nec = qa.get(Q_NECROSIS_TYPE, "").strip() or _necrosis_present_to_text(qa)
    bullets = []
    if arch:
        bullets.append(f"  - Type: {arch}")
    if ng:
        bullets.append(f"  - Nuclear grade: {ng}")
    if nec:
        bullets.append(f"  - Necrosis: {nec}")
    if bullets:
        return dx + "\n" + "\n".join(bullets)
    return dx


def _necrosis_present_to_text(qa: dict[str, str]) -> str:
    """If necrosis Q answered No, GT often says 'Absent'."""
    a = qa.get("Is there any necrosis present?", "").lower()
    if a.startswith("no"):
        return "Absent"
    return ""


def _format_prostate_adenoca(dx: str, qa: dict[str, str]) -> str:
    """Append Gleason score / grade group / Gleason pattern 4 % / tumor volume."""
    gs = qa.get(Q_GLEASON_SCORE, "").strip()
    gg = qa.get(Q_GRADE_GROUP, "").strip().lower()  # "Grade group 3" → "grade group 3"
    pct = qa.get(Q_PCT_PATTERN_4, "").strip()
    vol = qa.get(Q_TUMOR_VOLUME, "").strip()
    parts = [dx]
    if gs:
        parts.append(f"Gleason's score {gs}")
    if gg:
        if pct and "1" not in gg.split()[-1]:  # grade group >= 2 has pct
            parts.append(f"{gg} (Gleason pattern 4: {pct})")
        else:
            parts.append(gg)
    if vol:
        parts.append(f"tumor volume: {vol}")
    return ", ".join(parts)


# Inflammatory/benign findings that GT appends as "with X" (1 finding) or "with 1) X\n 2) Y"
# (>=2). Sourced from "Is there any X present? -> Yes" cascade Qs. Priority order matches GT
# (intestinal metaplasia listed first). Only applied to inflammatory dx (see _append_findings).
_FINDING_QS = [
    ("Is there any intestinal metaplasia present?", "intestinal metaplasia"),
    ("Is there any foveolar epithelial hyperplasia present?", "foveolar epithelial hyperplasia"),
    ("Is there any lymphoid aggregate present?", "lymphoid aggregate"),
    ("Is there any lymphoid follicle present?", "lymphoid follicle"),
    ("Is there any ulceration present?", "ulceration"),
]
_INFLAMMATORY_DX = ("gastritis", "nonspecific inflammation", "cervicitis", "colitis", "duodenitis", "esophagitis")


def _collect_findings(qa: dict[str, str]) -> list[str]:
    return [name for q, name in _FINDING_QS if qa.get(q, "").strip().lower().startswith("yes")]


def _append_findings(dx: str, qa: dict[str, str]) -> str:
    """For inflammatory dx (e.g. 'Chronic gastritis'), append the YES findings the GT report lists:
    1 finding -> inline 'dx with X'; >=2 -> 'dx\\n  with 1) X\\n       2) Y'. (build_report B-bug fix:
    these sub-findings were previously dropped, tanking Stomach/Colon FinalReport keyword score.)"""
    if not any(k in dx.lower() for k in _INFLAMMATORY_DX):
        return dx
    finds = _collect_findings(qa)
    if not finds:
        return dx
    if len(finds) == 1:
        return f"{dx} with {finds[0]}"
    lines = [f"  with 1) {finds[0]}"] + [f"       {i}) {f}" for i, f in enumerate(finds[1:], 2)]
    return dx + "\n" + "\n".join(lines)


def _append_dysplasia(dx: str, qa: dict[str, str]) -> str:
    """Adenomas: GT appends 'with low/high grade dysplasia' (38% of colon reports) sourced from
    'What is the grade of dysplasia?' (-> 'Low grade'/'High grade'). Previously dropped."""
    if "adenoma" not in dx.lower() or "dysplasia" in dx.lower():
        return dx
    g = qa.get("What is the grade of dysplasia?", "").strip().lower()  # 'Low grade' / 'High grade'
    if g.endswith("grade"):
        return f"{dx} with {g} dysplasia"
    return dx


def _append_granulomatous(dx: str, qa: dict[str, str]) -> str:
    """Lung granulomatous inflammation: GT appends 'with necrosis' (necrosis Q -> Yes). 34 lung reports."""
    if "granulomatous" in dx.lower() and "necrosis" not in dx.lower() and qa.get("Is there any necrosis present?", "").strip().lower().startswith("yes"):
        return f"{dx} with necrosis"
    return dx


def _format_bladder_invasive(dx: str, qa: dict[str, str]) -> str:
    """Bladder invasive urothelial ca: GT reformats to 'Invasive urothelial carcinoma,\\n  with 1) involvement
    of {extent}\\n      2) {differentiation}'. Sources: extent of invasion + histologic subtype/dx."""
    base = "Invasive urothelial carcinoma"
    finds: list[str] = []
    ext = qa.get("What is the extent of invasion?", "").lower()
    if "muscularis propria" in ext or "muscle proper" in ext:
        finds.append("involvement of muscle proper")
    elif "subepithelial connective tissue" in ext:
        finds.append("involvement of subepithelial connective tissue")
    elif "lamina propria" in ext:
        finds.append("involvement of lamina propria")
    blob = (dx + " " + qa.get("What is the histologic subtype of neoplasm?", "")).lower()
    if "squamous differentiation" in blob:
        finds.append("squamous differentiation")
    elif "glandular differentiation" in blob:
        finds.append("glandular differentiation")
    if len(finds) >= 2:
        lines = [f"  with 1) {finds[0]}"] + [f"      {i}) {f}" for i, f in enumerate(finds[1:], 2)]
        return base + ",\n" + "\n".join(lines)
    if len(finds) == 1:
        return f"{base},\n  with {finds[0]}"
    # predicted cascade bakes the extent into the #1 dx string ("... with involvement of X") instead of a separate "extent of invasion" Q; split at " with " so the 2-line GT form still matches.
    low = dx.lower()
    if low.startswith(base.lower()) and " with " in low:
        i = low.index(" with ")
        return dx[:i] + ",\n  " + dx[i + 1 :]
    return dx


def _format_dx(dx: str, organ: str, qa: dict[str, str]) -> str:
    """Apply per-organ formatting to a single dx text."""
    if not dx:
        return ""
    organ_l = (organ or "").lower()
    if "breast" in organ_l:
        if _is_invasive_breast_ca(dx):
            return _format_breast_invasive_ca(dx, qa)
        if _is_breast_dcis(dx):
            return _format_breast_dcis(dx, qa)
    elif "prostate" in organ_l:
        if _is_prostate_adenoca(dx):
            return _format_prostate_adenoca(dx, qa)
    elif "bladder" in organ_l and "invasive urothelial carcinoma" in dx.lower():
        return _format_bladder_invasive(dx, qa)
    # inflammatory findings (gastritis) + adenoma dysplasia (colon) + granulomatous necrosis (lung)
    return _append_granulomatous(_append_findings(_append_dysplasia(dx, qa), qa), qa)


def _suffix(organ: str, procedure: str, qa: dict[str, str]) -> str:
    """Organ-specific trailing notes (Urinary bladder TURBT: Note about muscle proper).

    Cascade Q 'Is there any muscularis propria present?' gives 1-to-1 mapping:
      Yes -> 'includes muscle proper'
      No  -> 'does not include muscle proper'
    Falls back to 'includes' (77% majority) if no cascade answer.
    """
    organ_l = (organ or "").lower()
    proc_l = (procedure or "").lower()
    if "bladder" in organ_l and ("transurethral" in proc_l or "turbt" in proc_l):
        mp = qa.get("Is there any muscularis propria present?", "").lower()
        if mp.startswith("no"):
            return "\n\nNote) The specimen does not include muscle proper."
        # default to 'includes' (Yes-answer or no cascade Q)
        return "\n\nNote) The specimen includes muscle proper."
    return ""


def build_report(cascade_qa: dict[str, str], organ: str | None = None, procedure: str | None = None, dxs: list[str] | None = None) -> str:
    """Build a CAP-style pathology report from cascade Q&A.

    Args:
        cascade_qa: dict mapping cascade Q text → A text (most recent answer per Q).
        organ: organ name; if None, read from cascade_qa[Q_ORGAN].
        procedure: procedure; if None, read from cascade_qa[Q_PROCEDURE].
        dxs: list of dx texts in order #1..#N; if None, read from cascade_qa[Q_DX_N].
    """
    organ = organ or cascade_qa.get(Q_ORGAN, "").strip()
    procedure = procedure or cascade_qa.get(Q_PROCEDURE, "").strip()
    if dxs is None:
        dxs = []
        for q in Q_DX_N:
            a = cascade_qa.get(q, "").strip()
            if a:
                dxs.append(a)

    if cascade_qa.get("Is there any glandular involvement present?", "").strip().lower().startswith("no"):
        dxs = [dx.replace(" with glandular involvement", "") for dx in dxs]

    if not organ or not procedure or not dxs:
        # Fallback minimal report
        organ_p = organ or "Unknown"
        proc_p = procedure or "biopsy"
        dx_p = dxs[0] if dxs else "Specimen evaluated"
        return f"{organ_p}, {_normalize_procedure(proc_p)};\n  {dx_p}".replace("\n", "\\n")

    header = f"{organ}, {_normalize_procedure(procedure)};\n"

    if len(dxs) == 1:
        body = "  " + _format_dx(dxs[0], organ, cascade_qa)
    else:
        lines = []
        for i, dx in enumerate(dxs):
            lines.append(f"  {i + 1}. {_format_dx(dx, organ, cascade_qa)}")
        body = "\n".join(lines)

    return (header + body + _suffix(organ, procedure, cascade_qa)).replace("\n", "\\n")
