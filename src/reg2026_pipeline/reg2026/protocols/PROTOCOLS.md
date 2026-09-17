Protocols — per-organ diagnostic SOP (the readable "instruction manual").

A concise, reviewer-facing description of how each CAP/WHO field is determined per
organ. The engine executes these via the recognizers (slide ABMIL + M1 answerers),
the grading derivations (protocols.grading), the co-finding gates (protocols.cofindings)
and the routing/reroutes tables. Detail is intentionally light — "Per pathology
definition" stands in for the standard derivation; see grading.py for the formulas.

Pipeline (all organs):
  trajectory : BFS over a hand-coded DAG (block structure + M2 learned next-Q edges)
  answering  : visual recognizer — slide-level ABMIL #1 dx + per-Q M1 answerers
               (cls head / MedCPT-cosine, standalone or context-aware; see routing.py)
  computed   : deterministic clinical derivations (grading.py) for grade/score fields
  co-findings: visual tile-head gates + in-situ entailment (cofindings.py); region
               integrator for general #2-dx (Phase-2 distribution-shift hedge)
  report     : CAP-protocol structured template from the resolved fields

========================================================================
PER-ORGAN PROTOCOLS
========================================================================

PROSTATE — acinar adenocarcinoma (CAP + WHO)
  abnormality / #1 diagnosis : visual recognizer (slide ABMIL + M1 gate)
  Gleason pattern 3 / 4 / 5  : visual presence (M1)
  predominant / secondary    : visual (M1)
  Gleason score              : predominant + secondary            # Per pathology definition
  grade group                : ISUP 2014 from patterns            # Per pathology definition
  grading system             : "Gleason grading system"           # deterministic from dx

BREAST — invasive carcinoma NST / DCIS (CAP + WHO)
  abnormality / #1 diagnosis : visual recognizer
  tubular / nuclear / mitotic: visual scores (M1)
  Nottingham score + grade   : sum of scores -> grade I/II/III    # Per pathology definition
  grading system             : "Nottingham combined histologic grade"
  co-finding: DCIS           : visual tile-head gate (focal -> top-k pooling)
  co-finding: microcalcification : visual tile-head gate (focal -> top-k)
  in-situ gates (DCIS)       : invasion / papillary entailed "No" # Per pathology definition

COLON — adenocarcinoma / tubulovillous adenoma (CAP + WHO)
  abnormality / #1 diagnosis : visual recognizer
  histologic type            : visual (M1)
  differentiation grade      : visual, 3-tier
  grading system of neoplasm : "3-tier grading system" (adenocarcinoma)
  grading system of dysplasia: "2-tier grading system" (adenoma)  # Per pathology definition

STOMACH — adenocarcinoma / dysplasia / NET (CAP + WHO)
  abnormality / #1 diagnosis : visual recognizer
  histologic type            : visual (Lauren / WHO)
  differentiation grade      : visual, 3-tier
  grading system of neoplasm : "3-tier grading system"
  neuroendocrine tumor       : "WHO grading system of neuroendocrine neoplasms" (if NET)
  grading system of dysplasia: "2-tier grading system"            # Per pathology definition

BLADDER — urothelial carcinoma / CIS (CAP + WHO)
  abnormality / #1 diagnosis : visual recognizer
  invasion present           : visual gate (M1)
  papillary lesion present   : visual gate (M1)
  grade                      : visual, low / high
  co-finding: CIS            : visual tile-head gate + pooled-carcinoma co-occurrence posterior
  in-situ gates (CIS)        : invasion / papillary entailed "No" # Per pathology definition

LUNG — non-small-cell carcinoma: adeno / squamous (CAP + WHO)
  abnormality / #1 diagnosis : visual recognizer
  histologic type            : visual (adenocarcinoma vs squamous cell carcinoma)
  favor / definitive         : reporting-convention hedge

CERVIX — squamous cell carcinoma / HSIL-LSIL (CAP + WHO)
  abnormality / #1 diagnosis : visual recognizer
  histologic type            : visual (M1)
  dysplasia grade            : visual
  grading system of dysplasia: "2-tier grading system"            # Per pathology definition
  grading system of atypia   : "3-tier grading system"            # Per pathology definition
