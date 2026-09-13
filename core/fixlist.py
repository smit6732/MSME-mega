"""
core/fixlist.py

Turns the raw numeric results from completeness.py, consistency.py,
duplication.py and structure.py into a plain-language, severity-sorted
list of Finding objects (see core/findings.py) -- the tool's answer to
the problem statement's "generates a plain-language fix list"
requirement.

Two-step design, in this order (the order matters -- see below):

  1. Collect every issue as a "raw candidate" -- what column, what type
     of problem, what % of rows/columns it affects -- WITHOUT deciding
     severity yet. This step is pure arithmetic on the four dimension
     results, nothing more.

  2. Calibrate the critical/moderate severity cutoffs for THIS table
     using the full set of percentages gathered in step 1 (see
     core/calibration.py), THEN classify each candidate into a severity
     tier using those cutoffs, pick a matching sentence template, and
     build the final Finding.

  We can't do this in one pass: to calibrate severity thresholds "off
  the distribution of % affected values across all the checks run on
  this table" (the project spec's wording), we need to know every
  candidate's percentage BEFORE we can decide what counts as critical
  for this table. Hence: collect first, calibrate once, classify second.

The rule-based sentence (Finding.rule_based_description) is always
built here via pure template string-formatting (templates/fix_templates.py)
-- no AI, no network access, guaranteed to work. Optionally, after
Findings are built, core/llm_phrasing.py is given a chance to fill in
Finding.ai_phrased_description for each one; if that module is
unavailable or fails for any Finding, that Finding's rule_based_description
remains exactly what the UI shows -- see core/llm_phrasing.py's docstring
for the full fallback story.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.completeness import CompletenessResult
from core.consistency import ConsistencyResult
from core.duplication import DuplicationResult
from core.structure import StructureResult
from core.validity import ValidityResult
from core.calibration import calibrate_severity_thresholds, ThresholdCalibration
from core.findings import (
    Finding,
    SEVERITY_CRITICAL,
    SEVERITY_MODERATE,
    SEVERITY_MINOR,
    CHECK_TYPE_COMPLETENESS,
    CHECK_TYPE_CONSISTENCY,
    CHECK_TYPE_DUPLICATION,
    CHECK_TYPE_STRUCTURE,
    CHECK_TYPE_VALIDITY,
)
from templates.fix_templates import FIX_TEMPLATES

# Sort order used when displaying the final list -- critical issues first.
SEVERITY_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_MODERATE: 1, SEVERITY_MINOR: 2}

# ---------------------------------------------------------------------------
# Severity calibration is NOT purely "% of rows affected" -- that alone
# inverts real-world severity (see the two policies below). A free-text
# column that is 100% empty is the single highest percentage a table can
# produce, so a pure percentage-driven cutoff makes it look like the
# worst problem in the file; meanwhile a handful of "TBD"/"unknown"
# strings sitting in a numeric Age column, or one row with Age=-5, might
# affect only 3% of rows yet represent a much more concrete, damaging
# problem. These two small policy layers correct that, on top of (not
# instead of) the existing per-table percentage calibration:
#
#   1. severity_cap -- an issue_type/column combination that is known to
#      be low-stakes even at 100% (an optional free-text column being
#      empty) is capped at that ceiling, and its percentage is excluded
#      from the pool calibrate_severity_thresholds() sees, so it can't
#      drag the "critical" cutoff up so high that a real problem
#      elsewhere in the table gets under-classified as a side effect.
#   2. severity_floor -- an issue_type that is inherently a strong signal
#      regardless of how many rows it touches (an exact duplicate row, a
#      literal "unknown" sitting in a numeric column, a domain-impossible
#      value like Age=-5) is never allowed to fall below a floor, even if
#      its percentage alone would classify it as minor.
# ---------------------------------------------------------------------------

# Column-name substrings (lowercased) for free-text columns whose entire
# purpose is optional commentary -- a human filling in "why", not core
# transactional data. Empty is the NORMAL state for these, not a defect.
_OPTIONAL_FREE_TEXT_NAME_HINTS = (
    "notes", "note", "comment", "comments", "remark", "remarks",
    "description", "additional_info", "additional_information", "feedback", "memo",
)

# If any of these appear in the column name, the column is explicitly
# calling itself out as required -- e.g. "Mandatory_Notes",
# "Required_Comment" -- so the optional-free-text cap above must NOT
# apply, even though the name also matches one of the hints above.
_MANDATORY_OVERRIDE_HINTS = ("mandatory", "required", "compulsory", "must_")


def _is_optional_free_text_column(column_name: str, column_type: Optional[str]) -> bool:
    """
    True only for a genuinely optional, free-text commentary column --
    see the two hint lists above. column_type is checked (when known)
    because this rule is specifically about free TEXT commentary fields;
    a numeric or date column matching one of these words by coincidence
    would be a different (and real) problem, not optional commentary.

    "unknown" is accepted alongside "text" (not just None) deliberately:
    core/ingestion.py's type inference has nothing to infer a type FROM
    when a column is 100% empty -- exactly the headline case this cap
    exists for (a Notes column with nothing in it at all) -- so it comes
    back typed "unknown", not "text". Only a POSITIVELY confirmed
    non-text type (numeric, date, boolean) rules the cap out.
    """
    name_lower = column_name.lower()
    if any(hint in name_lower for hint in _MANDATORY_OVERRIDE_HINTS):
        return False
    if column_type not in (None, "text", "unknown"):
        return False
    return any(hint in name_lower for hint in _OPTIONAL_FREE_TEXT_NAME_HINTS)


# issue_type -> the LOWEST severity that issue_type is ever allowed to be
# classified as, regardless of what percentage-based calibration alone
# would produce. These are all issue types that are a strong, concrete
# signal of a real problem even when they affect only a few rows:
#   - exact_duplicate_rows: an unambiguous, 100%-confidence data-
#     integrity problem the moment even one exists.
#   - invalid_type_in_numeric / invalid_type_in_date: a literal non-
#     numeric/non-date string ("TBD", "unknown", "adult", "long", "ok")
#     sitting in a column that's supposed to be numeric/date -- not a
#     formatting quirk, a genuinely wrong kind of value.
#   - domain_outlier: a value that parses fine but is impossible for
#     what the column means (Age=-5, Rating=10, Fee=-100) -- see
#     core/validity.py's own domain-rule catalog.
# Deliberately NOT applied to structural_issue (a column that merely
# LOOKS like an identifier having a few duplicates is still genuinely
# rare/low-impact at low %) or to categorical_inconsistency/
# fuzzy_duplicate_rows (their real-world severity legitimately tracks
# how much of the column/table is affected, which percentage-based
# calibration already handles once it isn't being skewed by capped
# candidates -- see severity_cap above).
_SEVERITY_FLOOR_BY_ISSUE_TYPE = {
    "exact_duplicate_rows": SEVERITY_MODERATE,
    "invalid_type_in_numeric": SEVERITY_MODERATE,
    "invalid_type_in_date": SEVERITY_MODERATE,
    "domain_outlier": SEVERITY_MODERATE,
}

# AI phrasing costs roughly 0.5-2 seconds per Finding (small local model,
# but still not instant), PLUS a one-time ~1-2s model-load cost the
# first time it runs in a given process. A table with many issues could
# otherwise stall the UI for well past the tool's own "scorecard in
# under 20 seconds" target -- measured directly: 15 findings pushed one
# run from 1.9s (no AI) to over 30s. So: only the most severe findings
# get the AI pass -- findings are already sorted critical-first by the
# time this cap is applied, so this means "spend the AI time budget on
# what matters most, and every finding beyond the cap simply keeps its
# guaranteed, instant, template-based sentence" -- never a missing or
# blank finding. See AI_PHRASING_TIME_BUDGET_SECONDS below for the
# second, wall-clock half of this same guarantee.
MAX_FINDINGS_TO_AI_PHRASE = 15

# Hard wall-clock ceiling on the WHOLE AI-phrasing pass, checked before
# starting each finding's rephrasing (never mid-generation -- llama.cpp
# gives no way to interrupt a single call partway through). This is what
# actually bounds worst-case total time, independent of MAX_FINDINGS_TO_AI_PHRASE
# above: a table that calibrates to only 3-4 findings but happens to hit
# a slow cold model load, or a table with exactly 15 findings each
# taking longer than the 0.5-2s estimate, both still finish in bounded
# time. Once the budget is spent, every remaining finding simply keeps
# its guaranteed, already-computed rule_based_description -- identical
# in effect to "model unavailable" for that finding, never a missing or
# blank one. Chosen so that, combined with the rest of the pipeline
# (typically well under 2s for a small/medium file -- see
# core/calibration.py, the only other non-trivial cost), the WHOLE
# scorecard -- detection, scoring, AND the optional AI phrasing layer --
# stays comfortably inside the tool's ~20 second target even on a cold
# model load, rather than only bounding generation time and letting a
# slow load blow the budget anyway.
AI_PHRASING_TIME_BUDGET_SECONDS = 8.0

# Which check_type each issue_type belongs to -- used to tag every
# Finding with the right one of the four scorecard dimensions.
_ISSUE_TYPE_TO_CHECK_TYPE = {
    "missing_data": CHECK_TYPE_COMPLETENESS,
    "inconsistent_format_numeric": CHECK_TYPE_CONSISTENCY,
    "inconsistent_format_date": CHECK_TYPE_CONSISTENCY,
    "inconsistent_format_text": CHECK_TYPE_CONSISTENCY,
    "exact_duplicate_rows": CHECK_TYPE_DUPLICATION,
    "fuzzy_duplicate_rows": CHECK_TYPE_DUPLICATION,
    "structural_issue": CHECK_TYPE_STRUCTURE,
    "invalid_type_in_numeric": CHECK_TYPE_VALIDITY,
    "invalid_type_in_date": CHECK_TYPE_VALIDITY,
    "categorical_inconsistency": CHECK_TYPE_VALIDITY,
    "domain_outlier": CHECK_TYPE_VALIDITY,
    "missing_value_representation": CHECK_TYPE_VALIDITY,
}


@dataclass
class _RawIssueCandidate:
    """
    An issue before severity has been decided -- everything needed to
    build a Finding, minus severity and the final templated sentence
    (both of which need the calibrated cutoffs, computed after every
    candidate has been collected). Private to this module.
    """
    issue_type: str
    column_name: str
    percentage_affected: float
    format_kwargs: dict  # keyword args for str.format() on the template
    # Carried straight through to the finished Finding's own .details --
    # see core/findings.py and core/validity.py's docstrings for what
    # this is for. None for every candidate type that doesn't need it
    # (every non-validity candidate, plus most validity ones).
    details: Optional[dict] = None
    # See the severity_cap module docstring above. None for the
    # overwhelming majority of candidates -- percentage-based
    # calibration alone decides their severity, unchanged. (The
    # companion severity FLOOR is looked up straight from
    # _SEVERITY_FLOOR_BY_ISSUE_TYPE by issue_type, in generate_fix_list
    # below -- it never varies per-column the way the cap does, so it
    # doesn't need its own field here.)
    severity_cap: Optional[str] = None


# ---- Step 1: collect raw candidates (no severity decided yet) ------------

def _collect_missing_data_candidates(
    completeness_result: CompletenessResult, column_types: Optional[Dict[str, str]] = None,
) -> List[_RawIssueCandidate]:
    candidates = []
    for column in completeness_result.per_column.values():
        if column.missing_percentage <= 0:
            continue
        column_type = column_types.get(column.column_name) if column_types else None
        is_optional_free_text = _is_optional_free_text_column(column.column_name, column_type)
        candidates.append(_RawIssueCandidate(
            issue_type="missing_data",
            column_name=column.column_name,
            percentage_affected=column.missing_percentage,
            format_kwargs={"field": column.column_name, "percentage": column.missing_percentage},
            severity_cap=SEVERITY_MINOR if is_optional_free_text else None,
        ))
    return candidates


def _collect_consistency_candidates(consistency_result: ConsistencyResult) -> List[_RawIssueCandidate]:
    candidates = []
    # Maps a column's inferred type to the matching template bucket --
    # boolean/unknown columns aren't checked by consistency.py, so they
    # simply have no entry here and are skipped below.
    type_to_issue_key = {
        "numeric": "inconsistent_format_numeric",
        "date": "inconsistent_format_date",
        "text": "inconsistent_format_text",
    }
    for column in consistency_result.per_column.values():
        if column.inconsistent_percentage <= 0:
            continue
        issue_key = type_to_issue_key.get(column.column_type)
        if issue_key is None:
            continue
        example = column.example_values[0] if column.example_values else "N/A"
        candidates.append(_RawIssueCandidate(
            issue_type=issue_key,
            column_name=column.column_name,
            percentage_affected=column.inconsistent_percentage,
            format_kwargs={
                "field": column.column_name,
                "percentage": column.inconsistent_percentage,
                "example": example,
            },
        ))
    return candidates


def _collect_duplication_candidates(duplication_result: DuplicationResult, total_rows: int) -> List[_RawIssueCandidate]:
    candidates = []

    if duplication_result.exact_duplicate_row_indexes:
        percentage = round(len(duplication_result.exact_duplicate_row_indexes) / total_rows * 100, 2) if total_rows else 0.0
        candidates.append(_RawIssueCandidate(
            issue_type="exact_duplicate_rows",
            column_name="(whole row)",
            percentage_affected=percentage,
            format_kwargs={"percentage": percentage},
        ))

    if duplication_result.fuzzy_duplicate_pairs:
        rows_involved = set()
        for pair in duplication_result.fuzzy_duplicate_pairs:
            rows_involved.add(pair.row_index_a)
            rows_involved.add(pair.row_index_b)
        percentage = round(len(rows_involved) / total_rows * 100, 2) if total_rows else 0.0
        example_pair = duplication_result.fuzzy_duplicate_pairs[0]
        example = f"{example_pair.value_a} vs {example_pair.value_b}"
        column_label = duplication_result.identity_column_used or "(row)"
        candidates.append(_RawIssueCandidate(
            issue_type="fuzzy_duplicate_rows",
            column_name=column_label,
            percentage_affected=percentage,
            format_kwargs={"percentage": percentage, "example": example},
        ))

    return candidates


def _collect_validity_candidates(validity_result: Optional[ValidityResult]) -> List[_RawIssueCandidate]:
    """
    core/validity.py already does its own "collect candidates" pass
    (ValidityFindingCandidate, built alongside its per-column scores --
    see that module's check_validity), so this is a thin adapter, not a
    second detection pass: it just re-shapes each ValidityFindingCandidate
    into this module's own _RawIssueCandidate so it flows through the
    same calibrate-then-classify-then-template pipeline as every other
    dimension's candidates. validity_result is None whenever a caller
    doesn't compute validity at all (see generate_fix_list's docstring)
    -- an empty list in that case, not an error.
    """
    if validity_result is None:
        return []
    candidates = []
    for item in validity_result.candidates:
        format_kwargs = {"field": item.column_name, "percentage": item.percentage_affected}
        if item.example is not None:
            format_kwargs["example"] = item.example
        candidates.append(_RawIssueCandidate(
            issue_type=item.issue_type,
            column_name=item.column_name,
            percentage_affected=item.percentage_affected,
            format_kwargs=format_kwargs,
            details=item.details,
        ))
    return candidates


def _collect_structure_candidates(structure_result: StructureResult) -> List[_RawIssueCandidate]:
    candidates = []
    for issue in structure_result.issues:
        column_label = issue.column_name or "(dataset)"
        candidates.append(_RawIssueCandidate(
            issue_type="structural_issue",
            column_name=column_label,
            percentage_affected=issue.affected_percentage,
            format_kwargs={"example": issue.description},
        ))
    return candidates


# ---- Step 2: calibrate severity, then classify + template each one -------

def _severity_for_percentage(percentage: float, critical_cutoff: float, moderate_cutoff: float) -> str:
    """
    Classify an issue's severity by comparing it against THIS table's
    calibrated cutoffs (falling back to the fixed 30%/10% defaults when
    calibration itself fell back -- see core/calibration.py). The
    comparison logic itself never changes; only the two numbers it
    compares against do.
    """
    if percentage > critical_cutoff:
        return SEVERITY_CRITICAL
    if percentage >= moderate_cutoff:
        return SEVERITY_MODERATE
    return SEVERITY_MINOR


def _pick_template(issue_type: str, severity: str, pick_counters: Dict[Tuple[str, str], int]) -> str:
    """
    Pick one phrasing variant for this issue type/severity, cycling
    round-robin through the options available for it, so a fix list with
    several issues of the same type doesn't repeat the exact same
    sentence structure over and over.

    pick_counters is passed in by the caller (generate_fix_list) as a
    fresh, empty dict for every call, rather than kept as module-level
    state. That's deliberate, not incidental: module-level state would
    mean the SAME file, scored twice in the same running app, could pick
    different phrasing variants the second time (the counters would
    already be partway through their cycle from the first call) -- which
    would violate "the same file processed twice must always produce the
    same fix list". A fresh dict per call keeps this function's output a
    pure function of its inputs.
    """
    variants = FIX_TEMPLATES[issue_type][severity]
    key = (issue_type, severity)
    pick_index = pick_counters.get(key, 0)
    pick_counters[key] = (pick_index + 1) % len(variants)
    return variants[pick_index]


def generate_fix_list(
    completeness_result: CompletenessResult,
    consistency_result: ConsistencyResult,
    duplication_result: DuplicationResult,
    structure_result: StructureResult,
    total_rows: int,
    table_name: str = "table",
    use_ai_phrasing: bool = True,
    validity_result: Optional[ValidityResult] = None,
    column_types: Optional[Dict[str, str]] = None,
) -> Tuple[List[Finding], ThresholdCalibration, ThresholdCalibration]:
    """
    Build the full, severity-sorted list of Findings for one table. This
    is the single function core/pipeline.py calls to get the final
    plain-language output for a table.

    Returns (findings, critical_cutoff, moderate_cutoff) -- the two
    ThresholdCalibration objects are returned alongside the findings so
    the UI can display exactly what cutoff was used for this table (and
    whether it was calibrated or the fixed fallback), per the project's
    "the actual calibrated values used must be returned alongside the
    results" requirement.

    use_ai_phrasing: set False to skip the (optional) local-LLM
    rephrasing pass entirely -- used by tests that want to verify the
    guaranteed template-only path without depending on whether a model
    happens to be downloaded on the machine running the test.

    validity_result: OPTIONAL core/validity.py output (the 5th,
    Validity, dimension). Defaults to None -- meaning no validity
    Findings are added and every pre-existing caller of this function
    gets byte-identical behavior to before this dimension existed. This
    is what keeps generate_fix_list backward-compatible: core/pipeline.py
    passes a real ValidityResult now, but nothing about this function's
    contract required that.

    column_types: OPTIONAL {column_name: "text"/"numeric"/"date"/...}
    map (straight from DatasetProfile). Used only to decide whether a
    100%-empty column is genuinely optional free-text commentary (Notes,
    Comments, ...) rather than a real data gap -- see
    _is_optional_free_text_column. None (the default) means every
    pre-existing caller keeps working exactly as before; the missing-data
    severity cap simply never applies without type information to confirm
    a column is actually text.
    """
    raw_candidates: List[_RawIssueCandidate] = []
    raw_candidates += _collect_missing_data_candidates(completeness_result, column_types)
    raw_candidates += _collect_consistency_candidates(consistency_result)
    raw_candidates += _collect_duplication_candidates(duplication_result, total_rows)
    raw_candidates += _collect_structure_candidates(structure_result)
    raw_candidates += _collect_validity_candidates(validity_result)

    # Candidates with a severity_cap already have a known, fixed ceiling
    # on their outcome -- feeding their (often very high, e.g. a 100%-
    # empty Notes column) percentage into calibration would only risk
    # dragging THIS table's critical/moderate cutoffs up so high that a
    # genuinely serious issue elsewhere gets under-classified as a side
    # effect. Excluded from the pool, not from the findings list itself.
    calibration_percentages = [
        candidate.percentage_affected for candidate in raw_candidates if candidate.severity_cap is None
    ]
    critical_cutoff, moderate_cutoff = calibrate_severity_thresholds(calibration_percentages)

    # Fresh per call -- see _pick_template's docstring for why this must
    # not be module-level state.
    template_pick_counters: Dict[Tuple[str, str], int] = {}

    findings: List[Finding] = []
    for candidate in raw_candidates:
        severity = _severity_for_percentage(candidate.percentage_affected, critical_cutoff.value, moderate_cutoff.value)

        # Floor: some issue types are a strong signal regardless of how
        # small a percentage they affect (see _SEVERITY_FLOOR_BY_ISSUE_TYPE) --
        # never let percentage-based calibration alone under-classify them.
        floor = _SEVERITY_FLOOR_BY_ISSUE_TYPE.get(candidate.issue_type)
        if floor is not None and SEVERITY_ORDER[severity] > SEVERITY_ORDER[floor]:
            severity = floor

        # Cap: the inverse case (see _is_optional_free_text_column) --
        # never let percentage-based calibration alone over-classify a
        # column that's low-stakes by construction, even at 100% empty.
        if candidate.severity_cap is not None and SEVERITY_ORDER[severity] < SEVERITY_ORDER[candidate.severity_cap]:
            severity = candidate.severity_cap

        template = _pick_template(candidate.issue_type, severity, template_pick_counters)
        rule_based_sentence = template.format(**candidate.format_kwargs)

        findings.append(Finding(
            table_name=table_name,
            column_name=candidate.column_name,
            check_type=_ISSUE_TYPE_TO_CHECK_TYPE[candidate.issue_type],
            issue_type=candidate.issue_type,
            percentage_affected=candidate.percentage_affected,
            severity=severity,
            rule_based_description=rule_based_sentence,
            # Not every issue type has a concrete example (e.g. plain
            # missing-data candidates don't) -- .get() naturally gives
            # None for those, which Finding.example already defaults to.
            example=candidate.format_kwargs.get("example"),
            details=candidate.details,
        ))

    # Sort by severity tier first (critical -> moderate -> minor), then
    # by percentage affected descending within the same tier, so the
    # most impactful issues always surface first.
    findings.sort(key=lambda finding: (SEVERITY_ORDER[finding.severity], -finding.percentage_affected))

    if use_ai_phrasing:
        _apply_ai_phrasing(findings)

    return findings, critical_cutoff, moderate_cutoff


def _apply_ai_phrasing(findings: List[Finding]) -> None:
    """
    Best-effort pass over the most severe findings (see
    MAX_FINDINGS_TO_AI_PHRASE), asking the local LLM to fill in
    ai_phrased_description -- bounded by BOTH a count cap and a
    wall-clock time budget (AI_PHRASING_TIME_BUDGET_SECONDS), whichever
    is hit first. Imported lazily (inside the function, not at module
    top) so that core/fixlist.py -- and everything that imports it --
    never fails to import just because llama-cpp-python isn't installed;
    the import only happens at call time, inside a module that already
    handles its own absence gracefully.

    The time check runs BEFORE each finding's call, never during one --
    there's no way to interrupt a single llama.cpp generation partway
    through, so this bounds how many NEW rephrasings get started once
    the budget is spent, not any individual call's own duration. Every
    finding this loop doesn't reach keeps its guaranteed
    rule_based_description untouched -- exactly the same fallback path
    as "AI unavailable", just triggered by the clock instead.
    """
    from core.llm_phrasing import phrase_finding

    start_time = time.monotonic()
    for finding in findings[:MAX_FINDINGS_TO_AI_PHRASE]:
        if time.monotonic() - start_time > AI_PHRASING_TIME_BUDGET_SECONDS:
            break
        finding.ai_phrased_description = phrase_finding(finding)
