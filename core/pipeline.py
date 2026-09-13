"""
core/pipeline.py

Orchestration layer: ties ingestion, all four checks, scoring, and the
fix list together for ONE table, and separately handles running that
same sequence across MULTIPLE tables (multiple uploaded files, and/or
multiple sheets within an uploaded Excel file) and combining the results
into one overall summary.

This module exists so app.py can stay "thin" -- app.py's job is only to
accept uploads and render results, never to know the exact sequence of
core/*.py calls needed to go from a DatasetProfile to a finished
scorecard. That sequence lives here, in one place, so it's testable
without Streamlit and easy to point to when explaining "what actually
happens when I upload a file."
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from core.ingestion import load_all_tables, DatasetProfile
from core.completeness import check_completeness
from core.consistency import check_consistency
from core.duplication import check_duplication, DuplicationResult
from core.structure import check_structure
from core.validity import check_validity
from core.scoring import build_scorecard, ScorecardResult
from core.fixlist import generate_fix_list
from core.calibration import ThresholdCalibration
from core.findings import Finding, SEVERITY_CRITICAL, SEVERITY_MODERATE, SEVERITY_MINOR

_SEVERITY_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_MODERATE: 1, SEVERITY_MINOR: 2}


@dataclass
class TableResult:
    """Everything the UI needs to show for ONE successfully-processed table."""
    table_name: str
    row_count: int
    column_count: int
    scorecard: ScorecardResult
    findings: List[Finding]
    duplicate_threshold: ThresholdCalibration   # similarity cutoff actually used
    critical_cutoff: ThresholdCalibration        # % cutoff actually used
    moderate_cutoff: ThresholdCalibration
    # The already-loaded dataframe THIS scorecard was computed from, and
    # the already-computed duplication result (which carries
    # exact_duplicate_row_indexes). Both are carried through here purely
    # so Tier 1 (core/remediation.py, called from app.py) can reuse them
    # directly -- "Finding-driven, not a fresh scan" also means not
    # re-loading or re-running duplication detection a second time.
    # Neither field is read by anything that computes a score: adding
    # them changes nothing about how `scorecard` above was built.
    dataframe: Optional[pd.DataFrame] = None
    duplication_result: Optional[DuplicationResult] = None
    # column_types, straight from DatasetProfile -- the same type
    # inference (numeric/text/date/boolean) core/chart_generation.py
    # reuses to decide each column's chart, rather than re-detecting types.
    column_types: dict = field(default_factory=dict)
    # Both straight from DatasetProfile (core/ingestion.py) -- carried
    # through here so app.py can show them without reaching past this
    # orchestration layer back into ingestion internals. 0 / None are the
    # "nothing to report" values -- see ingestion.py's own docstrings for
    # what each one means and why it's never silently swallowed.
    skipped_preamble_rows: int = 0
    encoding_warning: Optional[str] = None


@dataclass
class TableFailure:
    """One table (file or sheet) that could not be processed, and why.
    Kept separate from TableResult (rather than making every field
    Optional on one shared class) so the UI's "did this table succeed?"
    check is a simple isinstance/list-membership question, not a
    None-checking exercise."""
    table_name: str
    error_message: str


@dataclass
class MultiTableSummary:
    """
    The full result of processing a batch of uploaded files (each
    possibly multi-sheet). This is the single object app.py needs to
    render the whole page: the overall aggregate first, then each
    table's own detail.
    """
    table_results: List[TableResult] = field(default_factory=list)
    table_failures: List[TableFailure] = field(default_factory=list)
    overall_score: float = 0.0
    combined_findings: List[Finding] = field(default_factory=list)  # every table's findings, tagged, sorted


def run_pipeline_for_table(
    dataset_profile: DatasetProfile, table_name: str, use_ai_phrasing: bool = True, on_stage=None,
) -> TableResult:
    """
    Run completeness -> consistency -> duplication -> structure ->
    scoring -> fix list for ONE already-loaded table, and package the
    result. This is exactly the sequence a single-file upload has always
    run (see the original app.py's run_pipeline) -- multi-table support
    just means calling this once per table instead of once per upload.

    on_stage: an OPTIONAL callback, `on_stage(stage_name: str, table_name: str)`,
    fired right before each stage below starts running. Purely additive
    instrumentation for app.py's staged loading UI -- it changes nothing
    about what gets computed or returned (every caller that omits it, which
    includes every existing test, gets byte-identical behavior to before
    this parameter existed). Kept as a bare callable rather than a logging
    framework or event bus on purpose -- the simplest thing that lets the
    UI layer observe real progress without this module knowing anything
    about Streamlit.
    """
    def _report(stage: str) -> None:
        if on_stage:
            on_stage(stage, table_name)

    _report("completeness")
    completeness_result = check_completeness(dataset_profile.dataframe)
    _report("consistency")
    consistency_result = check_consistency(dataset_profile.dataframe, dataset_profile.column_types)
    _report("duplication")
    duplication_result = check_duplication(dataset_profile.dataframe, dataset_profile.column_types)
    _report("structure")
    structure_result = check_structure(dataset_profile.dataframe, dataset_profile.raw_text)
    _report("validity")
    validity_result = check_validity(dataset_profile.dataframe, dataset_profile.column_types)

    _report("scoring")
    scorecard = build_scorecard(
        completeness_result, consistency_result, duplication_result, structure_result, validity_result,
    )

    _report("fixlist")
    findings, critical_cutoff, moderate_cutoff = generate_fix_list(
        completeness_result,
        consistency_result,
        duplication_result,
        structure_result,
        dataset_profile.row_count,
        table_name=table_name,
        use_ai_phrasing=use_ai_phrasing,
        validity_result=validity_result,
        column_types=dataset_profile.column_types,
    )

    duplicate_threshold = ThresholdCalibration(
        value=duplication_result.similarity_threshold_used,
        was_calibrated=duplication_result.threshold_was_calibrated,
    )

    return TableResult(
        table_name=table_name,
        row_count=dataset_profile.row_count,
        column_count=dataset_profile.column_count,
        scorecard=scorecard,
        findings=findings,
        duplicate_threshold=duplicate_threshold,
        critical_cutoff=critical_cutoff,
        moderate_cutoff=moderate_cutoff,
        dataframe=dataset_profile.dataframe,
        duplication_result=duplication_result,
        column_types=dataset_profile.column_types,
        skipped_preamble_rows=dataset_profile.skipped_preamble_rows,
        encoding_warning=dataset_profile.encoding_warning,
    )


def run_multi_table_pipeline(uploaded_files: list, use_ai_phrasing: bool = True, on_stage=None) -> MultiTableSummary:
    """
    The multi-table entry point: takes a list of uploaded files (each
    either a path string or a Streamlit UploadedFile-like object with
    .name), loads every table inside every file (one CSV = one table,
    one Excel file = one table per sheet), runs the full pipeline on
    each table that loaded successfully, and combines everything into
    one MultiTableSummary.

    Failure isolation is the key property here: a single file's read
    error, or a single bad sheet inside an otherwise-fine workbook, or
    even an unexpected exception while SCORING one particular table's
    data, is caught and recorded as a TableFailure -- it never stops the
    rest of the batch from being processed. This directly implements the
    "one bad table should never crash the whole upload" requirement.

    on_stage: see run_pipeline_for_table's docstring -- passed straight
    through, plus one extra "loading" stage reported here for the file
    read itself (which happens above/outside that function). Optional;
    omitting it (every existing caller does) changes nothing.
    """
    summary = MultiTableSummary()

    for uploaded_file in uploaded_files:
        file_name = _get_file_name(uploaded_file)
        if on_stage:
            on_stage("loading", file_name)
        load_outcomes = load_all_tables(uploaded_file, file_name)

        for outcome in load_outcomes:
            if not outcome.succeeded:
                summary.table_failures.append(TableFailure(outcome.table_name, outcome.error))
                continue

            try:
                table_result = run_pipeline_for_table(
                    outcome.profile, outcome.table_name, use_ai_phrasing=use_ai_phrasing, on_stage=on_stage,
                )
                summary.table_results.append(table_result)
            except Exception as unexpected_error:
                # A check module choking on this particular table's data
                # shape (e.g. a column type combination we didn't
                # anticipate) still shouldn't take down the rest of the
                # batch -- record it the same way a load failure is
                # recorded, and move on.
                summary.table_failures.append(
                    TableFailure(outcome.table_name, f"Error while scoring this table: {unexpected_error}")
                )

    summary.overall_score = _compute_overall_score(summary.table_results)
    summary.combined_findings = _combine_and_sort_findings(summary.table_results)
    return summary


def _get_file_name(uploaded_file) -> str:
    """Streamlit's UploadedFile has a .name attribute; a plain path
    string (used in tests and scripts) IS the name. Handles both without
    the rest of this module needing to know which one it got."""
    return uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)


def _compute_overall_score(table_results: List[TableResult]) -> float:
    """
    The aggregate score across every successfully-processed table: a
    plain average of each table's own overall_score. Simple and easy to
    explain -- every table counts equally, regardless of its row count,
    same philosophy as completeness.py averaging every column equally
    (see that module's docstring) rather than weighting by size, which
    would need a business judgment call this tool doesn't try to make.
    """
    if not table_results:
        return 0.0
    return round(sum(result.scorecard.overall_score for result in table_results) / len(table_results), 2)


def _combine_and_sort_findings(table_results: List[TableResult]) -> List[Finding]:
    """
    Every table's findings, concatenated into one list and re-sorted by
    severity (critical -> moderate -> minor) across the WHOLE upload --
    not just within each table. Finding.table_name (set back in
    core/fixlist.py) is what lets the UI show which table each item in
    this combined list came from.
    """
    all_findings: List[Finding] = []
    for result in table_results:
        all_findings.extend(result.findings)

    all_findings.sort(key=lambda finding: (_SEVERITY_ORDER[finding.severity], -finding.percentage_affected))
    return all_findings
