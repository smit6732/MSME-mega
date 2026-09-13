"""
tests/test_pipeline.py

Smoke tests for the whole pipeline -- single-table backward compatibility,
multi-table/multi-sheet support, calibration behavior (including its
fallback path), determinism, and the AI-phrasing fallback. This isn't
meant to be exhaustive -- it exists so that after changing any core/*.py
module, running `pytest` quickly confirms nothing broke end-to-end.

Run with:
    pytest
or:
    python -m pytest tests/test_pipeline.py -v
"""

import os
import sys

# Make sure "core" and "templates" are importable when this file is run
# directly (pytest normally handles this via its rootdir/conftest
# discovery, but this keeps `python tests/test_pipeline.py` working too).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingestion import load_dataset, load_all_tables
from core.pipeline import run_pipeline_for_table, run_multi_table_pipeline
from core.calibration import calibrate_duplicate_threshold, calibrate_severity_thresholds
from core.findings import Finding
from core.remediation import remediate_table, _apply_or_decline_date
from core.chart_generation import generate_charts_for_table
from core.validity import check_validity, is_coercible_numeric_token, is_recognized_date_token, categorical_key
from core.completeness import check_completeness
from core.scoring import build_scorecard
from core.cleaning_config import CleaningConfig, CONSERVATIVE, STANDARD, AGGRESSIVE, MISSING_STRATEGY_DROP_ROW, MISSING_STRATEGY_CONSTANT
import core.llm_phrasing as llm_phrasing


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLE_TABLE_CSV = os.path.join(REPO_ROOT, "sample_data", "messy_msme_sample.csv")
MULTI_SHEET_XLSX = os.path.join(REPO_ROOT, "sample_data", "multi_sheet_msme_workbook.xlsx")
TEST_DATASET_CSV = os.path.join(REPO_ROOT, "test_dataset.csv")


# ---- Single-table backward compatibility ----------------------------------

def test_single_csv_still_loads_and_scores():
    """The original single-file flow (load_dataset + run_pipeline_for_table)
    must keep working exactly as it did before multi-table support existed."""
    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    assert dataset_profile.row_count == 20
    assert dataset_profile.column_count == 9
    assert dataset_profile.column_types["Annual_Revenue"] == "numeric"
    assert dataset_profile.column_types["Registration_Date"] == "date"

    result = run_pipeline_for_table(dataset_profile, "messy_msme_sample.csv", use_ai_phrasing=False)
    assert 0 <= result.scorecard.overall_score <= 100
    # This sample file is deliberately messy on completeness/consistency/
    # duplication -- it should NOT score as excellent. Its own numeric
    # and date columns happen to have no Validity-dimension problems
    # (see core/validity.py), so adding that 5th dimension legitimately
    # nudges the overall score up from this test's original <80 bar --
    # that's the new dimension correctly reporting "this specific facet
    # is fine", not a broken check. <90 keeps the real intent: this file
    # is clearly imperfect, not a false "everything's great" result.
    assert result.scorecard.overall_score < 90
    assert len(result.findings) > 0


def test_fix_list_is_sorted_by_severity():
    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    result = run_pipeline_for_table(dataset_profile, "messy_msme_sample.csv", use_ai_phrasing=False)

    severity_rank = {"critical": 0, "moderate": 1, "minor": 2}
    ranks = [severity_rank[f.severity] for f in result.findings]
    assert ranks == sorted(ranks), "Findings must be sorted critical -> moderate -> minor"


def test_duplication_check_finds_known_duplicates():
    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    result = run_pipeline_for_table(dataset_profile, "messy_msme_sample.csv", use_ai_phrasing=False)
    duplication_findings = [f for f in result.findings if f.check_type == "duplication"]
    # The sample file has one exact duplicate row and several near-
    # duplicate business names built in on purpose.
    assert len(duplication_findings) >= 1


# ---- Regression: AI-phrased duplication findings must keep their specifics --------

def test_fuzzy_duplicate_finding_carries_example_field():
    """
    Regression test for a bug where core.findings.Finding had no field
    at all for a check's concrete example (e.g. a named near-duplicate
    pair like "Shree Ganesh Traders vs Shri Ganesh Traders"), so
    core/llm_phrasing.py's prompt structurally could never include it no
    matter how the prompt was worded -- the fact simply never reached
    that module. This test doesn't need the actual LLM to be installed
    -- it only confirms the example reaches the Finding object itself,
    which is the fix that makes AI-phrasing able to keep it at all.
    """
    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    result = run_pipeline_for_table(dataset_profile, "messy_msme_sample.csv", use_ai_phrasing=False)

    fuzzy_dup_findings = [f for f in result.findings if f.issue_type == "fuzzy_duplicate_rows"]
    assert len(fuzzy_dup_findings) >= 1
    assert fuzzy_dup_findings[0].example is not None
    assert " vs " in fuzzy_dup_findings[0].example


def test_ai_phrased_duplication_finding_keeps_its_percentage_and_example():
    """
    End-to-end regression test for the same bug, this time actually
    exercising the local LLM: an AI-phrased duplication finding must
    still contain its specific percentage number, and (when the model is
    available at all) core/llm_phrasing.py's content-fidelity check
    means any AI phrasing that DID get produced is guaranteed to have
    kept it -- a vague rephrasing gets silently rejected in favor of the
    template sentence instead of ever reaching the UI. Skipped (not
    failed) when the model hasn't been downloaded on this machine --
    see core/llm_phrasing.py's module docstring for why that's the
    correct behavior, not a gap in coverage.
    """
    from core.llm_phrasing import is_model_available
    if not is_model_available():
        print("SKIP: local model not downloaded -- nothing to verify here, see scripts/download_model.py")
        return

    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    result = run_pipeline_for_table(dataset_profile, "messy_msme_sample.csv", use_ai_phrasing=True)

    fuzzy_dup_findings = [f for f in result.findings if f.issue_type == "fuzzy_duplicate_rows"]
    assert len(fuzzy_dup_findings) >= 1
    finding = fuzzy_dup_findings[0]

    if finding.ai_phrased_description is not None:
        whole_percent = str(int(finding.percentage_affected))
        assert whole_percent in finding.ai_phrased_description


# ---- Regression: raw issue_type codes must never leak into AI-phrased text --------

def test_issue_type_humanization_covers_every_known_issue_type():
    """
    Regression test for a bug where raw internal issue_type codes (e.g.
    "fuzzy_duplicate_rows") leaked verbatim into AI-phrased sentences --
    the LLM prompt handed the model this raw snake_case string with
    nothing telling it that it was a technical label, not language to
    repeat, so it sometimes echoed it straight into "plain-language"
    output. Fixed with core/llm_phrasing.py's _ISSUE_TYPE_HUMANIZED
    mapping. This test confirms that mapping stays in sync with
    fixlist.py's own list of issue_type codes (the actual source of
    truth for which codes exist), so a new issue_type added to one file
    later can't silently go unmapped in the other.
    """
    from core.fixlist import _ISSUE_TYPE_TO_CHECK_TYPE
    from core.llm_phrasing import _ISSUE_TYPE_HUMANIZED

    known_issue_types = set(_ISSUE_TYPE_TO_CHECK_TYPE.keys())
    mapped_issue_types = set(_ISSUE_TYPE_HUMANIZED.keys())
    missing = known_issue_types - mapped_issue_types
    assert not missing, f"issue_type(s) missing a humanized phrase: {missing}"

    # Every humanized phrase must itself contain no underscores -- a
    # phrase that still looks like a code would defeat the whole point.
    for issue_type, phrase in _ISSUE_TYPE_HUMANIZED.items():
        assert "_" not in phrase, f"{issue_type!r} maps to a still-snake_case phrase: {phrase!r}"


def test_leaked_issue_type_safety_net_catches_codes_and_ignores_prose():
    """
    Unit test for the safety-net regex itself (core/llm_phrasing.py's
    _contains_leaked_issue_type_code): must catch a raw snake_case code
    if one ever leaks through despite humanization, and must NOT flag
    genuine plain-language sentences (including ones that mention a
    percentage or a hyphenated phrase like "near-duplicate") as false
    positives -- a check this trigger-happy would reject good AI
    phrasings for no reason.
    """
    from core.llm_phrasing import _contains_leaked_issue_type_code

    assert _contains_leaked_issue_type_code(
        "The Business_Name column contains a fuzzy_duplicate_rows issue with 15.0% affected rows."
    ) is True
    assert _contains_leaked_issue_type_code(
        "This column has several near-duplicate records that should be reviewed."
    ) is False
    assert _contains_leaked_issue_type_code(
        "Inconsistent number formatting affects 45.0% of rows in this column."
    ) is False


def test_ai_phrased_fuzzy_duplicate_finding_never_leaks_raw_issue_type():
    """
    End-to-end regression test for the same bug, exercising the real
    local model when it's available: an AI-phrased fuzzy-duplicate
    finding must never contain the raw "fuzzy_duplicate_rows" code
    verbatim. This should hold even if the model tries to echo the
    (now-humanized) problem-type phrase back, since there's no longer a
    raw snake_case string anywhere in the prompt for it to echo. Skipped
    (not failed) when the model hasn't been downloaded on this machine.
    """
    from core.llm_phrasing import is_model_available
    if not is_model_available():
        print("SKIP: local model not downloaded -- nothing to verify here, see scripts/download_model.py")
        return

    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    result = run_pipeline_for_table(dataset_profile, "messy_msme_sample.csv", use_ai_phrasing=True)

    fuzzy_dup_findings = [f for f in result.findings if f.issue_type == "fuzzy_duplicate_rows"]
    assert len(fuzzy_dup_findings) >= 1
    finding = fuzzy_dup_findings[0]

    if finding.ai_phrased_description is not None:
        assert "fuzzy_duplicate_rows" not in finding.ai_phrased_description


# ---- Regression: severity classification for a low-percentage finding -------------

def test_low_percentage_structural_finding_classifies_and_displays_as_minor():
    """
    Regression-style test for the severity-classification path
    (core/fixlist.py's _severity_for_percentage): a structural finding
    whose percentage sits below both the critical and moderate cutoffs
    (calibrated or fixed-default, whichever this synthetic table's data
    triggers) must come out as "minor" -- and "displays as Minor" is
    verified by replicating app.py's exact grouping rule (group findings
    by finding.severity == tier), since that grouping is what actually
    decides which expander section a finding shows up under, without
    needing to drive Streamlit itself.
    """
    import pandas as pd
    from core.structure import check_structure
    from core.completeness import check_completeness
    from core.consistency import check_consistency
    from core.duplication import check_duplication
    from core.fixlist import generate_fix_list

    row_count = 40
    gst_numbers = [f"24GST{i:04d}Z1" for i in range(row_count)]
    gst_numbers[10] = gst_numbers[0]  # exactly one pair shares a GST -> a clearly low %
    dataframe = pd.DataFrame({
        "Business_Name": [f"Business {i}" for i in range(row_count)],
        "GST_Number": gst_numbers,
        "Notes": [None if i % 2 == 0 else "ok" for i in range(row_count)],  # a high-% contrast
    })
    column_types = {"Business_Name": "text", "GST_Number": "text", "Notes": "text"}

    completeness_result = check_completeness(dataframe)
    consistency_result = check_consistency(dataframe, column_types)
    duplication_result = check_duplication(dataframe, column_types)
    structure_result = check_structure(dataframe, raw_text=None)

    findings, critical_cutoff, moderate_cutoff = generate_fix_list(
        completeness_result, consistency_result, duplication_result, structure_result,
        row_count, table_name="t", use_ai_phrasing=False,
    )

    structural_findings = [f for f in findings if f.check_type == "structure"]
    assert len(structural_findings) >= 1
    finding = structural_findings[0]

    # The actual classification math -- this is what would catch a real
    # _severity_for_percentage bug.
    assert finding.percentage_affected < moderate_cutoff.value
    assert finding.severity == "minor"

    # "Displays as Minor" -- app.py's render_finding_list groups findings
    # by this exact comparison, tier by tier.
    minor_group = [f for f in findings if f.severity == "minor"]
    moderate_group = [f for f in findings if f.severity == "moderate"]
    critical_group = [f for f in findings if f.severity == "critical"]
    assert finding in minor_group
    assert finding not in moderate_group
    assert finding not in critical_group


# ---- Duplication performance: large tables use blocking (see core/duplication.py) --

def test_large_table_uses_bulk_cdist_and_matches_manual_computation():
    """
    core/duplication.py uses rapidfuzz.process.cdist (vectorized, in C)
    instead of a manual pair-by-pair Python loop for speed -- this test
    confirms that swap didn't change any actual similarity number, by
    comparing its output against fuzz.token_sort_ratio() called directly,
    pair by pair, the "obviously correct but slow" way, on a table just
    big enough to matter (150 rows, ~11,000 pairs -- still small enough
    to double-check by brute force in a test without being slow itself).
    """
    import pandas as pd
    from rapidfuzz import fuzz
    from core.duplication import check_duplication

    prefixes = ["Shree", "Om", "Jay", "New", "Modern", "National", "Krishna", "Laxmi"]
    cores = ["Traders", "Textiles", "Enterprises", "Hardware", "Foods", "Auto Parts"]
    names = [f"{prefixes[i % len(prefixes)]} {cores[(i * 3) % len(cores)]} {i}" for i in range(150)]
    dataframe = pd.DataFrame({"Business_Name": names, "City": ["Ahmedabad"] * len(names)})
    column_types = {"Business_Name": "text", "City": "text"}

    result = check_duplication(dataframe, column_types)

    # Brute-force recomputation of a handful of specific pairs, the slow
    # pair-by-pair way, to confirm cdist's numbers agree exactly.
    for pair in result.fuzzy_duplicate_pairs[:5]:
        expected = round(fuzz.token_sort_ratio(pair.value_a, pair.value_b), 2)
        assert pair.similarity_score == expected


def test_blocking_still_finds_planted_duplicates_on_a_large_table():
    """
    Above BLOCKING_ROW_COUNT_THRESHOLD rows, core/duplication.py groups
    rows into cheap "blocks" before comparing (see _blocking_key) instead
    of comparing every row against every other row. This test builds a
    table just over that threshold with a few planted near-duplicate
    pairs and confirms blocking doesn't cause them to be missed.
    """
    import pandas as pd
    from core.duplication import check_duplication, BLOCKING_ROW_COUNT_THRESHOLD

    prefixes = ["Shree", "Om", "Jay", "New", "Modern", "National", "Krishna", "Laxmi",
                "Bharat", "Royal", "Sona", "Star", "United", "City", "Prime", "Classic"]
    cores = ["Traders", "Textiles", "Enterprises", "Hardware", "Foods", "Auto Parts",
             "Garments", "Plastics", "Steel Works", "Spices", "Exports", "Electronics"]

    row_count = BLOCKING_ROW_COUNT_THRESHOLD + 50
    names = [f"{prefixes[i % len(prefixes)]} {cores[(i * 7) % len(cores)]} {i}" for i in range(row_count)]
    # Plant near-duplicate pairs at known positions -- same words, one
    # dropped letter, exactly the kind of typo this check exists to catch.
    names[10] = "Krishna Auto Parts"
    names[11] = "Krishna Auto Part"
    names[500] = "Om Sai Enterprises"
    names[501] = "Om Sai Enterprise"

    dataframe = pd.DataFrame({"Business_Name": names, "City": ["Ahmedabad"] * row_count})
    column_types = {"Business_Name": "text", "City": "text"}

    result = check_duplication(dataframe, column_types)
    matched_pairs = {(p.row_index_a, p.row_index_b) for p in result.fuzzy_duplicate_pairs}
    assert (10, 11) in matched_pairs
    assert (500, 501) in matched_pairs


# ---- Multi-table / multi-sheet support -------------------------------------

def test_multi_sheet_excel_loads_every_sheet_as_a_table():
    outcomes = load_all_tables(MULTI_SHEET_XLSX, "multi_sheet_msme_workbook.xlsx")
    # 4 sheets in the sample workbook: 3 good, 1 deliberately empty.
    assert len(outcomes) == 4
    succeeded = [o for o in outcomes if o.succeeded]
    failed = [o for o in outcomes if not o.succeeded]
    assert len(succeeded) == 3
    assert len(failed) == 1
    assert "Draft_Notes" in failed[0].table_name


def test_multi_table_pipeline_isolates_one_bad_sheet():
    """One malformed sheet (Draft_Notes, empty) must not prevent the
    other 3 sheets in the same workbook from being scored."""
    summary = run_multi_table_pipeline([MULTI_SHEET_XLSX], use_ai_phrasing=False)
    assert len(summary.table_results) == 3
    assert len(summary.table_failures) == 1
    assert 0 <= summary.overall_score <= 100
    # Every successfully-processed table's findings should be present in
    # the combined, cross-table fix list.
    assert len(summary.combined_findings) >= sum(len(r.findings) for r in summary.table_results)


def test_multiple_files_at_once():
    """Uploading the single CSV and the multi-sheet workbook together
    should process all of them as one combined batch."""
    summary = run_multi_table_pipeline([SINGLE_TABLE_CSV, MULTI_SHEET_XLSX], use_ai_phrasing=False)
    # 1 table from the CSV + 3 good sheets from the workbook = 4.
    assert len(summary.table_results) == 4
    assert len(summary.table_failures) == 1  # the empty Draft_Notes sheet


# ---- Calibration behavior ---------------------------------------------------

def test_duplicate_threshold_falls_back_on_pure_noise():
    """When similarity scores have no real high/low separation, the
    calibrated threshold must fall back to the fixed 90% default rather
    than inventing a low cutoff that would flag unrelated names."""
    noise_scores = [12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0]
    result = calibrate_duplicate_threshold(noise_scores)
    assert result.was_calibrated is False
    assert result.value == 90.0


def test_duplicate_threshold_calibrates_on_a_real_gap():
    """When there's a genuine bimodal split (a cluster of low scores, a
    cluster of high scores), calibration should trust it and land
    somewhere in the gap between the two clusters."""
    bimodal_scores = [12, 18, 22, 25, 30, 35, 15, 20, 28, 33, 92, 95, 97, 99, 93, 96]
    result = calibrate_duplicate_threshold(bimodal_scores)
    assert result.was_calibrated is True
    assert 35 < result.value < 92  # somewhere in the gap, not at either extreme


def test_severity_thresholds_fall_back_with_too_few_findings():
    result_critical, result_moderate = calibrate_severity_thresholds([5.0, 10.0, 15.0])
    assert result_critical.was_calibrated is False
    assert result_critical.value == 30.0
    assert result_moderate.was_calibrated is False
    assert result_moderate.value == 10.0


def test_calibration_is_deterministic():
    """The same input must always produce the same calibrated value --
    required so the same file, processed twice, scores identically."""
    scores = [12, 18, 22, 25, 30, 35, 15, 20, 28, 33, 92, 95, 97, 99, 93, 96]
    first_run = calibrate_duplicate_threshold(scores)
    second_run = calibrate_duplicate_threshold(scores)
    assert first_run == second_run


# ---- Determinism of the whole pipeline (same file, run twice) -------------

def test_same_file_processed_twice_gives_identical_results():
    """Hard project requirement: the same file, processed twice, must
    always produce the same score and the same fix list -- including the
    exact sentence text (this also guards against the round-robin
    template picker accidentally being process-global state again)."""
    def run_once():
        dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
        result = run_pipeline_for_table(dataset_profile, "messy_msme_sample.csv", use_ai_phrasing=False)
        return (
            result.scorecard.overall_score,
            [(f.severity, f.rule_based_description) for f in result.findings],
        )

    first_run = run_once()
    second_run = run_once()
    third_run = run_once()
    assert first_run == second_run == third_run


# ---- AI phrasing: verify the guaranteed fallback actually works -----------

def test_ai_phrasing_falls_back_cleanly_when_model_unavailable():
    """
    Forces core.llm_phrasing to behave as if the model file doesn't
    exist (by pointing MODEL_PATH at a path that doesn't exist and
    resetting its lazy-load cache), then confirms phrase_finding()
    returns None and Finding.display_description still falls back to
    the guaranteed rule_based_description. This is the test the project
    spec explicitly asks for: "verify this fallback actually works by
    testing with the model unavailable."
    """
    original_model_path = llm_phrasing.MODEL_PATH
    original_instance = llm_phrasing._model_instance
    original_attempted = llm_phrasing._model_load_attempted
    try:
        llm_phrasing.MODEL_PATH = os.path.join(REPO_ROOT, "models", "this-file-does-not-exist.gguf")
        llm_phrasing._model_instance = None
        llm_phrasing._model_load_attempted = False  # force a fresh load attempt

        from core.findings import Finding
        finding = Finding(
            table_name="t", column_name="Annual_Revenue", check_type="consistency",
            issue_type="inconsistent_format_numeric", percentage_affected=77.78,
            severity="critical", rule_based_description="The guaranteed template sentence.",
        )
        result = llm_phrasing.phrase_finding(finding)
        assert result is None
        assert finding.ai_phrased_description is None
        assert finding.display_description == "The guaranteed template sentence."
        assert finding.is_ai_phrased is False
    finally:
        # Restore real state so other tests in this file (which may run
        # after this one and DO want the real model, if present) aren't
        # affected by this test's simulated "model missing" condition.
        llm_phrasing.MODEL_PATH = original_model_path
        llm_phrasing._model_instance = original_instance
        llm_phrasing._model_load_attempted = original_attempted


def test_pipeline_works_end_to_end_with_ai_phrasing_disabled():
    """The whole tool must work completely normally with zero AI
    involvement -- this is the project's core promise, tested at the
    full-pipeline level, not just the llm_phrasing module in isolation."""
    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    result = run_pipeline_for_table(dataset_profile, "messy_msme_sample.csv", use_ai_phrasing=False)
    assert len(result.findings) > 0
    for finding in result.findings:
        assert finding.ai_phrased_description is None
        assert finding.display_description == finding.rule_based_description


# ---- Priority 1: preamble/title-row detection (core/ingestion.py) ---------

def _bytes_buffer(text: str, name: str):
    """Builds a Streamlit-uploader-shaped in-memory file (bytes + .name)
    from a text string, matching how load_dataset actually receives
    uploads in the app (not a path on disk)."""
    import io
    buffer = io.BytesIO(text.encode("utf-8"))
    buffer.name = name
    return buffer


def test_preamble_row_is_detected_and_skipped_on_csv():
    """A genuine Tally/POS-style export: a report-title line, a blank
    line, then the real header. The heuristic should find the clear
    width jump and skip both leading lines -- and DatasetProfile must
    report exactly how many, per the "never silent" requirement."""
    csv_text = (
        "Shree Balaji Traders - Sales Register\n"
        "\n"
        "Business_Name,GST_Number,City,Annual_Revenue\n"
        "Patel Hardware,24AAAAA0000A1Z5,Ahmedabad,1500000\n"
        "Shah Textiles,24BBBBB1111B1Z6,Surat,2200000\n"
        "Modi Foods,24CCCCC2222C1Z7,Vadodara,980000\n"
    )
    profile = load_dataset(_bytes_buffer(csv_text, "preamble.csv"), "preamble.csv")
    assert profile.skipped_preamble_rows == 2
    assert list(profile.dataframe.columns) == ["Business_Name", "GST_Number", "City", "Annual_Revenue"]
    assert profile.row_count == 3

    # The rest of the pipeline must work normally on the correctly-
    # parsed table -- and raw_text must be rebuilt from the real header
    # onward, or structure.py's ragged-row check would misfire on the
    # intentionally-skipped preamble line.
    result = run_pipeline_for_table(profile, "preamble.csv", use_ai_phrasing=False)
    assert 0 <= result.scorecard.overall_score <= 100
    ragged_row_findings = [f for f in result.findings if f.issue_type == "structural_issue" and "ragged" in f.rule_based_description.lower()]
    assert ragged_row_findings == []


def test_file_without_preamble_is_completely_unaffected():
    """No regression case: an ordinary file (header already on row 0)
    must behave exactly as before -- skipped_preamble_rows stays 0, and
    every existing assertion about messy_msme_sample.csv still holds."""
    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    assert dataset_profile.skipped_preamble_rows == 0
    assert dataset_profile.row_count == 20
    assert dataset_profile.column_count == 9


def test_narrow_table_does_not_trigger_preamble_detection():
    """Ambiguous case, per the guard-and-fallback design: a genuinely
    narrow (2-column) table gives the heuristic too little signal to
    trust, so it must fall back to "row 0 is the header" rather than
    guess -- exactly the same discipline core/calibration.py uses."""
    csv_text = "Name,Value\nA,1\nB,2\nC,3\n"
    profile = load_dataset(_bytes_buffer(csv_text, "narrow.csv"), "narrow.csv")
    assert profile.skipped_preamble_rows == 0
    assert list(profile.dataframe.columns) == ["Name", "Value"]


def test_preamble_row_is_detected_and_skipped_on_excel():
    """The same detection applies to Excel workbooks, not just CSV --
    Tally/POS Excel exports have the identical letterhead-then-header
    shape."""
    import io
    import pandas as pd

    data = pd.DataFrame({
        "Business_Name": ["Patel Hardware", "Shah Textiles", "Modi Foods"],
        "GST_Number": ["24AAAAA0000A1Z5", "24BBBBB1111B1Z6", "24CCCCC2222C1Z7"],
        "City": ["Ahmedabad", "Surat", "Vadodara"],
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        title_row = pd.DataFrame([["Shree Balaji Traders - Sales Register", None, None]])
        title_row.to_excel(writer, sheet_name="Sheet1", index=False, header=False, startrow=0)
        data.to_excel(writer, sheet_name="Sheet1", index=False, startrow=2)
    buffer.seek(0)
    buffer.name = "preamble.xlsx"

    profile = load_dataset(buffer, "preamble.xlsx")
    assert profile.skipped_preamble_rows == 2
    assert list(profile.dataframe.columns) == ["Business_Name", "GST_Number", "City"]
    assert profile.row_count == 3


# ---- Priority 2: encoding fallback (core/ingestion.py) --------------------

def test_non_utf8_csv_is_read_via_fallback_with_a_visible_warning():
    """A Windows-1252-encoded file (the realistic failure mode named in
    the project brief -- older/regional Windows exports) must not raise
    UnicodeDecodeError, and the fallback must be visible, never silent --
    a clean UTF-8 file must show no such warning at all."""
    import io

    text = (
        "Business_Name,City,Contact_Person\r\n"
        "Café Milano,Mumbai,André Fernandes\r\n"
        "Patel Hardware,Ahmedabad,Suresh Patel\r\n"
    )
    buffer = io.BytesIO(text.encode("windows-1252"))
    buffer.name = "non_utf8.csv"

    profile = load_dataset(buffer, "non_utf8.csv")
    assert profile.row_count == 2
    assert profile.encoding_warning is not None
    assert "UTF-8" in profile.encoding_warning
    # The accented business name must still come through readably, not
    # as a raw decode failure -- this is what makes the fallback worth
    # having at all, rather than just always using latin-1.
    assert "Caf" in profile.dataframe["Business_Name"].iloc[0]

    # Full pipeline must still run cleanly on the fallback-decoded table.
    result = run_pipeline_for_table(profile, "non_utf8.csv", use_ai_phrasing=False)
    assert 0 <= result.scorecard.overall_score <= 100


def test_clean_utf8_csv_gets_no_encoding_warning():
    dataset_profile = load_dataset(SINGLE_TABLE_CSV, "messy_msme_sample.csv")
    assert dataset_profile.encoding_warning is None


# ---- Regression: date-type inference false positives on non-date data -----
# Found via a real user-supplied sensor/IoT alerts log (not MSME business
# data): a bare zero-padded ID column ("0001" repeated) and a bare
# time-of-day column ("23:49:28", no date component) were both
# misclassified as COLUMN_TYPE_DATE, because core.ingestion._looks_like_date's
# pandas fallback silently accepts a 4-digit string as "the year" and a
# bare time as "today's date + that time". This cascaded into a bogus
# 100%-fuzzy-duplicate-rows finding (the identity-column picker fell back
# to a column that was wrongly excluded from being a real text column)
# and a meaningfully deflated overall score (62.0 vs the correct 87.0).

def test_zero_padded_id_column_is_not_misclassified_as_a_date():
    import io

    buffer = io.BytesIO(b"node_id,label\n0001,a\n0001,b\n0001,c\n0001,d\n0001,e\n")
    buffer.name = "ids.csv"
    profile = load_dataset(buffer, "ids.csv")
    assert profile.column_types["node_id"] == "numeric"


def test_bare_time_of_day_column_is_not_misclassified_as_a_date():
    """A bare time-of-day is not a date -- and treating it as one would
    make the inferred type depend on the wall-clock date the tool
    happens to run on, breaking the "same file twice -> same result"
    guarantee (pandas silently attaches TODAY's date to a bare time)."""
    import io

    buffer = io.BytesIO(
        b"timestamp,label\n23:49:28,a\n23:49:25,b\n23:49:22,c\n23:49:19,d\n23:49:01,e\n"
    )
    buffer.name = "times.csv"
    profile = load_dataset(buffer, "times.csv")
    assert profile.column_types["timestamp"] == "text"


def test_real_dates_still_classified_correctly_after_the_shape_guard():
    """The shape guard added for the two regressions above must not
    reject genuine dates that only match via the pandas fallback (not
    one of the explicit DATE_FORMAT_PATTERNS) -- e.g. a dot-separated
    or compact 8-digit date."""
    import io

    buffer = io.BytesIO(
        b"reg_date,label\n2020.01.15,a\n2019.03.22,b\n2021.06.30,c\n2018.11.05,d\n2022.02.28,e\n"
    )
    buffer.name = "dotted_dates.csv"
    profile = load_dataset(buffer, "dotted_dates.csv")
    assert profile.column_types["reg_date"] == "date"


# ---- Priority 3: degenerate file edge cases --------------------------------

def test_completely_empty_file_raises_a_clean_error():
    import io

    buffer = io.BytesIO(b"")
    buffer.name = "empty.csv"
    try:
        load_dataset(buffer, "empty.csv")
        assert False, "expected a ValueError for a completely empty file"
    except ValueError as error:
        assert "empty" in str(error).lower()


def test_header_only_file_raises_a_clean_error():
    """A file with column headers but zero data rows must not crash --
    it should raise the same kind of clean, friendly ValueError app.py
    already knows how to display, not an unhandled pandas exception."""
    import io

    buffer = io.BytesIO(b"Business_Name,City,Revenue\n")
    buffer.name = "header_only.csv"
    try:
        load_dataset(buffer, "header_only.csv")
        assert False, "expected a ValueError for a file with no data rows"
    except ValueError as error:
        assert "no data" in str(error).lower()


def test_single_column_file_runs_the_full_pipeline_without_crashing():
    """Every one of the four checks, scoring, and fix-list generation
    must handle a one-column table -- completeness/consistency work per-
    column regardless of column count, duplication's identity-column
    picker has only one candidate, and structure's checks don't assume
    more than one column exists."""
    import io

    buffer = io.BytesIO(b"Business_Name\nPatel Hardware\nShah Textiles\nPatel Hardware\n")
    buffer.name = "single_column.csv"
    profile = load_dataset(buffer, "single_column.csv")
    assert profile.column_count == 1

    result = run_pipeline_for_table(profile, "single_column.csv", use_ai_phrasing=False)
    assert 0 <= result.scorecard.overall_score <= 100
    # The planted exact duplicate ("Patel Hardware" twice) must still be
    # caught even with only one column to work with.
    duplication_findings = [f for f in result.findings if f.check_type == "duplication"]
    assert len(duplication_findings) >= 1


# ---- Tier 1: core/remediation.py -------------------------------------------

def _finding(issue_type: str, column_name: str = "Some_Column", percentage_affected: float = 50.0,
             example=None, rule_based_description: str = "A generic problem sentence.") -> Finding:
    """Builds a bare Finding directly -- these Tier 1 tests exercise
    core/remediation.py's own dispatch/guard logic in isolation, the
    same way core/llm_phrasing.py's tests build a Finding by hand
    (see test_ai_phrasing_falls_back_cleanly_when_model_unavailable)
    rather than running the whole pipeline just to get one."""
    return Finding(
        table_name="t", column_name=column_name, check_type="x", issue_type=issue_type,
        percentage_affected=percentage_affected, severity="moderate",
        rule_based_description=rule_based_description, example=example,
    )


def test_remediation_numeric_fix_strips_valid_grouping_but_leaves_ambiguous_values():
    """The spec's own headline example ("25,00,000" -> "2500000") must
    work, alongside Western grouping, while a grouping that doesn't fit
    either convention (e.g. "1,2,345") is left untouched, not guessed at."""
    import pandas as pd

    dataframe = pd.DataFrame({
        "Annual_Revenue": ["25,00,000", "2,500,000", "1,2,345", "980000"],
    })
    finding = _finding("inconsistent_format_numeric", column_name="Annual_Revenue")

    result = remediate_table(dataframe, [finding], [])

    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.action_type == "strip_numeric_formatting"
    assert action.count_affected == 2  # only the two valid-grouping values changed
    assert action.notes is not None and "1" in action.notes  # one ambiguous value flagged

    cleaned_values = result.cleaned_dataframe["Annual_Revenue"].tolist()
    assert "2500000" in cleaned_values
    assert "2500000" in cleaned_values  # both "25,00,000" and "2,500,000" clean to this
    assert "1,2,345" in cleaned_values  # left exactly as-is -- never guessed
    assert "980000" in cleaned_values   # already clean, unaffected
    assert not result.manual_review


def test_remediation_declines_a_fully_ambiguous_numeric_column():
    """When NOTHING in the column can be safely cleaned, the whole
    Finding must go to manual review rather than silently doing nothing."""
    import pandas as pd

    dataframe = pd.DataFrame({"Annual_Revenue": ["1,2,345", "9,87,6"]})
    finding = _finding("inconsistent_format_numeric", column_name="Annual_Revenue")

    result = remediate_table(dataframe, [finding], [])

    assert result.actions == []
    assert len(result.manual_review) == 1
    assert result.manual_review[0].finding is finding
    assert result.manual_review[0].reason  # non-empty, plain-language


def test_remediation_date_fix_normalizes_unambiguous_dates_but_leaves_ambiguous_ones():
    """03-04-2020 is the spec's own textbook ambiguous case (day/month
    could plausibly be either way round) -- it must be left untouched,
    while a date with a component > 12 is unambiguous and gets normalized."""
    import pandas as pd

    dataframe = pd.DataFrame({"Registration_Date": ["15-03-2019", "03-04-2020"]})
    finding = _finding("inconsistent_format_date", column_name="Registration_Date")

    result = remediate_table(dataframe, [finding], [])

    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.action_type == "normalize_date_format"
    assert action.count_affected == 1
    assert action.after_example == "2019-03-15"
    assert action.notes is not None and "1" in action.notes

    cleaned_values = result.cleaned_dataframe["Registration_Date"].tolist()
    assert "2019-03-15" in cleaned_values
    assert "03-04-2020" in cleaned_values  # ambiguous -- untouched


def test_remediation_leading_year_dates_with_small_day_and_month_are_not_flagged_ambiguous():
    """Regression test for a real bug found via live browser verification:
    a LEADING 4-digit year (already YYYY-MM-DD, e.g. "2021-01-10") must
    NOT be treated as ambiguous just because both remaining components
    happen to be <= 12 -- the year's position already tells us the
    month/day order, unlike a TRAILING year (e.g. "03-04-2020", which
    genuinely is ambiguous). Getting this wrong flooded the audit log
    with dozens of already-correct dates falsely reported as "ambiguous"."""
    import pandas as pd

    dataframe = pd.DataFrame({"Registration_Date": ["2021-01-10", "2020-07-22", "15-01-2020"]})
    finding = _finding("inconsistent_format_date", column_name="Registration_Date")

    result = remediate_table(dataframe, [finding], [])

    # Only the trailing-year value (unambiguous: 15 > 12) should have
    # changed -- the two already-ISO values must be left exactly as-is
    # AND must not be counted as ambiguous.
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.count_affected == 1
    assert action.after_example == "2020-01-15"
    assert action.notes is None  # nothing here was genuinely ambiguous

    cleaned_values = result.cleaned_dataframe["Registration_Date"].tolist()
    assert "2021-01-10" in cleaned_values
    assert "2020-07-22" in cleaned_values


def test_remediation_trims_whitespace_only_never_touches_casing():
    """Whitespace gets trimmed; a pure casing difference (deliberately
    out of scope -- there's no objectively "correct" case) is left alone."""
    import pandas as pd

    dataframe = pd.DataFrame({"City": [" Ahmedabad", "ahmedabad", "Surat"]})
    finding = _finding("inconsistent_format_text", column_name="City")

    result = remediate_table(dataframe, [finding], [])

    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.action_type == "trim_whitespace"
    assert action.count_affected == 1  # only the whitespace value changed

    cleaned_values = result.cleaned_dataframe["City"].tolist()
    assert "Ahmedabad" in cleaned_values  # trimmed
    assert "ahmedabad" in cleaned_values  # casing left exactly as-is


def test_remediation_declines_text_fix_when_only_casing_differs():
    """If a text Finding's inconsistency is entirely casing (no
    whitespace to trim), applying "the fix" would be a no-op -- this
    must be reported honestly as a decline, not a fix that did nothing."""
    import pandas as pd

    dataframe = pd.DataFrame({"City": ["Ahmedabad", "ahmedabad"]})
    finding = _finding("inconsistent_format_text", column_name="City")

    result = remediate_table(dataframe, [finding], [])

    assert result.actions == []
    assert len(result.manual_review) == 1
    assert result.manual_review[0].finding is finding


def test_remediation_removes_exact_duplicate_rows_reusing_given_indexes():
    """Must reuse the row indexes handed in (as duplication.py's own
    exact_duplicate_row_indexes would be) rather than recomputing them."""
    import pandas as pd

    dataframe = pd.DataFrame({
        "Business_Name": ["Krishna Enterprises", "Om Sai Distributors", "Krishna Enterprises"],
    })
    finding = _finding("exact_duplicate_rows", column_name="(whole row)")

    result = remediate_table(dataframe, [finding], exact_duplicate_row_indexes=[2])

    assert len(result.actions) == 1
    assert result.actions[0].action_type == "remove_exact_duplicate_rows"
    assert result.actions[0].count_affected == 1
    assert len(result.cleaned_dataframe) == 2
    assert "Krishna Enterprises" in result.cleaned_dataframe["Business_Name"].tolist()


def test_remediation_never_silently_drops_missing_data_fuzzy_or_structural_findings():
    """These three issue_types have NO safe automatic fix -- every one of
    them must land on the manual review list, never vanish."""
    import pandas as pd

    dataframe = pd.DataFrame({"Contact_Number": ["123", None]})
    findings = [
        _finding("missing_data", column_name="Contact_Number", percentage_affected=37.5),
        _finding("fuzzy_duplicate_rows", column_name="Business_Name", example="Shree Balaji Traders vs Shri Balaji Traders"),
        _finding("structural_issue", column_name="GST_Number", rule_based_description="Duplicate GST_Number found."),
    ]

    result = remediate_table(dataframe, findings, [])

    assert result.actions == []
    assert len(result.manual_review) == 3
    reviewed_findings = [item.finding for item in result.manual_review]
    assert reviewed_findings == findings  # same three, same order, none dropped
    assert all(item.reason for item in result.manual_review)  # every reason is non-empty


def test_remediation_never_mutates_the_caller_dataframe_or_touches_the_score():
    """The diagnostic score must stay identical whether or not Tier 1
    ever runs -- proven here by showing remediate_table works on a COPY,
    never the caller's own dataframe (the one the score was computed from)."""
    dataset_profile = load_dataset(TEST_DATASET_CSV, "test_dataset.csv")
    result = run_pipeline_for_table(dataset_profile, "test_dataset.csv", use_ai_phrasing=False)
    original_snapshot = result.dataframe.copy(deep=True)
    score_before = result.scorecard.overall_score

    remediate_table(result.dataframe, result.findings, result.duplication_result.exact_duplicate_row_indexes)

    assert result.dataframe.equals(original_snapshot)
    assert result.scorecard.overall_score == score_before


def test_remediation_matches_known_test_dataset_issues():
    """Runs Tier 1 against test_dataset.csv and checks it against this
    file's actual, hand-verified content: an exact duplicate row
    ("Krishna Enterprises", rows 8/9), Annual_Revenue's 5 comma-formatted
    values, and Contact_Number's missing data (37.5%, 15 of 40 rows)."""
    dataset_profile = load_dataset(TEST_DATASET_CSV, "test_dataset.csv")
    result = run_pipeline_for_table(dataset_profile, "test_dataset.csv", use_ai_phrasing=False)

    remediation = remediate_table(
        result.dataframe, result.findings, result.duplication_result.exact_duplicate_row_indexes,
    )

    # The exact duplicate row must actually be gone.
    assert len(remediation.cleaned_dataframe) == len(result.dataframe) - 1
    exact_dup_actions = [a for a in remediation.actions if a.action_type == "remove_exact_duplicate_rows"]
    assert len(exact_dup_actions) == 1
    assert exact_dup_actions[0].count_affected == 1

    # Annual_Revenue's comma-formatted values (5 of them) must be cleaned.
    revenue_actions = [a for a in remediation.actions if a.column_name == "Annual_Revenue"]
    assert len(revenue_actions) == 1
    assert revenue_actions[0].count_affected == 5
    assert "," not in revenue_actions[0].after_example

    # Contact_Number's missing data must be left on manual review, never auto-filled.
    contact_manual_review = [
        item for item in remediation.manual_review
        if item.finding.column_name == "Contact_Number" and item.finding.issue_type == "missing_data"
    ]
    assert len(contact_manual_review) == 1
    assert "37.5" in contact_manual_review[0].reason

    # Registration_Date: exactly 2 unambiguous DD-MM-YYYY values get
    # normalized ("15-03-2019", "20-12-2015" -- day > 12 rules out the
    # month reading); "05-11-2018" is genuinely ambiguous and must be
    # left alone. The other 37 rows are already YYYY-MM-DD and must NOT
    # be falsely flagged ambiguous just because month/day are both <= 12
    # (see test_remediation_leading_year_dates_with_small_day_and_month_are_not_flagged_ambiguous).
    date_actions = [a for a in remediation.actions if a.column_name == "Registration_Date"]
    assert len(date_actions) == 1
    assert date_actions[0].count_affected == 2
    assert date_actions[0].notes is not None and "1" in date_actions[0].notes


def test_downloaded_cleaned_csv_reloads_cleanly():
    """Definition-of-done requirement: the cleaned CSV must actually be
    valid when re-loaded, not just successfully generated."""
    import io

    dataset_profile = load_dataset(TEST_DATASET_CSV, "test_dataset.csv")
    result = run_pipeline_for_table(dataset_profile, "test_dataset.csv", use_ai_phrasing=False)
    remediation = remediate_table(
        result.dataframe, result.findings, result.duplication_result.exact_duplicate_row_indexes,
    )

    csv_bytes = remediation.cleaned_dataframe.to_csv(index=False).encode("utf-8")
    buffer = io.BytesIO(csv_bytes)
    buffer.name = "cleaned.csv"
    reloaded_profile = load_dataset(buffer, "cleaned.csv")

    assert reloaded_profile.row_count == len(remediation.cleaned_dataframe)
    assert reloaded_profile.column_count == len(remediation.cleaned_dataframe.columns)
    # The comma-formatted Annual_Revenue values must have actually been
    # written out cleaned, not just cleaned in memory.
    assert not reloaded_profile.dataframe["Annual_Revenue"].astype(str).str.contains(",").any()


# ---- Tier 1: core/chart_generation.py --------------------------------------

def test_chart_generation_makes_one_chart_per_column_with_matching_annotations():
    dataset_profile = load_dataset(TEST_DATASET_CSV, "test_dataset.csv")
    result = run_pipeline_for_table(dataset_profile, "test_dataset.csv", use_ai_phrasing=False)

    charts = generate_charts_for_table(result.dataframe, dataset_profile.column_types, result.findings)

    assert len(charts) == dataset_profile.column_count
    chart_by_column = {chart.column_name: chart for chart in charts}

    # Contact_Number has a missing_data Finding -- its chart must carry
    # an annotation built from that Finding's own percentage.
    contact_chart = chart_by_column["Contact_Number"]
    assert contact_chart.annotation is not None
    assert "37.5" in contact_chart.annotation

    # Annual_Revenue is numeric -- must get a distribution chart, not a
    # category-count chart.
    assert chart_by_column["Annual_Revenue"].chart_type == "numeric_distribution"
    assert chart_by_column["Registration_Date"].chart_type == "date_over_time"


# ---- Dimension 5: core/validity.py -----------------------------------------

def test_validity_flags_invalid_type_in_numeric_and_ignores_valid_values():
    """The spec's own headline examples ("unknown", "N/A", "high", "ask",
    "TBD", "??") in an otherwise-numeric column must all be flagged as a
    problem -- some as invalid_type_in_numeric (genuine garbage text: 'high',
    'ask', '??'), the rest as missing_value_representation ('unknown',
    'N/A', 'TBD' are all disguised-missing tokens -- see
    core/completeness.py's MISSING_VALUE_TOKENS, which validity.py reuses
    so the same cell is never double-flagged under both issue types).
    None of the genuinely valid numbers should be flagged either way."""
    import pandas as pd

    dataframe = pd.DataFrame({"Price": ["50000", "unknown", "N/A", "high", "ask", "TBD", "??", "62000", "71000"]})
    result = check_validity(dataframe, {"Price": "numeric"})

    invalid_candidates = [c for c in result.candidates if c.issue_type == "invalid_type_in_numeric"]
    assert len(invalid_candidates) == 1
    assert invalid_candidates[0].column_name == "Price"
    assert set(invalid_candidates[0].details["invalid_examples"]) == {"high", "ask", "??"}

    missing_repr_candidates = [c for c in result.candidates if c.issue_type == "missing_value_representation"]
    assert len(missing_repr_candidates) == 1
    assert set(missing_repr_candidates[0].details["tokens_found"]) == {"unknown", "N/A", "TBD"}


def test_validity_flags_invalid_type_in_date():
    import pandas as pd

    dataframe = pd.DataFrame({"Sale_Date": ["2020-01-15", "2021-03-02", "invalid", "not a date", "2019-11-30"]})
    result = check_validity(dataframe, {"Sale_Date": "date"})

    candidates = [c for c in result.candidates if c.issue_type == "invalid_type_in_date"]
    assert len(candidates) == 1
    assert result.per_column["Sale_Date"].invalid_count == 2


def test_validity_flags_categorical_typos_against_dominant_spelling():
    """The spec's own headline example: Diesel/diesel/Diesal/Dizel must
    all map to the dominant spelling 'Diesel'; BMW/bmw/B.M.W/BMV must
    all map to 'BMW'. A minority spelling that's genuinely unrelated
    (e.g. a real third category) must NOT be forced into either bucket."""
    import pandas as pd

    dataframe = pd.DataFrame({"Fuel_Type": [
        "Diesel", "Diesel", "Diesel", "Diesel", "Diesel", "Diesel", "diesel", "Diesal", "Dizel",
        "Petrol", "Petrol", "Petrol",
    ]})
    result = check_validity(dataframe, {"Fuel_Type": "text"})

    candidates = [c for c in result.candidates if c.issue_type == "categorical_inconsistency"]
    assert len(candidates) == 1
    canonical_map = candidates[0].details["canonical_map"]
    # "diesel" (lowercase, exact-case variant) is already caught by
    # consistency.py's own dominant-variant check, not this one -- but
    # it's still a MINORITY normalized form here too (count 1), so it's
    # fine either way; what matters is the genuine misspellings resolve
    # to the real dominant spelling, and Petrol (a real category) never
    # gets touched.
    assert canonical_map["diesal"]["canonical"] == "Diesel"
    assert canonical_map["dizel"]["canonical"] == "Diesel"
    assert "petrol" not in canonical_map


def test_validity_declines_categorical_check_on_high_cardinality_free_text():
    """A genuine free-text column (every value essentially unique, like a
    Business_Name column) must never trigger categorical_inconsistency --
    there's no real 'canonical form' to converge on, and fuzzy-matching
    it would just invent false mappings between unrelated businesses."""
    import pandas as pd

    names = [f"Business {i} Traders" for i in range(30)]
    dataframe = pd.DataFrame({"Business_Name": names})
    result = check_validity(dataframe, {"Business_Name": "text"})

    assert not [c for c in result.candidates if c.issue_type == "categorical_inconsistency"]


def test_validity_flags_domain_outlier_for_named_rule_doors():
    """Doors is a small fixed legal set {2,3,4,5} -- 9 doors is not a
    plausible car, and must be flagged with the exact set in details."""
    import pandas as pd

    dataframe = pd.DataFrame({"Doors": ["2", "4", "4", "5", "9"]})
    result = check_validity(dataframe, {"Doors": "numeric"})

    candidates = [c for c in result.candidates if c.issue_type == "domain_outlier"]
    assert len(candidates) == 1
    assert candidates[0].details["allowed_set"] == [2, 3, 4, 5]


def test_validity_flags_negative_mileage_and_impossible_year():
    import pandas as pd

    dataframe = pd.DataFrame({
        "Mileage": ["45000", "-500", "62000", "71000", "58000", "39000", "84000", "12000"],
        "Year": ["2015", "2018", "2050", "2012", "2020", "2016", "2019", "2011"],
    })
    result = check_validity(dataframe, {"Mileage": "numeric", "Year": "numeric"})

    mileage_candidates = [c for c in result.candidates if c.issue_type == "domain_outlier" and c.column_name == "Mileage"]
    assert len(mileage_candidates) == 1
    assert mileage_candidates[0].details["lower_bound"] == 0.0

    year_candidates = [c for c in result.candidates if c.issue_type == "domain_outlier" and c.column_name == "Year"]
    assert len(year_candidates) == 1  # 2050 is outside [1900, current_year+1]


def test_validity_flags_missing_value_representation_tokens():
    import pandas as pd

    dataframe = pd.DataFrame({"Owner_Notes": ["Good condition", "N/A", "Unknown", "Fine", "null"]})
    result = check_validity(dataframe, {"Owner_Notes": "text"})

    candidates = [c for c in result.candidates if c.issue_type == "missing_value_representation"]
    assert len(candidates) == 1
    assert set(candidates[0].details["tokens_found"]) == {"N/A", "Unknown", "null"}


def test_completeness_treats_disguised_missing_tokens_as_missing():
    """core/completeness.py's own missing definition must ALSO count
    these placeholder tokens -- not just a true blank cell -- per this
    feature's own requirement that completeness scoring reflect them."""
    import pandas as pd

    dataframe = pd.DataFrame({"Notes": ["Good", "N/A", "", "Fine", "unknown"]})
    result = check_completeness(dataframe)

    assert result.per_column["Notes"].missing_count == 3  # "N/A", "", "unknown"


def test_scoring_backward_compatible_without_validity_result():
    """build_scorecard called without a validity_result (every caller
    that predates this feature) must use the ORIGINAL four-weight
    formula and leave validity_score as None -- byte-identical to
    before the Validity dimension existed."""
    dataset_profile = load_dataset(TEST_DATASET_CSV, "test_dataset.csv")
    result = run_pipeline_for_table(dataset_profile, "test_dataset.csv", use_ai_phrasing=False)

    from core.completeness import check_completeness as _cc
    from core.consistency import check_consistency as _co
    from core.duplication import check_duplication as _du
    from core.structure import check_structure as _st

    completeness_result = _cc(dataset_profile.dataframe)
    consistency_result = _co(dataset_profile.dataframe, dataset_profile.column_types)
    duplication_result = _du(dataset_profile.dataframe, dataset_profile.column_types)
    structure_result = _st(dataset_profile.dataframe, dataset_profile.raw_text)

    legacy_scorecard = build_scorecard(completeness_result, consistency_result, duplication_result, structure_result)
    assert legacy_scorecard.validity_score is None
    for column_score in legacy_scorecard.per_column_scores.values():
        assert column_score.validity_score is None

    # And WITH a validity_result, the pipeline's own scorecard must carry
    # a real validity_score.
    assert result.scorecard.validity_score is not None


# ---- Tier 1 remediation: new issue types + CleaningConfig ------------------

def test_remediation_coerces_invalid_numeric_to_blank_never_guesses():
    import pandas as pd

    dataframe = pd.DataFrame({"Price": ["50000", "unknown", "62000"]})
    finding = _finding("invalid_type_in_numeric", column_name="Price")

    result = remediate_table(dataframe, [finding], [])

    assert len(result.actions) == 1
    assert result.actions[0].action_type == "coerce_invalid_numeric"
    assert result.actions[0].count_affected == 1
    cleaned = result.cleaned_dataframe["Price"]
    assert cleaned.iloc[0] == "50000" and cleaned.iloc[2] == "62000"
    assert pd.isna(cleaned.iloc[1])  # the invalid cell is now blank, never guessed at
    assert dataframe["Price"].tolist() == ["50000", "unknown", "62000"]  # original untouched


def test_remediation_coerces_invalid_date_to_blank():
    import pandas as pd

    dataframe = pd.DataFrame({"Sale_Date": ["2020-01-15", "invalid", "2019-11-30"]})
    finding = _finding("invalid_type_in_date", column_name="Sale_Date")

    result = remediate_table(dataframe, [finding], [])

    assert len(result.actions) == 1
    assert result.actions[0].action_type == "coerce_invalid_date"
    assert result.actions[0].count_affected == 1


def test_remediation_normalizes_missing_token_to_blank_unconditionally():
    """This fix needs no CleaningConfig gate -- it's always safe."""
    import pandas as pd

    dataframe = pd.DataFrame({"Notes": ["Good", "N/A", "Fine", "unknown"]})
    finding = _finding("missing_value_representation", column_name="Notes")

    result = remediate_table(dataframe, [finding], [], config=CONSERVATIVE)

    assert len(result.actions) == 1
    assert result.actions[0].action_type == "normalize_missing_token"
    assert result.actions[0].count_affected == 2
    cleaned = result.cleaned_dataframe["Notes"]
    assert pd.isna(cleaned.iloc[1]) and pd.isna(cleaned.iloc[3])


def test_remediation_categorical_standardization_respects_confidence_threshold():
    """The Finding's canonical_map carries per-mapping match scores from
    validity.py's own detection pass. remediation.py must apply ONLY the
    mappings whose score clears THIS run's config.categorical_fuzzy_threshold
    (CONSERVATIVE's default is 88) -- proving config, not a hardcoded
    constant, governs what gets auto-applied, and that a below-threshold
    match is skipped (noted) rather than silently applied or dropped."""
    import pandas as pd

    dataframe = pd.DataFrame({"Fuel_Type": ["Diesel"] * 8 + ["Diesal", "Dizel"]})
    finding = _finding("categorical_inconsistency", column_name="Fuel_Type")
    finding.details = {"canonical_map": {
        "diesal": {"canonical": "Diesel", "score": 92.0},  # clears CONSERVATIVE's 88 floor
        "dizel": {"canonical": "Diesel", "score": 85.0},   # does NOT clear it
    }}

    result = remediate_table(dataframe.copy(), [finding], [], config=CONSERVATIVE)
    action = result.actions[0]
    assert action.action_type == "standardize_categorical_spelling"
    assert action.count_affected == 1  # only "Diesal" (92) was trusted
    assert action.notes is not None  # the 85-score "Dizel" match was skipped and noted
    assert "Diesel" in result.cleaned_dataframe["Fuel_Type"].tolist()
    assert "Dizel" in result.cleaned_dataframe["Fuel_Type"].tolist()  # left unchanged, not dropped


def test_remediation_domain_outlier_default_blanks_clip_mode_clips():
    import pandas as pd

    dataframe = pd.DataFrame({"Doors": ["2", "4", "9"]})
    finding = _finding("domain_outlier", column_name="Doors")
    finding.details = {"allowed_set": [2, 3, 4, 5]}

    default_result = remediate_table(dataframe.copy(), [finding], [], config=CONSERVATIVE)
    assert default_result.actions[0].action_type == "neutralize_domain_outlier"
    assert pd.isna(default_result.cleaned_dataframe["Doors"].iloc[2])

    dataframe2 = pd.DataFrame({"Mileage": ["45000", "-500"]})
    finding2 = _finding("domain_outlier", column_name="Mileage")
    finding2.details = {"lower_bound": 0.0, "upper_bound": None}

    clip_config = CleaningConfig(outlier_action="clip")
    clip_result = remediate_table(dataframe2.copy(), [finding2], [], config=clip_config)
    assert clip_result.actions[0].action_type == "clip_domain_outlier"
    assert clip_result.cleaned_dataframe["Mileage"].iloc[1] == "0"


def test_remediation_missing_data_strategy_drop_row_and_constant():
    import pandas as pd

    dataframe = pd.DataFrame({"City": ["Ahmedabad", None, "Surat"]})
    finding = _finding("missing_data", column_name="City", percentage_affected=33.3)

    # Default (conservative) -- unchanged, still manual review.
    default_result = remediate_table(dataframe.copy(), [finding], [])
    assert default_result.actions == []
    assert len(default_result.manual_review) == 1

    # drop_row strategy actually removes the row.
    drop_config = CleaningConfig(missing_strategy=MISSING_STRATEGY_DROP_ROW)
    drop_result = remediate_table(dataframe.copy(), [finding], [], config=drop_config)
    assert len(drop_result.cleaned_dataframe) == 2
    assert drop_result.actions[0].action_type == "drop_rows_missing_value"

    # constant strategy fills instead.
    constant_config = CleaningConfig(missing_strategy=MISSING_STRATEGY_CONSTANT, missing_constant="Unknown City")
    constant_result = remediate_table(dataframe.copy(), [finding], [], config=constant_config)
    assert "Unknown City" in constant_result.cleaned_dataframe["City"].tolist()
    assert len(constant_result.cleaned_dataframe) == 3


# ---- End-to-end: messy car-price dataset (this feature's own success criteria) --

def test_end_to_end_messy_car_price_dataset():
    """
    A reduced version of the messy car-price dataset this feature was
    built for. Exercises every success criterion in one place: exact
    duplicates removed, dirty numeric strings coerced to blank,
    categorical variants standardized, impossible values neutralized,
    genuinely ambiguous cases left for manual review with a clear
    reason, a complete audit trail, and the original score computed
    only from the raw (unmodified) data.
    """
    import pandas as pd

    rows = [
        {"Brand": "BMW", "Fuel_Type": "Diesel", "Year": "2018", "Mileage": "45000", "Price": "500000", "Doors": "4", "Owner_Count": "1"},
        {"Brand": "BMW", "Fuel_Type": "diesel", "Year": "2016", "Mileage": "62000", "Price": "420000", "Doors": "4", "Owner_Count": "2"},
        {"Brand": "B.M.W", "Fuel_Type": "Diesal", "Year": "2019", "Mileage": "31000", "Price": "610000", "Doors": "4", "Owner_Count": "1"},
        {"Brand": "Toyota", "Fuel_Type": "Petrol", "Year": "2020", "Mileage": "18000", "Price": "750000", "Doors": "4", "Owner_Count": "1"},
        {"Brand": "Toyota", "Fuel_Type": "Petrol", "Year": "2017", "Mileage": "ask", "Price": "680000", "Doors": "5", "Owner_Count": "1"},
        {"Brand": "Honda", "Fuel_Type": "Petrol", "Year": "2050", "Mileage": "22000", "Price": "N/A", "Doors": "4", "Owner_Count": "1"},
        {"Brand": "Honda", "Fuel_Type": "Petrol", "Year": "2015", "Mileage": "-500", "Price": "390000", "Doors": "9", "Owner_Count": "1"},
        # exact duplicate of row index 3 (Toyota Petrol 2020 18000 750000 4 1)
        {"Brand": "Toyota", "Fuel_Type": "Petrol", "Year": "2020", "Mileage": "18000", "Price": "750000", "Doors": "4", "Owner_Count": "1"},
    ]
    dataframe = pd.DataFrame(rows)

    column_types = {
        "Brand": "text", "Fuel_Type": "text", "Year": "numeric", "Mileage": "numeric",
        "Price": "numeric", "Doors": "numeric", "Owner_Count": "numeric",
    }
    from core.completeness import check_completeness as _cc
    from core.consistency import check_consistency as _co
    from core.duplication import check_duplication as _du
    from core.structure import check_structure as _st

    completeness_result = _cc(dataframe)
    consistency_result = _co(dataframe, column_types)
    duplication_result = _du(dataframe, column_types)
    structure_result = _st(dataframe)
    validity_result = check_validity(dataframe, column_types)

    from core.scoring import build_scorecard as _build_scorecard
    from core.fixlist import generate_fix_list as _generate_fix_list

    scorecard = _build_scorecard(completeness_result, consistency_result, duplication_result, structure_result, validity_result)
    findings, _crit, _mod = _generate_fix_list(
        completeness_result, consistency_result, duplication_result, structure_result,
        len(dataframe), table_name="car_price", use_ai_phrasing=False, validity_result=validity_result,
    )
    original_score = scorecard.overall_score

    remediation = remediate_table(dataframe, findings, duplication_result.exact_duplicate_row_indexes, config=STANDARD)

    # The score computed above must never change because remediation ran.
    assert scorecard.overall_score == original_score
    assert dataframe.equals(pd.DataFrame(rows))  # original dataframe untouched

    # Exact duplicate removed.
    assert len(remediation.cleaned_dataframe) == len(rows) - 1

    # Dirty numeric strings coerced to blank, never guessed at.
    mileage_actions = [a for a in remediation.actions if a.column_name == "Mileage" and a.action_type == "coerce_invalid_numeric"]
    assert len(mileage_actions) == 1

    # Categorical standardization: BMW / Diesel variants converge.
    brand_actions = [a for a in remediation.actions if a.column_name == "Brand"]
    fuel_actions = [a for a in remediation.actions if a.column_name == "Fuel_Type" and a.action_type == "standardize_categorical_spelling"]
    assert brand_actions or fuel_actions  # at least one categorical column got standardized at Standard intensity

    # Impossible values (2050 year, negative mileage, 9 doors) flagged/neutralized.
    outlier_actions = [a for a in remediation.actions if a.action_type in ("neutralize_domain_outlier", "clip_domain_outlier")]
    assert outlier_actions

    # "N/A" price normalized to blank.
    price_actions = [a for a in remediation.actions if a.column_name == "Price" and a.action_type == "normalize_missing_token"]
    assert len(price_actions) == 1

    # Nothing silently dropped: every Finding is in exactly one place.
    acted_on_findings = {id(action.source_finding) for action in remediation.actions}
    reviewed_findings = {id(item.finding) for item in remediation.manual_review}
    for finding in findings:
        assert id(finding) in acted_on_findings or id(finding) in reviewed_findings
    for item in remediation.manual_review:
        assert item.reason  # every manual-review reason is non-empty, plain language


# ---- Phase 4: false-positive fixes on transactional/order data ------------

def test_repeating_customer_id_in_order_data_is_not_a_structural_issue():
    """The classic false positive this phase fixes: Customer_ID repeating
    across many different orders (normal transactional data) must NOT be
    flagged as 'should be a unique identifier' -- only a real per-row
    transaction key (Order_ID) should be."""
    import pandas as pd
    from core.structure import check_structure

    dataframe = pd.DataFrame({
        "Order_ID": ["ORD-001", "ORD-002", "ORD-003", "ORD-004"],
        "Customer_ID": ["CUST-100", "CUST-100", "CUST-100", "CUST-101"],
        "Customer_Name": ["Amit Shah", "Amit Shah", "Amit Shah", "Priya Rao"],
    })
    result = check_structure(dataframe)

    issue_columns = {issue.column_name for issue in result.issues}
    assert "Customer_ID" not in issue_columns
    assert "Customer_Name" not in issue_columns


def test_duplicate_order_id_still_flagged_as_structural_issue():
    """The fix must not blunt real detection -- a genuinely repeating
    transaction key is still a structural problem."""
    import pandas as pd
    from core.structure import check_structure

    dataframe = pd.DataFrame({"Order_ID": ["ORD-001", "ORD-002", "ORD-001"], "Amount": ["100", "200", "300"]})
    result = check_structure(dataframe)

    issue_columns = {issue.column_name for issue in result.issues}
    assert "Order_ID" in issue_columns


def test_repeat_customer_across_orders_is_not_a_fuzzy_duplicate():
    """The other half of the same false positive: the SAME customer name
    appearing on several different orders (different amounts/dates) must
    not be reported as a near-duplicate RECORD -- it's a normal repeat
    customer, not two copies of one order."""
    import pandas as pd
    from core.duplication import check_duplication

    dataframe = pd.DataFrame({
        "Customer_Name": ["Amit Shah", "Amit Shah", "Amit Shah", "Priya Rao", "Priya Rao", "Neha Verma"],
        "Order_Amount": ["1000", "2500", "750", "3000", "1200", "500"],
        "Order_Date": ["2024-01-05", "2024-02-11", "2024-03-02", "2024-01-19", "2024-02-28", "2024-01-30"],
    })
    result = check_duplication(dataframe, {"Customer_Name": "text", "Order_Amount": "numeric", "Order_Date": "date"})

    assert result.exact_duplicate_row_indexes == []
    assert result.fuzzy_duplicate_pairs == []


def test_near_duplicate_record_with_matching_row_is_still_caught():
    """The fix must not blunt real detection -- two rows with a genuine
    SPELLING VARIANT on the identity column AND a matching rest-of-row
    are still exactly the near-duplicate scenario this check exists for."""
    import pandas as pd
    from core.duplication import check_duplication

    dataframe = pd.DataFrame({
        "Business_Name": [
            "Shree Ganesh Traders", "Shri Ganesh Traders", "Om Sai Distributors",
            "Krishna Enterprises", "Patel Hardware", "Singh Textiles", "Sharma Foods",
        ],
        "City": ["Ahmedabad", "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Anand"],
    })
    result = check_duplication(dataframe, {"Business_Name": "text", "City": "text"})

    assert len(result.fuzzy_duplicate_pairs) >= 1


# ---- Phase 2: new domain rules (Rating, Discount) --------------------------

def test_validity_flags_rating_and_discount_out_of_bounds():
    import pandas as pd

    dataframe = pd.DataFrame({
        "Rating": ["4", "5", "3", "15"],
        "Discount_Percent": ["10", "20", "150", "5"],
    })
    result = check_validity(dataframe, {"Rating": "numeric", "Discount_Percent": "numeric"})

    rating_candidates = [c for c in result.candidates if c.column_name == "Rating"]
    assert len(rating_candidates) == 1 and rating_candidates[0].issue_type == "domain_outlier"

    discount_candidates = [c for c in result.candidates if c.column_name == "Discount_Percent"]
    assert len(discount_candidates) == 1 and discount_candidates[0].issue_type == "domain_outlier"


def test_validity_known_alias_dictionary_maps_abbreviations_and_city_variants():
    """DC/NB/city-name aliases are curated domain knowledge, applied even
    though they share almost no characters with their canonical form (so
    no similarity score would ever safely link them)."""
    import pandas as pd

    dataframe = pd.DataFrame({
        "Payment_Method": [
            "Debit Card", "Debit Card", "Debit Card", "Debit Card", "DC", "DC",
            "Net Banking", "Net Banking", "Net Banking", "NB",
        ],
        "City": ["Delhi", "Delhi", "Delhi", "Delhi", "Dilli", "Dilli", "Chennai", "Chennai", "Chennai", "Madras"],
    })
    result = check_validity(dataframe, {"Payment_Method": "text", "City": "text"})

    payment_candidate = next(c for c in result.candidates if c.column_name == "Payment_Method")
    payment_map = payment_candidate.details["canonical_map"]
    assert payment_map[categorical_key("DC")]["canonical"] == "Debit Card"
    assert payment_map[categorical_key("NB")]["canonical"] == "Net Banking"

    city_candidate = next(c for c in result.candidates if c.column_name == "City")
    city_map = city_candidate.details["canonical_map"]
    assert city_map[categorical_key("Dilli")]["canonical"] == "Delhi"
    assert city_map[categorical_key("Madras")]["canonical"] == "Chennai"


def test_validity_new_delhi_only_folds_into_delhi_when_delhi_is_dominant():
    """The conditional alias must NOT fire when its target isn't actually
    this column's own established spelling -- avoids silently renaming a
    genuinely distinct place."""
    import pandas as pd

    dataframe = pd.DataFrame({
        "City": ["New Delhi", "New Delhi", "New Delhi", "Mumbai", "Pune"],
    })
    result = check_validity(dataframe, {"City": "text"})
    city_candidates = [c for c in result.candidates if c.column_name == "City"]
    if city_candidates:
        canonical_map = city_candidates[0].details.get("canonical_map", {})
        assert categorical_key("New Delhi") not in canonical_map


def test_remediation_normalizes_month_name_dates_unambiguously():
    """DD-Mon-YYYY (and similar) formats are unambiguous by construction
    -- a month NAME resolves day/month order completely, unlike an
    all-numeric date -- so these must normalize, not land on manual review."""
    import pandas as pd
    from core.findings import Finding

    dataframe = pd.DataFrame({"Order_Date": ["15-Jan-2020", "3 Feb 2021", "20-12-2015", "2020-05-01"]})
    finding = Finding(
        table_name="t", issue_type="inconsistent_format_date", column_name="Order_Date", check_type="consistency",
        severity="minor", percentage_affected=75.0, example="15-Jan-2020",
        rule_based_description="Mixed date formats.",
    )
    actions, manual_review = [], []
    _apply_or_decline_date(dataframe, finding, actions, manual_review)

    assert len(actions) == 1
    assert dataframe.loc[0, "Order_Date"] == "2020-01-15"
    assert dataframe.loc[1, "Order_Date"] == "2021-02-03"
    assert dataframe.loc[2, "Order_Date"] == "2015-12-20"


def test_validity_flags_extreme_quantity_via_iqr_but_not_wide_legitimate_revenue():
    """Quantity/count-shaped columns get an extra statistical extreme-
    value check (999/5000 among mostly single-digit orders is a strong
    error signal); Revenue/Amount-shaped columns deliberately do NOT, so
    a company with legitimately much higher revenue than its peers isn't
    misflagged as a data error."""
    import pandas as pd

    quantity_df = pd.DataFrame({
        "Quantity": ["2", "3", "1", "4", "5", "6", "7", "8", "2", "3", "999"],
    })
    quantity_result = check_validity(quantity_df, {"Quantity": "numeric"})
    quantity_candidates = [c for c in quantity_result.candidates if c.issue_type == "domain_outlier"]
    assert len(quantity_candidates) == 1

    revenue_df = pd.DataFrame({
        "Annual_Revenue": ["500000", "600000", "550000", "480000", "5200000", "530000", "610000", "490000"],
    })
    revenue_result = check_validity(revenue_df, {"Annual_Revenue": "numeric"})
    revenue_candidates = [c for c in revenue_result.candidates if c.issue_type == "domain_outlier"]
    assert len(revenue_candidates) == 0


def test_validity_flags_residual_typo_in_high_cardinality_categorical_column():
    """A column with many legitimate, genuinely-repeated categories (real
    cities) plus one close-typo variant must still be caught -- the
    cardinality guard protects genuine free text (nothing repeats), not
    a categorical column that merely has several real categories."""
    import pandas as pd

    dataframe = pd.DataFrame({
        "City": [
            "Chennai", "Chennai", "Chennai", "Chennnai",
            "Mumbai", "Mumbai", "Bangalore", "Bangalore", "Delhi", "Delhi",
        ],
    })
    result = check_validity(dataframe, {"City": "text"})
    candidate = next(c for c in result.candidates if c.column_name == "City")
    canonical_map = candidate.details["canonical_map"]
    assert canonical_map[categorical_key("Chennnai")]["canonical"] == "Chennai"
    assert canonical_map[categorical_key("Chennnai")]["score"] >= 88.0


def test_validity_flags_customer_rating_outside_one_to_five():
    """Customer_Rating is unambiguously a 1-5 scale by name -- narrower
    and more specific than the generic 0-10 'rating' catch-all."""
    import pandas as pd

    dataframe = pd.DataFrame({"Customer_Rating": ["4", "5", "1", "3", "9"]})
    result = check_validity(dataframe, {"Customer_Rating": "numeric"})
    candidate = next(c for c in result.candidates if c.column_name == "Customer_Rating")
    assert candidate.issue_type == "domain_outlier"
    assert candidate.details["lower_bound"] == 1.0
    assert candidate.details["upper_bound"] == 5.0


def test_validity_flags_zero_quantity_and_remediation_blanks_it():
    """Quantity <= 0 (not just < 0) is invalid -- an order line can't
    have zero items -- and remediation must reuse this exact bound."""
    import pandas as pd
    from core.findings import Finding

    dataframe = pd.DataFrame({"Quantity": ["2", "3", "1", "4", "5", "6", "7", "0", "-1"]})
    result = check_validity(dataframe, {"Quantity": "numeric"})
    candidate = next(c for c in result.candidates if c.column_name == "Quantity")
    assert candidate.details["min_exclusive"] is True

    finding = Finding(
        table_name="t", issue_type="domain_outlier", column_name="Quantity", check_type="validity",
        severity="critical", percentage_affected=candidate.percentage_affected, example=candidate.example,
        rule_based_description="Quantity out of range.", details=candidate.details,
    )
    remediation = remediate_table(dataframe, [finding], [])
    cleaned = remediation.cleaned_dataframe["Quantity"]
    assert pd.isna(cleaned.iloc[7])  # the "0"
    assert pd.isna(cleaned.iloc[8])  # the "-1"
    assert cleaned.iloc[0] == "2"    # untouched


def test_validity_flags_extreme_total_amount_via_iqr():
    """Total_Amount (unlike Revenue) is transaction-scoped within one
    file and gets the IQR extreme-outlier check too -- one wildly
    inflated amount among consistent order totals is a strong error
    signal here, not legitimate wide business variance."""
    import pandas as pd

    dataframe = pd.DataFrame({
        "Total_Amount": ["1000", "1200", "950", "1100", "1050", "990", "1150", "999999"],
    })
    result = check_validity(dataframe, {"Total_Amount": "numeric"})
    candidates = [c for c in result.candidates if c.issue_type == "domain_outlier"]
    assert len(candidates) == 1


# ---- Scoring/severity/uniqueness overhaul: false-Critical & inverted-severity fixes ----

def test_empty_optional_free_text_column_is_capped_at_minor_not_critical():
    """A 100%-empty Notes/Comments/Remarks/Description/Feedback column is
    the single highest percentage a table can produce -- pure percentage
    calibration would make it look like the worst problem in the file.
    It must be capped at minor, never moderate or critical."""
    import pandas as pd
    from core.consistency import check_consistency
    from core.duplication import check_duplication
    from core.structure import check_structure
    from core.fixlist import generate_fix_list

    row_count = 20
    dataframe = pd.DataFrame({
        "Order_ID": [f"ORD-{i:04d}" for i in range(row_count)],
        "Amount": [100 + i for i in range(row_count)],
        "Notes": [None] * row_count,
        "Comments": [""] * row_count,
        "Remarks": [None] * row_count,
        "Description": [None] * row_count,
        "Additional_Info": [None] * row_count,
        "Feedback": [None] * row_count,
    })
    column_types = {
        "Order_ID": "text", "Amount": "numeric", "Notes": "unknown", "Comments": "unknown",
        "Remarks": "unknown", "Description": "unknown", "Additional_Info": "unknown", "Feedback": "unknown",
    }

    completeness_result = check_completeness(dataframe)
    consistency_result = check_consistency(dataframe, column_types)
    duplication_result = check_duplication(dataframe, column_types)
    structure_result = check_structure(dataframe)

    findings, _crit, _mod = generate_fix_list(
        completeness_result, consistency_result, duplication_result, structure_result,
        row_count, table_name="t", use_ai_phrasing=False, column_types=column_types,
    )

    optional_column_names = {"Notes", "Comments", "Remarks", "Description", "Additional_Info", "Feedback"}
    optional_findings = [f for f in findings if f.column_name in optional_column_names]
    assert len(optional_findings) == len(optional_column_names)
    for finding in optional_findings:
        assert finding.severity == "minor", f"{finding.column_name} was {finding.severity}, expected minor"


def test_mandatory_named_empty_column_is_not_suppressed():
    """A column that names itself as required (Mandatory_Notes,
    Required_Comment) must NOT get the optional-free-text cap, even
    though it also matches the free-text hint words."""
    from core.fixlist import _is_optional_free_text_column

    assert _is_optional_free_text_column("Notes", "text") is True
    assert _is_optional_free_text_column("Notes", "unknown") is True
    assert _is_optional_free_text_column("Mandatory_Notes", "text") is False
    assert _is_optional_free_text_column("Required_Comment", "text") is False
    assert _is_optional_free_text_column("Amount", "numeric") is False  # not a free-text hint at all


def test_repeating_patient_name_with_stable_attributes_is_not_a_duplicate():
    """The exact scenario this fix targets: Patient_042 visits several
    times, and its OWN attributes (age, standard fee) are naturally the
    same across visits -- that must not be mistaken for a duplicated
    appointment record. Appointment_ID (the real per-row key) differs
    every time."""
    import pandas as pd
    from core.duplication import check_duplication

    dataframe = pd.DataFrame({
        "Appointment_ID": [f"APT-{i:04d}" for i in range(10)],
        "Patient_Name": (["Patient_042"] * 3) + (["Patient_017"] * 3) + (["Patient_099"] * 4),
        "Doctor": ["Dr. Sharma"] * 8 + ["Dr. Gupta"] * 2,
        "Status": ["Completed"] * 9 + ["Cancelled"],
        "Age": [45, 45, 45, 62, 62, 62, 33, 33, 33, 33],
        "Fee": [500, 500, 500, 600, 600, 650, 550, 550, 550, 550],
    })
    column_types = {
        "Appointment_ID": "text", "Patient_Name": "text", "Doctor": "text",
        "Status": "text", "Age": "numeric", "Fee": "numeric",
    }
    result = check_duplication(dataframe, column_types)

    assert result.exact_duplicate_row_indexes == []
    assert result.fuzzy_duplicate_pairs == []
    assert result.duplication_score == 100.0


def test_domain_outlier_and_dirty_numeric_get_a_moderate_floor_even_at_low_percentage():
    """Clear domain outliers (Age=-5) and dirty strings in a numeric
    column ("adult") are a strong signal of a real problem even when
    they affect only a small % of rows -- must never be classified
    below moderate, unlike a pure percentage-calibrated result would."""
    import pandas as pd
    from core.consistency import check_consistency
    from core.duplication import check_duplication
    from core.structure import check_structure
    from core.fixlist import generate_fix_list

    row_count = 50
    ages = ["30"] * (row_count - 2) + ["-5", "adult"]
    dataframe = pd.DataFrame({
        "Order_ID": [f"ORD-{i:04d}" for i in range(row_count)],
        "Age": ages,
    })
    column_types = {"Order_ID": "text", "Age": "numeric"}

    completeness_result = check_completeness(dataframe)
    consistency_result = check_consistency(dataframe, column_types)
    duplication_result = check_duplication(dataframe, column_types)
    structure_result = check_structure(dataframe)
    validity_result = check_validity(dataframe, column_types)

    findings, _crit, _mod = generate_fix_list(
        completeness_result, consistency_result, duplication_result, structure_result,
        row_count, table_name="t", use_ai_phrasing=False,
        validity_result=validity_result, column_types=column_types,
    )

    outlier_findings = [f for f in findings if f.issue_type in ("domain_outlier", "invalid_type_in_numeric")]
    assert outlier_findings
    for finding in outlier_findings:
        assert finding.percentage_affected < 10.0  # confirms this is genuinely a LOW-percentage case
        assert finding.severity in ("moderate", "critical")  # never "minor" despite the low percentage


def test_exact_duplicate_rows_get_a_moderate_floor():
    """An exact duplicate row is a 100%-confidence data-integrity problem
    the moment even one exists -- must never be classified as minor just
    because it's a small share of a large table."""
    import pandas as pd
    from core.consistency import check_consistency
    from core.duplication import check_duplication
    from core.structure import check_structure
    from core.fixlist import generate_fix_list

    row_count = 200
    rows = {"Order_ID": [f"ORD-{i:04d}" for i in range(row_count)], "Amount": list(range(row_count))}
    dataframe = pd.DataFrame(rows)
    # Plant exactly one exact-duplicate row (a small % of a 200-row table).
    dataframe.loc[199] = dataframe.loc[0]
    column_types = {"Order_ID": "text", "Amount": "numeric"}

    completeness_result = check_completeness(dataframe)
    consistency_result = check_consistency(dataframe, column_types)
    duplication_result = check_duplication(dataframe, column_types)
    structure_result = check_structure(dataframe)

    findings, _crit, _mod = generate_fix_list(
        completeness_result, consistency_result, duplication_result, structure_result,
        row_count, table_name="t", use_ai_phrasing=False, column_types=column_types,
    )

    exact_dup_findings = [f for f in findings if f.issue_type == "exact_duplicate_rows"]
    assert len(exact_dup_findings) == 1
    assert exact_dup_findings[0].percentage_affected < 10.0
    assert exact_dup_findings[0].severity in ("moderate", "critical")


def test_appointment_id_and_invoice_no_still_enforce_uniqueness():
    """Appointment_ID/Invoice_No are genuine primary-key-style
    transaction keys -- a repeated value is still a real structural
    problem and must still be caught."""
    import pandas as pd
    from core.structure import check_structure

    dataframe = pd.DataFrame({
        "Appointment_ID": ["APT-0001", "APT-0002", "APT-0001", "APT-0003"],
        "Invoice_No": ["INV-01", "INV-02", "INV-03", "INV-01"],
    })
    result = check_structure(dataframe)
    flagged_columns = {issue.column_name for issue in result.issues if issue.issue_type == "duplicate_identifier"}
    assert "Appointment_ID" in flagged_columns
    assert "Invoice_No" in flagged_columns


def test_patient_doctor_department_do_not_trigger_uniqueness_errors():
    """Patient_Name, Customer_ID, Doctor, Department, and Status all
    legitimately repeat in transactional/clinical data and must never be
    treated as if they need to be unique."""
    import pandas as pd
    from core.structure import check_structure

    dataframe = pd.DataFrame({
        "Patient_Name": ["Patient_042"] * 4,
        "Customer_ID": ["CUST-01"] * 4,
        "Doctor": ["Dr. Sharma"] * 4,
        "Department": ["Cardiology"] * 4,
        "Status": ["Completed"] * 4,
    })
    result = check_structure(dataframe)
    flagged_columns = {issue.column_name for issue in result.issues if issue.issue_type == "duplicate_identifier"}
    assert flagged_columns == set()


# ---- Performance: AI phrasing must never blow the "scorecard under 20s" target ----

def test_ai_phrasing_respects_its_time_budget_even_with_many_slow_findings():
    """
    Regression test for a real, measured bug: a table with the full
    MAX_FINDINGS_TO_AI_PHRASE worth of findings pushed generation from
    ~2s (no AI) to over 30s, because the AI-phrasing pass had a COUNT
    cap but no WALL-CLOCK cap -- a slow model call (or just a lot of
    findings) had no bound on total time. Verified directly against the
    time-budget mechanism (core/fixlist.py's AI_PHRASING_TIME_BUDGET_SECONDS)
    with a fake, deliberately slow phrase_finding -- not the real local
    model, which may not even be downloaded in this environment (see
    core/llm_phrasing.py's own is_model_available skip pattern) and
    whose real timing would make this test itself slow and flaky.
    """
    import time
    import core.fixlist as fixlist
    import core.llm_phrasing as llm_phrasing
    from core.findings import Finding

    call_count = {"n": 0}

    def _slow_phrase_finding(finding):
        call_count["n"] += 1
        time.sleep(0.05)  # stands in for a real ~1-4s model call, scaled down so this test stays fast
        return "a phrased sentence"

    findings = [
        Finding(
            table_name="t", column_name=f"col_{i}", check_type="completeness", issue_type="missing_data",
            percentage_affected=10.0, severity="minor", rule_based_description="placeholder",
        )
        for i in range(50)  # far more than MAX_FINDINGS_TO_AI_PHRASE
    ]

    original_phrase_finding = llm_phrasing.phrase_finding
    original_budget = fixlist.AI_PHRASING_TIME_BUDGET_SECONDS
    llm_phrasing.phrase_finding = _slow_phrase_finding
    fixlist.AI_PHRASING_TIME_BUDGET_SECONDS = 0.12  # shrunk so the test itself doesn't take the real 8s budget
    try:
        start = time.monotonic()
        fixlist._apply_ai_phrasing(findings)
        elapsed = time.monotonic() - start
    finally:
        llm_phrasing.phrase_finding = original_phrase_finding
        fixlist.AI_PHRASING_TIME_BUDGET_SECONDS = original_budget

    # The pass must stop close to the budget (plus at most one in-flight
    # call), never anywhere near "all 50 findings, one at a time".
    assert elapsed < 1.0
    assert call_count["n"] < len(findings)
    # Every finding beyond the budget keeps its guaranteed template
    # sentence untouched -- nothing silently blank or missing.
    for finding in findings:
        assert finding.rule_based_description == "placeholder"
        assert finding.ai_phrased_description in (None, "a phrased sentence")


if __name__ == "__main__":
    # Allow running as a plain script too: python tests/test_pipeline.py
    import traceback

    test_functions = [
        test_single_csv_still_loads_and_scores,
        test_fix_list_is_sorted_by_severity,
        test_duplication_check_finds_known_duplicates,
        test_fuzzy_duplicate_finding_carries_example_field,
        test_ai_phrased_duplication_finding_keeps_its_percentage_and_example,
        test_issue_type_humanization_covers_every_known_issue_type,
        test_leaked_issue_type_safety_net_catches_codes_and_ignores_prose,
        test_ai_phrased_fuzzy_duplicate_finding_never_leaks_raw_issue_type,
        test_low_percentage_structural_finding_classifies_and_displays_as_minor,
        test_large_table_uses_bulk_cdist_and_matches_manual_computation,
        test_blocking_still_finds_planted_duplicates_on_a_large_table,
        test_multi_sheet_excel_loads_every_sheet_as_a_table,
        test_multi_table_pipeline_isolates_one_bad_sheet,
        test_multiple_files_at_once,
        test_duplicate_threshold_falls_back_on_pure_noise,
        test_duplicate_threshold_calibrates_on_a_real_gap,
        test_severity_thresholds_fall_back_with_too_few_findings,
        test_calibration_is_deterministic,
        test_same_file_processed_twice_gives_identical_results,
        test_ai_phrasing_falls_back_cleanly_when_model_unavailable,
        test_pipeline_works_end_to_end_with_ai_phrasing_disabled,
        test_preamble_row_is_detected_and_skipped_on_csv,
        test_file_without_preamble_is_completely_unaffected,
        test_narrow_table_does_not_trigger_preamble_detection,
        test_preamble_row_is_detected_and_skipped_on_excel,
        test_non_utf8_csv_is_read_via_fallback_with_a_visible_warning,
        test_clean_utf8_csv_gets_no_encoding_warning,
        test_zero_padded_id_column_is_not_misclassified_as_a_date,
        test_bare_time_of_day_column_is_not_misclassified_as_a_date,
        test_real_dates_still_classified_correctly_after_the_shape_guard,
        test_completely_empty_file_raises_a_clean_error,
        test_header_only_file_raises_a_clean_error,
        test_single_column_file_runs_the_full_pipeline_without_crashing,
        test_remediation_numeric_fix_strips_valid_grouping_but_leaves_ambiguous_values,
        test_remediation_declines_a_fully_ambiguous_numeric_column,
        test_remediation_date_fix_normalizes_unambiguous_dates_but_leaves_ambiguous_ones,
        test_remediation_leading_year_dates_with_small_day_and_month_are_not_flagged_ambiguous,
        test_remediation_trims_whitespace_only_never_touches_casing,
        test_remediation_declines_text_fix_when_only_casing_differs,
        test_remediation_removes_exact_duplicate_rows_reusing_given_indexes,
        test_remediation_never_silently_drops_missing_data_fuzzy_or_structural_findings,
        test_remediation_never_mutates_the_caller_dataframe_or_touches_the_score,
        test_remediation_matches_known_test_dataset_issues,
        test_downloaded_cleaned_csv_reloads_cleanly,
        test_chart_generation_makes_one_chart_per_column_with_matching_annotations,
        test_validity_flags_invalid_type_in_numeric_and_ignores_valid_values,
        test_validity_flags_invalid_type_in_date,
        test_validity_flags_categorical_typos_against_dominant_spelling,
        test_validity_declines_categorical_check_on_high_cardinality_free_text,
        test_validity_flags_domain_outlier_for_named_rule_doors,
        test_validity_flags_negative_mileage_and_impossible_year,
        test_validity_flags_missing_value_representation_tokens,
        test_completeness_treats_disguised_missing_tokens_as_missing,
        test_scoring_backward_compatible_without_validity_result,
        test_remediation_coerces_invalid_numeric_to_blank_never_guesses,
        test_remediation_coerces_invalid_date_to_blank,
        test_remediation_normalizes_missing_token_to_blank_unconditionally,
        test_remediation_categorical_standardization_respects_confidence_threshold,
        test_remediation_domain_outlier_default_blanks_clip_mode_clips,
        test_remediation_missing_data_strategy_drop_row_and_constant,
        test_end_to_end_messy_car_price_dataset,
        test_repeating_customer_id_in_order_data_is_not_a_structural_issue,
        test_duplicate_order_id_still_flagged_as_structural_issue,
        test_repeat_customer_across_orders_is_not_a_fuzzy_duplicate,
        test_near_duplicate_record_with_matching_row_is_still_caught,
        test_validity_flags_rating_and_discount_out_of_bounds,
        test_validity_known_alias_dictionary_maps_abbreviations_and_city_variants,
        test_validity_new_delhi_only_folds_into_delhi_when_delhi_is_dominant,
        test_remediation_normalizes_month_name_dates_unambiguously,
        test_validity_flags_extreme_quantity_via_iqr_but_not_wide_legitimate_revenue,
        test_validity_flags_residual_typo_in_high_cardinality_categorical_column,
        test_validity_flags_customer_rating_outside_one_to_five,
        test_validity_flags_zero_quantity_and_remediation_blanks_it,
        test_validity_flags_extreme_total_amount_via_iqr,
        test_empty_optional_free_text_column_is_capped_at_minor_not_critical,
        test_mandatory_named_empty_column_is_not_suppressed,
        test_repeating_patient_name_with_stable_attributes_is_not_a_duplicate,
        test_domain_outlier_and_dirty_numeric_get_a_moderate_floor_even_at_low_percentage,
        test_exact_duplicate_rows_get_a_moderate_floor,
        test_appointment_id_and_invoice_no_still_enforce_uniqueness,
        test_patient_doctor_department_do_not_trigger_uniqueness_errors,
        test_ai_phrasing_respects_its_time_budget_even_with_many_slow_findings,
    ]
    failures = 0
    for test_function in test_functions:
        try:
            test_function()
            print(f"PASS: {test_function.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL: {test_function.__name__}")
            traceback.print_exc()
    print(f"\n{len(test_functions) - failures}/{len(test_functions)} tests passed.")
