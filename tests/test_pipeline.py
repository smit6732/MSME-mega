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
import core.llm_phrasing as llm_phrasing


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLE_TABLE_CSV = os.path.join(REPO_ROOT, "sample_data", "messy_msme_sample.csv")
MULTI_SHEET_XLSX = os.path.join(REPO_ROOT, "sample_data", "multi_sheet_msme_workbook.xlsx")


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
    # This sample file is deliberately messy -- it should NOT score as
    # "good" (>=80). If it does, a check is probably broken.
    assert result.scorecard.overall_score < 80
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
