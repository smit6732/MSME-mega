"""
templates/fix_templates.py

Plain-language sentence templates used to build the "fix list" -- the
tool's answer to the problem statement's requirement for a plain-language
list of what needs fixing.

Design:
- Templates are grouped by issue_type, then by severity tier.
- Each group has 2-3 phrasing VARIANTS, so the same issue type doesn't
  always produce the exact same sentence. core/fixlist.py cycles through
  them round-robin so the output doesn't read like it came from a rigid
  template.
- Templates use Python's str.format() placeholders: {field}, {percentage},
  {example}. core/fixlist.py fills these in with real values.

This file, together with core/fixlist.py, is the ENTIRE "plain language
generation" layer of the tool -- pure string formatting, no AI/LLM call,
no network access. That's a hard requirement for this project (see the
README), not just a convenience.
"""

# Severity tier labels, used as dict keys here and throughout
# core/fixlist.py. Imported from core/findings.py rather than redefined
# here -- that keeps exactly one spelling of each tier used everywhere
# in the codebase (the Finding dataclass, this template dict, and the
# severity-calibration logic in core/calibration.py all agree).
from core.findings import SEVERITY_CRITICAL, SEVERITY_MODERATE, SEVERITY_MINOR


FIX_TEMPLATES = {

    "missing_data": {
        SEVERITY_CRITICAL: [
            "'{field}' is missing in {percentage}% of rows — this field is close to unusable until it's filled in.",
            "Critical gap: {percentage}% of '{field}' values are blank. Consider whether this field can realistically be collected going forward.",
            "'{field}' has {percentage}% missing data, well above a safe threshold — treat this as a priority before using the dataset for analysis.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% missing values — worth filling in before this dataset feeds an AI/BI pipeline.",
            "{percentage}% of '{field}' entries are blank. A manual review to fill these gaps would meaningfully improve the dataset.",
            "'{field}' is missing for {percentage}% of records — not urgent, but should be addressed for reliable analysis.",
        ],
        SEVERITY_MINOR: [
            "'{field}' is missing in a small number of rows ({percentage}%) — low priority, but still worth a quick check.",
            "A few '{field}' values ({percentage}%) are blank. Minor, but easy to fix while the data is being cleaned anyway.",
            "'{field}' has {percentage}% missing data — minor issue, unlikely to affect most analysis.",
        ],
    },

    "inconsistent_format_numeric": {
        SEVERITY_CRITICAL: [
            "'{field}' should be numeric, but {percentage}% of values use formatting (like commas or currency symbols, e.g. \"{example}\") that breaks number parsing.",
            "{percentage}% of '{field}' values can't be read as plain numbers as-is (e.g. \"{example}\") — this needs a formatting cleanup before any calculation or model can use it.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% of values in a non-numeric format (e.g. \"{example}\") — strip commas/currency symbols to make this column calculation-ready.",
            "About {percentage}% of '{field}' entries, like \"{example}\", need reformatting to be treated as numbers.",
        ],
        SEVERITY_MINOR: [
            "A small share ({percentage}%) of '{field}' values, e.g. \"{example}\", have formatting that stops them from being read as numbers.",
            "'{field}' is mostly clean, but {percentage}% of values (e.g. \"{example}\") need minor formatting fixes.",
        ],
    },

    "inconsistent_format_date": {
        SEVERITY_CRITICAL: [
            "'{field}' mixes multiple date formats — {percentage}% of values (e.g. \"{example}\") don't match the most common format used in this column.",
            "{percentage}% of dates in '{field}' are in a different format than the rest (e.g. \"{example}\") — standardize to one format before this column can be reliably parsed.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% of dates (e.g. \"{example}\") in a different format from the rest of the column — worth standardizing.",
            "About {percentage}% of '{field}' values, like \"{example}\", use an inconsistent date format.",
        ],
        SEVERITY_MINOR: [
            "A few dates in '{field}' ({percentage}%), e.g. \"{example}\", don't match the column's usual format.",
            "'{field}' is mostly one date format, with {percentage}% of exceptions like \"{example}\".",
        ],
    },

    "inconsistent_format_text": {
        SEVERITY_CRITICAL: [
            "'{field}' has {percentage}% of values with inconsistent capitalization or spacing (e.g. \"{example}\") — this will cause the same real-world value to be treated as different categories.",
            "{percentage}% of '{field}' entries, such as \"{example}\", have casing/whitespace issues that should be standardized before grouping or filtering on this column.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% of values, like \"{example}\", with inconsistent capitalization or extra spacing.",
            "About {percentage}% of '{field}' entries need casing/whitespace cleanup, e.g. \"{example}\".",
        ],
        SEVERITY_MINOR: [
            "A small number of '{field}' values ({percentage}%), e.g. \"{example}\", have minor casing or spacing inconsistencies.",
            "'{field}' is mostly consistent, aside from {percentage}% of entries like \"{example}\".",
        ],
    },

    "exact_duplicate_rows": {
        SEVERITY_CRITICAL: [
            "{percentage}% of rows are exact duplicates of another row — remove these before using the dataset, they will double-count in any aggregation.",
            "This dataset has exact duplicate rows making up {percentage}% of all records — de-duplicate first, they will skew totals and averages.",
        ],
        SEVERITY_MODERATE: [
            "{percentage}% of rows are exact duplicates — worth removing to avoid double-counting.",
            "A noticeable share ({percentage}%) of rows are repeated exactly and should be de-duplicated.",
        ],
        SEVERITY_MINOR: [
            "A small number of rows ({percentage}%) are exact duplicates — low impact, but easy to remove.",
            "{percentage}% of rows are exact repeats of another row. Minor, but worth a quick de-duplication pass.",
        ],
    },

    "fuzzy_duplicate_rows": {
        SEVERITY_CRITICAL: [
            "{percentage}% of rows look like near-duplicate records (e.g. \"{example}\") — likely the same business entered more than once with small spelling/formatting differences.",
            "Several records ({percentage}% of rows), such as \"{example}\", appear to be the same entity recorded inconsistently — review and merge these before analysis.",
        ],
        SEVERITY_MODERATE: [
            "{percentage}% of rows appear to be near-duplicates (e.g. \"{example}\") — worth a manual review to confirm and merge.",
            "Some records ({percentage}%), like \"{example}\", look like the same business entered slightly differently.",
        ],
        SEVERITY_MINOR: [
            "A few rows ({percentage}%) look like possible near-duplicates, e.g. \"{example}\" — low priority, but worth a glance.",
            "{percentage}% of rows are possible near-duplicates, such as \"{example}\". Minor, easy to verify manually.",
        ],
    },

    "invalid_type_in_numeric": {
        SEVERITY_CRITICAL: [
            "'{field}' should be a number, but {percentage}% of values (e.g. \"{example}\") are not numeric at all — these can't be calculated on until they're fixed or cleared.",
            "{percentage}% of '{field}' contains non-numeric text like \"{example}\" in a column that should be numbers — this needs attention before any calculation uses this column.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% of values, like \"{example}\", that aren't valid numbers at all (not just badly formatted).",
            "About {percentage}% of '{field}' entries, e.g. \"{example}\", are non-numeric text sitting in a numeric column.",
        ],
        SEVERITY_MINOR: [
            "A small number of '{field}' values ({percentage}%), e.g. \"{example}\", aren't valid numbers.",
            "'{field}' is mostly valid numbers, aside from {percentage}% of entries like \"{example}\".",
        ],
    },

    "invalid_type_in_date": {
        SEVERITY_CRITICAL: [
            "'{field}' should be a date, but {percentage}% of values (e.g. \"{example}\") aren't recognizable as a date at all.",
            "{percentage}% of '{field}' contains text like \"{example}\" that isn't any known date format — not just a different format, not a date at all.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% of values, like \"{example}\", that don't parse as a date in any recognized format.",
            "About {percentage}% of '{field}' entries, e.g. \"{example}\", aren't valid dates at all.",
        ],
        SEVERITY_MINOR: [
            "A small number of '{field}' values ({percentage}%), e.g. \"{example}\", aren't recognizable dates.",
            "'{field}' is mostly valid dates, aside from {percentage}% of entries like \"{example}\".",
        ],
    },

    "categorical_inconsistency": {
        SEVERITY_CRITICAL: [
            "'{field}' has {percentage}% of values that look like typos or variant spellings of another value in the same column (e.g. \"{example}\") — these will silently split one real category into several.",
            "{percentage}% of '{field}' entries appear to be misspelled versions of a more common value (e.g. \"{example}\") — worth standardizing before grouping or counting by this column.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% of values, like \"{example}\", that look like spelling variants of a more common category.",
            "About {percentage}% of '{field}' entries, e.g. \"{example}\", may be typos of an existing category in this column.",
        ],
        SEVERITY_MINOR: [
            "A small number of '{field}' values ({percentage}%), e.g. \"{example}\", look like possible spelling variants.",
            "'{field}' is mostly consistent categories, aside from {percentage}% of possible variants like \"{example}\".",
        ],
    },

    "domain_outlier": {
        SEVERITY_CRITICAL: [
            "'{field}' has {percentage}% of values that are outside a realistic range for this field (e.g. \"{example}\") — these look like data-entry errors, not real values.",
            "{percentage}% of '{field}' values, such as \"{example}\", fall well outside what's realistically possible for this field.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% of values, like \"{example}\", outside the range that's realistic for this field — worth a second look.",
            "About {percentage}% of '{field}' entries, e.g. \"{example}\", look like they may be data-entry errors.",
        ],
        SEVERITY_MINOR: [
            "A small number of '{field}' values ({percentage}%), e.g. \"{example}\", sit outside this field's usual range.",
            "'{field}' is mostly within a realistic range, aside from {percentage}% of values like \"{example}\".",
        ],
    },

    "missing_value_representation": {
        SEVERITY_CRITICAL: [
            "'{field}' has {percentage}% of rows using a placeholder like \"{example}\" instead of being left truly blank — these are missing data in disguise.",
            "{percentage}% of '{field}' values are placeholder text (e.g. \"{example}\") rather than real data — treat these the same as blank cells.",
        ],
        SEVERITY_MODERATE: [
            "'{field}' has {percentage}% of values, like \"{example}\", that are placeholder text standing in for a real (missing) value.",
            "About {percentage}% of '{field}' entries use a placeholder such as \"{example}\" instead of a real value.",
        ],
        SEVERITY_MINOR: [
            "A small number of '{field}' values ({percentage}%), e.g. \"{example}\", are placeholder text rather than real data.",
            "'{field}' is mostly real values, aside from {percentage}% of placeholders like \"{example}\".",
        ],
    },

    "structural_issue": {
        SEVERITY_CRITICAL: [
            "Structural problem: {example} This affects how reliably the file can be loaded and processed at all.",
            "{example} This is a structural issue that should be fixed before the file is used in any pipeline.",
        ],
        SEVERITY_MODERATE: [
            "Structural issue: {example} Worth fixing to avoid confusion downstream.",
            "{example} Consider cleaning this up before relying on the file's structure.",
        ],
        SEVERITY_MINOR: [
            "Minor structural note: {example}",
            "{example} Low impact, but worth knowing about.",
        ],
    },
}
