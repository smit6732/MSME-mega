"""
core/consistency.py

Dimension 2 of the scorecard: CONSISTENCY.

completeness.py asks "is the cell filled in?". Consistency asks a harder
question: "even when it IS filled in, is it recorded the same way every
time?" A column can be 100% complete and still be useless for AI/BI if
half its numbers are "12,50,000" and half are "1250000", or half its
dates are DD/MM/YYYY and half are YYYY-MM-DD.

We reuse the column type ingestion.py already inferred (numeric / date /
text / boolean) and run a different check depending on that type, because
"consistent" means something different for each kind of data.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from core.ingestion import (
    COLUMN_TYPE_NUMERIC,
    COLUMN_TYPE_DATE,
    COLUMN_TYPE_TEXT,
    DATE_FORMAT_PATTERNS,
)


@dataclass
class ColumnConsistency:
    """Consistency result for a single column."""
    column_name: str
    column_type: str
    inconsistent_count: int
    total_checked: int              # non-missing values that were actually checked
    inconsistent_percentage: float  # 0-100
    consistency_score: float        # 0-100
    example_values: List[str]       # a few offending values, used by fixlist.py


@dataclass
class ConsistencyResult:
    per_column: Dict[str, ColumnConsistency]
    overall_score: float


# Characters a human commonly adds to a "number" that stop it parsing as
# one: currency symbols, thousands separators, stray whitespace. Same
# idea as ingestion.py's numeric-detection cleanup, applied here per value
# instead of per column.
_NUMERIC_JUNK_PATTERN = re.compile(r"[₹$,\s]")

# Named date patterns, built from the same regex list ingestion.py uses
# to decide a column IS a date column -- reused here to work out WHICH
# format each individual value is in, so we can spot the odd one out.
_NAMED_DATE_PATTERNS = [
    (re.compile(DATE_FORMAT_PATTERNS[0]), "YYYY-MM-DD"),
    (re.compile(DATE_FORMAT_PATTERNS[1]), "DD/MM/YYYY"),
    (re.compile(DATE_FORMAT_PATTERNS[2]), "DD-MM-YYYY"),
    (re.compile(DATE_FORMAT_PATTERNS[3]), "YYYY/MM/DD"),
    (re.compile(DATE_FORMAT_PATTERNS[4]), "DD-Mon-YYYY"),
    (re.compile(DATE_FORMAT_PATTERNS[5]), "Month DD, YYYY"),
]


# ---- Numeric formatting ---------------------------------------------------

def _is_plain_number(value: str) -> bool:
    """True only if Python's float() can parse the value with zero cleanup."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def _check_numeric_column(values: List[str]) -> Tuple[int, List[str]]:
    """
    Flags any value that needs commas/currency symbols/whitespace
    stripped before it parses as a number -- that stripping is exactly
    the formatting inconsistency we want to surface.
    """
    inconsistent_count = 0
    examples: List[str] = []
    for raw_value in values:
        if not _is_plain_number(raw_value):
            inconsistent_count += 1
            if len(examples) < 3:
                examples.append(raw_value)
    return inconsistent_count, examples


# ---- Date formats ----------------------------------------------------------

def _detect_date_format(raw_value: str) -> str:
    """Return the name of the first date pattern that matches, or
    'unrecognized' if the value doesn't match any known format at all."""
    for pattern, format_name in _NAMED_DATE_PATTERNS:
        if pattern.match(raw_value):
            return format_name
    return "unrecognized"


def _check_date_column(values: List[str]) -> Tuple[int, List[str]]:
    """
    Finds the column's "house style" (its most common date format) and
    flags every value that isn't in that format -- including ones we
    can't recognise at all. A column that's already 100% one format
    naturally scores as fully consistent.
    """
    detected_formats = [_detect_date_format(v) for v in values]
    if not detected_formats:
        return 0, []

    dominant_format = pd.Series(detected_formats).value_counts().index[0]

    inconsistent_count = 0
    examples: List[str] = []
    for raw_value, detected_format in zip(values, detected_formats):
        if detected_format != dominant_format:
            inconsistent_count += 1
            if len(examples) < 3:
                examples.append(raw_value)
    return inconsistent_count, examples


# ---- Text casing / whitespace ----------------------------------------------

def _has_whitespace_issue(raw_value: str) -> bool:
    """Leading/trailing spaces, or doubled-up internal spaces, count as a
    whitespace inconsistency."""
    return raw_value != raw_value.strip() or "  " in raw_value


def _check_text_column(values: List[str]) -> Tuple[int, List[str]]:
    """
    Groups values by a normalized (trimmed, lowercased) form -- e.g.
    "Ahmedabad" and "ahmedabad" both normalize to "ahmedabad". Within
    each group, whichever exact spelling appears most often is treated as
    the column's "house style" for that value (same idea as
    _check_date_column picking the dominant date format); anything that
    doesn't match it is flagged.

    We deliberately flag only the minority spelling, not every occurrence
    in the group -- if 5 rows say "Ahmedabad" and 1 says "ahmedabad", the
    1 typo is the inconsistency, not all 6 rows.
    """
    normalized_to_variant_counts: Dict[str, Dict[str, int]] = {}
    for raw_value in values:
        normalized = raw_value.strip().lower()
        variant_counts = normalized_to_variant_counts.setdefault(normalized, {})
        variant_counts[raw_value] = variant_counts.get(raw_value, 0) + 1

    dominant_variant_by_normalized = {
        normalized: max(variant_counts, key=variant_counts.get)
        for normalized, variant_counts in normalized_to_variant_counts.items()
    }

    inconsistent_count = 0
    examples: List[str] = []
    for raw_value in values:
        normalized = raw_value.strip().lower()
        does_not_match_house_style = raw_value != dominant_variant_by_normalized[normalized]
        if does_not_match_house_style or _has_whitespace_issue(raw_value):
            inconsistent_count += 1
            if len(examples) < 3:
                examples.append(raw_value)
    return inconsistent_count, examples


# ---- Main entry point -------------------------------------------------------

def check_consistency(dataframe: pd.DataFrame, column_types: Dict[str, str]) -> ConsistencyResult:
    """
    Run the check appropriate to each column's inferred type. Boolean and
    unknown columns are skipped (treated as fully consistent) -- there's
    no meaningful formatting check for a two-value boolean column, and
    "unknown" means ingestion.py couldn't even tell what the column
    should look like, so we have nothing to compare against.
    """
    per_column: Dict[str, ColumnConsistency] = {}

    for column_name in dataframe.columns:
        column_type = column_types.get(column_name, "unknown")

        non_missing = dataframe[column_name].dropna().astype(str).str.strip()
        non_missing = non_missing[non_missing != ""]
        # Consistency checks below expect trimmed strings for pattern
        # matching (dates/numbers), but the *whitespace* check needs the
        # untrimmed original -- so text columns use the raw values instead.
        if column_type == COLUMN_TYPE_TEXT:
            raw_series = dataframe[column_name].dropna().astype(str)
            values = [v for v in raw_series if v.strip() != ""]
        else:
            values = non_missing.tolist()

        total_checked = len(values)

        if total_checked == 0:
            inconsistent_count, examples = 0, []
        elif column_type == COLUMN_TYPE_NUMERIC:
            inconsistent_count, examples = _check_numeric_column(values)
        elif column_type == COLUMN_TYPE_DATE:
            inconsistent_count, examples = _check_date_column(values)
        elif column_type == COLUMN_TYPE_TEXT:
            inconsistent_count, examples = _check_text_column(values)
        else:
            inconsistent_count, examples = 0, []

        inconsistent_percentage = (
            (inconsistent_count / total_checked * 100) if total_checked else 0.0
        )
        consistency_score = 100.0 - inconsistent_percentage

        per_column[column_name] = ColumnConsistency(
            column_name=column_name,
            column_type=column_type,
            inconsistent_count=inconsistent_count,
            total_checked=total_checked,
            inconsistent_percentage=round(inconsistent_percentage, 2),
            consistency_score=round(consistency_score, 2),
            example_values=examples,
        )

    overall_score = (
        sum(c.consistency_score for c in per_column.values()) / len(per_column)
        if per_column else 0.0
    )

    return ConsistencyResult(per_column=per_column, overall_score=round(overall_score, 2))
