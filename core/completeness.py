"""
core/completeness.py

Dimension 1 of the scorecard: COMPLETENESS.

Answers "how much of this data is actually filled in?" for each column,
then rolls that up into one score for the whole dataset.

We treat a value as "missing" if it's a real null (NaN/None), an
empty/whitespace-only string, OR one of a small set of common "disguised
missing" placeholder tokens (MISSING_VALUE_TOKENS below) -- a cell that
literally contains the text "N/A" or "Unknown" is not usably-filled-in
data any more than a blank cell is, and counting only true blanks would
understate how much of the column is actually missing. ingestion.py
reads every cell as a string, so a blank cell can show up as "" or "  "
rather than a clean NaN -- if we only checked pd.isna(), we'd undercount
this even before placeholder tokens enter the picture.

MISSING_VALUE_TOKENS is the single source of truth for which literal
strings count as a disguised-missing placeholder -- core/validity.py
imports it directly (rather than redefining its own list) so its
missing_value_representation Finding can never disagree with what this
module already counted as missing for the completeness score.
"""

from dataclasses import dataclass
from typing import Dict

import pandas as pd

# Common ways a human or export tool writes "nothing here" instead of
# leaving a cell truly blank. Matched case-insensitively after
# stripping whitespace -- "N/A", "n/a", " N/A " all match. Deliberately
# a small, conservative list of tokens that are essentially never a
# legitimate real-world VALUE for any column (unlike, say, "0" or "-",
# which genuinely can be real data) -- see core/validity.py's own
# docstring for why the list stops there.
MISSING_VALUE_TOKENS = frozenset({
    "n/a", "na", "null", "none", "nan", "?", "unknown", "tbd", "n.a.", "n.a",
})


@dataclass
class ColumnCompleteness:
    """Completeness result for a single column."""
    column_name: str
    missing_count: int
    total_count: int
    missing_percentage: float   # 0-100, share of rows missing a value
    completeness_score: float   # 0-100, the inverse of missing_percentage


@dataclass
class CompletenessResult:
    """Completeness result for the whole dataset."""
    per_column: Dict[str, ColumnCompleteness]
    overall_score: float  # 0-100, average of every column's completeness_score


def _is_missing(value) -> bool:
    """
    A single cell counts as missing if it's a null, a string that is
    empty/only whitespace once trimmed, or a recognized disguised-missing
    placeholder token (see MISSING_VALUE_TOKENS above). Kept as its own
    function so this exact definition of "missing" stays consistent
    everywhere it's used (this module, consistency.py, and validity.py
    all need to skip missing values the same way).
    """
    if pd.isna(value):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return True
        if stripped.lower() in MISSING_VALUE_TOKENS:
            return True
    return False


def check_completeness(dataframe: pd.DataFrame) -> CompletenessResult:
    """
    Compute missing-value stats for every column, then average them into
    one overall score. A plain (unweighted) mean is used on purpose --
    every column matters equally here; deciding some columns are "more
    important" than others would need business context this tool doesn't
    have.
    """
    per_column: Dict[str, ColumnCompleteness] = {}
    total_rows = len(dataframe)

    for column_name in dataframe.columns:
        missing_count = int(dataframe[column_name].apply(_is_missing).sum())
        missing_percentage = (missing_count / total_rows * 100) if total_rows else 0.0
        completeness_score = 100.0 - missing_percentage

        per_column[column_name] = ColumnCompleteness(
            column_name=column_name,
            missing_count=missing_count,
            total_count=total_rows,
            missing_percentage=round(missing_percentage, 2),
            completeness_score=round(completeness_score, 2),
        )

    overall_score = (
        sum(c.completeness_score for c in per_column.values()) / len(per_column)
        if per_column else 0.0
    )

    return CompletenessResult(per_column=per_column, overall_score=round(overall_score, 2))
