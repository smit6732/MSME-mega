"""
core/chart_generation.py

Tier 1's dashboard half: for each column in a table, produces the DATA a
chart needs -- one chart per column, keyed on the column type
core/ingestion.py already inferred, so "what kind of chart does this
column get" is decided in exactly one place, the same way every other
column-type-driven decision in this project already works (see
core/consistency.py, core/duplication.py's identity-column pick).

Why this returns plain pandas DataFrames, not a rendered chart: every
core/*.py module in this project has zero Streamlit imports on purpose
(see core/pipeline.py, core/fixlist.py) so it can be tested headlessly
with plain pytest, no Streamlit test harness needed. app.py is the only
place a chart actually gets drawn (st.bar_chart / st.line_chart) --
this module's job stops at handing it the right numbers.

Ties back to Tier 0: when a column has a matching Finding, this module
attaches a short annotation string built directly from that Finding's
own already-computed percentage_affected (e.g. "⚠ 37.5% missing") --
never recomputed here -- so the dashboard and the diagnosis can never
tell two different stories about the same column.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from core.ingestion import COLUMN_TYPE_NUMERIC, COLUMN_TYPE_DATE
from core.findings import Finding

# Categorical/text columns with more distinct values than this get
# collapsed to "top 10 + Other" -- a bar chart with hundreds of bars is
# unreadable and defeats the point of a quick-glance dashboard.
TOP_N_CATEGORIES = 10
OTHER_LABEL = "Other"

# How many buckets a numeric distribution chart uses at most -- enough
# to show the shape of the data without turning into unreadable noise
# on a column with many distinct values.
MAX_NUMERIC_BUCKETS = 10

# Short, human phrase per issue_type, used to build a chart annotation
# without repeating a Finding's full sentence. Deliberately covers only
# the issue types a single COLUMN can carry (duplication/structure
# findings are about rows or the whole table, not one column's chart).
_ISSUE_SHORT_LABELS = {
    "missing_data": "missing",
    "inconsistent_format_numeric": "inconsistently formatted",
    "inconsistent_format_date": "inconsistently formatted",
    "inconsistent_format_text": "inconsistently formatted",
}


@dataclass
class ColumnChart:
    """Everything app.py needs to draw one column's chart, plus the
    Tier 0 annotation to show alongside it (see module docstring)."""
    column_name: str
    chart_type: str                      # "numeric_distribution" | "category_counts" | "date_over_time" | "unsupported"
    chart_data: Optional[pd.DataFrame]    # None only for "unsupported" (e.g. an all-blank column)
    annotation: Optional[str] = None      # e.g. "⚠ 37.5% missing" -- from a matching Finding, if any


def _annotation_for_column(column_name: str, findings: List[Finding]) -> Optional[str]:
    """
    Looks up the first Finding about this exact column and turns it into
    a short annotation string, using ONLY that Finding's own
    percentage_affected -- never recomputed. Findings are already
    severity-sorted (core/fixlist.py), so if a column somehow has more
    than one, the most severe one wins.
    """
    for finding in findings:
        if finding.column_name == column_name and finding.issue_type in _ISSUE_SHORT_LABELS:
            return f"⚠ {finding.percentage_affected:.1f}% {_ISSUE_SHORT_LABELS[finding.issue_type]}"
    return None


def _numeric_distribution(series: pd.Series) -> Optional[pd.DataFrame]:
    """
    A plain histogram: bucket the column's numeric values into evenly-
    spaced ranges and count how many fall in each. Values that don't
    parse as plain numbers (already-flagged formatting issues -- see the
    matching consistency Finding) are simply excluded from the shape of
    the distribution, not coerced into something they aren't.
    """
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_values.empty:
        return None

    if numeric_values.nunique() == 1:
        # A single repeated value can't be split into ranges -- pd.cut
        # would fail on a zero-width bin edge, so show one bucket
        # directly instead.
        value = numeric_values.iloc[0]
        return pd.DataFrame({"range": [str(value)], "count": [len(numeric_values)]}).set_index("range")

    bucket_count = min(MAX_NUMERIC_BUCKETS, numeric_values.nunique())
    buckets = pd.cut(numeric_values, bins=bucket_count)
    counts = buckets.value_counts().sort_index()
    return pd.DataFrame({
        "range": [str(interval) for interval in counts.index],
        "count": counts.values,
    }).set_index("range")


def _category_counts(series: pd.Series) -> Optional[pd.DataFrame]:
    """
    Value-count bar chart data for a categorical/text column, collapsed
    to the top N + "Other" for high-cardinality columns (e.g. a
    Business_Name column with 40 distinct values) so the chart stays
    readable at a glance instead of drawing one bar per row.
    """
    non_blank = series.dropna().astype(str).str.strip()
    non_blank = non_blank[non_blank != ""]
    if non_blank.empty:
        return None

    value_counts = non_blank.value_counts()
    if len(value_counts) > TOP_N_CATEGORIES:
        top = value_counts.iloc[:TOP_N_CATEGORIES]
        other_count = int(value_counts.iloc[TOP_N_CATEGORIES:].sum())
        value_counts = pd.concat([top, pd.Series({OTHER_LABEL: other_count})])

    return pd.DataFrame({"value": value_counts.index, "count": value_counts.values}).set_index("value")


def _date_over_time(series: pd.Series) -> Optional[pd.DataFrame]:
    """
    Record count per month -- day-level granularity would be too sparse
    for a typical MSME-sized file, month is a readable, still-meaningful
    bucket for "how does this data's volume change over time".
    """
    parsed = pd.to_datetime(series, errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return None

    monthly_counts = parsed.dt.to_period("M").value_counts().sort_index()
    return pd.DataFrame({
        "month": [str(period) for period in monthly_counts.index],
        "count": monthly_counts.values,
    }).set_index("month")


def generate_charts_for_table(
    dataframe: pd.DataFrame,
    column_types: Dict[str, str],
    findings: List[Finding],
) -> List[ColumnChart]:
    """
    One ColumnChart per column, in the dataframe's own column order --
    the single entry point app.py calls to build a table's whole
    dashboard. Uses the column types core/ingestion.py already inferred
    (never re-detects them), same "reuse Tier 0, don't rescan" principle
    as core/remediation.py.
    """
    charts: List[ColumnChart] = []

    for column_name in dataframe.columns:
        column_type = column_types.get(column_name, "unknown")
        series = dataframe[column_name]

        if column_type == COLUMN_TYPE_NUMERIC:
            chart_type, chart_data = "numeric_distribution", _numeric_distribution(series)
        elif column_type == COLUMN_TYPE_DATE:
            chart_type, chart_data = "date_over_time", _date_over_time(series)
        else:
            # Text, boolean, and unknown columns all get the same
            # value-count treatment -- a boolean column is just a
            # category with (usually) two values, and "unknown" is
            # still worth a chart rather than nothing.
            chart_type, chart_data = "category_counts", _category_counts(series)

        if chart_data is None:
            chart_type = "unsupported"

        charts.append(ColumnChart(
            column_name=column_name,
            chart_type=chart_type,
            chart_data=chart_data,
            annotation=_annotation_for_column(column_name, findings),
        ))

    return charts
