"""
core/structure.py

Dimension 4 of the scorecard: STRUCTURE.

This checks the "shape" of the data, separate from the values inside it:
are the column headers usable, is every row the same width, and do
columns that are supposed to hold a unique identifier (like a GST number)
actually hold unique values? A file can pass every completeness and
consistency check and still be structurally broken -- e.g. two rows
sharing the same GST number is a data-integrity problem no amount of
"fill in the blanks" fixes.
"""

import csv
import io
import re
from dataclasses import dataclass
from typing import List, Optional

# Word-boundary-aware hint matching -- a plain `hint in name` substring
# check (the naive version this replaces) also matches "gst" inside
# "furnishingstatus" (the "...ing" + "status" boundary spells "ngst"),
# or "pan" inside "company", or "code" inside "postcode" -- short hints
# are exactly the ones a bare substring check misfires on. Bounded by
# the string's start/end or any non-alphanumeric character (underscore
# included, unlike regex's own \b, which treats underscore as a "word"
# character and would NOT stop at it) -- so "GST_Number" and "PAN123"
# still match (boundary is start-of-string or an underscore/non-letter),
# but a hint sitting in the middle of an unrelated, unbroken word never
# does. Verified live against this exact false positive on a real
# (non-MSME) housing dataset.
def _hint_matches(hint: str, name_lower: str) -> bool:
    # A boundary is only REQUIRED on a side where the hint's own edge
    # character is alphanumeric -- a hint that already starts/ends with
    # a separator (like "_id" or "id_") is self-bounding on that side,
    # and demanding an additional non-alnum character right before/after
    # the underscore itself would wrongly reject the exact "Customer_ID"
    # shape this hint exists to match.
    left = r"(?<![a-z0-9])" if hint[:1].isalnum() else ""
    right = r"(?![a-z0-9])" if hint[-1:].isalnum() else ""
    pattern = left + re.escape(hint) + right
    return re.search(pattern, name_lower) is not None

import pandas as pd


# Column-name substrings (lowercased) that suggest a column should hold a
# PER-ROW-UNIQUE identifier -- a real primary/transaction key, not a
# repeating dimension. Used to decide which columns get checked for
# duplicate values -- checking every column would be noisy (a City column
# is SUPPOSED to repeat; a GST number is not).
IDENTIFIER_COLUMN_NAME_HINTS = [
    "gst", "pan", "order_id", "order_no", "invoice_id", "invoice_no",
    "transaction_id", "txn_id", "receipt_no", "bill_no", "registration_no", "license_no",
    "appointment_id", "appointment_no", "booking_id", "booking_no",
]

# Column-name substrings that mean "this column names WHO/WHAT the row is
# about" -- a foreign-key-style reference into a master/dimension table
# (a customer, vendor, product, employee, ...). These LEGITIMATELY repeat
# many times in transactional/order-level data (the same customer places
# many orders), so even a bare "_id"/"code" hint below must never flag
# one of these as "should be unique". This is the fix for the false-
# positive "Customer_ID looks like a duplicate identifier" finding on
# normal order data.
_DIMENSION_REFERENCE_NAME_HINTS = [
    "customer", "client", "vendor", "supplier", "user", "account",
    "employee", "staff", "product", "item", "category", "party",
    "patient", "doctor", "department", "physician", "practitioner",
]

# A bare "_id"/"id_"/trailing "id"/"code" hint (unlike the explicit
# transaction-key names above) is only trusted as "should be unique" when
# the column name does NOT also look like a dimension reference -- e.g.
# a plain "ID" or "Record_Code" column still counts, but "Customer_ID"
# and "Product_Code" do not.
_GENERIC_ID_HINTS = ["_id", "id_", "code"]

# Flat penalty subtracted from 100 per structural issue found. A flat
# penalty (rather than a percentage-based one, like completeness/
# consistency use) is deliberate: structure issues are typically few and
# each one matters on its own -- even ONE duplicate GST number is worth
# flagging at real weight, not diluted by dividing over the whole dataset.
PENALTY_PER_ISSUE = 15


@dataclass
class StructureIssue:
    """One specific structural problem found in the dataset."""
    issue_type: str          # e.g. "unnamed_column", "ragged_rows", "duplicate_identifier"
    column_name: Optional[str]
    description: str
    affected_count: int
    affected_percentage: float


@dataclass
class StructureResult:
    issues: List[StructureIssue]
    structure_score: float  # 0-100


def _find_unnamed_or_empty_columns(dataframe: pd.DataFrame) -> List[StructureIssue]:
    """
    Flags columns with no real name -- either literally empty, or
    pandas' auto-generated "Unnamed: N" placeholder, which shows up when
    a CSV/Excel export has a stray extra column (often a leftover row
    index) with no header text of its own.
    """
    issues = []
    total_columns = len(dataframe.columns)
    for column_name in dataframe.columns:
        column_name_str = str(column_name).strip()
        is_blank = column_name_str == ""
        is_pandas_placeholder = bool(re.match(r"^Unnamed:\s*\d+$", column_name_str))
        if is_blank or is_pandas_placeholder:
            issues.append(
                StructureIssue(
                    issue_type="unnamed_column",
                    column_name=column_name,
                    description=f"Column '{column_name}' has no usable header name.",
                    affected_count=1,
                    affected_percentage=round(100 / total_columns, 2) if total_columns else 0.0,
                )
            )
    return issues


def _find_ragged_rows(raw_text: str) -> List[StructureIssue]:
    """
    Re-reads the raw CSV text line by line (via csv.reader, not pandas)
    to check every row has the same number of fields as the header.
    pandas will silently pad short rows with NaN or error out on long
    ones -- either way it can hide the fact the SOURCE file was
    inconsistent, so we inspect the raw text directly instead of the
    already-normalized dataframe.
    """
    reader = csv.reader(io.StringIO(raw_text))
    rows = list(reader)
    if len(rows) < 2:
        return []

    header_field_count = len(rows[0])
    ragged_row_numbers = [
        row_number for row_number, row in enumerate(rows[1:], start=2)
        if len(row) != header_field_count
    ]
    if not ragged_row_numbers:
        return []

    total_data_rows = len(rows) - 1
    return [
        StructureIssue(
            issue_type="ragged_rows",
            column_name=None,
            description=(
                f"{len(ragged_row_numbers)} row(s) have a different number of fields than "
                f"the header ({header_field_count} columns expected) -- likely an export or "
                f"formatting error."
            ),
            affected_count=len(ragged_row_numbers),
            affected_percentage=round(len(ragged_row_numbers) / total_data_rows * 100, 2),
        )
    ]


def _find_identifier_duplicates(dataframe: pd.DataFrame) -> List[StructureIssue]:
    """
    For columns whose name suggests they hold a unique identifier (GST
    number, PAN, ID, code, etc.), checks whether any value repeats. A
    repeated identifier usually means two different records got merged/
    miscoded, or the same record was entered twice under the same ID.
    """
    issues = []
    total_rows = len(dataframe)

    for column_name in dataframe.columns:
        column_name_lower = str(column_name).lower()
        is_dimension_reference = any(_hint_matches(hint, column_name_lower) for hint in _DIMENSION_REFERENCE_NAME_HINTS)
        looks_like_transaction_key = any(_hint_matches(hint, column_name_lower) for hint in IDENTIFIER_COLUMN_NAME_HINTS)
        looks_like_generic_id = (
            any(_hint_matches(hint, column_name_lower) for hint in _GENERIC_ID_HINTS) and not is_dimension_reference
        )
        if not (looks_like_transaction_key or looks_like_generic_id):
            continue

        non_missing_values = dataframe[column_name].dropna()
        non_missing_values = non_missing_values[non_missing_values.astype(str).str.strip() != ""]
        duplicate_count = int(non_missing_values.duplicated(keep=False).sum())

        if duplicate_count > 0:
            issues.append(
                StructureIssue(
                    issue_type="duplicate_identifier",
                    column_name=column_name,
                    description=(
                        f"Column '{column_name}' looks like a unique identifier but has "
                        f"{duplicate_count} row(s) sharing a value that should be unique."
                    ),
                    affected_count=duplicate_count,
                    affected_percentage=round(duplicate_count / total_rows * 100, 2) if total_rows else 0.0,
                )
            )
    return issues


def check_structure(dataframe: pd.DataFrame, raw_text: Optional[str] = None) -> StructureResult:
    """
    Run all structural checks and combine them into one score.

    raw_text: the original CSV text (see DatasetProfile.raw_text). Only
    needed for the ragged-row check; pass None (the default) to skip that
    one check -- e.g. for Excel files, where it doesn't apply.
    """
    issues: List[StructureIssue] = []
    issues.extend(_find_unnamed_or_empty_columns(dataframe))
    issues.extend(_find_identifier_duplicates(dataframe))
    if raw_text is not None:
        issues.extend(_find_ragged_rows(raw_text))

    structure_score = max(0.0, 100.0 - len(issues) * PENALTY_PER_ISSUE)

    return StructureResult(issues=issues, structure_score=round(structure_score, 2))
