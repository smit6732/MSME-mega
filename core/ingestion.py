"""
core/ingestion.py

Handles loading a raw MSME data file (CSV or Excel) into a pandas
DataFrame and figuring out, column by column, what *kind* of data it
actually holds (numeric / text / date / boolean / unknown).

Why we infer types ourselves instead of trusting pandas' own dtypes:
pandas will happily read "12,50,000" or "₹12,50,000" as a text (object)
column, because the commas/currency symbol stop it from parsing as a
number. If we just used df.dtypes after a normal pd.read_csv(), we'd
think Annual_Revenue is a text column and never flag it as "numeric data
trapped in bad formatting" -- which is exactly the kind of problem this
tool exists to catch. So every column is read in as plain text first,
and we decide its "real" type ourselves by looking at the values.
"""

import io
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


# Column type labels used everywhere else in the pipeline (completeness,
# consistency, duplication, structure all read dataset_profile.column_types).
COLUMN_TYPE_NUMERIC = "numeric"
COLUMN_TYPE_TEXT = "text"
COLUMN_TYPE_DATE = "date"
COLUMN_TYPE_BOOLEAN = "boolean"
COLUMN_TYPE_UNKNOWN = "unknown"

# Fraction of a column's non-missing values that must match a pattern
# before we commit to that type for the whole column. Set below 1.0 on
# purpose -- a handful of badly-formatted values shouldn't stop us from
# recognising the column's overall intent (e.g. a mostly-numeric revenue
# column with a couple of typos is still a numeric column, just an
# inconsistent one -- see core/consistency.py for catching those typos).
TYPE_MATCH_THRESHOLD = 0.7


@dataclass
class DatasetProfile:
    """
    A bundle of everything the rest of the pipeline needs to know about
    the file the user uploaded. We pass this one object around instead of
    the raw dataframe plus a handful of loose variables, so every module
    downstream (completeness, consistency, duplication, structure) works
    off the same source of truth.
    """
    dataframe: pd.DataFrame
    file_name: str
    row_count: int
    column_count: int
    column_types: Dict[str, str] = field(default_factory=dict)
    # Raw CSV text, kept around only so structure.py can check for ragged
    # rows (rows with a different field count than the header) by reading
    # the file's actual text, not the already-normalized dataframe pandas
    # produces. None for Excel files, where this check doesn't apply --
    # Excel is grid-based, so there's no equivalent "ragged line" concept.
    raw_text: Optional[str] = None


def _read_all_bytes(file_path_or_buffer) -> bytes:
    """
    Accepts either a file path (str/Path) or a file-like buffer (e.g.
    what Streamlit's file_uploader gives us) and returns its raw bytes
    either way. Centralizing this here means the rest of load_dataset
    doesn't need to care which one it was handed.
    """
    if isinstance(file_path_or_buffer, (str, os.PathLike)):
        with open(file_path_or_buffer, "rb") as file_handle:
            return file_handle.read()
    file_path_or_buffer.seek(0)
    return file_path_or_buffer.read()


def load_dataset(file_path_or_buffer, file_name: str) -> DatasetProfile:
    """
    Load a CSV or Excel file into a DatasetProfile.

    file_path_or_buffer: a path string, or a file-like object (what
    Streamlit's file_uploader gives us). Both work the same way here.
    file_name: the original file name -- used to decide CSV vs Excel, and
    to show the user what they uploaded.

    Raises ValueError for anything that stops us from producing a usable
    dataset (wrong extension, empty file, unparseable content), so the
    caller (app.py) can catch one exception type and show a friendly
    error instead of a raw traceback.
    """
    extension = os.path.splitext(file_name)[1].lower()
    if extension not in (".csv", ".xlsx", ".xls"):
        raise ValueError(
            f"Unsupported file type '{extension}'. Please upload a .csv, .xlsx, or .xls file."
        )

    raw_bytes = _read_all_bytes(file_path_or_buffer)
    if len(raw_bytes) == 0:
        raise ValueError("The uploaded file is empty.")

    if extension == ".csv":
        # dtype=str + keep_default_na=True: read every cell as text, but
        # still let pandas recognise conventional blank markers (empty
        # string, "NaN", "N/A", ...) as missing. We do our own type
        # inference below rather than trusting pandas' auto-typing.
        dataframe = _parse_csv_bytes(raw_bytes, extension)
        raw_text = raw_bytes.decode("utf-8", errors="replace")
    else:
        # Only the FIRST sheet, matching pandas' own default -- this is
        # the single-table entry point kept for backward compatibility.
        # For multi-sheet Excel files, use load_all_tables() instead,
        # which loads every sheet as its own table.
        dataframe = _parse_excel_bytes(raw_bytes, extension, sheet_name=0)
        raw_text = None

    return _build_profile(dataframe, file_name, raw_text)


def _parse_csv_bytes(raw_bytes: bytes, extension: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=True)
    except Exception as parse_error:
        # pandas raises several different exception types for a corrupted/
        # malformed file (ParserError, EmptyDataError, etc.) -- we don't
        # need to distinguish them here, just turn "this file won't
        # parse" into one clear message the caller can show the user.
        raise ValueError(f"Could not read this file as a valid {extension} file ({parse_error}).")


def _parse_excel_bytes(raw_bytes: bytes, extension: str, sheet_name) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(raw_bytes), dtype=str, sheet_name=sheet_name)
    except Exception as parse_error:
        raise ValueError(f"Could not read this file as a valid {extension} file ({parse_error}).")


def _build_profile(dataframe: pd.DataFrame, table_name: str, raw_text: Optional[str]) -> "DatasetProfile":
    """
    Shared final step for both load_dataset (single table) and
    load_all_tables (multi-table): validate the parsed dataframe isn't
    empty, infer every column's type, and package it into a
    DatasetProfile. Centralizing this here means the empty-check and
    type-inference logic can't drift between the single- and multi-table
    code paths.
    """
    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError("This table has no data rows to analyze.")

    column_types = {column: infer_column_type(dataframe[column]) for column in dataframe.columns}

    return DatasetProfile(
        dataframe=dataframe,
        file_name=table_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        column_types=column_types,
        raw_text=raw_text,
    )


# ---- Multi-table / multi-sheet loading ------------------------------------

@dataclass
class TableLoadOutcome:
    """
    The result of trying to load ONE table (one CSV file, or one sheet
    within an uploaded Excel file). Exactly one of `profile` / `error`
    is set -- never both, never neither. Using this "outcome" wrapper
    instead of just raising an exception is what lets load_all_tables
    keep going after one sheet/file fails, instead of the whole batch
    aborting on the first bad table (see this dataclass's use in
    core/pipeline.py's run_multi_table_pipeline).
    """
    table_name: str
    profile: Optional[DatasetProfile] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.profile is not None


def load_all_tables(file_path_or_buffer, file_name: str) -> List[TableLoadOutcome]:
    """
    Load every table contained in one uploaded file:
      - a .csv file is always exactly one table.
      - a .xlsx/.xls file is one table PER SHEET -- an Excel workbook
        with 3 sheets produces 3 TableLoadOutcomes, each named
        "<file_name> -> <sheet_name>" so the UI/fix list can always show
        which specific sheet a finding came from.

    This function is designed to NEVER raise. Every possible failure --
    wrong extension, empty file, a corrupted file, one bad sheet inside
    an otherwise-fine workbook -- becomes a TableLoadOutcome with `error`
    set, not an exception. That is what lets the caller process a whole
    batch of files/sheets and have "one bad table" only cost that one
    table, per the multi-table requirement that a single malformed
    sheet/file must never crash the rest of the upload.
    """
    extension = os.path.splitext(file_name)[1].lower()
    if extension not in (".csv", ".xlsx", ".xls"):
        return [TableLoadOutcome(
            table_name=file_name,
            error=f"Unsupported file type '{extension}'. Please upload a .csv, .xlsx, or .xls file.",
        )]

    try:
        raw_bytes = _read_all_bytes(file_path_or_buffer)
    except Exception as read_error:
        return [TableLoadOutcome(table_name=file_name, error=f"Could not read this file ({read_error}).")]

    if len(raw_bytes) == 0:
        return [TableLoadOutcome(table_name=file_name, error="The uploaded file is empty.")]

    if extension == ".csv":
        return [_load_one_table(raw_bytes, extension, table_name=file_name, sheet_name=None)]

    # Excel: discover every sheet name first (a separate, cheap parse
    # pass via pandas.ExcelFile), then load each sheet independently so
    # one broken sheet can't stop the others from being processed.
    try:
        excel_file = pd.ExcelFile(io.BytesIO(raw_bytes))
        sheet_names = excel_file.sheet_names
    except Exception as parse_error:
        return [TableLoadOutcome(
            table_name=file_name,
            error=f"Could not read this file as a valid {extension} file ({parse_error}).",
        )]

    outcomes = []
    for sheet_name in sheet_names:
        table_name = f"{file_name} -> {sheet_name}"
        outcomes.append(_load_one_table(raw_bytes, extension, table_name=table_name, sheet_name=sheet_name))
    return outcomes


def _load_one_table(raw_bytes: bytes, extension: str, table_name: str, sheet_name) -> TableLoadOutcome:
    """
    Parses one table's raw bytes into a TableLoadOutcome -- success
    (profile set) or failure (error set), never an exception. sheet_name
    is None for CSV (single table, no sheets) or a sheet name/index for
    Excel.
    """
    try:
        if sheet_name is None:
            dataframe = _parse_csv_bytes(raw_bytes, extension)
            raw_text = raw_bytes.decode("utf-8", errors="replace")
        else:
            dataframe = _parse_excel_bytes(raw_bytes, extension, sheet_name=sheet_name)
            raw_text = None
        profile = _build_profile(dataframe, table_name, raw_text)
        return TableLoadOutcome(table_name=table_name, profile=profile)
    except ValueError as known_error:
        return TableLoadOutcome(table_name=table_name, error=str(known_error))
    except Exception as unexpected_error:
        # Belt-and-braces: anything that slips through as a non-ValueError
        # (an unanticipated pandas/openpyxl exception type on a
        # particularly weird sheet) still becomes a clean per-table
        # failure instead of crashing the whole batch.
        return TableLoadOutcome(table_name=table_name, error=f"Unexpected error reading this table: {unexpected_error}")


# ---- Type inference -------------------------------------------------------

BOOLEAN_VALUES = {"true", "false", "yes", "no", "y", "n"}

# Each pattern below matches ONE specific date format. Kept as a list
# (not one giant regex) so consistency.py can reuse the same idea to name
# *which* format a value is in, not just whether it looks date-like.
DATE_FORMAT_PATTERNS = [
    r"^\d{4}-\d{1,2}-\d{1,2}$",              # 2020-01-15
    r"^\d{1,2}/\d{1,2}/\d{4}$",              # 15/03/2019
    r"^\d{1,2}-\d{1,2}-\d{4}$",              # 22-01-2021
    r"^\d{4}/\d{1,2}/\d{1,2}$",              # 2020/07/12
    r"^\d{1,2}-[A-Za-z]{3}-\d{4}$",          # 10-Feb-2018
    r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$",  # Jan 22, 2021
]
_COMBINED_DATE_PATTERN = "|".join(DATE_FORMAT_PATTERNS)

# Characters a human commonly adds to a "number" that stop it from parsing
# as one: currency symbols, thousands separators, surrounding whitespace.
_NUMERIC_FORMATTING_CHARACTERS = re.compile(r"[₹$,\s]")


def infer_column_type(column: pd.Series) -> str:
    """
    Guess what kind of data a column holds, by testing its non-missing
    values against a few patterns. Checked in this order: boolean, date,
    numeric, falling back to text. Order matters -- e.g. we want a column
    of "true"/"false" caught as boolean before anything else has a chance
    to misread it.
    """
    non_missing = column.dropna()
    non_missing = non_missing[non_missing.astype(str).str.strip() != ""]

    if len(non_missing) == 0:
        return COLUMN_TYPE_UNKNOWN

    sample = non_missing.astype(str).str.strip()

    if _looks_boolean(sample):
        return COLUMN_TYPE_BOOLEAN
    if _looks_like_date(sample):
        return COLUMN_TYPE_DATE
    if _looks_numeric(sample):
        return COLUMN_TYPE_NUMERIC
    return COLUMN_TYPE_TEXT


def _looks_boolean(sample: pd.Series) -> bool:
    """A column is boolean-like if almost every value is a recognised
    true/false word AND there are at most 2 distinct values -- this stops
    a text column that happens to contain the word "yes" once from being
    misclassified."""
    lowered = sample.str.lower()
    matches_boolean_word = lowered.isin(BOOLEAN_VALUES)
    return matches_boolean_word.mean() >= TYPE_MATCH_THRESHOLD and lowered.nunique() <= 2


def _looks_like_date(sample: pd.Series) -> bool:
    matches_known_pattern = sample.str.match(_COMBINED_DATE_PATTERN)
    if matches_known_pattern.mean() >= TYPE_MATCH_THRESHOLD:
        return True
    # Fallback: let pandas itself try to parse anything we didn't write a
    # regex for. dayfirst=True matches the DD/MM/YYYY convention common
    # in Indian business data. This is run on every non-date column too
    # (e.g. names, GST numbers) as part of ruling date out, so pandas'
    # "could not infer format" warning is expected noise here, not a
    # real problem -- we suppress it rather than let it spam the console.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return parsed.notna().mean() >= TYPE_MATCH_THRESHOLD


def _looks_numeric(sample: pd.Series) -> bool:
    """
    Deliberately lenient: we strip common number-formatting characters
    (commas, currency symbols, spaces) before checking if what's left
    parses as a plain number. This is intentional -- we WANT "12,50,000"
    to still count as "this column is numeric", so consistency.py can
    flag it as numeric data with bad formatting, rather than ingestion.py
    quietly writing the whole column off as text.
    """
    cleaned = sample.str.replace(_NUMERIC_FORMATTING_CHARACTERS, "", regex=True)
    is_plain_number = cleaned.str.match(r"^-?\d+\.?\d*$")
    return is_plain_number.mean() >= TYPE_MATCH_THRESHOLD
