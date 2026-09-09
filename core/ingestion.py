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

Two robustness passes live in this file, both following the same
guarded philosophy as core/calibration.py -- take confident action, or
fall back to the safe default, but never silently guess and never hide
that a fallback happened:

  - Preamble/title-row detection (_detect_header_row_index): real Tally/
    POS exports commonly put a report title or company name in row 1,
    with the actual column headers a row or two below it. We scan the
    first few rows for a clear width jump (a real header, and the rows
    after it, are consistently wider than a title/preamble line) before
    trusting anything other than "row 0 is the header".
  - Encoding fallback (_decode_csv_bytes): older/regional exports are
    sometimes saved in Windows-1252 or another non-UTF-8 encoding. We
    try UTF-8 first, then ask charset-normalizer to detect the actual
    encoding, then fall back to latin-1 (which never raises) as a last
    resort -- and ALWAYS surface a note when we had to use anything
    other than clean UTF-8, because empirically (see this module's own
    tests) even a "confident" detector result can pick the wrong one of
    several similar single-byte encodings, so pretending certainty we
    don't actually have would be dishonest.
"""

import csv
import io
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import charset_normalizer


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
    # If a preamble was skipped (see skipped_preamble_rows below), this
    # is the text from the real header onward, NOT the original file --
    # otherwise structure.py's ragged-row check would misfire on the
    # intentionally-skipped preamble line(s).
    raw_text: Optional[str] = None
    # How many leading rows (a report title, company letterhead, etc.)
    # were detected and skipped before the real header. 0 means "no
    # preamble detected -- row 0 was used as the header", which is also
    # the honest fallback whenever the detection heuristic isn't
    # confident. NEVER silent: app.py surfaces this whenever it's > 0.
    skipped_preamble_rows: int = 0
    # Set only when the file wasn't clean UTF-8 -- see _decode_csv_bytes.
    # None means "read as UTF-8, no fallback needed". Always None for
    # Excel (openpyxl handles Excel's own internal text encoding; this
    # is a CSV-specific concern).
    encoding_warning: Optional[str] = None


def _read_all_bytes(file_path_or_buffer) -> bytes:
    """
    Accepts either a file path (str/Path) or a file-like object (e.g.
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
        dataframe, raw_text, skipped_rows, encoding_warning = _parse_csv_bytes(raw_bytes, extension)
    else:
        # Only the FIRST sheet, matching pandas' own default -- this is
        # the single-table entry point kept for backward compatibility.
        # For multi-sheet Excel files, use load_all_tables() instead,
        # which loads every sheet as its own table.
        dataframe, skipped_rows = _parse_excel_bytes(raw_bytes, extension, sheet_name=0)
        raw_text = None
        encoding_warning = None

    return _build_profile(
        dataframe, file_name, raw_text,
        skipped_preamble_rows=skipped_rows, encoding_warning=encoding_warning,
    )


# ---- Encoding fallback (Priority 2) ---------------------------------------

# charset-normalizer's "chaos" score (0.0 = no decode inconsistencies at
# all, higher = more). IMPORTANT, verified empirically (see
# tests/test_pipeline.py's encoding test and this module's own dev
# notes): chaos == 0.0 does NOT reliably mean "this is definitely the
# right encoding" -- for short-to-medium business-data files, several
# similar single-byte Western/Central-European codepages (e.g. cp1250
# vs the far more common cp1252) can ALL decode a file with zero
# chaos, because they only differ in a handful of accented-character
# mappings that a small sample might never exercise. So we still use
# chaos to decide whether charset-normalizer found anything plausible
# at all, but we do NOT treat "chaos was low" as license to stay
# silent -- see _decode_csv_bytes below.
_MAX_ACCEPTABLE_CHAOS = 0.2


def _decode_csv_bytes(raw_bytes: bytes) -> Tuple[str, Optional[str]]:
    """
    Decodes raw CSV bytes to text. Returns (text, encoding_warning) --
    the warning is None only when the file was clean UTF-8 and needed no
    fallback at all.

    1. Try UTF-8 (utf-8-sig handles a leading byte-order-mark too) --
       the vast majority of modern exports, and the only case where we
       have real certainty.
    2. If that fails, ask charset-normalizer to detect the actual
       encoding from the bytes themselves.
    3. If detection finds nothing plausible, fall back to latin-1,
       which can decode ANY byte sequence and so never raises -- but
       "never raises" is not the same as "correct".

    Every non-UTF-8 path returns a warning, deliberately more cautious
    than "only warn when detection confidence is low": our own testing
    showed charset-normalizer's confidence score isn't a reliable signal
    for telling apart similar single-byte encodings on typical business
    data, so claiming silent certainty whenever the score merely looked
    good would be overclaiming. Being upfront that SOME fallback
    happened, every time one did, matches the honesty this project
    already holds calibration and AI-phrasing fallbacks to.
    """
    try:
        return raw_bytes.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        pass

    best_match = charset_normalizer.from_bytes(raw_bytes).best()
    if best_match is not None and best_match.chaos <= _MAX_ACCEPTABLE_CHAOS:
        detected_encoding = best_match.encoding
        warning = (
            f"This file wasn't UTF-8 -- it was read using the detected encoding "
            f"'{detected_encoding}'. If any accented or special characters look "
            f"wrong, that's why: automatic encoding detection can't always tell "
            f"apart similar encodings with full certainty."
        )
        return str(best_match), warning

    warning = (
        "This file's encoding could not be confidently detected, so it was read "
        "as Latin-1 (a fallback that never fails, but isn't guaranteed correct) -- "
        "some characters may not display correctly."
    )
    return raw_bytes.decode("latin-1"), warning


# ---- Preamble / title-row detection (Priority 1) ---------------------------

# How many of a table's leading rows to look at when checking for a
# report title/company letterhead line before the real header. Generous
# enough for any realistic preamble (company name, address, report
# title, date range -- rarely more than 2-3 lines) while staying cheap.
_PREAMBLE_SCAN_ROW_LIMIT = 10

# A row counts as part of the table (the header, or a data row right
# after it) if it's at least this fraction as populated as the widest
# row seen -- lenient enough that a data row with a couple of blanks
# still counts, strict enough that a mostly-empty row doesn't.
_TABLE_ROW_WIDTH_RATIO = 0.6

# A row counts as a safely-skippable preamble line only if it's
# populated well BELOW this fraction of the widest row -- a clear gap,
# not a borderline call.
_PREAMBLE_ROW_WIDTH_RATIO = 0.5

# The heuristic only engages when the candidate header itself has at
# least this many populated cells -- a table with only 1-2 columns
# doesn't give it enough signal to tell a preamble line from a genuinely
# narrow table.
_MIN_HEADER_WIDTH_FOR_DETECTION = 3

# How many rows immediately after the candidate header must ALSO stay
# consistently wide before we trust it's really where the table starts,
# not just one coincidentally-wide line in the middle of a preamble.
_MIN_CONSISTENT_ROWS_AFTER_HEADER = 2


def _count_non_empty_cells(row: List[str]) -> int:
    return sum(1 for cell in row if cell is not None and str(cell).strip() != "")


def _detect_header_row_index(raw_rows: List[List[str]]) -> int:
    """
    Looks at a table's first few raw rows for a report title/company
    letterhead line before the real header. Returns the row index to
    treat as the header -- 0 if no confident preamble is found, meaning
    "use row 0", exactly the previous, unconditional behavior.

    HOW (same guarded philosophy as core/calibration.py -- confident
    action or safe fallback, never a guess):
      1. Count non-empty cells in each of the first
         _PREAMBLE_SCAN_ROW_LIMIT rows.
      2. Find the first row reaching the maximum width seen -- our
         header candidate.
      3. Only trust it if ALL of the following hold: the candidate is
         comfortably wide on its own terms; EVERY row before it is
         clearly narrower (a preamble signature); and enough rows AFTER
         it stay consistently close to that width (an actual table
         starting, not a one-off wide line).
      4. If any of that isn't true -- no clear width jump, a "before"
         row that's suspiciously wide too, not enough evidence after the
         candidate -- fall back to 0. Ambiguous cases are exactly what
         this fallback exists for; we do not guess.
    """
    scanned_rows = raw_rows[:_PREAMBLE_SCAN_ROW_LIMIT]
    if len(scanned_rows) < 2:
        return 0  # not enough rows to compare against -- nothing to detect

    widths = [_count_non_empty_cells(row) for row in scanned_rows]
    max_width = max(widths)

    if max_width < _MIN_HEADER_WIDTH_FOR_DETECTION:
        return 0  # too narrow a table for this heuristic to mean anything

    candidate_index = widths.index(max_width)  # first row reaching max width
    if candidate_index == 0:
        return 0  # row 0 is already the widest -- nothing to skip

    rows_before = widths[:candidate_index]
    if any(width > max_width * _PREAMBLE_ROW_WIDTH_RATIO for width in rows_before):
        return 0  # not a clean gap -- some "before" row is suspiciously wide too

    rows_after = widths[candidate_index + 1:]
    if len(rows_after) < _MIN_CONSISTENT_ROWS_AFTER_HEADER:
        return 0  # not enough rows after the candidate to confirm a real table
    if any(width < max_width * _TABLE_ROW_WIDTH_RATIO for width in rows_after[:_MIN_CONSISTENT_ROWS_AFTER_HEADER]):
        return 0  # widths right after the candidate aren't consistent -- ambiguous

    return candidate_index


def _rows_to_csv_text(rows: List[List[str]]) -> str:
    """Re-serializes a list of raw CSV rows back into CSV text. Used only
    to rebuild raw_text starting from the real header after a preamble is
    skipped, so structure.py's ragged-row check (which reads raw_text)
    sees the same table pandas parsed, not the original file's preamble
    lines. Field-count-wise this is exactly equivalent to slicing the
    original text; re-serializing (rather than hunting for byte offsets
    in the original string) is what makes this correct even when a
    quoted field contains an embedded newline before the header."""
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


# ---- CSV / Excel parsing ----------------------------------------------------

def _parse_csv_bytes(raw_bytes: bytes, extension: str) -> Tuple[pd.DataFrame, str, int, Optional[str]]:
    """Returns (dataframe, raw_text_from_the_real_header_onward,
    skipped_preamble_rows, encoding_warning)."""
    raw_text, encoding_warning = _decode_csv_bytes(raw_bytes)

    all_rows = list(csv.reader(io.StringIO(raw_text)))
    header_row_index = _detect_header_row_index(all_rows)

    try:
        if header_row_index == 0:
            # Identical to this module's previous behavior -- zero risk
            # of changing anything for the (overwhelmingly common) case
            # where row 0 already is the header.
            dataframe = pd.read_csv(io.StringIO(raw_text), dtype=str, keep_default_na=True)
            effective_raw_text = raw_text
        else:
            # skip_blank_lines=False here (deliberately different from
            # pandas' own default of True) so "skip N rows" means
            # exactly the same thing to pandas as it did to our own
            # row-counting above -- otherwise a blank line inside the
            # preamble could make pandas and our own detection disagree
            # about which physical row is the header.
            dataframe = pd.read_csv(
                io.StringIO(raw_text), dtype=str, keep_default_na=True,
                skiprows=header_row_index, header=0, skip_blank_lines=False,
            )
            effective_raw_text = _rows_to_csv_text(all_rows[header_row_index:])
    except Exception as parse_error:
        raise ValueError(f"Could not read this file as a valid {extension} file ({parse_error}).")

    return dataframe, effective_raw_text, header_row_index, encoding_warning


def _parse_excel_bytes(raw_bytes: bytes, extension: str, sheet_name) -> Tuple[pd.DataFrame, int]:
    """Returns (dataframe, skipped_preamble_rows). No encoding concern
    here -- openpyxl handles Excel's own internal text encoding, so
    Priority 2 (CSV byte-encoding fallback) doesn't apply."""
    try:
        preview = pd.read_excel(
            io.BytesIO(raw_bytes), sheet_name=sheet_name, header=None,
            nrows=_PREAMBLE_SCAN_ROW_LIMIT, dtype=str,
        )
        preview_rows = preview.fillna("").astype(str).values.tolist()
        header_row_index = _detect_header_row_index(preview_rows)

        dataframe = pd.read_excel(
            io.BytesIO(raw_bytes), dtype=str, sheet_name=sheet_name, header=header_row_index,
        )
    except Exception as parse_error:
        raise ValueError(f"Could not read this file as a valid {extension} file ({parse_error}).")

    return dataframe, header_row_index


def _build_profile(
    dataframe: pd.DataFrame,
    table_name: str,
    raw_text: Optional[str],
    skipped_preamble_rows: int = 0,
    encoding_warning: Optional[str] = None,
) -> "DatasetProfile":
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
        skipped_preamble_rows=skipped_preamble_rows,
        encoding_warning=encoding_warning,
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
            dataframe, raw_text, skipped_rows, encoding_warning = _parse_csv_bytes(raw_bytes, extension)
        else:
            dataframe, skipped_rows = _parse_excel_bytes(raw_bytes, extension, sheet_name=sheet_name)
            raw_text = None
            encoding_warning = None
        profile = _build_profile(
            dataframe, table_name, raw_text,
            skipped_preamble_rows=skipped_rows, encoding_warning=encoding_warning,
        )
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
