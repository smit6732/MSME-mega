"""
core/duplication.py

Dimension 3 of the scorecard: DUPLICATION.

Two different problems live under "duplication":
1. Exact duplicate rows -- the same row appears twice, byte for byte.
   Easy to catch with pandas' built-in duplicated() check.
2. Near-duplicate rows -- e.g. "Shree Ganesh Traders" and "Shri Ganesh
   Traders" are almost certainly the same business, entered slightly
   differently by two different people. Pandas can't catch this on its
   own, so we use rapidfuzz to compare text similarity between rows on
   whichever column looks most like a business/record identifier.

No ML anomaly detection here by design -- just pandas' exact-match check
plus rapidfuzz string similarity, per this project's scope.

Adaptive threshold: instead of always requiring a fixed 90% similarity
score to call two names a likely duplicate, we compute EVERY pairwise
similarity score for this table's identity column first, then hand that
full list to core/calibration.py, which finds where THIS table's scores
naturally split into "similar" vs "not similar" (see that file's
docstring for exactly how). We only filter pairs down to the "likely
duplicate" list AFTER calibration decides the cutoff -- calibrating on an
already-filtered list would only ever see "similar" pairs and have
nothing to compare them against.

Performance note (read this before touching the pairwise comparison):
this used to be a manual Python double loop calling
fuzz.token_sort_ratio() once per pair -- correct, but an O(n^2) algorithm
implemented with O(n^2) *Python-level* work is exactly the kind of thing
that's fine at hundreds of rows and unusably slow at thousands (a 3,000
row file has ~4.5 million candidate pairs). Two changes fix this without
changing a single similarity number:
  1. rapidfuzz.process.cdist does the whole batch of comparisons in ONE
     C call instead of millions of individual Python-to-C calls -- same
     algorithm, same scores, far less Python-loop overhead.
  2. Everything downstream of that (finding the flat list of scores for
     calibration, finding which pairs clear the threshold) stays in
     numpy array operations instead of a Python loop -- a Python loop
     that just re-packages an already-computed n^2 matrix into n^2
     tuples would put the O(n^2) Python cost right back.
For genuinely large tables (see BLOCKING_ROW_COUNT_THRESHOLD), a cheap
"blocking" step also runs first so we only ever compare rows that have a
real chance of being near-duplicates, instead of every row against every
other row -- see _blocking_key's docstring.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from core.calibration import calibrate_duplicate_threshold

# Column-name substrings (lowercased), checked in priority order, used to
# auto-pick which text column to run fuzzy matching on. Comparing every
# text column would be slow and mostly meaningless -- fuzzy-matching a
# City column doesn't tell us anything about duplicate business records.
IDENTITY_COLUMN_NAME_HINTS = ["business_name", "company_name", "vendor_name", "name"]

# Below this many candidate rows, comparing every possible pair (using
# rapidfuzz's bulk C implementation, see module docstring) is already
# well under a second, so we don't bother with the added complexity of
# blocking -- we just compare everyone against everyone. Above it, rows
# are grouped into cheap "blocks" first (see _blocking_key) so the
# comparison volume doesn't grow as the full square of the row count.
BLOCKING_ROW_COUNT_THRESHOLD = 800

# How many characters of a row's "sorted words" signature to use as its
# block key. Short enough that a genuine near-duplicate (a typo, extra
# punctuation, a word out of order) almost always still lands in the
# same block; long enough that blocks stay small on a large file.
BLOCKING_PREFIX_LENGTH = 4


@dataclass
class FuzzyDuplicatePair:
    """One pair of rows whose identity-column values look like the same
    business/entity, recorded slightly differently."""
    row_index_a: int
    row_index_b: int
    value_a: str
    value_b: str
    similarity_score: float  # 0-100, higher = more similar


@dataclass
class DuplicationResult:
    identity_column_used: Optional[str]
    exact_duplicate_row_indexes: List[int]
    fuzzy_duplicate_pairs: List[FuzzyDuplicatePair]
    duplicate_row_percentage: float  # % of rows involved in any duplicate (exact or fuzzy)
    duplication_score: float         # 0-100, 100 = no duplicates found
    similarity_threshold_used: float  # the cutoff actually applied for this table
    threshold_was_calibrated: bool    # False if we fell back to the fixed 90% default


def _choose_identity_column(column_types: Dict[str, str]) -> Optional[str]:
    """
    Pick the text column most likely to identify a unique business
    record (e.g. Business_Name), so fuzzy matching runs on something
    meaningful. Falls back to the first text column if nothing matches a
    known naming hint, rather than skipping fuzzy detection entirely.
    """
    text_columns = [name for name, col_type in column_types.items() if col_type == "text"]
    if not text_columns:
        return None

    for hint in IDENTITY_COLUMN_NAME_HINTS:
        for column_name in text_columns:
            if hint in column_name.lower():
                return column_name

    return text_columns[0]


def _find_exact_duplicate_rows(dataframe: pd.DataFrame) -> List[int]:
    """
    Row indexes that are a byte-for-byte duplicate of an earlier row.
    keep="first" means the first occurrence of a repeated row is NOT
    flagged, only the repeat(s) -- which is what we want when telling the
    user how many rows they could safely delete.
    """
    is_duplicate = dataframe.duplicated(keep="first")
    return dataframe.index[is_duplicate].tolist()


def _candidate_rows(
    dataframe: pd.DataFrame, identity_column: str, exact_duplicate_indexes: List[int]
) -> List[Tuple[int, str]]:
    """
    The list of (row_index, identity_value) pairs actually worth
    comparing: rows already flagged as an exact whole-row duplicate are
    excluded (they're already accounted for by the exact-duplicate check
    -- comparing them again here would just report the same relationship
    a second time under a different label), and so are rows with a blank
    identity value (nothing to compare).
    """
    values = dataframe[identity_column].fillna("").astype(str).str.strip()
    exact_duplicate_index_set = set(exact_duplicate_indexes)
    return [
        (index, value)
        for index, value in values.items()
        if index not in exact_duplicate_index_set and value != ""
    ]


def _blocking_key(value: str) -> str:
    """
    A cheap, O(len(value)) signature used to group rows before the
    expensive pairwise comparison on a large table. The value's words
    are sorted alphabetically first -- the same normalization
    fuzz.token_sort_ratio itself does internally -- so "Ganesh Shree
    Traders" and "Shree Ganesh Traders" land in the same block even
    though their word order differs, then we take the first few
    characters of that sorted form as the key.

    This is a trade-off, not a perfect filter: a typo in the very first
    couple of characters of a name (e.g. "Krishna" vs "Kirshna") could,
    in principle, push two genuinely similar rows into different blocks,
    so they'd never get compared. For the row counts this matters at
    (thousands+), that's the right trade to make -- the same blocking
    idea real deduplication pipelines use -- and it turns an unusable
    O(n^2) comparison into one that finishes in seconds instead of
    minutes. Below BLOCKING_ROW_COUNT_THRESHOLD rows, this function is
    never called at all -- every row is compared against every other
    row, so this trade-off only applies where it's actually needed.
    """
    normalized = " ".join(sorted(value.lower().split()))
    return normalized[:BLOCKING_PREFIX_LENGTH]


def _build_blocks(candidates: List[Tuple[int, str]]) -> List[List[Tuple[int, str]]]:
    """
    Split candidate rows into blocks for a large table (see
    BLOCKING_ROW_COUNT_THRESHOLD), or return everyone in a single block
    for a small/medium one -- in which case every row IS compared
    against every other row, identical in effect to the pre-optimization
    version of this module.
    """
    if len(candidates) <= BLOCKING_ROW_COUNT_THRESHOLD:
        return [candidates]

    blocks: Dict[str, List[Tuple[int, str]]] = {}
    for index, value in candidates:
        blocks.setdefault(_blocking_key(value), []).append((index, value))
    return list(blocks.values())


def _score_block(block: List[Tuple[int, str]]) -> np.ndarray:
    """
    RapidFuzz's optimized bulk comparison (implemented in C) across
    every pair within one block, in a single call instead of one
    Python-level call per pair -- this is the actual speed fix described
    in this module's docstring. Returns an NxN matrix of similarity
    scores; only the upper triangle (i<j) is meaningful to the caller
    (the matrix is symmetric, and the diagonal is each value against
    itself). Produces numerically IDENTICAL scores to calling
    fuzz.token_sort_ratio() one pair at a time -- same underlying
    algorithm, just invoked in one batched call instead of many.
    """
    values = [value for (_, value) in block]
    return process.cdist(values, values, scorer=fuzz.token_sort_ratio, workers=-1)


def check_duplication(dataframe: pd.DataFrame, column_types: Dict[str, str]) -> DuplicationResult:
    """
    Run both duplicate checks and combine them into one duplication
    score. The fuzzy-match cutoff used here is calibrated fresh for this
    table (see core/calibration.py), falling back to the fixed 90%
    default when there isn't enough data to calibrate meaningfully.
    """
    total_rows = len(dataframe)
    exact_duplicate_indexes = _find_exact_duplicate_rows(dataframe)
    identity_column = _choose_identity_column(column_types)

    # scored_blocks holds every block alongside its already-computed
    # similarity matrix, so we score each block exactly once and reuse
    # the result both for calibration (below) and for extracting the
    # actual matched pairs afterward -- never recomputing anything.
    scored_blocks: List[Tuple[List[Tuple[int, str]], np.ndarray]] = []
    all_similarity_scores: List[float] = []

    if identity_column:
        candidates = _candidate_rows(dataframe, identity_column, exact_duplicate_indexes)
        for block in _build_blocks(candidates):
            if len(block) < 2:
                continue  # nothing to compare a single row in a block against
            matrix = _score_block(block)
            scored_blocks.append((block, matrix))
            upper_i, upper_j = np.triu_indices(len(block), k=1)
            all_similarity_scores.extend(np.round(matrix[upper_i, upper_j], 2).tolist())

    # Calibrate the threshold using every pairwise score computed above,
    # THEN filter down to the pairs that clear it -- calibrating on an
    # already-filtered list would only ever see "similar" pairs and have
    # nothing to compare them against.
    calibration = calibrate_duplicate_threshold(all_similarity_scores)

    fuzzy_pairs = _extract_matching_pairs(scored_blocks, calibration.value)

    # A row "counts" toward the duplicate percentage if it's an exact
    # duplicate, or it appears on either side of a fuzzy-duplicate pair.
    rows_involved = set(exact_duplicate_indexes)
    for pair in fuzzy_pairs:
        rows_involved.add(pair.row_index_a)
        rows_involved.add(pair.row_index_b)

    duplicate_row_percentage = (len(rows_involved) / total_rows * 100) if total_rows else 0.0
    duplication_score = max(0.0, 100.0 - duplicate_row_percentage)

    return DuplicationResult(
        identity_column_used=identity_column,
        exact_duplicate_row_indexes=exact_duplicate_indexes,
        fuzzy_duplicate_pairs=fuzzy_pairs,
        duplicate_row_percentage=round(duplicate_row_percentage, 2),
        duplication_score=round(duplication_score, 2),
        similarity_threshold_used=calibration.value,
        threshold_was_calibrated=calibration.was_calibrated,
    )


def _extract_matching_pairs(
    scored_blocks: List[Tuple[List[Tuple[int, str]], np.ndarray]], threshold: float
) -> List[FuzzyDuplicatePair]:
    """
    Turns each block's similarity matrix into actual FuzzyDuplicatePair
    objects, but ONLY for pairs at or above the calibrated threshold --
    normally a tiny fraction of all the pairs that were scored. This is
    deliberately the one place a Python-level loop runs over individual
    pairs: by this point numpy has already narrowed things down to just
    the matches (via the boolean comparison + np.where below), so this
    loop runs a handful of times, not n^2 times.
    """
    fuzzy_pairs: List[FuzzyDuplicatePair] = []
    for block, matrix in scored_blocks:
        upper_i, upper_j = np.triu_indices(len(block), k=1)
        scores = matrix[upper_i, upper_j]
        matched_positions = np.where(scores >= threshold)[0]

        for position in matched_positions:
            i, j = upper_i[position], upper_j[position]
            index_a, value_a = block[i]
            index_b, value_b = block[j]
            fuzzy_pairs.append(FuzzyDuplicatePair(
                row_index_a=index_a,
                row_index_b=index_b,
                value_a=value_a,
                value_b=value_b,
                similarity_score=round(float(scores[position]), 2),
            ))
    return fuzzy_pairs
