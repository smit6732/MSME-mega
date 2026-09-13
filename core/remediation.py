"""
core/remediation.py

Tier 1: turns Tier 0's already-computed Findings (core/pipeline.py's
TableResult.findings) into an actual cleaned dataset.

Finding-driven, not a fresh scan -- this is the central design rule for
this whole module. Every decision here starts from a Finding that
core/fixlist.py already built (issue_type, column_name,
percentage_affected, example) -- this module never re-inspects the raw
data to decide something is "wrong" in the first place. It only decides,
per Finding, whether the problem that Finding already identified can be
fixed SAFELY and AUTOMATICALLY, using that Finding's own facts.

Two non-negotiable rules, unchanged from every guard in this project
(see core/calibration.py, and the Priority 1/2 fixes in
core/ingestion.py):
  1. Never guess when correctness is genuinely ambiguous. A value that
     could plausibly mean two different things after cleaning (is
     "03-04-2020" the 3rd of April or the 4th of March? is "12,34" a
     typo or a 2-digit-grouped number?) is left exactly as it was, not
     "fixed" with a coin flip.
  2. Nothing silent. Every Finding ends up in exactly one place: either
     it produced a RemediationAction (something was actually changed,
     with a real before/after pair to prove it), or it's on the manual
     review list with a plain-language reason. No Finding is ever just
     dropped.

Finding.issue_type -> what happens (exact mapping, see the project's
Tier 1 spec):
    missing_data                 -> manual review by default; or, if
                                     config.missing_strategy isn't "leave",
                                     a configurable fill/drop (never the
                                     default -- see CleaningConfig)
    missing_value_representation -> auto-fix (unconditional): a disguised-
                                     missing placeholder token ("N/A",
                                     "Unknown", ...) is replaced with a
                                     true blank -- this is normalizing a
                                     REPRESENTATION, never a guess at a
                                     real value, so it needs no guard
    inconsistent_format_numeric  -> guarded auto-fix (strip thousands/currency)
    inconsistent_format_date     -> guarded auto-fix (normalize to YYYY-MM-DD)
    inconsistent_format_text     -> auto-fix (trim whitespace always; casing
                                     only when config allows it AND one
                                     spelling already dominates the column)
    invalid_type_in_numeric      -> auto-fix (clear to blank -- NEVER imputed)
    invalid_type_in_date         -> auto-fix (clear to blank -- NEVER imputed)
    categorical_inconsistency    -> auto-fix, but ONLY the specific minority
                                     spellings validity.py already matched to
                                     a canonical form at or above this run's
                                     confidence threshold (config) -- reuses
                                     that exact map, never re-derives it
    domain_outlier                -> auto-fix: set to blank (default) or clip
                                     to the nearest already-computed bound
                                     (config.outlier_action) -- reuses the
                                     exact bounds validity.py computed
    exact_duplicate_rows         -> auto-fix (drop rows via duplication.py's
                                     own exact_duplicate_row_indexes -- never
                                     recomputed here)
    fuzzy_duplicate_rows         -> manual review (which one is "correct"?)
    structural_issue             -> manual review (e.g. duplicate identifier --
                                     no way to know which row holds the real one)

Every one of the new auto-fixes above still obeys this module's two
non-negotiable rules from the top of this docstring: type-coercion never
guesses a replacement value (it clears to blank, exactly like a
formatting fix that can't confidently normalize a value declines rather
than guesses); categorical standardization and domain-outlier handling
only ever reuse facts core/validity.py already computed (Finding.details
-- see that module's docstring), never re-deriving them independently
against a possibly-already-modified copy of the data.

This module is scoring-blind on purpose: it never touches, calls, or
imports anything from core/scoring.py, and it always receives the
dataframe as an already-independent copy to mutate (see remediate_table).
The diagnostic score is computed once, upstream, from the ORIGINAL data
-- cleaning is a separate output path that runs afterward and can never
feed back into it. That is the one property every test in this module
exists to protect.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

from core.findings import Finding
from core.validity import (
    is_coercible_numeric_token, is_recognized_date_token, is_missing_representation_token, categorical_key,
)
from core.cleaning_config import (
    CleaningConfig,
    CONSERVATIVE,
    MISSING_STRATEGY_LEAVE,
    MISSING_STRATEGY_DROP_ROW,
    MISSING_STRATEGY_CONSTANT,
    MISSING_STRATEGY_MEDIAN,
    MISSING_STRATEGY_MODE,
    OUTLIER_ACTION_NAN,
    OUTLIER_ACTION_CLIP,
)

# Numeric-junk characters stripped before parsing an already-validated
# numeric token to a float -- same character class as
# core/validity.py's own _NUMERIC_JUNK_PATTERN (kept as a separate
# constant, not imported, for the same reason core/consistency.py
# duplicates rather than imports this project's other numeric-junk
# patterns: each module's copy is allowed to evolve independently
# without silently changing another module's behavior).
_NUMERIC_JUNK_PATTERN = re.compile(r"[₹$,\s]")

# A column's dominant EXACT spelling must cover at least this share of
# its non-missing values before config.allow_categorical_case_normalization
# will standardize the rest to match it -- "confident majority, not a
# coin flip", the same bar this project holds every auto-fix to.
_DOMINANT_CASE_SHARE_THRESHOLD = 0.8

# Currency symbols and stray whitespace a human commonly adds around a
# number -- same idea as core/consistency.py's own _NUMERIC_JUNK_PATTERN,
# duplicated here (not imported) because that pattern strips commas too,
# and this module needs to reason about commas separately (see
# _guarded_clean_numeric) rather than just discard them blindly.
_CURRENCY_AND_WHITESPACE = re.compile(r"[₹$€£\s]")

# A simple 3-part, all-digit date, split on the usual separators. Any
# date that doesn't look like this (a month name, a 2-part date, extra
# junk) is left alone -- normalizing it would require guessing at a
# format we have no evidence for.
_DATE_SEPARATOR = re.compile(r"[\-/.]")


@dataclass
class RemediationAction:
    """
    One safe, automatic fix that was actually applied to a column (or,
    for exact-duplicate removal, to the whole table). source_finding
    keeps the Tier 0 connection explicit -- every action here can always
    be traced back to the exact Finding that triggered it.
    """
    column_name: str
    action_type: str          # e.g. "strip_numeric_formatting", "normalize_date_format",
                               #   "trim_whitespace", "remove_exact_duplicate_rows"
    count_affected: int        # how many cells/rows were ACTUALLY changed (never a claimed estimate)
    before_example: str
    after_example: str
    source_finding: Finding
    # Not part of the original four-field sketch, but required by this
    # project's own "nothing silent" rule: a column can be MOSTLY
    # fixable but still contain a few genuinely ambiguous values that
    # were correctly left untouched (see _guarded_clean_numeric /
    # _guarded_normalize_date). Reporting count_affected alone would
    # make that column look identical, in the audit log, to one with no
    # ambiguous values at all -- this field is what keeps that visible.
    notes: Optional[str] = None


@dataclass
class ManualReviewItem:
    """One Finding that was deliberately NOT auto-fixed, with a plain-
    language reason a non-technical user can actually understand."""
    finding: Finding
    reason: str


@dataclass
class RemediationResult:
    """Everything app.py needs to show for one table's Tier 1 output."""
    cleaned_dataframe: pd.DataFrame
    actions: List[RemediationAction] = field(default_factory=list)
    manual_review: List[ManualReviewItem] = field(default_factory=list)


# ---- Guarded numeric cleaning ----------------------------------------------

def _guarded_clean_numeric(raw_value: str) -> Optional[str]:
    """
    Strips thousands separators and currency symbols from a numeric-
    looking string (e.g. "₹25,00,000" -> "2500000"), but ONLY when the
    result is unambiguous. Returns None -- meaning "leave this value
    alone" -- whenever it isn't.

    Two guard rules, both there to stop a guess:
      1. More than one "." after stripping currency/whitespace could
         mean this is decimal-comma notation (e.g. European-style
         "2.500.000,75", where "." is the THOUSANDS separator and ","
         is the decimal point) rather than the comma-thousands/dot-
         decimal format this function assumes. We can't tell which from
         the string alone, so we decline rather than silently strip
         digits that might belong on the wrong side of a decimal point.
      2. Comma groups must fit a real thousands-grouping convention: the
         LAST group must be exactly 3 digits (both Indian and Western
         grouping agree on this -- it's always the hundreds/tens/units),
         and every group BETWEEN the first and last (if any) must be
         consistently 2 digits wide (Indian, e.g. "25,00,000" means
         25 | 00 | 000) or consistently 3 digits wide (Western, e.g.
         "2,500,000" means 2 | 500 | 000). A grouping that fits neither
         -- e.g. "1,2,345" -- means the commas might not be thousands
         separators at all, so we decline rather than guess.
    """
    text = _CURRENCY_AND_WHITESPACE.sub("", raw_value.strip())
    if text == "":
        return None

    sign = ""
    if text[:1] in ("-", "+"):
        sign = "-" if text[0] == "-" else ""
        text = text[1:]
    if text == "":
        return None

    if text.count(".") > 1:
        return None  # ambiguous -- could be decimal-comma notation

    if "." in text:
        integer_part, decimal_part = text.rsplit(".", 1)
        if decimal_part == "" or not decimal_part.isdigit():
            return None
    else:
        integer_part, decimal_part = text, None

    if "," in integer_part:
        groups = integer_part.split(",")
        if len(groups) < 2 or not all(group.isdigit() for group in groups):
            return None
        first_group, last_group = groups[0], groups[-1]
        middle_groups = groups[1:-1]
        if not (1 <= len(first_group) <= 3):
            return None
        if len(last_group) != 3:
            return None  # the final group is always 3 digits in a real thousands grouping
        if middle_groups:
            middle_widths = {len(group) for group in middle_groups}
            if middle_widths not in ({2}, {3}):
                return None  # middle groups must be consistently Indian (2) or Western (3)
        integer_part = "".join(groups)

    if not integer_part.isdigit():
        return None

    cleaned = integer_part if decimal_part is None else f"{integer_part}.{decimal_part}"
    return sign + cleaned


# ---- Guarded date normalization --------------------------------------------

def _guarded_normalize_date(raw_value: str) -> Optional[str]:
    """
    Normalizes a simple numeric date (e.g. "15-01-2020" or "2020/01/15")
    to canonical YYYY-MM-DD, but ONLY when the day/month order is
    unambiguous. Returns None -- "leave this value alone" -- otherwise.

    The year's POSITION matters, not just its value:
      - A LEADING 4-digit year (e.g. "2020-07-22") is, by overwhelming
        convention, already YYYY-MM-DD -- the two remaining components
        are month-then-day in that order, no guessing needed. We only
        decline here if the "month" slot is actually > 12 (e.g. a
        malformed "2020-13-05"), since that can't be a real YYYY-MM-DD
        value at all.
      - A TRAILING 4-digit year (e.g. "15-01-2020") could be DD-MM-YYYY
        or MM-DD-YYYY -- genuinely two different conventions -- so the
        order is only safe to infer when exactly one of the two
        components is greater than 12 (which rules out it being a
        month). "03-04-2020" is the textbook ambiguous case the project
        spec calls out by name: both 3 and 4 are valid months AND valid
        days, so there is no way to know if this means 3 April or
        4 March without guessing -- it is left untouched.
      - A year in the MIDDLE (e.g. "03-2020-04") isn't a format this
        function recognizes at all -- declined, not guessed at.
    """
    text = raw_value.strip()
    if text == "":
        return None

    parts = _DATE_SEPARATOR.split(text)
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None  # not a plain 3-part numeric date -- e.g. "15-Jan-2020"

    year_positions = [i for i, part in enumerate(parts) if len(part) == 4]
    if len(year_positions) != 1:
        return None  # 0 or 2+ four-digit parts -- can't confidently place the year
    year_index = year_positions[0]
    year = parts[year_index]

    if year_index == 0:
        # YYYY-?-? -- trust the positional convention directly, rather
        # than re-deriving month/day order from an ambiguity test that
        # would otherwise flag perfectly good, already-ISO dates (e.g.
        # "2021-01-10", where both remaining parts are <= 12) as
        # "ambiguous" just because they happen to both be small numbers.
        month, day = int(parts[1]), int(parts[2])
    elif year_index == 2:
        first_value, second_value = int(parts[0]), int(parts[1])
        if first_value > 12 and second_value > 12:
            return None  # neither can be a month -- not a valid date
        if first_value > 12:
            day, month = first_value, second_value
        elif second_value > 12:
            month, day = first_value, second_value
        else:
            return None  # both components are valid as day OR month -- genuinely ambiguous
    else:
        return None  # year in the middle -- not a format this function recognizes

    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None

    return f"{year}-{month:02d}-{day:02d}"


# ---- Per-Finding dispatch ----------------------------------------------------

def _apply_guarded_column_fix(
    dataframe: pd.DataFrame,
    finding: Finding,
    guarded_cleaner,
    action_type: str,
    all_ambiguous_reason: str,
    ambiguous_note_word: str,
    manual_review: List[ManualReviewItem],
    actions: List[RemediationAction],
) -> None:
    """
    Shared loop for both guarded fixers (numeric and date): try the
    guarded cleaner on every non-missing value in the column, mutating
    only the ones it accepts, and leaving everything it declines exactly
    as-is. Factored out because the numeric and date fixers are
    otherwise identical in shape -- only the cleaning function and the
    wording differ.
    """
    column = finding.column_name
    if column not in dataframe.columns:
        manual_review.append(ManualReviewItem(finding, f"Column '{column}' not found in the data -- cannot clean."))
        return

    before_example: Optional[str] = None
    after_example: Optional[str] = None
    changed_count = 0
    ambiguous_count = 0

    for row_index, raw_value in dataframe[column].items():
        if pd.isna(raw_value):
            continue
        raw_text = str(raw_value)
        cleaned_value = guarded_cleaner(raw_text)
        if cleaned_value is None:
            ambiguous_count += 1
            continue
        if cleaned_value != raw_text:
            dataframe.at[row_index, column] = cleaned_value
            changed_count += 1
            if before_example is None:
                before_example, after_example = raw_text, cleaned_value

    if changed_count == 0:
        manual_review.append(ManualReviewItem(finding, all_ambiguous_reason))
        return

    notes = None
    if ambiguous_count:
        word = ambiguous_note_word if ambiguous_count == 1 else ambiguous_note_word + "s"
        was_or_were = "was" if ambiguous_count == 1 else "were"
        notes = f"{ambiguous_count} {word} in this column {was_or_were} ambiguous and left unchanged."

    actions.append(RemediationAction(
        column_name=column,
        action_type=action_type,
        count_affected=changed_count,
        before_example=before_example,
        after_example=after_example,
        source_finding=finding,
        notes=notes,
    ))


def _apply_or_decline_numeric(dataframe, finding, actions, manual_review) -> None:
    reason = (
        "Every value in this column was too ambiguous to safely reformat "
        "(the comma/decimal grouping didn't clearly match a known pattern) "
        "-- couldn't tell without guessing."
    )
    _apply_guarded_column_fix(
        dataframe, finding, _guarded_clean_numeric, "strip_numeric_formatting",
        reason, "value", manual_review, actions,
    )


def _apply_or_decline_date(dataframe, finding, actions, manual_review) -> None:
    reason = (
        "Every date in this column was ambiguous (day/month could plausibly "
        "be either way round, e.g. 03-04-2020) -- couldn't tell which without guessing."
    )
    _apply_guarded_column_fix(
        dataframe, finding, _guarded_normalize_date, "normalize_date_format",
        reason, "date", manual_review, actions,
    )


def _apply_text_trim(dataframe, finding, config: CleaningConfig, actions, manual_review) -> None:
    """
    Whitespace trimming is unconditional (no ambiguity guard needed) --
    but "unconditional" doesn't mean this column definitely has
    whitespace to trim. When config.allow_categorical_case_normalization
    is True, this ALSO standardizes casing to the column's dominant
    exact spelling, but only when that one spelling already covers at
    least _DOMINANT_CASE_SHARE_THRESHOLD of the column's non-missing
    values (see that constant's docstring) -- a genuine, confident
    majority, not a coin flip between two similarly-common spellings.
    With the default (conservative) config, this function's behavior is
    unchanged from before this feature existed: trim only, casing always
    left alone. If nothing at all changes, decline rather than claim a
    fix that did nothing.
    """
    column = finding.column_name
    if column not in dataframe.columns:
        manual_review.append(ManualReviewItem(finding, f"Column '{column}' not found in the data -- cannot clean."))
        return

    before_example: Optional[str] = None
    after_example: Optional[str] = None
    trimmed_count = 0

    for row_index, raw_value in dataframe[column].items():
        if pd.isna(raw_value):
            continue
        raw_text = str(raw_value)
        stripped = raw_text.strip()
        if stripped != raw_text:
            dataframe.at[row_index, column] = stripped
            trimmed_count += 1
            if before_example is None:
                before_example, after_example = raw_text, stripped

    case_changed_count = 0
    dominant_spelling: Optional[str] = None
    if config.allow_categorical_case_normalization:
        non_missing = dataframe[column].dropna().astype(str)
        non_missing = non_missing[non_missing.str.strip() != ""]
        if len(non_missing) > 0:
            value_counts = non_missing.value_counts()
            candidate_spelling, candidate_count = value_counts.index[0], int(value_counts.iloc[0])
            if candidate_count / len(non_missing) >= _DOMINANT_CASE_SHARE_THRESHOLD:
                dominant_spelling = candidate_spelling
                for row_index, raw_value in dataframe[column].items():
                    if pd.isna(raw_value):
                        continue
                    raw_text = str(raw_value)
                    if raw_text != dominant_spelling and raw_text.lower() == dominant_spelling.lower():
                        dataframe.at[row_index, column] = dominant_spelling
                        case_changed_count += 1
                        if before_example is None:
                            before_example, after_example = raw_text, dominant_spelling

    total_changed = trimmed_count + case_changed_count
    if total_changed == 0:
        manual_review.append(ManualReviewItem(
            finding,
            "No leading/trailing whitespace found to trim -- the formatting "
            "difference here is something this tool doesn't automatically fix (e.g. casing).",
        ))
        return

    notes = None
    if case_changed_count:
        notes = f"Also standardized {case_changed_count} value(s)' casing to the dominant spelling '{dominant_spelling}'."

    # repr() here (not the plain strings) is deliberate: a whitespace-only
    # difference like "Foo " vs "Foo" is otherwise invisible once rendered
    # as plain text -- repr() keeps the actual trimmed spaces visible in
    # the audit log.
    actions.append(RemediationAction(
        column_name=column,
        action_type="trim_whitespace",
        count_affected=total_changed,
        before_example=repr(before_example),
        after_example=repr(after_example),
        source_finding=finding,
        notes=notes,
    ))


def _remove_exact_duplicates(dataframe, finding, exact_duplicate_row_indexes, actions, manual_review) -> pd.DataFrame:
    """
    Drops exactly the rows core/duplication.py already identified as
    exact duplicates -- reusing exact_duplicate_row_indexes directly,
    never recomputing it. If, for any reason, the Finding says there are
    duplicates but the index list handed in is empty (should not happen
    in practice -- both come from the same duplication_result), this is
    treated as a "can't safely act" case and goes to manual review rather
    than silently doing nothing.
    """
    if not exact_duplicate_row_indexes:
        manual_review.append(ManualReviewItem(
            finding, "This table's duplicate-row list was empty when remediation ran -- nothing to remove.",
        ))
        return dataframe

    cleaned = dataframe.drop(index=exact_duplicate_row_indexes)
    actions.append(RemediationAction(
        column_name="(whole row)",
        action_type="remove_exact_duplicate_rows",
        count_affected=len(exact_duplicate_row_indexes),
        before_example=f"{len(exact_duplicate_row_indexes)} duplicate row(s) present (e.g. row {exact_duplicate_row_indexes[0]})",
        after_example="row(s) removed",
        source_finding=finding,
    ))
    return cleaned


# ---- Type coercion (invalid_type_in_numeric / invalid_type_in_date) --------

def _apply_type_coercion(dataframe, finding, is_valid_token, action_type, actions, manual_review) -> None:
    """
    Shared loop for both type-coercion fixers: clears (sets to blank)
    every non-missing cell that ISN'T a valid instance of the column's
    type, using the exact same predicate core/validity.py used to
    detect the problem in the first place (is_coercible_numeric_token /
    is_recognized_date_token -- see this module's imports). Never
    imputes a replacement value -- "unknown" in a Price column becomes
    blank, never a guessed number, per this module's "never guess" rule.
    """
    column = finding.column_name
    if column not in dataframe.columns:
        manual_review.append(ManualReviewItem(finding, f"Column '{column}' not found in the data -- cannot clean."))
        return

    before_example: Optional[str] = None
    after_example: Optional[str] = None
    changed_count = 0

    for row_index, raw_value in dataframe[column].items():
        if pd.isna(raw_value):
            continue
        raw_text = str(raw_value)
        if is_valid_token(raw_text):
            continue
        dataframe.at[row_index, column] = None
        changed_count += 1
        if before_example is None:
            before_example, after_example = raw_text, "(blank)"

    if changed_count == 0:
        manual_review.append(ManualReviewItem(
            finding, "No non-valid values found in this column when remediation ran -- nothing to clear.",
        ))
        return

    actions.append(RemediationAction(
        column_name=column,
        action_type=action_type,
        count_affected=changed_count,
        before_example=before_example,
        after_example=after_example,
        source_finding=finding,
        notes="Invalid values were cleared to blank rather than guessed at -- fill them in manually if the real value is known.",
    ))


# ---- missing_value_representation ------------------------------------------

def _apply_missing_token_normalization(dataframe, finding, actions, manual_review) -> None:
    """
    Unconditional, unguarded auto-fix: a cell holding a disguised-missing
    placeholder token ("N/A", "Unknown", ...) becomes a true blank. This
    is never a guess -- it's normalizing how "nothing here" is
    represented, not inventing a value -- so unlike missing_data itself,
    this fix needs no CleaningConfig gate.
    """
    column = finding.column_name
    if column not in dataframe.columns:
        manual_review.append(ManualReviewItem(finding, f"Column '{column}' not found in the data -- cannot clean."))
        return

    before_example: Optional[str] = None
    after_example: Optional[str] = None
    changed_count = 0

    for row_index, raw_value in dataframe[column].items():
        if pd.isna(raw_value):
            continue
        raw_text = str(raw_value)
        if is_missing_representation_token(raw_text):
            dataframe.at[row_index, column] = None
            changed_count += 1
            if before_example is None:
                before_example, after_example = raw_text, "(blank)"

    if changed_count == 0:
        manual_review.append(ManualReviewItem(
            finding, "No placeholder tokens found in this column when remediation ran -- nothing to normalize.",
        ))
        return

    actions.append(RemediationAction(
        column_name=column,
        action_type="normalize_missing_token",
        count_affected=changed_count,
        before_example=before_example,
        after_example=after_example,
        source_finding=finding,
    ))


# ---- categorical_inconsistency ---------------------------------------------

def _apply_categorical_standardization(dataframe, finding, config: CleaningConfig, actions, manual_review) -> None:
    """
    Rewrites ONLY the specific minority spellings core/validity.py
    already matched to a canonical form (Finding.details["canonical_map"]
    -- see that module's _check_categorical_inconsistency), and only the
    ones whose match score clears THIS run's confidence threshold
    (config.categorical_fuzzy_threshold). Never re-runs fuzzy matching
    here, never invents a mapping validity.py didn't already find --
    this function's whole job is applying an already-computed decision,
    not making a new one.
    """
    column = finding.column_name
    if column not in dataframe.columns:
        manual_review.append(ManualReviewItem(finding, f"Column '{column}' not found in the data -- cannot clean."))
        return

    canonical_map = (finding.details or {}).get("canonical_map") or {}
    trusted_map = {
        normalized: mapping for normalized, mapping in canonical_map.items()
        if mapping["score"] >= config.categorical_fuzzy_threshold
    }
    if not trusted_map:
        manual_review.append(ManualReviewItem(
            finding,
            f"No spelling-variant match cleared this run's confidence threshold "
            f"({config.categorical_fuzzy_threshold:.0f}%) -- left unchanged.",
        ))
        return

    before_example: Optional[str] = None
    after_example: Optional[str] = None
    changed_count = 0
    skipped_count = 0

    for row_index, raw_value in dataframe[column].items():
        if pd.isna(raw_value):
            continue
        raw_text = str(raw_value)
        key = categorical_key(raw_text)
        if key in trusted_map:
            canonical = trusted_map[key]["canonical"]
            if raw_text.strip() != canonical:
                dataframe.at[row_index, column] = canonical
                changed_count += 1
                if before_example is None:
                    before_example, after_example = raw_text, canonical
        elif key in canonical_map:
            # Matched at detection time, but this run's config trusts
            # only higher-confidence matches -- left alone, not silently
            # dropped (see the notes on the resulting action, or the
            # manual-review reason if nothing else in the column changed).
            skipped_count += 1

    if changed_count == 0:
        manual_review.append(ManualReviewItem(
            finding, "No spelling-variant values in this column could be confidently standardized this run.",
        ))
        return

    notes = None
    if skipped_count:
        word = "value" if skipped_count == 1 else "values"
        notes = (
            f"{skipped_count} {word} matched a possible variant but scored below this run's "
            f"confidence threshold and were left unchanged."
        )

    actions.append(RemediationAction(
        column_name=column,
        action_type="standardize_categorical_spelling",
        count_affected=changed_count,
        before_example=before_example,
        after_example=after_example,
        source_finding=finding,
        notes=notes,
    ))


# ---- domain_outlier ---------------------------------------------------------

def _apply_domain_outlier_fix(dataframe, finding, config: CleaningConfig, actions, manual_review) -> None:
    """
    Reuses the exact bounds core/validity.py already computed for this
    column (Finding.details -- "lower_bound"/"upper_bound", or
    "allowed_set" for a small fixed legal set like Doors) to decide
    which cells violate them; never re-derives a bound here. Neutralizes
    a violation by setting it to blank (config.outlier_action ==
    OUTLIER_ACTION_NAN, the default) or clipping it to the nearest bound
    (OUTLIER_ACTION_CLIP) -- an allowed_set rule is always neutralized to
    blank regardless of config, since "clip to the nearest legal value"
    isn't a meaningful operation on a small discrete set (is 9 doors
    closer to 5 or is it just wrong? -- we don't guess).
    """
    column = finding.column_name
    if column not in dataframe.columns:
        manual_review.append(ManualReviewItem(finding, f"Column '{column}' not found in the data -- cannot clean."))
        return

    details = finding.details or {}
    allowed_set = set(details["allowed_set"]) if details.get("allowed_set") is not None else None
    lower_bound, upper_bound = details.get("lower_bound"), details.get("upper_bound")
    if allowed_set is None and lower_bound is None and upper_bound is None:
        manual_review.append(ManualReviewItem(
            finding, "This column's safe numeric bounds couldn't be determined -- needs manual review.",
        ))
        return

    clip_mode = allowed_set is None and config.outlier_action == OUTLIER_ACTION_CLIP

    before_example: Optional[str] = None
    after_example: Optional[str] = None
    changed_count = 0

    for row_index, raw_value in dataframe[column].items():
        if pd.isna(raw_value):
            continue
        raw_text = str(raw_value)
        if not is_coercible_numeric_token(raw_text):
            continue  # not a valid number at all -- invalid_type_in_numeric's job, not this one's
        value = float(_NUMERIC_JUNK_PATTERN.sub("", raw_text.strip()))

        if allowed_set is not None:
            violates = value not in allowed_set
        else:
            violates = (lower_bound is not None and value < lower_bound) or (upper_bound is not None and value > upper_bound)
        if not violates:
            continue

        if clip_mode:
            if lower_bound is not None and value < lower_bound:
                new_value = lower_bound
            else:
                new_value = upper_bound
            new_text = f"{new_value:g}"
            dataframe.at[row_index, column] = new_text
        else:
            new_text = "(blank)"
            dataframe.at[row_index, column] = None

        changed_count += 1
        if before_example is None:
            before_example, after_example = raw_text, new_text

    if changed_count == 0:
        manual_review.append(ManualReviewItem(
            finding, "No values in this column violated its known bounds when remediation ran -- nothing to change.",
        ))
        return

    notes = "Out-of-range values were clipped to the nearest realistic bound." if clip_mode else \
        "Out-of-range values were set to blank rather than guessed at."
    actions.append(RemediationAction(
        column_name=column,
        action_type="clip_domain_outlier" if clip_mode else "neutralize_domain_outlier",
        count_affected=changed_count,
        before_example=before_example,
        after_example=after_example,
        source_finding=finding,
        notes=notes,
    ))


# ---- missing_data (configurable strategy, default stays manual review) -----

def _apply_missing_data_strategy(
    dataframe, finding, config: CleaningConfig, actions, manual_review, rows_pending_drop: Set[int],
) -> None:
    """
    Default (config.missing_strategy == "leave", i.e. CONSERVATIVE)
    behaves EXACTLY as this project always has: missing data always goes
    to manual review, because there is no universally-safe way to guess
    what belongs in a blank cell. Only when a caller explicitly opts
    into a different strategy (Standard/Aggressive presets, or a custom
    CleaningConfig) does this function actually fill or drop anything --
    and even then, it only ever fills with a plain constant/median/mode,
    never a model-based guess.

    Row drops are NOT applied immediately (see rows_pending_drop) --
    dropping mid-loop, while other Findings for OTHER columns on the
    same row are still being processed by index, would risk a stale-
    index error. remediate_table applies every pending drop once, after
    every other Finding has already been handled.
    """
    if config.missing_strategy == MISSING_STRATEGY_LEAVE:
        manual_review.append(ManualReviewItem(
            finding,
            f"{finding.percentage_affected:.1f}% of '{finding.column_name}' is missing -- there's no safe, "
            "universal way to guess what belongs in a blank cell.",
        ))
        return

    column = finding.column_name
    if column not in dataframe.columns:
        manual_review.append(ManualReviewItem(finding, f"Column '{column}' not found in the data -- cannot clean."))
        return

    def _is_blank(value) -> bool:
        return pd.isna(value) or (isinstance(value, str) and value.strip() == "")

    missing_indexes = [index for index, value in dataframe[column].items() if _is_blank(value)]
    if not missing_indexes:
        manual_review.append(ManualReviewItem(
            finding, "No missing values found in this column when remediation ran -- nothing to fill.",
        ))
        return

    if config.missing_strategy == MISSING_STRATEGY_DROP_ROW:
        rows_pending_drop.update(missing_indexes)
        actions.append(RemediationAction(
            column_name=column,
            action_type="drop_rows_missing_value",
            count_affected=len(missing_indexes),
            before_example=f"{len(missing_indexes)} row(s) with '{column}' blank",
            after_example="row(s) removed",
            source_finding=finding,
        ))
        return

    if config.missing_strategy == MISSING_STRATEGY_CONSTANT:
        fill_value = config.missing_constant
    elif config.missing_strategy in (MISSING_STRATEGY_MEDIAN, MISSING_STRATEGY_MODE):
        non_missing = dataframe[column].dropna().astype(str)
        non_missing = non_missing[non_missing.apply(lambda v: not _is_blank(v))]
        if config.missing_strategy == MISSING_STRATEGY_MEDIAN:
            numeric = pd.to_numeric(
                non_missing[non_missing.apply(is_coercible_numeric_token)].str.replace(_NUMERIC_JUNK_PATTERN, "", regex=True),
                errors="coerce",
            ).dropna()
            if numeric.empty:
                manual_review.append(ManualReviewItem(
                    finding, "This column has no valid numeric values to compute a median from -- left for manual review.",
                ))
                return
            fill_value = f"{numeric.median():g}"
        else:
            if non_missing.empty:
                manual_review.append(ManualReviewItem(
                    finding, "This column has no non-missing values to compute a mode from -- left for manual review.",
                ))
                return
            fill_value = non_missing.mode().iloc[0]
    else:
        manual_review.append(ManualReviewItem(finding, f"Unrecognized missing-value strategy '{config.missing_strategy}' -- needs manual review."))
        return

    for row_index in missing_indexes:
        dataframe.at[row_index, column] = fill_value

    actions.append(RemediationAction(
        column_name=column,
        action_type=f"fill_missing_{config.missing_strategy}",
        count_affected=len(missing_indexes),
        before_example="(blank)",
        after_example=str(fill_value),
        source_finding=finding,
        notes=f"Filled using this run's configured '{config.missing_strategy}' strategy -- not the conservative default.",
    ))


# ---- Main entry point --------------------------------------------------------

def remediate_table(
    dataframe: pd.DataFrame,
    findings: List[Finding],
    exact_duplicate_row_indexes: List[int],
    config: CleaningConfig = CONSERVATIVE,
) -> RemediationResult:
    """
    The single entry point app.py calls for one table's Tier 1 output.
    Works on a COPY of dataframe -- the caller's original (the one the
    score was computed from) is never touched, which is what guarantees
    the score stays byte-for-byte identical whether or not this function
    is ever called.

    Every Finding in `findings` is dispatched by its issue_type to
    exactly one outcome: an applied RemediationAction, or a
    ManualReviewItem with a reason. Nothing in `findings` is ever
    dropped silently -- see the module docstring's "nothing silent" rule.

    config: OPTIONAL CleaningConfig (see core/cleaning_config.py).
    Defaults to CONSERVATIVE, which reproduces this function's exact
    original behavior -- every caller written before this parameter
    existed (including every pre-existing test) gets byte-identical
    results. app.py's "Cleaning intensity" selector is the one thing
    that changes this argument in practice.
    """
    cleaned = dataframe.copy()
    actions: List[RemediationAction] = []
    manual_review: List[ManualReviewItem] = []
    # missing_data row-drops are deferred and applied once, at the very
    # end -- see _apply_missing_data_strategy's docstring for why.
    rows_pending_drop: Set[int] = set()

    for finding in findings:
        if finding.issue_type == "missing_data":
            _apply_missing_data_strategy(cleaned, finding, config, actions, manual_review, rows_pending_drop)
        elif finding.issue_type == "missing_value_representation":
            _apply_missing_token_normalization(cleaned, finding, actions, manual_review)
        elif finding.issue_type == "inconsistent_format_numeric":
            _apply_or_decline_numeric(cleaned, finding, actions, manual_review)
        elif finding.issue_type == "inconsistent_format_date":
            _apply_or_decline_date(cleaned, finding, actions, manual_review)
        elif finding.issue_type == "inconsistent_format_text":
            _apply_text_trim(cleaned, finding, config, actions, manual_review)
        elif finding.issue_type == "invalid_type_in_numeric":
            _apply_type_coercion(cleaned, finding, is_coercible_numeric_token, "coerce_invalid_numeric", actions, manual_review)
        elif finding.issue_type == "invalid_type_in_date":
            _apply_type_coercion(cleaned, finding, is_recognized_date_token, "coerce_invalid_date", actions, manual_review)
        elif finding.issue_type == "categorical_inconsistency":
            _apply_categorical_standardization(cleaned, finding, config, actions, manual_review)
        elif finding.issue_type == "domain_outlier":
            _apply_domain_outlier_fix(cleaned, finding, config, actions, manual_review)
        elif finding.issue_type == "exact_duplicate_rows":
            cleaned = _remove_exact_duplicates(cleaned, finding, exact_duplicate_row_indexes, actions, manual_review)
        elif finding.issue_type == "fuzzy_duplicate_rows":
            manual_review.append(ManualReviewItem(
                finding,
                f"Two records look like they might be the same business (example: {finding.example}) -- "
                "deciding which spelling is correct, or how to merge them, needs a human, not a guess.",
            ))
        elif finding.issue_type == "structural_issue":
            manual_review.append(ManualReviewItem(
                finding,
                f"{finding.rule_based_description} There's no way to tell which row holds the correct value "
                "without a human decision.",
            ))
        else:
            # Every issue_type core/fixlist.py can produce is handled
            # above -- this branch exists only so a Finding of some
            # future/unrecognized type still lands somewhere visible
            # instead of vanishing, per this module's "nothing silent" rule.
            manual_review.append(ManualReviewItem(finding, f"Unrecognized issue type '{finding.issue_type}' -- needs manual review."))

    if rows_pending_drop:
        # Only ever indexes still present in `cleaned` -- exact-duplicate
        # removal above may already have dropped some of the same rows,
        # and re-dropping an already-gone index would raise.
        still_present = [index for index in rows_pending_drop if index in cleaned.index]
        if still_present:
            cleaned = cleaned.drop(index=still_present)

    cleaned = cleaned.reset_index(drop=True)
    return RemediationResult(cleaned_dataframe=cleaned, actions=actions, manual_review=manual_review)
