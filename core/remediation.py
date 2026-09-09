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
    missing_data                 -> manual review (no safe universal fill-in)
    inconsistent_format_numeric  -> guarded auto-fix (strip thousands/currency)
    inconsistent_format_date     -> guarded auto-fix (normalize to YYYY-MM-DD)
    inconsistent_format_text     -> auto-fix (trim whitespace only, never casing)
    exact_duplicate_rows         -> auto-fix (drop rows via duplication.py's
                                     own exact_duplicate_row_indexes -- never
                                     recomputed here)
    fuzzy_duplicate_rows         -> manual review (which one is "correct"?)
    structural_issue             -> manual review (e.g. duplicate identifier --
                                     no way to know which row holds the real one)

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
from typing import Dict, List, Optional

import pandas as pd

from core.findings import Finding

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


def _apply_text_trim(dataframe, finding, actions, manual_review) -> None:
    """
    Unconditional fix (no ambiguity guard needed) -- but "unconditional"
    only means casing is out of scope, not that this column definitely
    has whitespace to trim. If it turns out this Finding's inconsistency
    was entirely about casing (deliberately out of scope, see module
    docstring), there is nothing to actually change -- decline rather
    than claim a fix that did nothing.
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
        stripped = raw_text.strip()
        if stripped != raw_text:
            dataframe.at[row_index, column] = stripped
            changed_count += 1
            if before_example is None:
                before_example, after_example = raw_text, stripped

    if changed_count == 0:
        manual_review.append(ManualReviewItem(
            finding,
            "No leading/trailing whitespace found to trim -- the formatting "
            "difference here is something this tool doesn't automatically fix (e.g. casing).",
        ))
        return

    # repr() here (not the plain strings) is deliberate: a whitespace-only
    # difference like "Foo " vs "Foo" is otherwise invisible once rendered
    # as plain text -- repr() keeps the actual trimmed spaces visible in
    # the audit log.
    actions.append(RemediationAction(
        column_name=column,
        action_type="trim_whitespace",
        count_affected=changed_count,
        before_example=repr(before_example),
        after_example=repr(after_example),
        source_finding=finding,
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


# ---- Main entry point --------------------------------------------------------

def remediate_table(
    dataframe: pd.DataFrame,
    findings: List[Finding],
    exact_duplicate_row_indexes: List[int],
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
    """
    cleaned = dataframe.copy()
    actions: List[RemediationAction] = []
    manual_review: List[ManualReviewItem] = []

    for finding in findings:
        if finding.issue_type == "missing_data":
            manual_review.append(ManualReviewItem(
                finding,
                f"{finding.percentage_affected:.1f}% of '{finding.column_name}' is missing -- there's no safe, "
                "universal way to guess what belongs in a blank cell.",
            ))
        elif finding.issue_type == "inconsistent_format_numeric":
            _apply_or_decline_numeric(cleaned, finding, actions, manual_review)
        elif finding.issue_type == "inconsistent_format_date":
            _apply_or_decline_date(cleaned, finding, actions, manual_review)
        elif finding.issue_type == "inconsistent_format_text":
            _apply_text_trim(cleaned, finding, actions, manual_review)
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

    cleaned = cleaned.reset_index(drop=True)
    return RemediationResult(cleaned_dataframe=cleaned, actions=actions, manual_review=manual_review)
