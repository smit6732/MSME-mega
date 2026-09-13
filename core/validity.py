"""
core/validity.py

Dimension 5 of the scorecard: VALIDITY -- a value can be perfectly
*present* (completeness.py is happy) and perfectly *consistently
formatted* (consistency.py is happy) and still be simply wrong: the
string "unknown" sitting in a numeric Price column, a Year of 2050, a
Doors count of 9, or "Diesal" where every other row says "Diesel".
Those are three related but distinct questions this module answers:

  1. invalid_type_in_numeric / invalid_type_in_date -- the column is
     (predominantly) one type, but this specific value isn't even a
     malformed instance of that type, it's a different kind of thing
     entirely (a word, a placeholder, garbage).
  2. categorical_inconsistency -- a text column has multiple surface
     spellings of what is almost certainly the SAME real-world category
     (a typo, not a genuinely different value). Deliberately separate
     from consistency.py's own text check: that one only catches
     case/whitespace variants of the exact same normalized string
     ("Ahmedabad" vs "ahmedabad"); this one catches variants that don't
     even normalize the same way ("Diesel" vs "Diesal").
  3. domain_outlier / impossible_value -- a numeric value that parses
     fine but violates a basic real-world bound for what that column
     means (a negative mileage, a 99-litre engine).
  4. missing_value_representation -- a cell that isn't truly blank but
     holds a placeholder token (see core/completeness.py's
     MISSING_VALUE_TOKENS) instead of a real value.

Same guarded philosophy as every other check module in this project
(core/calibration.py, core/ingestion.py): flag confidently, or don't
flag at all -- never guess at what the "correct" value should be. This
module only ever DETECTS and DESCRIBES a problem; it never edits data
(that's core/remediation.py's job, once these Findings reach it).

Finding-driven remediation, made possible here: for the two issue types
where a downstream auto-fix needs more than a plain example string to
act safely (categorical_inconsistency's minority->canonical spelling
map, domain_outlier's numeric bounds), this module puts those exact
already-computed facts into Finding.details (see core/findings.py) so
core/remediation.py can reuse them directly instead of re-deriving them
against a possibly-already-modified copy of the data. For the other
issue types, core/remediation.py re-applies the SAME predicate function
this module used to detect the problem (is_coercible_numeric_token,
is_recognized_date_token, is_missing_representation_token, all exported
below) -- identical in spirit to how core/consistency.py's own
inconsistency check and core/remediation.py's guarded numeric/date
fixers already share one definition of "does this value parse cleanly".
"""

import datetime
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from core.completeness import MISSING_VALUE_TOKENS
from core.ingestion import COLUMN_TYPE_NUMERIC, COLUMN_TYPE_DATE, COLUMN_TYPE_TEXT, DATE_FORMAT_PATTERNS

# ---------------------------------------------------------------------------
# Shared value-level predicates -- the same functions used here to DETECT
# a problem are reused, unmodified, by core/remediation.py to find which
# specific cells to act on. One definition, two call sites, never two.
# ---------------------------------------------------------------------------

_NUMERIC_JUNK_PATTERN = re.compile(r"[₹$,\s]")
_COMBINED_DATE_PATTERN = "|".join(DATE_FORMAT_PATTERNS)
_DATE_LIKE_SHAPE_PATTERN = re.compile(r"[-/.]|[A-Za-z]{3}")


def is_coercible_numeric_token(raw_value: str) -> bool:
    """
    True if raw_value can be read as a plain number once common
    formatting characters (currency symbols, thousands separators,
    whitespace) are stripped -- the same lenient cleanup
    core/ingestion.py's own numeric-detection uses. False for anything
    else: a stray word ("unknown", "high", "ask", "TBD"), a placeholder
    ("N/A", "??"), or genuinely non-numeric text.
    """
    cleaned = _NUMERIC_JUNK_PATTERN.sub("", raw_value.strip())
    if cleaned in ("", "-", "+"):
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def is_recognized_date_token(raw_value: str) -> bool:
    """
    True if raw_value is recognizable as SOME date, in SOME known
    format -- it doesn't need to match the column's dominant format
    (that distinction is core/consistency.py's job); it just needs to be
    a date at all, as opposed to a word or placeholder sitting in a date
    column. Reuses core/ingestion.py's own DATE_FORMAT_PATTERNS plus its
    pandas-parse fallback, applied to one value instead of a whole
    column, with the same shape guard ingestion.py uses (a bare digit
    string or bare time-of-day is never treated as a date -- see that
    module's docstring for why that guard exists).
    """
    text = raw_value.strip()
    if text == "":
        return False
    if re.match(_COMBINED_DATE_PATTERN, text):
        return True
    if not _DATE_LIKE_SHAPE_PATTERN.search(text) and not re.match(r"^\d{8}$", text):
        return False
    try:
        parsed = pd.to_datetime(text, errors="raise", dayfirst=True)
    except (ValueError, TypeError):
        return False
    return not pd.isna(parsed)


def is_missing_representation_token(value) -> bool:
    """
    True only for the "disguised missing" case: a non-null, non-blank
    string whose trimmed/lowercased form is one of
    core/completeness.py's MISSING_VALUE_TOKENS -- e.g. "N/A" or
    "Unknown" sitting where a real value should be. Deliberately
    excludes a plain empty string ("" already IS missing, plainly --
    this predicate is about the SURPRISING case of a cell that looks
    filled in but isn't) and excludes real nulls (nothing to "represent"
    -- there's no literal token to normalize away).
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped == "":
        return False
    return stripped.lower() in MISSING_VALUE_TOKENS


# ---------------------------------------------------------------------------
# Domain-rule catalog for domain_outlier / impossible_value
# ---------------------------------------------------------------------------

# Column-name substrings (lowercased), checked in this order (first match
# wins), each naming a real-world bound. "allowed_set" is for a small
# fixed set of legal values (doors); "min"/"max" are plain numeric
# bounds. None means "no bound in that direction". These are intentionally
# a short, named, EXPLAINABLE list (never a black-box model) -- exactly
# the project's existing "explainable > clever" stance (see
# core/calibration.py's docstring). A caller can widen or override any of
# these per-column via the `domain_rule_overrides` parameter below.
_NAMED_DOMAIN_RULES: Dict[str, dict] = {
    "doors": {"allowed_set": {2, 3, 4, 5}},
    "owner": {"min": 0.0, "max": 10.0},          # owner_count, num_owners, ...
    "engine_size": {"min": 0.0, "max": 15.0},    # litres
    "engine_cc": {"min": 0.0, "max": 8000.0},
    "mileage": {"min": 0.0, "max": 1_000_000.0},
    "odometer": {"min": 0.0, "max": 1_000_000.0},
    "km": {"min": 0.0, "max": 1_000_000.0},
    "age": {"min": 0.0, "max": 120.0},
    "year": {"min": 1900.0, "max": None},  # upper bound filled in dynamically, see _year_upper_bound
}

# Column-name substrings that imply "this can never legitimately be
# negative", used ONLY when no more specific named rule above already
# matched. A generic, conservative fallback -- min=0 is the one bound
# this project is comfortable inferring from a column's NAME alone,
# because a negative count/price/distance is never a real value for any
# MSME dataset this tool is likely to see, regardless of the specific
# column.
_NON_NEGATIVE_NAME_HINTS = (
    "price", "amount", "revenue", "salary", "cost", "quantity", "qty",
    "count", "distance", "weight", "fee", "charge", "discount", "stock",
)

# Numeric columns with at least this many distinct values are eligible
# for the generic IQR-based "extreme outlier" fallback (too few distinct
# values makes an interquartile range meaningless). Deliberately the
# same floor core/calibration.py uses before it trusts a clustering
# result -- "not enough data to say anything meaningful" applies here
# for the same reason.
_MIN_DISTINCT_VALUES_FOR_IQR = 8

# How many IQRs beyond Q1/Q3 a value must sit before the GENERIC (no
# named rule) fallback flags it. Deliberately much wider than the
# textbook 1.5x IQR "mild outlier" convention -- this fallback has no
# real-world semantic backing a named rule has, so it only speaks up for
# values that are extreme by ANY reasonable reading of the column,
# minimizing false positives on data that's simply skewed, not wrong.
_IQR_EXTREME_MULTIPLIER = 3.0

# categorical_inconsistency: similarity floor (0-100, Jaro-Winkler
# scale -- see _categorical_similarity below) before a minority spelling
# is considered a likely typo of a dominant one, rather than a
# genuinely different category. Jaro-Winkler, not rapidfuzz's edit-
# distance-based fuzz.WRatio, on purpose: WRatio scores short category
# words (e.g. "Diesal" vs "Diesel", both 6 characters) far lower than
# their edit distance actually warrants (~83, below any safe floor),
# because it's tuned for longer free-text comparisons; Jaro-Winkler's
# extra weight on a shared PREFIX is the better fit for exactly this
# "one interior character off" typo pattern, at a threshold still high
# enough to keep this module's "never guess" stance -- see module
# docstring. A single-character brand-code substitution on a 3-letter
# acronym (e.g. "BMV" vs "BMW") still won't clear this floor by design:
# that's indistinguishable from a genuinely different short code without
# domain knowledge, which is exactly what a synonym dictionary (see this
# module's docstring) is for, not a generic distance metric.
CATEGORICAL_FUZZY_THRESHOLD = 85.0


def _categorical_similarity(a: str, b: str, **_kwargs) -> float:
    """rapidfuzz.process scorer wrapper -- see CATEGORICAL_FUZZY_THRESHOLD's
    docstring for why Jaro-Winkler over the rapidfuzz.fuzz.* family here.
    Scaled to 0-100 to match every other similarity score in this
    project (core/duplication.py's rapidfuzz scores are also 0-100)."""
    return JaroWinkler.normalized_similarity(a, b) * 100

# How many of a column's most frequent normalized spellings are treated
# as "dominant" candidates a minority spelling can be matched against.
# Comparing against every distinct value (not just the frequent ones)
# would let two equally-rare typos "correct" each other arbitrarily.
_DOMINANT_VARIANT_TOP_N = 12

# A column is only eligible for categorical-typo detection when it has
# at most this many distinct normalized values relative to its row
# count -- a genuine free-text column (Business_Name, Address) has no
# real "canonical form" to converge on, and running fuzzy matching there
# would just invent false-positive mappings between unrelated values.
_MAX_DISTINCT_RATIO_FOR_CATEGORICAL = 0.5
_MAX_ABSOLUTE_DISTINCT_FOR_CATEGORICAL = 40


def _year_upper_bound() -> float:
    """current_year + 1, computed at call time (never cached at import
    time) so this stays correct regardless of when the process started
    -- deliberately narrow in scope (a sanity bound on a YEAR value, not
    a date-parsing decision), unlike the "today's date" pitfall
    core/ingestion.py's docstring warns about for date TYPE inference."""
    return float(datetime.date.today().year + 1)


@dataclass
class ColumnValidity:
    """Validity result for a single column -- percentage/score shaped
    identically to ColumnConsistency (core/consistency.py) and
    ColumnCompleteness (core/completeness.py) so core/scoring.py can
    treat all three the same way."""
    column_name: str
    invalid_count: int
    total_checked: int
    invalid_percentage: float
    validity_score: float


@dataclass
class ValidityFindingCandidate:
    """One raw validity problem, already shaped almost exactly like
    core/fixlist.py's own _RawIssueCandidate -- kept as a separate,
    public dataclass (rather than reusing fixlist's private one) so this
    module has zero dependency on core/fixlist.py, matching every other
    check module in this project (completeness/consistency/duplication/
    structure also know nothing about fixlist.py)."""
    issue_type: str
    column_name: str
    percentage_affected: float
    example: Optional[str]
    details: Optional[dict] = None


@dataclass
class ValidityResult:
    per_column: Dict[str, ColumnValidity]
    overall_score: float
    candidates: List[ValidityFindingCandidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# invalid_type_in_numeric / invalid_type_in_date
# ---------------------------------------------------------------------------

def _check_numeric_validity(values: List[str]) -> Tuple[int, List[str]]:
    invalid = [v for v in values if not is_coercible_numeric_token(v)]
    return len(invalid), invalid[:3]


def _check_date_validity(values: List[str]) -> Tuple[int, List[str]]:
    invalid = [v for v in values if not is_recognized_date_token(v)]
    return len(invalid), invalid[:3]


# ---------------------------------------------------------------------------
# missing_value_representation
# ---------------------------------------------------------------------------

def _check_missing_representation(raw_series: pd.Series) -> Tuple[int, List[str]]:
    """Counted against the column's FULL row count (not just non-missing
    values, unlike the other checks in this module) -- a disguised-
    missing token is itself a form of missingness, so its percentage
    should read on the same scale as completeness.py's own missing %."""
    flagged = raw_series.dropna().astype(str)
    flagged = flagged[flagged.apply(is_missing_representation_token)]
    tokens_seen = sorted(set(v.strip() for v in flagged))
    return len(flagged), tokens_seen[:5]


# ---------------------------------------------------------------------------
# domain_outlier / impossible_value
# ---------------------------------------------------------------------------

def _resolve_domain_rule(column_name: str, overrides: Optional[Dict[str, dict]]) -> Optional[dict]:
    name_lower = column_name.lower()
    if overrides:
        for hint, rule in overrides.items():
            if hint.lower() in name_lower:
                return rule
    for hint, rule in _NAMED_DOMAIN_RULES.items():
        if hint in name_lower:
            resolved = dict(rule)
            if hint == "year" and resolved.get("max") is None:
                resolved["max"] = _year_upper_bound()
            return resolved
    if any(hint in name_lower for hint in _NON_NEGATIVE_NAME_HINTS):
        return {"min": 0.0, "max": None}
    return None


def _values_violating_rule(numeric_values: pd.Series, rule: dict) -> Tuple[pd.Series, str]:
    allowed_set = rule.get("allowed_set")
    if allowed_set is not None:
        violating = numeric_values[~numeric_values.isin(allowed_set)]
        description = f"outside the expected set {sorted(allowed_set)}"
        return violating, description

    lower_bound, upper_bound = rule.get("min"), rule.get("max")
    mask = pd.Series(False, index=numeric_values.index)
    if lower_bound is not None:
        mask |= numeric_values < lower_bound
    if upper_bound is not None:
        mask |= numeric_values > upper_bound
    bound_text = " and ".join(
        text for text in [
            f">= {lower_bound:g}" if lower_bound is not None else None,
            f"<= {upper_bound:g}" if upper_bound is not None else None,
        ] if text
    )
    return numeric_values[mask], f"expected {bound_text}"


def _check_domain_outliers(
    numeric_values: pd.Series, column_name: str, domain_rule_overrides: Optional[Dict[str, dict]],
) -> Optional[Tuple[pd.Series, str, dict]]:
    """
    Returns (violating_values, rule_description, details_for_finding) or
    None when nothing violates any rule -- or when no rule applies at
    all. Tries a NAMED rule first (real-world semantics, from the
    column's name); only falls back to the generic statistical IQR
    guard when no named rule matched, so a column we can actually reason
    about by name is never second-guessed by a purely statistical bound.
    """
    named_rule = _resolve_domain_rule(column_name, domain_rule_overrides)
    if named_rule is not None:
        violating, description = _values_violating_rule(numeric_values, named_rule)
        if violating.empty:
            return None
        details = {"rule": description, "lower_bound": named_rule.get("min"), "upper_bound": named_rule.get("max")}
        if named_rule.get("allowed_set") is not None:
            details = {"rule": description, "allowed_set": sorted(named_rule["allowed_set"])}
        return violating, description, details

    if numeric_values.nunique() < _MIN_DISTINCT_VALUES_FOR_IQR:
        return None
    q1, q3 = numeric_values.quantile(0.25), numeric_values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    lower_bound = q1 - _IQR_EXTREME_MULTIPLIER * iqr
    upper_bound = q3 + _IQR_EXTREME_MULTIPLIER * iqr
    violating = numeric_values[(numeric_values < lower_bound) | (numeric_values > upper_bound)]
    if violating.empty:
        return None
    description = f"statistically extreme (outside {lower_bound:.2f} to {upper_bound:.2f}, based on this column's own distribution)"
    details = {"rule": description, "lower_bound": round(float(lower_bound), 4), "upper_bound": round(float(upper_bound), 4)}
    return violating, description, details


# ---------------------------------------------------------------------------
# categorical_inconsistency
# ---------------------------------------------------------------------------

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def categorical_key(raw_value: str) -> str:
    """
    The grouping key for categorical-typo detection: lowercased, and
    every non-alphanumeric character (spaces, dots, hyphens) stripped
    out entirely -- so "BMW", "bmw", and "B.M.W" all collapse to the
    SAME key ("bmw") deterministically, with no fuzzy matching (and no
    "never guess" risk) involved at all. This is intentionally a
    DIFFERENT, coarser key than core/consistency.py's own
    strip()+lower() normalization (which deliberately keeps "B.M.W" and
    "BMW" as different groups, since punctuation differences are a
    formatting inconsistency, not a category typo) -- the two modules
    are answering different questions about the same values.

    Exported (not module-private) so core/remediation.py can reuse this
    EXACT key function to look up Finding.details["canonical_map"] --
    the map's keys were built with this function, so applying it must
    use the same one, never re-derive an equivalent-looking one.
    """
    return _NON_ALPHANUMERIC.sub("", raw_value.strip().lower())


def _check_categorical_inconsistency(values: List[str]) -> Optional[Tuple[dict, List[str], int]]:
    """
    Returns (canonical_map, example_variants, affected_count) or None.

    canonical_map: {categorical_key_of_minority_form: {"canonical": str,
    "score": float}} -- built in two stages:
      1. Group values by categorical_key (punctuation-stripped,
         lowercased) -- this alone merges pure punctuation/casing
         variants ("B.M.W" -> "BMW") with full confidence, no fuzzy
         matching needed.
      2. Fuzzy-match every remaining MINORITY key (one that isn't
         already among the column's established/repeated forms -- see
         the "dominant" comment below) against those established forms,
         keeping a match ONLY when it clears CATEGORICAL_FUZZY_THRESHOLD.
    Declines (returns None) entirely for a column that doesn't look
    categorical at all -- see _MAX_DISTINCT_RATIO_FOR_CATEGORICAL's
    docstring above.
    """
    if not values:
        return None

    normalized_counts: Dict[str, int] = {}
    variant_counts_by_key: Dict[str, Dict[str, int]] = {}
    for raw_value in values:
        key = categorical_key(raw_value)
        normalized_counts[key] = normalized_counts.get(key, 0) + 1
        counts = variant_counts_by_key.setdefault(key, {})
        stripped = raw_value.strip()
        counts[stripped] = counts.get(stripped, 0) + 1

    distinct_count = len(normalized_counts)
    if distinct_count < 2:
        return None
    if distinct_count > _MAX_ABSOLUTE_DISTINCT_FOR_CATEGORICAL:
        return None
    if distinct_count / len(values) > _MAX_DISTINCT_RATIO_FOR_CATEGORICAL:
        return None

    # The canonical form for each key is the most-frequent EXACT
    # spelling within that group (same idea as core/consistency.py's own
    # dominant-variant logic) -- so the form we map typos/punctuation-
    # variants TO is genuinely this column's own "house style".
    canonical_spelling = {key: max(counts, key=counts.get) for key, counts in variant_counts_by_key.items()}

    # "Dominant" means genuinely established -- a key that appears MORE
    # THAN ONCE. A key seen exactly once is the classic typo/variant
    # candidate (e.g. one "Diesal" among eight "Diesel"s), so it must
    # never end up in the pool a minority key gets matched AGAINST --
    # capping by top-N alone (without this >=2 floor) would otherwise
    # swallow every rare value into "dominant" whenever a column has few
    # distinct keys total, leaving nothing left to flag. Falls back to
    # just the single most common key if NOTHING repeats (every value in
    # the column is unique).
    sorted_keys = sorted(normalized_counts.items(), key=lambda item: -item[1])
    repeated_keys = [key for key, count in sorted_keys if count >= 2]
    dominant_keys = repeated_keys[:_DOMINANT_VARIANT_TOP_N] or [sorted_keys[0][0]]
    minority_keys = [key for key, _count in sorted_keys if key not in dominant_keys]
    if not dominant_keys:
        return None
    # NOTE: deliberately no early return when minority_keys is empty --
    # a column can have ZERO minority keys (every value's key is already
    # "established") and still need a fix, via the dominant-key-variant
    # step just below (e.g. "BMW"/"BMW"/"B.M.W" are ALL the dominant
    # "bmw" key -- there's no minority key here at all, but "B.M.W" still
    # needs correcting to "BMW"). Returning None here would skip that
    # step entirely.

    canonical_map: Dict[str, dict] = {}
    for minority_key in minority_keys:
        match = process.extractOne(
            minority_key, dominant_keys, scorer=_categorical_similarity, score_cutoff=CATEGORICAL_FUZZY_THRESHOLD,
        )
        if match is None:
            continue
        matched_key, score, _index = match
        canonical_map[minority_key] = {
            "canonical": canonical_spelling[matched_key],
            "score": round(float(score), 2),
        }

    # A DOMINANT key can still have more than one distinct EXACT
    # spelling sharing it -- e.g. "BMW" and "B.M.W" both collapse to the
    # key "bmw" (see categorical_key's docstring), but "B.M.W" is still
    # its own literal string sitting in the data. That's a full-
    # confidence, non-fuzzy fix (same key -- not a guess at all), so it
    # gets added at score 100.0 regardless of CATEGORICAL_FUZZY_THRESHOLD,
    # separate from the fuzzy minority-key matching above.
    for key in dominant_keys:
        if len(variant_counts_by_key[key]) > 1:
            canonical_map[key] = {"canonical": canonical_spelling[key], "score": 100.0}

    if not canonical_map:
        return None

    def _affected_count(key: str, canonical: str) -> int:
        if key in dominant_keys:
            # Only the non-canonically-spelled members of this
            # established group actually change.
            return normalized_counts[key] - variant_counts_by_key[key].get(canonical, 0)
        return normalized_counts[key]  # every member of a minority key is being remapped away

    def _example_before_text(key: str, canonical: str) -> str:
        non_canonical_spellings = [spelling for spelling in variant_counts_by_key[key] if spelling != canonical]
        return non_canonical_spellings[0] if non_canonical_spellings else next(iter(variant_counts_by_key[key]))

    affected_count = sum(_affected_count(key, mapping["canonical"]) for key, mapping in canonical_map.items())
    example_variants = [
        f"{_example_before_text(key, mapping['canonical'])} -> {mapping['canonical']}"
        for key, mapping in list(canonical_map.items())[:3]
    ]
    return canonical_map, example_variants, affected_count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_validity(
    dataframe: pd.DataFrame,
    column_types: Dict[str, str],
    domain_rule_overrides: Optional[Dict[str, dict]] = None,
) -> ValidityResult:
    """
    Run every validity check appropriate to each column's inferred type,
    same "reuse the type ingestion.py already decided" pattern
    core/consistency.py follows. Produces both a per-column score (for
    core/scoring.py) and a flat list of ValidityFindingCandidate objects
    (for core/fixlist.py) in one pass, since both need to walk the same
    columns and the candidates already carry everything fixlist.py needs
    to turn them into Findings.

    domain_rule_overrides: optional {column_name_substring: rule_dict}
    to widen/replace a built-in domain rule for a specific dataset --
    see _NAMED_DOMAIN_RULES above for the rule shape. None (the default)
    uses only the built-in catalog.
    """
    per_column: Dict[str, ColumnValidity] = {}
    candidates: List[ValidityFindingCandidate] = []

    for column_name in dataframe.columns:
        column_type = column_types.get(column_name, "unknown")
        raw_series = dataframe[column_name]

        missing_repr_count, missing_repr_tokens = _check_missing_representation(raw_series)
        total_rows = len(raw_series)
        if missing_repr_count > 0 and total_rows:
            percentage = round(missing_repr_count / total_rows * 100, 2)
            candidates.append(ValidityFindingCandidate(
                issue_type="missing_value_representation",
                column_name=column_name,
                percentage_affected=percentage,
                example=missing_repr_tokens[0] if missing_repr_tokens else None,
                details={"tokens_found": missing_repr_tokens},
            ))

        non_missing = raw_series.dropna().astype(str).str.strip()
        non_missing = non_missing[non_missing != ""]
        # Values already recognized as a disguised-missing token are
        # excluded from every check below -- they're accounted for by
        # the missing_value_representation candidate above, and
        # including them here would double-flag the same cell as also
        # "not a valid number/date", which is a different diagnostic
        # signal than the one that's actually useful to show.
        checkable = non_missing[~non_missing.apply(is_missing_representation_token)]
        values = checkable.tolist()
        total_checked = len(values)

        invalid_count = 0
        if total_checked and column_type == COLUMN_TYPE_NUMERIC:
            invalid_count, examples = _check_numeric_validity(values)
            if invalid_count:
                candidates.append(ValidityFindingCandidate(
                    issue_type="invalid_type_in_numeric",
                    column_name=column_name,
                    percentage_affected=round(invalid_count / total_checked * 100, 2),
                    example=examples[0],
                    details={"invalid_examples": examples},
                ))
            numeric_values = pd.to_numeric(
                checkable[checkable.apply(is_coercible_numeric_token)].str.replace(_NUMERIC_JUNK_PATTERN, "", regex=True),
                errors="coerce",
            ).dropna()
            outlier_result = _check_domain_outliers(numeric_values, column_name, domain_rule_overrides)
            if outlier_result is not None:
                violating, description, details = outlier_result
                candidates.append(ValidityFindingCandidate(
                    issue_type="domain_outlier",
                    column_name=column_name,
                    percentage_affected=round(len(violating) / total_checked * 100, 2),
                    example=f"{violating.iloc[0]:g} ({description})",
                    details=details,
                ))
                invalid_count += len(violating)

        elif total_checked and column_type == COLUMN_TYPE_DATE:
            invalid_count, examples = _check_date_validity(values)
            if invalid_count:
                candidates.append(ValidityFindingCandidate(
                    issue_type="invalid_type_in_date",
                    column_name=column_name,
                    percentage_affected=round(invalid_count / total_checked * 100, 2),
                    example=examples[0],
                    details={"invalid_examples": examples},
                ))

        elif total_checked and column_type == COLUMN_TYPE_TEXT:
            categorical_result = _check_categorical_inconsistency(values)
            if categorical_result is not None:
                canonical_map, example_variants, affected_count = categorical_result
                candidates.append(ValidityFindingCandidate(
                    issue_type="categorical_inconsistency",
                    column_name=column_name,
                    percentage_affected=round(affected_count / total_checked * 100, 2),
                    example=example_variants[0] if example_variants else None,
                    details={"canonical_map": canonical_map, "examples": example_variants},
                ))
                invalid_count += affected_count

        combined_invalid_for_column = invalid_count + missing_repr_count
        combined_total_for_column = total_checked + missing_repr_count
        invalid_percentage = (
            (combined_invalid_for_column / combined_total_for_column * 100) if combined_total_for_column else 0.0
        )
        validity_score = 100.0 - invalid_percentage

        per_column[column_name] = ColumnValidity(
            column_name=column_name,
            invalid_count=combined_invalid_for_column,
            total_checked=combined_total_for_column,
            invalid_percentage=round(invalid_percentage, 2),
            validity_score=round(validity_score, 2),
        )

    overall_score = (
        sum(c.validity_score for c in per_column.values()) / len(per_column) if per_column else 0.0
    )

    return ValidityResult(per_column=per_column, overall_score=round(overall_score, 2), candidates=candidates)
