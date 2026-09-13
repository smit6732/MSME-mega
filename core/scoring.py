"""
core/scoring.py

Combines the dimension scores (completeness, consistency, duplication,
structure, and optionally validity) into a single overall data-quality
score for the dataset, plus a per-column score where that makes sense.

The weights below are the one thing most likely to come up in a live
Q&A about this project, so they're kept as plain constants right at the
top -- not buried inside a function -- and are deliberately simple to
justify:

- Completeness (27%): the largest weight. A field an AI/BI pipeline can't
  even see, because it's blank, is the most fundamental blocker -- you
  can't fix formatting on data that was never captured.
- Consistency (22%): messy-but-present data is still usable after
  cleanup, so it's a real problem but a comparatively fixable one.
- Duplication (20%): duplicate/near-duplicate records directly bias any
  aggregation or model trained on the data (double-counted revenue,
  inflated customer counts), so it stays close to consistency's weight.
- Structure (16%): a smaller weight. Structural issues (bad headers,
  ragged rows, duplicate IDs) are real problems but tend to affect fewer
  rows/columns than completeness or consistency issues do.
- Validity (15%): a value that's present and consistently formatted but
  still plain wrong (a negative mileage, "unknown" in a price column, a
  99-litre engine) is a real, distinct failure mode -- see
  core/validity.py -- but it's typically the smallest share of a
  realistic MSME file's problems, hence the smallest non-structure weight.

These five (WEIGHT_VALIDITY_ENABLED) are what build_scorecard uses when
it's given a ValidityResult. Pass None for validity_result (the default)
to keep the original four-dimension formula (WEIGHT_*_LEGACY below)
byte-for-byte unchanged -- this is what keeps build_scorecard backward-
compatible for any caller that predates the Validity dimension.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from core.completeness import CompletenessResult
from core.consistency import ConsistencyResult
from core.duplication import DuplicationResult
from core.structure import StructureResult
from core.validity import ValidityResult


# The original four-dimension weights -- used ONLY when build_scorecard
# is called without a validity_result, so that code path's output never
# changes because of this feature. Must add up to 1.0.
WEIGHT_COMPLETENESS_LEGACY = 0.30
WEIGHT_CONSISTENCY_LEGACY = 0.25
WEIGHT_DUPLICATION_LEGACY = 0.25
WEIGHT_STRUCTURE_LEGACY = 0.20

_legacy_total = WEIGHT_COMPLETENESS_LEGACY + WEIGHT_CONSISTENCY_LEGACY + WEIGHT_DUPLICATION_LEGACY + WEIGHT_STRUCTURE_LEGACY
assert abs(_legacy_total - 1.0) < 1e-9, f"Legacy scoring weights must sum to 1.0, got {_legacy_total}"

# The five-dimension weights, used whenever a validity_result IS passed
# in. Change these five numbers (they must still add up to 1.0, checked
# below at import time) to change how the overall score is calculated
# once Validity is in play.
WEIGHT_COMPLETENESS = 0.27
WEIGHT_CONSISTENCY = 0.22
WEIGHT_DUPLICATION = 0.20
WEIGHT_STRUCTURE = 0.16
WEIGHT_VALIDITY = 0.15

_weight_total = WEIGHT_COMPLETENESS + WEIGHT_CONSISTENCY + WEIGHT_DUPLICATION + WEIGHT_STRUCTURE + WEIGHT_VALIDITY
assert abs(_weight_total - 1.0) < 1e-9, f"Scoring weights must sum to 1.0, got {_weight_total}"


@dataclass
class ColumnScore:
    """
    Per-column score, combining completeness and consistency (and,
    where computed, validity) -- the dimensions that are naturally
    measured per-column. Duplication and structure are mostly dataset-
    level checks (or apply to specific identifier columns only), so
    they aren't folded into every column's score here.
    """
    column_name: str
    completeness_score: float
    consistency_score: float
    combined_score: float
    # None whenever build_scorecard was called without a validity_result
    # (the legacy four-dimension path) -- see that function's docstring.
    validity_score: Optional[float] = None


@dataclass
class ScorecardResult:
    overall_score: float
    completeness_score: float
    consistency_score: float
    duplication_score: float
    structure_score: float
    per_column_scores: Dict[str, ColumnScore]
    # None whenever build_scorecard was called without a validity_result.
    validity_score: Optional[float] = None


def build_scorecard(
    completeness_result: CompletenessResult,
    consistency_result: ConsistencyResult,
    duplication_result: DuplicationResult,
    structure_result: StructureResult,
    validity_result: Optional[ValidityResult] = None,
) -> ScorecardResult:
    """
    Combine the dimension results into one ScorecardResult.

    validity_result: OPTIONAL core/validity.py output. Defaults to None,
    in which case this function uses exactly the original four-weight
    formula (WEIGHT_*_LEGACY) and leaves validity_score as None on both
    the returned ScorecardResult and every ColumnScore -- byte-for-byte
    identical to this function's behavior before the Validity dimension
    existed. Passing a real ValidityResult switches to the five-weight
    formula (WEIGHT_COMPLETENESS..WEIGHT_VALIDITY) and folds a validity
    contribution into both the overall score and every column that has
    one. core/pipeline.py always passes a real ValidityResult; this
    parameter stays optional purely so build_scorecard's own public
    contract never breaks for any other caller.
    """
    if validity_result is None:
        overall_score = (
            completeness_result.overall_score * WEIGHT_COMPLETENESS_LEGACY
            + consistency_result.overall_score * WEIGHT_CONSISTENCY_LEGACY
            + duplication_result.duplication_score * WEIGHT_DUPLICATION_LEGACY
            + structure_result.structure_score * WEIGHT_STRUCTURE_LEGACY
        )
    else:
        overall_score = (
            completeness_result.overall_score * WEIGHT_COMPLETENESS
            + consistency_result.overall_score * WEIGHT_CONSISTENCY
            + duplication_result.duplication_score * WEIGHT_DUPLICATION
            + structure_result.structure_score * WEIGHT_STRUCTURE
            + validity_result.overall_score * WEIGHT_VALIDITY
        )

    per_column_scores: Dict[str, ColumnScore] = {}
    all_column_names = set(completeness_result.per_column.keys()) | set(consistency_result.per_column.keys())

    for column_name in all_column_names:
        completeness_score = completeness_result.per_column[column_name].completeness_score
        consistency_score = consistency_result.per_column[column_name].consistency_score
        validity_column = validity_result.per_column.get(column_name) if validity_result else None

        if validity_column is not None:
            validity_score = validity_column.validity_score
            # Simple average across the three per-column dimensions --
            # kept unweighted, same reasoning as the two-way average
            # below: the per-column number stays easy to explain on its
            # own, the weighted formula above is reserved for the one
            # headline overall score.
            combined_score = (completeness_score + consistency_score + validity_score) / 3
        else:
            validity_score = None
            combined_score = (completeness_score + consistency_score) / 2

        per_column_scores[column_name] = ColumnScore(
            column_name=column_name,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            combined_score=round(combined_score, 2),
            validity_score=validity_score,
        )

    return ScorecardResult(
        overall_score=round(overall_score, 2),
        completeness_score=completeness_result.overall_score,
        consistency_score=consistency_result.overall_score,
        duplication_score=duplication_result.duplication_score,
        structure_score=structure_result.structure_score,
        per_column_scores=per_column_scores,
        validity_score=validity_result.overall_score if validity_result else None,
    )
