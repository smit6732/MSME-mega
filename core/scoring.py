"""
core/scoring.py

Combines the four dimension scores (completeness, consistency,
duplication, structure) into a single overall data-quality score for the
dataset, plus a per-column score where that makes sense.

The weights below are the one thing most likely to come up in a live
Q&A about this project, so they're kept as plain constants right at the
top -- not buried inside a function -- and are deliberately simple to
justify:

- Completeness (30%): the largest weight. A field an AI/BI pipeline can't
  even see, because it's blank, is the most fundamental blocker -- you
  can't fix formatting on data that was never captured.
- Consistency (25%): messy-but-present data is still usable after
  cleanup, so it's a real problem but a comparatively fixable one.
- Duplication (25%): duplicate/near-duplicate records directly bias any
  aggregation or model trained on the data (double-counted revenue,
  inflated customer counts), so it's weighted evenly with consistency.
- Structure (20%): the smallest weight. Structural issues (bad headers,
  ragged rows, duplicate IDs) are real problems but tend to affect fewer
  rows/columns than completeness or consistency issues do.
"""

from dataclasses import dataclass
from typing import Dict

from core.completeness import CompletenessResult
from core.consistency import ConsistencyResult
from core.duplication import DuplicationResult
from core.structure import StructureResult


# Change these four numbers to change how the overall score is
# calculated. They must add up to 1.0 (100%) -- checked below at import
# time so a typo here fails loudly instead of silently skewing the score.
WEIGHT_COMPLETENESS = 0.30
WEIGHT_CONSISTENCY = 0.25
WEIGHT_DUPLICATION = 0.25
WEIGHT_STRUCTURE = 0.20

_weight_total = WEIGHT_COMPLETENESS + WEIGHT_CONSISTENCY + WEIGHT_DUPLICATION + WEIGHT_STRUCTURE
assert abs(_weight_total - 1.0) < 1e-9, f"Scoring weights must sum to 1.0, got {_weight_total}"


@dataclass
class ColumnScore:
    """
    Per-column score, combining completeness and consistency -- the two
    dimensions that are naturally measured per-column. Duplication and
    structure are mostly dataset-level checks (or apply to specific
    identifier columns only), so they aren't folded into every column's
    score here.
    """
    column_name: str
    completeness_score: float
    consistency_score: float
    combined_score: float


@dataclass
class ScorecardResult:
    overall_score: float
    completeness_score: float
    consistency_score: float
    duplication_score: float
    structure_score: float
    per_column_scores: Dict[str, ColumnScore]


def build_scorecard(
    completeness_result: CompletenessResult,
    consistency_result: ConsistencyResult,
    duplication_result: DuplicationResult,
    structure_result: StructureResult,
) -> ScorecardResult:
    """Combine the four dimension results into one ScorecardResult."""

    overall_score = (
        completeness_result.overall_score * WEIGHT_COMPLETENESS
        + consistency_result.overall_score * WEIGHT_CONSISTENCY
        + duplication_result.duplication_score * WEIGHT_DUPLICATION
        + structure_result.structure_score * WEIGHT_STRUCTURE
    )

    per_column_scores: Dict[str, ColumnScore] = {}
    all_column_names = set(completeness_result.per_column.keys()) | set(consistency_result.per_column.keys())

    for column_name in all_column_names:
        completeness_score = completeness_result.per_column[column_name].completeness_score
        consistency_score = consistency_result.per_column[column_name].consistency_score
        # Simple average of the two per-column dimensions -- kept
        # unweighted so the per-column number stays easy to explain on
        # its own. The weighted formula above is reserved for the one
        # headline overall score.
        combined_score = (completeness_score + consistency_score) / 2

        per_column_scores[column_name] = ColumnScore(
            column_name=column_name,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            combined_score=round(combined_score, 2),
        )

    return ScorecardResult(
        overall_score=round(overall_score, 2),
        completeness_score=completeness_result.overall_score,
        consistency_score=consistency_result.overall_score,
        duplication_score=duplication_result.duplication_score,
        structure_score=structure_result.structure_score,
        per_column_scores=per_column_scores,
    )
