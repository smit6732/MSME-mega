"""
core/calibration.py

The tool's only "AI" that touches numbers, and it is scoped tightly: it
calibrates two THRESHOLDS to the shape of the specific table being
scored, instead of using the same fixed cutoff for every file. It never
changes what a check measures, and it never changes how the final
weighted score is computed (that stays 100% fixed constants -- see
core/scoring.py). Calibration only decides "where's the line" for two
things: (1) how similar do two names need to be to count as a likely
duplicate, and (2) what % of rows affected counts as critical vs. minor
for THIS table's own data.

Both functions below use the same idea: k-means clustering
(scikit-learn) splits a 1-D list of numbers into natural groups, and we
use the midpoint between neighbouring cluster centers as the calibrated
cutoff. Why k-means instead of, say, a fixed percentile: k-means finds
where the DATA actually splits into groups (e.g. "these values are all
bunched up near 60-70%, these others are bunched up near 95-100%")
rather than imposing an arbitrary rank cut regardless of the numbers'
actual shape.

IMPORTANT caveat we specifically guard against: k-means ALWAYS returns
exactly k clusters, even when the data has no real separation at all --
if you ask it for 2 clusters in a single unstructured blob of noise, it
will still hand back 2 centers, just an arbitrary cut through the middle
of that blob. Blindly trusting that split would be actively harmful here
(e.g. it could invent a low "duplicate" threshold on a table that has no
real duplicates, flagging unrelated business names as likely matches).
So after clustering, both functions check that the groups it found are
actually separated by a real gap -- not just "k-means produced two
numbers" -- before trusting the result. If the gap check fails, we fall
back to the fixed default, same as when there isn't enough data at all.

Both functions are:
  - Deterministic. KMeans' default initialization is randomized, so we
    fix random_state -- required so "the same file processed twice
    produces the same score and the same fix list" stays true even
    though calibration is technically a clustering algorithm.
  - Fit fresh every call, on only the numbers passed in for THIS table.
    Nothing is saved, cached, or reused between tables or between runs
    -- no model file, no pretrained weights, matching the "fit fresh on
    each table, every time" requirement.
  - Guarded TWICE before being trusted: first a minimum-data check
    before even attempting to cluster, then a minimum-gap check on the
    clustering result itself. Either guard failing means "fall back to
    the fixed default" -- we would rather use the simple, well-tested
    default than trust a clustering result that isn't backed by a real
    pattern in this table's data.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
# scikit-learn is intentionally NOT imported here at module level.
# Importing sklearn.cluster pulls in scipy/joblib/threadpoolctl behind
# it and costs roughly a full second on a typical machine -- paid at
# app.py's very first `import core.pipeline` (before a user has even
# uploaded a file) if this were a top-level import, since
# core/pipeline.py -> core/duplication.py/core/fixlist.py -> this
# module gets imported eagerly at Streamlit startup. Deferred into
# _fit_kmeans_and_get_sorted_cluster_ranges below -- the ONLY place
# KMeans is actually used -- so the cost is paid at most once, the
# first time a file is actually calibrated, and never at all for a
# table whose data is too small/uniform to reach that function (see
# both calibrate_* functions' early "not enough data" fallback, which
# also needs no import at all).

# Fixed random seed for every KMeans call in this file. See module
# docstring: determinism is a hard requirement, and KMeans' default
# k-means++ initialization is randomized run to run unless seeded.
_KMEANS_RANDOM_STATE = 42
_KMEANS_N_INIT = 10  # run k-means 10 times with different seeds internally
                      # and keep the best fit -- still fully deterministic
                      # because random_state is fixed, just more stable.

# ---- Fixed fallback defaults (unchanged from before this feature) --------
# These are exactly the flat cutoffs the tool used before adaptive
# calibration existed. Every calibration function below falls back to
# these when there isn't enough data to calibrate meaningfully, or when
# the data doesn't show a real enough split to trust (see module docstring).
DEFAULT_DUPLICATE_SIMILARITY_THRESHOLD = 90.0
DEFAULT_CRITICAL_PERCENT = 30.0
DEFAULT_MODERATE_PERCENT = 10.0

# Minimum number of DISTINCT values required before we even attempt to
# cluster -- k-means on a handful of points (or a column where every
# pair scores the same) can't find a meaningful boundary at all.
_MIN_DISTINCT_SCORES_FOR_DUPLICATE_CALIBRATION = 6
_MIN_DISTINCT_PERCENTAGES_FOR_SEVERITY_CALIBRATION = 6

# Minimum gap (in percentage points) required BETWEEN neighbouring
# clusters before we trust the split as real, rather than an arbitrary
# cut through one continuous blob of noise. Chosen generously on
# purpose: a small, ambiguous gap is exactly the case where guessing
# wrong is most likely, so we'd rather fall back to the safe default
# than calibrate on a borderline signal.
_MIN_GAP_FOR_DUPLICATE_CALIBRATION = 15.0
_MIN_GAP_FOR_SEVERITY_CALIBRATION = 5.0


@dataclass
class ThresholdCalibration:
    """
    Result of one calibration attempt: the value to actually use, and
    whether it was genuinely computed from this table's data or is just
    the fixed fallback default. The UI always shows was_calibrated, so a
    fallback value is never presented as if it were a "calibrated for
    you" result.
    """
    value: float
    was_calibrated: bool


def calibrate_duplicate_threshold(similarity_scores: List[float]) -> ThresholdCalibration:
    """
    Given every pairwise similarity score (0-100, from rapidfuzz) computed
    across candidate pairs in a table's identity column, find the natural
    boundary between "probably the same business" and "probably not the
    same business" pairs -- instead of using a flat 90% cutoff for every
    table regardless of how that table's names actually compare to each
    other.

    HOW (explain this out loud without notes):
      1. Reshape the similarity scores into a k-means problem with
         k=2 -- we are deliberately looking for exactly two groups:
         "similar pairs" and "not similar pairs".
      2. Fit k-means on this table's scores only.
      3. Check that the two groups it found are actually separated by a
         real gap (see module docstring) -- if a table has no genuinely
         similar pairs at all, k-means would otherwise just slice its
         one blob of low scores in half and invent a meaningless
         "threshold". If the gap check fails, fall back to 90%.
      4. Otherwise, take the two cluster centers and use the MIDPOINT
         between them as the threshold. Any pair scoring above that
         midpoint sits closer to the "similar" cluster's center than the
         "not similar" one -- a natural dividing line, not a guess.

    Falls back to the fixed 90% default when there are fewer than
    _MIN_DISTINCT_SCORES_FOR_DUPLICATE_CALIBRATION distinct scores, OR
    when the clustering doesn't show a real gap (step 3 above).
    """
    distinct_scores = sorted(set(similarity_scores))
    if len(distinct_scores) < _MIN_DISTINCT_SCORES_FOR_DUPLICATE_CALIBRATION:
        return ThresholdCalibration(DEFAULT_DUPLICATE_SIMILARITY_THRESHOLD, False)

    cluster_ranges = _fit_kmeans_and_get_sorted_cluster_ranges(similarity_scores, n_clusters=2)
    (low_center, _low_min, low_max), (high_center, high_min, _high_max) = cluster_ranges

    gap = high_min - low_max
    if gap < _MIN_GAP_FOR_DUPLICATE_CALIBRATION:
        return ThresholdCalibration(DEFAULT_DUPLICATE_SIMILARITY_THRESHOLD, False)

    midpoint = (low_center + high_center) / 2

    # Sanity clamp: never let calibration produce a threshold so low
    # that it would flag almost every pair as a likely duplicate (below
    # 50%), or so high it's effectively "only exact matches" (above 99%).
    # Belt-and-braces alongside the gap check above.
    calibrated_threshold = max(50.0, min(99.0, midpoint))
    return ThresholdCalibration(round(float(calibrated_threshold), 2), True)


def calibrate_severity_thresholds(percentages: List[float]) -> Tuple[ThresholdCalibration, ThresholdCalibration]:
    """
    Given every "% of rows/columns affected" value produced by ALL of a
    table's checks (missing %, inconsistent %, duplicate %, structural
    issue %), find two natural breakpoints that separate low-impact,
    medium-impact, and high-impact issues for THIS table -- instead of
    every table using a flat 30% / 10% split regardless of how its own
    issues are actually distributed.

    HOW (explain this out loud without notes):
      1. Reshape the percentages into a k-means problem with k=3 -- we
         are deliberately looking for three groups: "low impact",
         "medium impact", "high impact".
      2. Fit k-means on this table's percentages only.
      3. Check both neighbouring gaps (low-to-medium, medium-to-high)
         are real, same reasoning as the duplicate-threshold function
         above. If EITHER gap is too small, fall back entirely to the
         fixed 30%/10% split -- a partially-trusted 3-way split would be
         harder to explain and reason about than "calibrated" or "not".
      4. Otherwise: the MODERATE cutoff is the midpoint between the low
         and medium centers; the CRITICAL cutoff is the midpoint between
         the medium and high centers.

    Falls back to the fixed 30% / 10% default when there are fewer than
    _MIN_DISTINCT_PERCENTAGES_FOR_SEVERITY_CALIBRATION distinct
    percentages, OR when either neighbouring gap isn't real (step 3).
    """
    distinct_percentages = sorted(set(percentages))
    if len(distinct_percentages) < _MIN_DISTINCT_PERCENTAGES_FOR_SEVERITY_CALIBRATION:
        return (
            ThresholdCalibration(DEFAULT_CRITICAL_PERCENT, False),
            ThresholdCalibration(DEFAULT_MODERATE_PERCENT, False),
        )

    cluster_ranges = _fit_kmeans_and_get_sorted_cluster_ranges(percentages, n_clusters=3)
    (low_center, _low_min, low_max), (mid_center, mid_min, mid_max), (high_center, high_min, _high_max) = cluster_ranges

    low_to_mid_gap = mid_min - low_max
    mid_to_high_gap = high_min - mid_max
    if low_to_mid_gap < _MIN_GAP_FOR_SEVERITY_CALIBRATION or mid_to_high_gap < _MIN_GAP_FOR_SEVERITY_CALIBRATION:
        return (
            ThresholdCalibration(DEFAULT_CRITICAL_PERCENT, False),
            ThresholdCalibration(DEFAULT_MODERATE_PERCENT, False),
        )

    moderate_cutoff = (low_center + mid_center) / 2
    critical_cutoff = (mid_center + high_center) / 2

    # Sanity clamp + ordering guarantee: moderate must always sit below
    # critical by at least 1 percentage point, and both must stay inside
    # a sane 0-100% range. Belt-and-braces alongside the gap check above.
    moderate_cutoff = max(1.0, min(95.0, moderate_cutoff))
    critical_cutoff = max(moderate_cutoff + 1.0, min(99.0, critical_cutoff))

    return (
        ThresholdCalibration(round(float(critical_cutoff), 2), True),
        ThresholdCalibration(round(float(moderate_cutoff), 2), True),
    )


def _fit_kmeans_and_get_sorted_cluster_ranges(
    values: List[float], n_clusters: int
) -> List[Tuple[float, float, float]]:
    """
    Shared helper: fits scikit-learn's KMeans on a 1-D list of numbers
    and returns, for each cluster (sorted by center, ascending), a tuple
    of (center, min_value_in_cluster, max_value_in_cluster). The min/max
    per cluster is what lets the calling function check whether
    neighbouring clusters actually have a gap between them, or whether
    they touch/overlap (meaning the "split" isn't a real pattern -- see
    module docstring).

    KMeans expects a 2-D array (rows = samples, columns = features) even
    when there's only one feature -- reshape(-1, 1) turns our flat list
    of numbers into a column of single-value "points", which is the
    standard way to cluster 1-D data with this API.
    """
    from sklearn.cluster import KMeans  # lazy -- see this module's top-of-file note for why

    values_array = np.array(values, dtype=float).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=_KMEANS_RANDOM_STATE, n_init=_KMEANS_N_INIT)
    cluster_labels = kmeans.fit_predict(values_array)
    centers = kmeans.cluster_centers_.reshape(-1)

    ranges = []
    for cluster_index in np.argsort(centers):
        members = values_array[cluster_labels == cluster_index].reshape(-1)
        ranges.append((float(centers[cluster_index]), float(members.min()), float(members.max())))
    return ranges
