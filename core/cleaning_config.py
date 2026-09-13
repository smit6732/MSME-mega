"""
core/cleaning_config.py

CleaningConfig: the single, explicit place every Tier 1
(core/remediation.py) auto-fix decision reads its safety knobs from --
e.g. "is casing normalization allowed", "how strict does a categorical-
typo match need to be before it's auto-applied", "what happens to a
domain outlier". Centralizing these knobs here (rather than scattering
constants across remediation.py) is what lets app.py's "Cleaning
intensity" selector change behavior by swapping ONE object, and what
keeps every automated decision auditable/explainable: any two runs with
the same CleaningConfig will always make the same choices.

What this file does NOT control: core/validity.py's DETECTION never
reads a CleaningConfig at all -- what counts as a Finding is fixed and
deterministic, computed once, upstream, before a user ever picks a
cleaning intensity (see that module's own docstring). CleaningConfig
only governs what core/remediation.py is ALLOWED to attempt once a
Finding already exists, and it never loosens remediation's own
correctness guards (the ambiguous-date / ambiguous-numeric-grouping
checks stay exactly as strict regardless of intensity -- those are
"never guess" guards, not policy choices; see core/remediation.py's
module docstring).

Every field below only ever WIDENS what remediation is allowed to
attempt as intensity increases. The default (CONSERVATIVE) is
deliberately the safest -- everything this project already did before
this feature, plus the new fixes that are safe unconditionally
(type-coercion to NaN, missing-token normalization).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ---- Missing-data strategies -------------------------------------------
# What remediation is allowed to do about a column's OWN missing_data
# Finding (blank cells, not the separate missing_value_representation
# token-normalization fix, which is always safe and always applied).
# "leave" (the conservative default) matches this project's original,
# unconditional stance: there is no universally-safe way to guess what
# belongs in a blank cell, so by default nothing is auto-filled.
MISSING_STRATEGY_LEAVE = "leave"
MISSING_STRATEGY_DROP_ROW = "drop_row"
MISSING_STRATEGY_CONSTANT = "constant"
MISSING_STRATEGY_MEDIAN = "median"
MISSING_STRATEGY_MODE = "mode"

# ---- Domain-outlier handling --------------------------------------------
OUTLIER_ACTION_NAN = "set_to_nan"   # default: flag as missing rather than invent a value
OUTLIER_ACTION_CLIP = "clip"        # clamp to the nearest bound instead


@dataclass
class CleaningConfig:
    """
    One cleaning run's full set of policy choices. See module docstring
    for what this does and doesn't control.

    allow_categorical_case_normalization: when True, core/remediation.py
        may also standardize a column's exact CASING to its dominant
        spelling (e.g. every "diesel"/"DIESEL" -> "Diesel") for values
        that already normalize identically -- this is a stricter version
        of the existing whitespace-only text fix. False (the default)
        keeps this project's original stance: casing is left alone, only
        whitespace is trimmed, because there's no objectively "correct"
        case.
    categorical_fuzzy_threshold: the minimum rapidfuzz match score (see
        core/validity.py's own CATEGORICAL_FUZZY_THRESHOLD, the DETECTION-
        time floor) a specific minority->canonical mapping must have
        BEEN DETECTED AT before remediation trusts it enough to actually
        rewrite cells. Raising this (a more conservative intensity) means
        remediation only acts on the most confident matches from the
        already-detected map; it can never lower detection's own floor.
    outlier_action: OUTLIER_ACTION_NAN (default) or OUTLIER_ACTION_CLIP.
    missing_strategy: one of the MISSING_STRATEGY_* constants above.
    missing_constant: the literal value used when missing_strategy is
        MISSING_STRATEGY_CONSTANT.
    domain_rule_overrides: optional {column_name_substring: rule_dict}
        passed straight through to core/validity.py's check_validity --
        see that module's _NAMED_DOMAIN_RULES for the rule shape. Lets a
        specific dataset widen/replace a built-in bound (e.g. this
        table's "Mileage" is genuinely allowed up to 2,000,000 km)
        without touching code.
    """
    allow_categorical_case_normalization: bool = False
    categorical_fuzzy_threshold: float = 88.0
    outlier_action: str = OUTLIER_ACTION_NAN
    missing_strategy: str = MISSING_STRATEGY_LEAVE
    missing_constant: str = ""
    domain_rule_overrides: Dict[str, dict] = field(default_factory=dict)


# Three named presets for app.py's "Cleaning intensity" selector (see
# that file's render_remediation_section). Conservative is the default
# everywhere a caller doesn't explicitly choose one -- matches this
# project's existing "the safest option is always the unconfigured
# default" stance (see core/calibration.py's fallback defaults).
CONSERVATIVE = CleaningConfig()

STANDARD = CleaningConfig(
    allow_categorical_case_normalization=True,
    categorical_fuzzy_threshold=90.0,  # trust ONLY very confident typo matches
)

AGGRESSIVE = CleaningConfig(
    allow_categorical_case_normalization=True,
    categorical_fuzzy_threshold=88.0,  # trust every match validity.py's own floor allows
    outlier_action=OUTLIER_ACTION_CLIP,
    missing_strategy=MISSING_STRATEGY_LEAVE,  # imputation is opt-in per-run, never bundled into a preset silently
)

INTENSITY_PRESETS: Dict[str, CleaningConfig] = {
    "Conservative": CONSERVATIVE,
    "Standard": STANDARD,
    "Aggressive": AGGRESSIVE,
}
