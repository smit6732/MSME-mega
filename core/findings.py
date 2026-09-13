"""
core/findings.py

Defines Finding -- the one shared, structured "here's a problem" record
that every check dimension's output eventually becomes, and that both
the fix-list generator (core/fixlist.py) and the optional local-LLM
phrasing layer (core/llm_phrasing.py) consume.

Why one shared structure instead of each dimension inventing its own
shape: it means the multi-table pipeline (core/pipeline.py) and the UI
(app.py) can always work with a single, predictable type -- "a list of
Findings" -- no matter which of the four checks a problem came from, or
which table (file/sheet) it's about. It also gives the LLM phrasing
layer one narrow, controlled input: a Finding's already-computed facts
(column, percentage, severity) and nothing else -- never the raw data.

This file has zero pandas/sklearn/llama_cpp imports on purpose. It's
pure data definitions, so every other module can depend on it without
pulling in anything heavy.
"""

from dataclasses import dataclass
from typing import Optional

# Canonical severity tier labels -- the single source of truth every
# other module imports these from (templates/fix_templates.py,
# core/fixlist.py, core/calibration.py), so there is exactly one
# spelling of each tier used everywhere in the codebase.
SEVERITY_CRITICAL = "critical"
SEVERITY_MODERATE = "moderate"
SEVERITY_MINOR = "minor"

# The scorecard dimensions, as machine-readable labels. Matches the check
# modules: core/completeness.py, core/consistency.py, core/duplication.py,
# core/structure.py, core/validity.py.
CHECK_TYPE_COMPLETENESS = "completeness"
CHECK_TYPE_CONSISTENCY = "consistency"
CHECK_TYPE_DUPLICATION = "duplication"
CHECK_TYPE_STRUCTURE = "structure"
CHECK_TYPE_VALIDITY = "validity"


@dataclass
class Finding:
    """
    One data-quality problem found in one table (one uploaded file, or
    one sheet within an uploaded Excel file), described in both a
    guaranteed deterministic sentence and an optional AI-rephrased one.

    rule_based_description is ALWAYS present -- it's produced by pure
    template string-formatting (templates/fix_templates.py) and is what
    the tool falls back to whenever the local LLM isn't available or
    fails. This is what keeps the tool's core promise ("works completely
    normally with zero AI") true no matter what.

    ai_phrased_description stays None until (and unless) the local model
    successfully rephrases this exact Finding. The UI uses whether this
    is None to decide whether to show an "AI-enhanced phrasing" tag --
    so AI-touched text is never visually indistinguishable from the
    guaranteed deterministic core.
    """
    table_name: str            # which file/sheet this problem was found in
    column_name: str           # column name, or a placeholder like "(whole row)"
    check_type: str            # one of the CHECK_TYPE_* constants above
    issue_type: str            # finer-grained label, e.g. "missing_data",
                                #   used to pick the right sentence template
    percentage_affected: float
    severity: str               # one of the SEVERITY_* constants above
    rule_based_description: str
    ai_phrased_description: Optional[str] = None
    # A concrete example backing this finding, when the check that found
    # it produced one -- e.g. "Shree Balaji Traders vs Shri Balaji
    # Traders" for a fuzzy-duplicate finding, or a sample bad value for a
    # consistency finding. None for finding types that don't have one
    # (e.g. a plain missing-data count). This is the one field that
    # NEEDS to reach core/llm_phrasing.py's prompt for the AI-phrased
    # sentence to be able to keep it -- see that module's docstring.
    example: Optional[str] = None
    # Optional structured facts a Finding needs to carry so a LATER stage
    # can act on it WITHOUT re-deriving them from scratch -- e.g. the
    # exact minority-spelling -> canonical-spelling map core/validity.py
    # already computed for a categorical_inconsistency Finding, or the
    # numeric bounds it already computed for a domain_outlier Finding.
    # This keeps core/remediation.py "Finding-driven, not a fresh scan"
    # (see that module's docstring) even for fixes that need more than a
    # plain example string to apply safely: it reuses these exact
    # already-computed facts rather than re-fitting them against a
    # (possibly already-modified) copy of the data. None for every issue
    # type that doesn't need this -- which is most of them.
    details: Optional[dict] = None

    @property
    def display_description(self) -> str:
        """
        What the UI should actually show for this Finding: the AI
        phrasing if we successfully got one, otherwise the guaranteed
        rule-based sentence. Centralizing this "pick one of two strings"
        choice here means app.py never has to duplicate the
        if-AI-text-exists-use-it logic.
        """
        return self.ai_phrased_description or self.rule_based_description

    @property
    def is_ai_phrased(self) -> bool:
        """Convenience flag for the UI's "AI-enhanced phrasing" tag."""
        return self.ai_phrased_description is not None
