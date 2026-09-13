"""
app.py

Streamlit dashboard for the MSME Data Quality Scorecard.

Run with:
    streamlit run app.py

This file is intentionally "thin" -- it doesn't contain any scoring
logic, calibration logic, or AI-calling logic of its own. Its only job
is to accept file upload(s), call core/pipeline.py, and lay the results
out on screen. Everything that actually decides a score, a threshold, or
a sentence lives in core/*.py so it can be tested and explained on its
own, independent of the UI. This redesign pass touched ONLY this file --
zero changes to core/*.py, zero changes to what any function returns.

Multi-table support: the uploader accepts multiple files at once, and
any Excel file with multiple sheets has each sheet treated as its own
table automatically (see core/ingestion.py's load_all_tables). The page
always shows one combined overall summary first, then an expandable
section per table underneath.

Design system: colors, spacing, and type sizes are defined ONCE as CSS
custom properties (see _theme_tokens below) and referenced everywhere
else via var(--x) -- never a hardcoded hex value repeated in five
places. Two token sets (light/dark) feed the same CSS rules, so a
component only ever needs to be styled once to work in both themes.

Theme switching: Streamlit (checked: v1.61, no public runtime "set
theme" API -- st.context.theme is read-only) has no built-in way to flip
themes at runtime without a full server restart, so this file holds the
current choice in st.session_state and re-injects the CSS block (with
the other token set) on every rerun -- see render_header() and
render_theme_toggle(). This is exactly the documented fallback for when
a cleaner API isn't available.

Every dynamic value dropped into raw HTML (column names, table names,
AI-phrased sentences -- all ultimately from the user's own uploaded
data) is passed through html.escape() first, via the _html() helper --
so a business name containing `<`, `>`, or `&` can never break the
page's markup.
"""

import datetime
import html
import io
import json
import re
from typing import Optional
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.pipeline import run_multi_table_pipeline, run_pipeline_for_table
from core.ingestion import load_dataset
from core.remediation import remediate_table
from core.cleaning_config import INTENSITY_PRESETS
from core.chart_generation import generate_charts_for_table


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Score thresholds used to color-code every score in the app. Kept as
# constants at the top so they're easy to point to and justify on demo
# day -- same approach as the dimension weights in core/scoring.py.
SCORE_GOOD_THRESHOLD = 80
SCORE_NEEDS_ATTENTION_THRESHOLD = 50

# Severity display metadata -- label and emoji only. Deliberately NO
# color here: colors live exclusively in the theme token dicts below, so
# there is exactly one place (per theme) where "critical = this red"
# is decided, never redefined ad hoc in a specific component.
SEVERITY_META = {
    "critical": {"label": "Critical", "emoji": "🔴"},
    "moderate": {"label": "Moderate", "emoji": "🟡"},
    "minor": {"label": "Minor", "emoji": "🟢"},
}

# The icon field is a KEY into _DIMENSION_ICONS below, not an emoji --
# both the landing page's "What this tool checks" cards and the
# per-table dimension grid read from this SAME list, so the same four
# icons render identically (same glyph, same weight) everywhere the
# four dimensions appear. Before this, each call site picked its own
# emoji independently, and different emoji at different sizes read as
# visibly inconsistent between the two screens -- confirmed by direct
# screenshot comparison, not a style guess.
DIMENSION_META = [
    ("completeness_score", "Completeness", "completeness", "How much of the data is actually filled in."),
    ("consistency_score", "Consistency", "consistency", "Whether values are formatted the same way throughout."),
    ("duplication_score", "Duplication", "duplication", "Exact and near-duplicate records."),
    ("structure_score", "Structure", "structure", "Whether the file itself is sound -- headers, columns, IDs."),
    ("validity_score", "Validity", "validity", "Whether present, well-formatted values are actually realistic."),
]

# One small, hand-authored inline SVG per dimension -- deliberately NOT
# emoji (which renders differently across OSes/browsers/sizes, which is
# exactly what caused the inconsistency above) and deliberately NOT a
# third-party icon font or library (no new dependency). Each uses
# stroke="currentColor" so a single CSS `color` rule (already themed
# via var(--x), see _build_theme_css) is the only thing that controls
# its color in both light and dark mode -- no icon-specific color logic
# anywhere. viewBox is a consistent 24x24 for all four, so they share
# the same visual weight at any size.
_DIMENSION_ICONS = {
    "completeness": (
        '<svg class="mdq-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="5" y="4" width="14" height="17" rx="2"/>'
        '<path d="M9 4V3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v1"/>'
        '<path d="M9 12.5l2 2 4-4.5"/>'
        "</svg>"
    ),
    "consistency": (
        '<svg class="mdq-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="9" width="18" height="6" rx="1"/>'
        '<path d="M7 9v2.5M11 9v2.5M15 9v2.5M19 9v2.5"/>'
        "</svg>"
    ),
    "duplication": (
        '<svg class="mdq-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="8" y="8" width="13" height="13" rx="2"/>'
        '<path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>'
        "</svg>"
    ),
    "structure": (
        '<svg class="mdq-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<line x1="3" y1="9" x2="21" y2="9"/>'
        '<line x1="3" y1="15" x2="21" y2="15"/>'
        '<line x1="9" y1="3" x2="9" y2="21"/>'
        "</svg>"
    ),
    "validity": (
        '<svg class="mdq-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="9"/>'
        '<path d="M9.5 12.5l1.8 1.8 3.2-4.2"/>'
        "</svg>"
    ),
}


# A small, hand-authored mark for the tool itself -- used next to the
# header title (see render_header). Deliberately simple: an ascending
# bar-chart shape with a checkmark, echoing "data quality, verified" in
# one glyph rather than a stock emoji. currentColor + the accent token
# so it re-themes for free like every other icon in this file.
_MARK_SVG = (
    '<svg class="mdq-mark" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="32" height="32" rx="8" fill="var(--accent)"/>'
    '<g stroke="var(--accent-contrast)" stroke-width="2.4" stroke-linecap="round">'
    '<line x1="9" y1="22" x2="9" y2="17"/>'
    '<line x1="16" y1="22" x2="16" y2="12"/>'
    '<line x1="23" y1="22" x2="23" y2="15"/>'
    "</g>"
    '<path d="M9 13.5l3.5 3.5L16 13l3 3 3.5-4" stroke="var(--accent-contrast)" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    "</svg>"
)

# The empty state's small illustration (requirement 6c) -- messy,
# scattered lines on the left resolving, through an upload arrow, into
# one clean, checked sheet on the right. Reinforces "upload your messy
# data here" rather than decorating for its own sake. Uses var(--x)
# theme tokens directly (this renders in the main page DOM via
# render_html, not an iframe, so those references resolve normally).
# Favicon version of the same mark, as a standalone SVG with LITERAL hex
# colors rather than var(--x) tokens -- a favicon renders outside our
# page's DOM entirely (the browser paints it into its own tab chrome),
# so CSS custom properties from :root never reach it; this is the one
# place in the whole file a color is hardcoded on purpose, and it stays
# the same regardless of the in-page theme toggle (browser tab chrome
# isn't themed by us either way). Colors match the light theme's accent
# (kept in sync by hand with _LIGHT_TOKENS["accent"] -- see the color-
# system replacement note above for why this can't just reference the
# token dict directly).
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="8" fill="#077B97"/>'
    '<g stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round">'
    '<line x1="9" y1="22" x2="9" y2="17"/>'
    '<line x1="16" y1="22" x2="16" y2="12"/>'
    '<line x1="23" y1="22" x2="23" y2="15"/>'
    "</g>"
    '<path d="M9 13.5l3.5 3.5L16 13l3 3 3.5-4" stroke="#FFFFFF" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    "</svg>"
)

_UPLOAD_ILLUSTRATION_SVG = """
<svg viewBox="0 0 240 130" width="100%" height="130" xmlns="http://www.w3.org/2000/svg">
  <g stroke="var(--text-tertiary)" stroke-width="2" stroke-linecap="round" fill="none" opacity="0.55">
    <path d="M18 38 Q34 27 50 40 T82 36"/>
    <path d="M18 60 Q40 69 60 58 T92 63"/>
    <path d="M18 82 Q30 73 56 83"/>
  </g>
  <g stroke="var(--accent)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="M120 104 V52"/>
    <path d="M105 67 L120 52 L135 67"/>
  </g>
  <rect x="151" y="28" width="72" height="92" rx="9" fill="var(--bg-elevated)" stroke="var(--accent)" stroke-width="2.5"/>
  <g stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" opacity="0.55">
    <line x1="163" y1="49" x2="211" y2="49"/>
    <line x1="163" y1="63" x2="211" y2="63"/>
    <line x1="163" y1="77" x2="197" y2="77"/>
  </g>
  <circle cx="201" cy="99" r="10" fill="var(--minor-bg)" stroke="var(--minor)" stroke-width="2"/>
  <path d="M196.5 99l3 3 7-7" stroke="var(--minor)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>
"""

# Sequential, real pipeline stages (requirement 2) -- these codes are
# exactly the stage names core/pipeline.py's on_stage callback reports,
# so this label map can never drift out of sync with what's actually
# running: if pipeline.py's stage names ever change, the label for that
# stage simply falls back to the raw code (see _make_stage_reporter)
# rather than silently showing a stale/wrong description.
_STAGE_LABELS = {
    "loading": "Reading and profiling file structure…",
    "completeness": "Checking completeness — how much data is filled in…",
    "consistency": "Checking consistency — formatting patterns per column…",
    "duplication": "Scanning for exact and near-duplicate rows…",
    "structure": "Validating headers and file structure…",
    "validity": "Checking validity — realistic values and known types…",
    "scoring": "Combining the five dimension scores…",
    "fixlist": "Generating the fix list (loading local AI model on first run)…",
}

# A categorical/numeric column where almost every value is the same one
# thing (e.g. a repeated device ID) produces a bar chart that's just one
# giant bar -- technically correct, not informative. Below this many
# distinct values, show a plain sentence instead of a chart.
LOW_CARDINALITY_THRESHOLD = 3


def _html(value) -> str:
    """
    Escape any value before it goes into a raw HTML string. Every piece
    of dynamic text we render this way (a business name, a column name,
    an AI-phrased sentence) ultimately traces back to the user's own
    uploaded file -- this is what stops a stray `<`, `>`, or `&` in
    someone's data from ever breaking the page's markup.
    """
    return html.escape(str(value), quote=True)


def _flatten_html(raw: str) -> str:
    """
    st.markdown(..., unsafe_allow_html=True) is NOT a raw-HTML passthrough
    -- it still runs its input through a Markdown parser first, which then
    hands the result to the browser. That matters here because standard
    Markdown treats any line indented by 4+ spaces as an INDENTED CODE
    BLOCK, rendered as literal escaped text rather than parsed as HTML.
    Every HTML-building helper in this file uses nicely-indented,
    multi-line triple-quoted strings purely for OUR OWN readability in the
    source -- so before any of that HTML reaches st.markdown, we strip
    every line down to zero leading whitespace and join everything onto
    one line. That sidesteps the code-block rule entirely (a single line
    can't be "indented"), and it's harmless for HTML, since browsers
    already collapse whitespace between tags.
    """
    return " ".join(line.strip() for line in raw.strip().splitlines())


def render_html(raw: str) -> None:
    """
    Single choke point for every unsafe_allow_html render in this file --
    see _flatten_html's docstring for exactly why this matters. Using one
    shared function (instead of calling st.markdown(..., unsafe_allow_html=True)
    directly all over the file) means the indentation fix can't accidentally
    be forgotten on a new component added later.
    """
    st.markdown(_flatten_html(raw), unsafe_allow_html=True)


def _score_tier(score: float) -> str:
    """Map a 0-100 score to one of three tier keys ("good"/"warn"/"bad"),
    used to pick a color throughout the UI from the active theme's
    token set. Same thresholds everywhere a score needs a color."""
    if score >= SCORE_GOOD_THRESHOLD:
        return "good"
    if score >= SCORE_NEEDS_ATTENTION_THRESHOLD:
        return "warn"
    return "bad"


_TIER_LABELS = {"good": ("🟢", "Good"), "warn": ("🟡", "Needs Attention"), "bad": ("🔴", "Poor")}


def score_indicator(score: float) -> str:
    """Plain-text tier label with emoji, used inside expander labels
    (Streamlit expander labels don't render custom HTML)."""
    emoji, label = _TIER_LABELS[_score_tier(score)]
    return f"{emoji} {label}"


# ---------------------------------------------------------------------------
# Design system: color tokens (light + dark), spacing, and type scale --
# defined ONCE here, referenced everywhere else as CSS var(--x).
# ---------------------------------------------------------------------------

# Warm zinc-toned neutrals + a single cyan/electric-teal signature
# accent (purple/violet deliberately excluded -- the generic "AI product"
# cliche as of 2026). Every value below was run through a real WCAG
# relative-luminance contrast check (not eyeballed) against EVERY
# background surface it actually appears behind in this file -- not just
# the lightest/easiest one. That check caught three of the values that
# would look fine in isolation but fail in practice:
#   - text-tertiary: the proposed #8A8177 passed against white but fell
#     to ~3.2:1 against bg-muted (#EDEAE5, the darkest of the light-mode
#     surfaces it's used on) -- darkened to #6F675F, which now clears
#     4.5:1 against all four light surfaces. Dark mode needed its OWN
#     value (#91897F) rather than reusing the light one, which was too
#     dark to read against dark-mode surfaces.
#   - accent: #0891B2 read as only 3.5-3.7:1 as TEXT (the tagline, and
#     white-on-accent in the mark/step-num pills) -- darkened to #077B97,
#     same hue, which clears 4.5:1 both as text-on-background and as the
#     background under white text.
#   - critical: #DC2626 on its own critical-bg was 4.41:1, just under
#     the line -- nudged to #DA2323.
# moderate/minor were already compliant at the proposed values and are
# unchanged. See the contrast check this was verified with for the exact
# numbers (every pairing >=4.5:1 in both light and dark, zero failures).
_LIGHT_TOKENS = {
    # Slate/Sky-Blue refactor (light mode only -- dark mode below is
    # untouched, kept from the prior verified pass). Every value here
    # was contrast-checked the same way as before: against the HARDEST
    # surface it sits on, not the easiest. accent (#0284C7, the exact
    # "Sky Blue" requested) is 4.10:1 as text -- fine for the large
    # circular step-badge digit (which the request also gave this exact
    # color for) but under 4.5:1 for flowing text, so the tagline uses
    # accent-text (#0369A1, same hue) instead. text-tertiary needed
    # darkening from the obvious Slate-500 (#64748B, 3.86:1 on
    # bg-muted) to #59687C to clear 4.5:1 everywhere it's used.
    "bg": "#F8FAFC",
    "bg-elevated": "#FFFFFF",
    "bg-subtle": "#F1F5F9",
    "bg-muted": "#E2E8F0",
    "border": "#E2E8F0",
    "border-subtle": "#EEF2F6",
    "text-primary": "#0F172A",
    "text-secondary": "#475569",
    "text-tertiary": "#59687C",
    "accent": "#0284C7",
    "accent-text": "#0369A1",
    "accent-bg": "#E0F2FE",
    "accent-border": "#BAE6FD",
    "accent-contrast": "#FFFFFF",
    "critical": "#DA2323", "critical-bg": "#FEF2F2", "critical-border": "#FECACA",
    "moderate": "#B45309", "moderate-bg": "#FFFBEB", "moderate-border": "#FDE68A",
    "minor": "#15803D", "minor-bg": "#F0FDF4", "minor-border": "#BBF7D0",
    "shadow-sm": "0 1px 2px rgba(15,23,42,0.05)",
    "shadow-md": "0 4px 12px rgba(15,23,42,0.06)",
    # Card separation (requirement 8): light mode already gets natural
    # lift from shadow contrast against a plain page. This is that same
    # lift made explicit as a token, so cards use ONE rule in both
    # themes -- see the dark set below for why dark mode's version is
    # different (a border-only card barely reads as "lifted" against a
    # dark page; light mode's is mostly shadow, dark mode's leans more
    # on a crisper border plus a soft glow).
    "card-shadow": "0 1px 2px rgba(28,25,23,0.04), 0 1px 1px rgba(28,25,23,0.03)",
}

_DARK_TOKENS = {
    "bg": "#121110",
    "bg-elevated": "#1A1816",
    "bg-subtle": "#201D1A",
    "bg-muted": "#272320",
    "border": "#322D29",
    "border-subtle": "#282420",
    "text-primary": "#F5F2EF",
    "text-secondary": "#B8B0A8",
    "text-tertiary": "#91897F",
    "accent": "#22D3EE",
    "accent-text": "#22D3EE",  # already passes as text at this lightness; no separate shade needed
    "accent-bg": "#103239",
    "accent-border": "#1B4D56",
    "accent-contrast": "#0B1220",
    "critical": "#F87171", "critical-bg": "#3A1518", "critical-border": "#5B2226",
    "moderate": "#FBBF24", "moderate-bg": "#3A2C0C", "moderate-border": "#5C4315",
    "minor": "#4ADE80", "minor-bg": "#0F2E1C", "minor-border": "#1D4D31",
    "shadow-sm": "0 1px 2px rgba(0,0,0,0.35)",
    "shadow-md": "0 4px 16px rgba(0,0,0,0.4)",
    "card-shadow": "0 0 0 1px rgba(255,255,255,0.03), 0 3px 10px rgba(0,0,0,0.35)",
}

# Aliasing the "good"/"warn"/"bad" score tiers onto the exact same colors
# as "minor"/"moderate"/"critical" severity -- they mean the same thing
# (low/medium/high concern), so they should look the same, not almost-
# the-same-but-slightly-off in a way that undermines "red always means
# the same thing everywhere."
_TIER_TOKEN_ALIAS = {"good": "minor", "warn": "moderate", "bad": "critical"}


def _theme_tokens() -> dict:
    """The active theme's token dict, based on st.session_state (see
    render_theme_toggle). Defaults to light -- the right default for an
    occasional-use business tool, per this redesign's own brief."""
    return _DARK_TOKENS if st.session_state.get("theme") == "dark" else _LIGHT_TOKENS


def _build_theme_css() -> str:
    """
    Builds the ENTIRE page's CSS in one block: a :root declaration
    binding every token above to a CSS variable, followed by component
    rules that all reference those variables (never a literal hex code
    below this point) so re-running this function with the other
    token set is the whole story of switching themes -- no per-component
    logic needs to know which theme is active.
    """
    tokens = _theme_tokens()
    root_vars = "".join(f"--{name}:{value};" for name, value in tokens.items())

    return f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    :root {{ {root_vars} }}

    /* ---- Typography (requirement 1) ------------------------------------
       Inter for everything data-dense/body -- optimized for legibility at
       small sizes. Space Grotesk (more character than a generic system
       sans, without being loud) reserved for the score number and section
       headings ONLY, so it reads as a deliberate accent, not a full
       typeface swap. Both are set once, on .stApp, and inherit down --
       never overridden per-component. Contrast is unaffected by this
       change (it's a font-family swap only, no color/weight change to
       anything already contrast-checked). */
    .stApp {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    .mdq-hero h1, .mdq-score-number, .mdq-section-title, .mdq-dim-value {{
        font-family: 'Space Grotesk', 'Inter', sans-serif;
    }}

    /* ---- Visible keyboard focus (requirement 3b) ------------------------
       The accent token, not the browser default outline -- applied via
       :focus-visible so it shows for keyboard navigation without adding a
       ring on every mouse click (the standard, correct way to do this;
       :focus-visible is what actually distinguishes the two). Covers
       every interactive element in the app: theme toggle, uploader,
       expanders, popovers, buttons, download button, and the sortable
       table headers rendered inside the components.html iframe (that
       iframe carries its own copy of this rule -- see
       _sortable_column_table_html). */
    .stApp a:focus-visible, .stApp button:focus-visible, .stApp [role="button"]:focus-visible,
    .stApp input:focus-visible, .stApp [tabindex]:focus-visible,
    [data-testid="stFileUploaderDropzone"]:focus-within, [data-baseweb="switch"]:focus-within,
    [data-testid="stExpander"] summary:focus-visible, [data-testid="stPopoverButton"]:focus-visible,
    button[kind="popover"]:focus-visible {{
        outline: 2px solid var(--accent) !important;
        outline-offset: 2px !important;
        border-radius: 6px;
    }}

    /* ---- Native Streamlit chrome, re-themed to match ------------------- */
    [data-testid="stAppViewContainer"], .stApp {{ background: var(--bg); }}
    [data-testid="stHeader"] {{ background: var(--bg); }}
    [data-testid="stSidebar"] {{ background: var(--bg-subtle); border-right: 1px solid var(--border); }}
    [data-testid="stSidebarContent"] {{ color: var(--text-primary); }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }}

    /* Ensure the core layout frame provides clean spacing around the main title */
    .main .block-container {{
        padding-top: 4rem !important;
        padding-bottom: 3rem !important;
    }}

    /* Fix sidebar header padding so content drops gracefully below default icons */
    [data-testid="stSidebarUserContent"] {{
        padding-top: 1.5rem !important;
    }}
    /* Scoped to .stApp (not a bare "body,p,span,div") so this can't bleed
       into elements that need their OWN color rule to win instead --
       native buttons and the uploader caption below are exactly that
       case: Streamlit gives them a color from its own generated CSS, so
       a blanket "every span/div is text-primary" rule was overriding
       button text without also fixing button background, producing
       near-invisible text. Those specific elements get their own,
       higher-specificity rule instead of relying on this general one. */
    .stApp, .stApp [data-testid="stMarkdownContainer"], .stApp p, .stApp span, .stApp div, .stApp li {{ color: var(--text-primary); }}
    [data-testid="stCaptionContainer"], .stCaption {{ color: var(--text-tertiary) !important; }}
    hr {{ border-color: var(--border); margin: 1.6rem 0; }}
    a, a:visited {{ color: var(--accent); }}

    /* Native buttons (the uploader's "Browse files", any st.button) --
       background AND text set together so they can never end up as
       near-invisible text on an unthemed background. */
    [data-testid^="stBaseButton"] {{
        background: var(--bg-elevated) !important; color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background: var(--bg-elevated); border: 2px dashed var(--border); border-radius: 12px;
        padding: 1.6rem; transition: border-color 0.2s ease;
    }}
    [data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploaderDropzone"]:focus-within {{
        border-color: var(--accent);
    }}
    /* The "uploaded file" chip that appears after a file is added --
       same bug pattern as the button above (native white background,
       my text-primary color on top of it) if left untouched. */
    [data-testid="stFileChip"] {{
        background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    }}
    [data-testid="stFileChipName"] {{ color: var(--text-primary) !important; }}
    /* !important: this specific element gets its color from Streamlit's
       own generated CSS as an rgba() tied to the STATIC config.toml
       theme, which otherwise outranks the rule above. */
    [data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzoneInstructions"] * {{
        color: var(--text-tertiary) !important;
    }}

    [data-testid="stExpander"] {{
        border-radius: 12px; border: 1px solid var(--border); background: var(--bg-elevated);
    }}
    /* !important: Streamlit gives an OPEN expander's <summary> its own
       hardcoded light background tied to the static config.toml theme
       (confirmed by inspecting the live DOM -- a collapsed expander's
       summary is transparent and inherits fine, but an expanded one
       isn't, which is exactly the kind of "half-switched" dark-mode bug
       this file's own brief calls out). Forcing it here covers both
       states consistently. */
    [data-testid="stExpander"] summary {{
        font-weight: 600; color: var(--text-primary) !important; background: var(--bg-elevated) !important;
    }}
    [data-testid="stExpander"] summary:hover {{ background: var(--bg-muted) !important; }}

    /* Popovers (the AI-provenance and calibration-badge content) render
       in a React portal OUTSIDE .stApp in the DOM -- confirmed by
       inspecting the live DOM, not assumed -- so none of the .stApp-
       scoped text rules above ever reach them. Everything a popover
       needs is scoped to stPopoverBody instead, which (unlike .stApp)
       genuinely IS an ancestor of everything rendered inside it. */
    [data-testid="stPopoverBody"] {{ background: var(--bg-elevated); border: 1px solid var(--border); }}
    [data-testid="stPopoverBody"] p, [data-testid="stPopoverBody"] li, [data-testid="stPopoverBody"] span,
    [data-testid="stPopoverBody"] div, [data-testid="stPopoverBody"] strong {{ color: var(--text-primary); }}
    [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] {{ color: var(--text-tertiary) !important; }}
    [data-testid="stPopoverBody"] blockquote {{ color: var(--text-secondary) !important; border-left-color: var(--border) !important; }}
    [data-testid="stPopoverBody"] code {{ background: var(--bg-muted) !important; color: var(--accent) !important; }}
    [data-testid="stPopoverButton"], button[kind="popover"] {{
        background: var(--bg-muted) !important; color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important; border-radius: 999px !important;
        font-size: 0.78rem !important; font-weight: 600 !important; padding: 0.15rem 0.75rem !important;
    }}

    [data-testid="stAlert"] {{ border-radius: 12px; }}

    [data-testid="stStatusWidget"] {{ background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 12px; }}

    [data-baseweb="switch"] {{ accent-color: var(--accent); }}

    /* ---- Hero header ---------------------------------------------------- */
    .mdq-hero {{ padding: 0 0 .25rem; }}
    .mdq-hero h1 {{ font-size: 1.9rem; font-weight: 800; margin: 0 0 .35rem; letter-spacing: -0.01em; color: var(--text-primary); }}
    .mdq-hero p {{ color: var(--text-secondary); font-size: 0.98rem; line-height: 1.6; max-width: 780px; margin: 0; }}
    .mdq-steps {{ display: flex; gap: .6rem; margin: 1rem 0 .25rem; flex-wrap: wrap; }}
    .mdq-step {{ display: flex; align-items: center; gap: .45rem; background: var(--bg-subtle); border: 1px solid var(--border);
                border-radius: 999px; padding: .35rem .85rem .35rem .55rem; font-size: .82rem; color: var(--text-secondary); }}
    .mdq-step-num {{ display: inline-flex; align-items: center; justify-content: center; width: 1.3rem; height: 1.3rem;
                    border-radius: 999px; background: var(--accent); color: var(--accent-contrast); font-size: .72rem; font-weight: 700; }}

    /* ---- Section headings ------------------------------------------------ */
    .mdq-section-title {{ font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin: 1.8rem 0 .7rem; }}

    /* ---- Score hero card ---------------------------------------------------
       Requirement 4 (restrained "AI-era" accent): a soft diagonal
       gradient hinting at the accent color in one corner -- never
       filling the whole card, and never anywhere near the severity
       red/amber/green system, so it can't be mistaken for carrying its
       own meaning. Requirement 3 (score-reveal entrance): one CSS
       @keyframes rise-in, ~420ms -- within the "nothing over ~400-500ms"
       guidance, and it only ever plays once per render, never on a
       repeated interaction. */
    @keyframes mdq-rise-in {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .mdq-score-hero {{ display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;
                       border-radius: 18px; padding: 1.75rem 2rem; border: 1px solid var(--border);
                       background: linear-gradient(135deg, var(--bg-elevated) 55%, var(--accent-bg) 145%);
                       box-shadow: var(--shadow-md); animation: mdq-rise-in 420ms ease-out; }}
    .mdq-score-number {{ font-size: 3.4rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }}
    .mdq-score-of100 {{ font-size: 1.15rem; font-weight: 500; opacity: .55; margin-left: .2rem; }}
    .mdq-score-tier-pill {{ display: inline-flex; align-items: center; gap: .35rem; font-weight: 700; font-size: .95rem;
                            padding: .3rem .8rem; border-radius: 999px; margin-bottom: .5rem; }}
    .mdq-score-caption {{ font-size: .87rem; color: var(--text-secondary); max-width: 420px; }}
    .mdq-score-legend {{ display:flex; gap: .9rem; font-size: .78rem; color: var(--text-tertiary); margin-top: .6rem; flex-wrap: wrap; }}

    /* ---- Issue-count chips (executive summary strip) ---------------------- */
    .mdq-count-row {{ display: flex; gap: .7rem; flex-wrap: wrap; margin-top: 1.1rem; }}
    .mdq-count-chip {{ display: flex; align-items: center; gap: .55rem; border-radius: 12px; padding: .7rem 1.05rem;
                       border: 1px solid var(--border); background: var(--bg-elevated); min-width: 140px;
                       box-shadow: var(--card-shadow); }}
    .mdq-count-chip .n {{ font-size: 1.5rem; font-weight: 800; line-height: 1; }}
    .mdq-count-chip .lbl {{ font-size: .78rem; color: var(--text-secondary); font-weight: 600; }}

    /* ---- Dimension score grid ---------------------------------------------- */
    .mdq-dim-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .9rem; margin: .6rem 0 1rem; }}
    @media (max-width: 900px) {{ .mdq-dim-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
    @media (max-width: 700px) {{ .mdq-dim-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .mdq-dim-card {{ background: var(--bg-muted); border: 1px solid var(--border-subtle); border-radius: 12px; padding: .9rem 1.05rem;
                     box-shadow: var(--card-shadow); }}
    .mdq-dim-label {{ font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--text-tertiary);
                      display: flex; align-items: center; gap: .35rem; }}
    .mdq-dim-value {{ font-size: 1.6rem; font-weight: 800; color: var(--text-primary); margin: .15rem 0 .55rem; }}
    .mdq-bar-track {{ height: 7px; background: var(--border); border-radius: 999px; overflow: hidden; }}
    .mdq-bar-fill {{ height: 100%; border-radius: 999px; }}

    /* ---- Severity badge: color + icon + text label, always together --------- */
    .mdq-sev-badge {{ display: inline-flex; align-items: center; gap: .35rem; font-weight: 700; font-size: .78rem;
                      padding: .22rem .65rem; border-radius: 999px; border: 1px solid; }}
    .mdq-sev-critical {{ color: var(--critical); background: var(--critical-bg); border-color: var(--critical-border); }}
    .mdq-sev-moderate {{ color: var(--moderate); background: var(--moderate-bg); border-color: var(--moderate-border); }}
    .mdq-sev-minor {{ color: var(--minor); background: var(--minor-bg); border-color: var(--minor-border); }}

    /* ---- Finding text + neutral tags --------------------------------------- */
    .mdq-finding-text {{ font-size: .93rem; color: var(--text-primary); line-height: 1.55; margin: .55rem 0 .3rem; }}
    .mdq-tag-table {{ display: inline-block; font-size: .72rem; font-weight: 600; padding: .12rem .55rem; border-radius: 6px;
                      background: var(--bg-muted); color: var(--text-secondary); border: 1px solid var(--border); }}

    /* ---- Status banners (requirement 7) ---------------------------------
       Was a solid tinted fill with colored body text -- read as a muddy
       block in dark mode, and put color-as-meaning on the whole
       paragraph rather than on one clear signal. Rebuilt to match the
       pattern finding cards already use well: a neutral card (bg-elevated,
       normal text-primary body copy) with ONE colored left-border accent
       bar carrying the severity meaning, paired with the emoji already in
       every banner's own text -- so color still never carries meaning
       alone, it's just concentrated in one clean accent instead of
       smeared across the whole surface. */
    .mdq-banner {{ border-radius: 10px; padding: .85rem 1.15rem; border: 1px solid var(--border); border-left-width: 4px;
                  font-size: .9rem; margin: .6rem 0 1rem; background: var(--bg-elevated); color: var(--text-primary);
                  box-shadow: var(--card-shadow); }}
    .mdq-banner-warn {{ border-left-color: var(--moderate); }}
    .mdq-banner-good {{ border-left-color: var(--minor); }}
    .mdq-banner-error {{ border-left-color: var(--critical); }}

    /* ---- Table sub-caption ------------------------------------------------- */
    .mdq-table-sub {{ font-size: .82rem; color: var(--text-tertiary); margin-bottom: .3rem; }}

    /* ---- Hand-built HTML table (per-column detail) -- see render_table_section
       for why this isn't st.dataframe: its canvas-rendered grid can't be
       re-themed by CSS, which would break dark mode for exactly this table. --- */
    .mdq-table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; margin-top: .3rem; }}
    .mdq-table {{ width: 100%; border-collapse: collapse; font-size: .87rem; }}
    .mdq-table thead th {{ text-align: left; font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
                           color: var(--text-tertiary); background: var(--bg-muted); padding: .6rem .9rem; border-bottom: 1px solid var(--border); }}
    .mdq-table thead th.num {{ text-align: right; }}
    .mdq-table tbody td {{ padding: .55rem .9rem; border-bottom: 1px solid var(--border-subtle); color: var(--text-primary); }}
    .mdq-table tbody tr:last-child td {{ border-bottom: none; }}
    .mdq-table tbody tr:hover {{ background: var(--bg-subtle); }}
    .mdq-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .mdq-table td.num.combined {{ font-weight: 700; }}

    /* ---- Empty / first-run state -------------------------------------------- */
    .mdq-empty-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .9rem; margin: 1.4rem 0; }}
    @media (max-width: 900px) {{ .mdq-empty-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
    @media (max-width: 700px) {{ .mdq-empty-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .mdq-empty-card, .mdq-feature-card {{ background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px; padding: 1.1rem 1.2rem; }}
    .mdq-empty-card .icon, .mdq-feature-card .icon {{ font-size: 1.5rem; }}
    .mdq-empty-card h4, .mdq-feature-card h4 {{ font-size: .95rem; font-weight: 700; margin: .5rem 0 .3rem; color: var(--text-primary); }}
    .mdq-empty-card p, .mdq-feature-card p {{ font-size: .82rem; color: var(--text-secondary); line-height: 1.5; margin: 0; }}
    .mdq-empty-hint {{ background: var(--accent-bg); border: 1px solid var(--accent-border); border-radius: 12px;
                       padding: .8rem 1.1rem; font-size: .85rem; color: var(--text-secondary); margin-top: 1rem; }}
    .mdq-empty-hint code {{ background: var(--bg-muted); padding: .1rem .4rem; border-radius: 4px; color: var(--text-primary); }}
    .mdq-empty-illustration {{ display: flex; justify-content: center; margin: .4rem 0 1.2rem; opacity: .95; }}

    /* ---- Shared icon/mark sizing --------------------------------------------
       One rule per context, all pointing at the SAME svg markup
       (_DIMENSION_ICONS) -- see that dict's own docstring for why this is
       the fix for requirement 0b (mismatched icons between screens). */
    .mdq-dim-label .mdq-icon {{ width: .95rem; height: .95rem; flex-shrink: 0; }}
    .mdq-empty-card .icon .mdq-icon, .mdq-feature-card .icon .mdq-icon {{ width: 1.6rem; height: 1.6rem; color: var(--accent); }}
    .mdq-empty-card .icon, .mdq-feature-card .icon {{ line-height: 0; }}
    .mdq-mark {{ width: 2.1rem; height: 2.1rem; flex-shrink: 0; vertical-align: middle; }}
    .mdq-hero-title-row {{ display: flex; align-items: center; gap: .65rem; }}
    .mdq-tagline {{ font-size: 1.05rem; font-weight: 600; color: var(--accent-text); margin: -.1rem 0 .6rem; font-family: 'Space Grotesk', 'Inter', sans-serif; }}

    /* ---- Inline technical-keyword badge (e.g. `pandas`, `RapidFuzz`) ---- */
    .mdq-kw {{ background: var(--bg-muted); border: 1px solid var(--border); padding: 2px 6px;
              border-radius: 4px; font-family: monospace; font-size: 0.85em; color: var(--text-primary); }}

    /* ---- Sidebar top buffer + info badges (replacing plain bullets) ---- */
    .mdq-sidebar-spacer {{ height: .5rem; }}
    .mdq-info-badge {{ background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px;
                       padding: .7rem .85rem; margin-bottom: .55rem; font-size: .85rem; color: var(--text-secondary); line-height: 1.5; }}
    .mdq-info-badge b {{ color: var(--text-primary); }}

    /* ---- Workflow stepper cards (st.columns(3)) with connector ---- */
    .mdq-step-card {{ display: flex; align-items: center; gap: .55rem; background: var(--bg-elevated);
                      border: 1px solid var(--border); border-radius: 8px; padding: .6rem .8rem; position: relative; }}
    .mdq-step-num {{ flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%; background: var(--accent);
                     color: var(--accent-contrast); font-weight: 700; font-size: .8rem;
                     display: inline-flex; justify-content: center; align-items: center; }}
    .mdq-step-label {{ font-size: .85rem; font-weight: 600; color: var(--text-primary); }}
    .mdq-step-connector {{ text-align: center; color: var(--border); font-size: 1.2rem; padding-top: .2rem; }}
    @media (max-width: 700px) {{ .mdq-step-connector {{ display: none; }} }}

    /* ---- Feature-check grid (st.columns(4)) hover lift ---- */
    .mdq-feature-card {{ background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px;
                         padding: 1.1rem 1rem; box-shadow: var(--shadow-sm); transition: box-shadow 0.2s ease, transform 0.2s ease; }}
    .mdq-feature-card:hover {{ box-shadow: var(--shadow-md); transform: translateY(-2px); }}

    /* ---- Card separation in dark mode (requirement 8) -----------------------
       st.container(border=True) is what every finding card and manual-
       review item renders through -- its native border alone reads as too
       subtle against a dark page to feel "lifted". Same card-shadow token
       every other card uses, applied to Streamlit's own bordered-container
       wrapper so it never needs a bespoke class of its own. */
    [data-testid="stVerticalBlockBorderWrapper"] {{ box-shadow: var(--card-shadow); border-radius: 10px; }}

    /* ---- Skeleton loading placeholder (Section C) ---------------------------
       Shown the instant a file is dropped, before st.status even starts --
       "establishes structure upfront" rather than a blank pause, per the
       loading-state research this pass is grounded in. Shape matches the
       real score-hero + 4-dimension-card layout it's standing in for. */
    @keyframes mdq-shimmer {{ 0% {{ background-position: -200% 0; }} 100% {{ background-position: 200% 0; }} }}
    .mdq-skeleton {{ background: linear-gradient(90deg, var(--bg-muted) 25%, var(--border-subtle) 50%, var(--bg-muted) 75%);
                     background-size: 200% 100%; animation: mdq-shimmer 1.4s ease-in-out infinite; }}
    .mdq-skel-hero {{ height: 118px; border-radius: 18px; margin-bottom: .9rem; }}
    .mdq-skel-dimgrid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .9rem; }}
    @media (max-width: 900px) {{ .mdq-skel-dimgrid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
    @media (max-width: 700px) {{ .mdq-skel-dimgrid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .mdq-skel-dim {{ height: 92px; border-radius: 12px; }}

    /* ---- Sortable per-column table (requirement 6b) --------------------------
       Rendered inside its own components.html iframe (see
       _sortable_column_table_html) because it needs real click-to-sort JS
       -- Streamlit's markdown sanitizes out <script> tags, confirmed
       directly rather than assumed (a script injected via the same
       unsafe_allow_html path this file already uses for everything else
       never executes). This wrapper just keeps its outer frame visually
       consistent with every other .mdq-table on the page. */
    .mdq-sortable-frame {{ border: none; width: 100%; }}
    </style>
    """


# ---------------------------------------------------------------------------
# Small HTML component builders -- each returns a ready-to-render string.
# ---------------------------------------------------------------------------

def _dimension_grid_html(scorecard) -> str:
    cards = []
    for attr_name, label, icon_key, _description in DIMENSION_META:
        score = getattr(scorecard, attr_name)
        tier_alias = _TIER_TOKEN_ALIAS[_score_tier(score)]
        width = max(0.0, min(100.0, score))
        cards.append(f"""
            <div class="mdq-dim-card">
              <div class="mdq-dim-label">{_DIMENSION_ICONS[icon_key]} {_html(label)}</div>
              <div class="mdq-dim-value">{score:.1f}</div>
              <div class="mdq-bar-track"><div class="mdq-bar-fill" style="width:{width}%;background:var(--{tier_alias});"></div></div>
            </div>
        """)
    return f'<div class="mdq-dim-grid">{"".join(cards)}</div>'


def _severity_badge_html(severity: str) -> str:
    """The ONE place a severity badge (color + icon + text label,
    combined -- never color alone) is built, so every call site looks
    identical. Used for findings; the score tier pill is built inline in
    render_overall_summary since it also carries the headline number."""
    meta = SEVERITY_META[severity]
    return f'<span class="mdq-sev-badge mdq-sev-{severity}">{meta["emoji"]} {meta["label"]}</span>'


def _sortable_column_table_html(scorecard, tokens: dict) -> str:
    """
    Requirement 6b: the per-column score table, click-to-sort on any
    header, ascending/descending, with a visible active-sort indicator.

    Rendered as a fully self-contained HTML document via
    st.components.v1.html (a real iframe, see render_table_section) --
    NOT through this file's usual render_html() path, because that path
    goes through Streamlit's markdown renderer, which strips <script>
    tags (confirmed directly: a script injected that way never executes,
    not assumed). A components.html iframe is Streamlit's own built-in
    primitive for exactly this case -- genuine custom JS -- so this adds
    no new dependency, just uses a different (correct) Streamlit API for
    the one piece of the page that actually needs to run script.

    Because the iframe is a separate document with no access to this
    page's :root CSS variables, every color below is the CURRENT theme's
    already-resolved value from `tokens`, not a var(--x) reference --
    this is what keeps it in sync with the light/dark toggle: app.py
    re-calls this function (with the other token dict) on every rerun,
    same as every other themed component in this file.
    """
    rows = list(scorecard.per_column_scores.values())
    body_rows = "".join(
        f"<tr>"
        f'<td data-sort="{_html(column_score.column_name.lower())}">{_html(column_score.column_name)}</td>'
        f'<td class="num" data-sort="{column_score.completeness_score}">{column_score.completeness_score:.1f}</td>'
        f'<td class="num" data-sort="{column_score.consistency_score}">{column_score.consistency_score:.1f}</td>'
        f'<td class="num combined" data-sort="{column_score.combined_score}">{column_score.combined_score:.1f}</td>'
        f"</tr>"
        for column_score in rows
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: 'Inter', -apple-system, sans-serif; background: {tokens["bg-elevated"]}; color: {tokens["text-primary"]}; }}
  .wrap {{ overflow-x: auto; border: 1px solid {tokens["border"]}; border-radius: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  thead th {{ text-align: left; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
              color: {tokens["text-tertiary"]}; background: {tokens["bg-muted"]}; padding: 9px 14px; border-bottom: 1px solid {tokens["border"]};
              cursor: pointer; user-select: none; white-space: nowrap; }}
  thead th:hover {{ color: {tokens["accent"]}; }}
  thead th:focus-visible {{ outline: 2px solid {tokens["accent"]}; outline-offset: -2px; }}
  thead th.num {{ text-align: right; }}
  .sort-ind {{ font-size: 9px; color: {tokens["accent"]}; }}
  tbody td {{ padding: 8px 14px; border-bottom: 1px solid {tokens["border-subtle"]}; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: {tokens["bg-subtle"]}; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.num.combined {{ font-weight: 700; }}
</style></head>
<body>
  <div class="wrap">
    <table id="t">
      <thead><tr>
        <th data-key="0" tabindex="0">Column<span class="sort-ind"></span></th>
        <th data-key="1" tabindex="0" class="num">Completeness<span class="sort-ind"></span></th>
        <th data-key="2" tabindex="0" class="num">Consistency<span class="sort-ind"></span></th>
        <th data-key="3" tabindex="0" class="num">Combined<span class="sort-ind"></span></th>
      </tr></thead>
      <tbody>{body_rows}</tbody>
    </table>
  </div>
  <script>
    function sortBy(th) {{
      const table = document.getElementById('t');
      const tbody = table.querySelector('tbody');
      const colIndex = parseInt(th.dataset.key, 10);
      const nextDir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
      table.querySelectorAll('th').forEach(h => {{ h.dataset.dir = ''; h.querySelector('.sort-ind').textContent = ''; }});
      th.dataset.dir = nextDir;
      th.querySelector('.sort-ind').textContent = nextDir === 'asc' ? ' \\u25B2' : ' \\u25BC';
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {{
        const cellA = a.children[colIndex], cellB = b.children[colIndex];
        const rawA = cellA.dataset.sort, rawB = cellB.dataset.sort;
        const numA = parseFloat(rawA), numB = parseFloat(rawB);
        const bothNumeric = !isNaN(numA) && !isNaN(numB) && colIndex !== 0;
        const va = bothNumeric ? numA : rawA;
        const vb = bothNumeric ? numB : rawB;
        if (va < vb) return nextDir === 'asc' ? -1 : 1;
        if (va > vb) return nextDir === 'asc' ? 1 : -1;
        return 0;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }}
    document.querySelectorAll('th[data-key]').forEach(th => {{
      th.addEventListener('click', () => sortBy(th));
      th.addEventListener('keydown', (e) => {{ if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); sortBy(th); }} }});
    }});
  </script>
</body></html>"""


def _count_chip_html(count: int, label: str, tier_alias: str) -> str:
    return f"""
        <div class="mdq-count-chip">
          <span class="n" style="color:var(--{tier_alias});">{count}</span>
          <span class="lbl">{_html(label)}</span>
        </div>
    """


# ---------------------------------------------------------------------------
# Tier 1: cleaning summary, cleaned-CSV download, and the per-column
# dashboard. See core/remediation.py and core/chart_generation.py for the
# actual logic -- this section is only ever presentation, same split as
# every other part of this file.
# ---------------------------------------------------------------------------

# Plain-language label for each action_type core/remediation.py can
# produce -- one place to keep this readable-name mapping, same idea as
# SEVERITY_META above.
_ACTION_TYPE_LABELS = {
    "strip_numeric_formatting": "Stripped number formatting",
    "normalize_date_format": "Normalized date format",
    "trim_whitespace": "Trimmed whitespace",
    "remove_exact_duplicate_rows": "Removed exact duplicate row(s)",
    "coerce_invalid_numeric": "Cleared non-numeric values",
    "coerce_invalid_date": "Cleared non-date values",
    "normalize_missing_token": "Normalized placeholder to blank",
    "standardize_categorical_spelling": "Standardized spelling variant",
    "neutralize_domain_outlier": "Cleared out-of-range value",
    "clip_domain_outlier": "Clipped out-of-range value",
    "drop_rows_missing_value": "Dropped row(s) with missing value",
    "fill_missing_constant": "Filled missing value (constant)",
    "fill_missing_median": "Filled missing value (median)",
    "fill_missing_mode": "Filled missing value (mode)",
}


def _safe_filename_stub(table_name: str) -> str:
    """Turns a table name (a filename or sheet name, straight from the
    user's own upload) into something safe to put in a download
    filename -- strips anything that isn't a letter, digit, underscore
    or hyphen, so odd characters in an uploaded file's name can never
    break the generated download."""
    stub = re.sub(r"[^A-Za-z0-9_-]+", "_", table_name).strip("_")
    return stub or "table"


def _remediation_actions_table_html(actions) -> str:
    """Same hand-built HTML table pattern as _per_column_table_html --
    kept as one row per action, with a stacked note row underneath when
    an action left some genuinely ambiguous values untouched (see
    RemediationAction.notes' docstring for why that's shown, not hidden)."""
    rows = []
    for action in actions:
        label = _ACTION_TYPE_LABELS.get(action.action_type, action.action_type)
        rows.append(f"""
            <tr>
              <td>{_html(action.column_name)}</td>
              <td>{_html(label)}</td>
              <td class="num">{action.count_affected}</td>
              <td>{_html(action.before_example)}</td>
              <td>{_html(action.after_example)}</td>
            </tr>
        """)
        if action.notes:
            rows.append(f'<tr><td colspan="5"><div class="mdq-table-sub" style="margin:0;">ℹ️ {_html(action.notes)}</div></td></tr>')
    return f"""
        <div class="mdq-table-wrap">
          <table class="mdq-table">
            <thead>
              <tr><th>Column</th><th>Fix Applied</th><th class="num">Count</th><th>Before</th><th>After</th></tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>
    """


# ---------------------------------------------------------------------------
# Requirement 0 (top priority in this pass): themed Dashboard charts.
#
# st.bar_chart / st.line_chart render through Vega-Lite with Vega-Lite's
# OWN default light theme -- confirmed directly by screenshot, not
# assumed: every chart in the Dashboard section rendered as a plain white
# rectangle against a dark page, because that Vega-Lite default never
# sees this file's CSS tokens at all. Same root cause, same fix pattern,
# as the reason results aren't shown via st.dataframe elsewhere in this
# file: build the chart explicitly (here, as an alt.Chart) and hand it
# real colors instead of trusting a native widget's own default theme.
# ---------------------------------------------------------------------------

def _chart_axis(tokens: dict, label_angle: int = None) -> alt.Axis:
    """One consistent axis style, built from the CURRENT theme's already-
    resolved colors -- called fresh for every chart, every render, so it
    always matches whichever theme is active (light or dark) at that
    moment, exactly like every other themed element on the page."""
    kwargs = dict(
        labelColor=tokens["text-secondary"], titleColor=tokens["text-secondary"],
        domainColor=tokens["border"], tickColor=tokens["border"], gridColor=tokens["border-subtle"],
    )
    if label_angle is not None:
        kwargs["labelAngle"] = label_angle
    return alt.Axis(**kwargs)


def _altair_bar_chart(chart_data: pd.DataFrame, x_field: str, tokens: dict) -> alt.Chart:
    """Themed replacement for st.bar_chart -- used for both the numeric-
    distribution and categorical charts (chart_generation.py already
    shaped chart_data identically for both: an index plus one "count"
    column), so one function covers both chart_type values that aren't
    "date_over_time"."""
    data = chart_data.reset_index()
    return (
        alt.Chart(data)
        .mark_bar(color=tokens["accent"], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{x_field}:N", sort=None, title=None, axis=_chart_axis(tokens, label_angle=-40)),
            y=alt.Y("count:Q", title=None, axis=_chart_axis(tokens)),
            tooltip=[alt.Tooltip(f"{x_field}:N", title=x_field.replace("_", " ").title()), alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=220, background=tokens["bg-elevated"])
        .configure_view(strokeWidth=0)
    )


def _altair_line_chart(chart_data: pd.DataFrame, x_field: str, tokens: dict) -> alt.Chart:
    """Themed replacement for st.line_chart -- used only for the
    date-over-time chart type."""
    data = chart_data.reset_index()
    return (
        alt.Chart(data)
        .mark_line(color=tokens["accent"], point=alt.OverlayMarkDef(color=tokens["accent"], filled=True, size=45))
        .encode(
            x=alt.X(f"{x_field}:N", sort=None, title=None, axis=_chart_axis(tokens, label_angle=-40)),
            y=alt.Y("count:Q", title=None, axis=_chart_axis(tokens)),
            tooltip=[alt.Tooltip(f"{x_field}:N", title=x_field.replace("_", " ").title()), alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=220, background=tokens["bg-elevated"])
        .configure_view(strokeWidth=0)
    )


def _low_cardinality_note(dataframe: pd.DataFrame, column_name: str) -> str:
    """
    Section C: a column where almost every value is the same one thing
    (e.g. a repeated device ID) produces a chart that's just one giant
    bar -- technically correct, not informative. Below
    LOW_CARDINALITY_THRESHOLD distinct values, return a plain sentence
    to show INSTEAD of a chart; returns None when the column has enough
    variety that a chart is still the right call. Reads directly from
    the table's own dataframe (never chart_data, which for a categorical
    column is already bucketed into top-10 + "Other" by
    core/chart_generation.py) so this sees the column's REAL distinct
    count, not a post-bucketing approximation of it.
    """
    non_blank = dataframe[column_name].dropna().astype(str).str.strip()
    non_blank = non_blank[non_blank != ""]
    if non_blank.empty:
        return None
    distinct = non_blank.nunique()
    if distinct >= LOW_CARDINALITY_THRESHOLD:
        return None
    value_counts = non_blank.value_counts()
    top_value, top_count = value_counts.idxmax(), int(value_counts.max())
    value_word = "value" if distinct == 1 else "values"
    return f'This column has only {distinct} unique {value_word} — e.g. "{top_value}" appears in {top_count} of {len(non_blank)} rows.'


_CLEANING_INTENSITY_HELP = (
    "**Conservative** (default): exactly this project's original safe fixes, plus type-coercion "
    "and placeholder normalization -- both unconditionally safe. Casing is never changed, outliers "
    "are cleared not guessed at, and missing data always goes to manual review.\n\n"
    "**Standard**: also standardizes very-high-confidence spelling variants (e.g. \"Diesal\" -> "
    "\"Diesel\") and dominant-case casing.\n\n"
    "**Aggressive**: trusts every spelling-variant match validity.py's own detection floor allows, "
    "and clips out-of-range numeric values to the nearest realistic bound instead of clearing them. "
    "⚠️ Review the audit log carefully after using this setting."
)


def _audit_report_text(table_result, remediation, intensity_choice: str) -> str:
    """
    A plain-text, human-readable export of everything render_remediation_section
    shows on screen -- every RemediationAction (with before/after) and
    every ManualReviewItem (with its reason) -- so the audit trail this
    project promises can leave the browser tab (attach it to an email,
    keep it as a compliance record) without needing a screenshot. Built
    from the SAME `remediation` object the on-screen audit log renders
    from, never a re-derived summary, so the two can never disagree.
    """
    lines = [
        f"Audit report -- {table_result.table_name}",
        f"Cleaning intensity: {intensity_choice}",
        f"Original score (from raw data): {table_result.scorecard.overall_score:.1f}/100",
        "",
        f"Fixes applied ({len(remediation.actions)}):",
    ]
    if not remediation.actions:
        lines.append("  (none)")
    for action in remediation.actions:
        label = _ACTION_TYPE_LABELS.get(action.action_type, action.action_type)
        lines.append(f"  - [{action.column_name}] {label} -- {action.count_affected} value(s): {action.before_example!r} -> {action.after_example!r}")
        if action.notes:
            lines.append(f"      note: {action.notes}")

    lines += ["", f"Needs manual review ({len(remediation.manual_review)}):"]
    if not remediation.manual_review:
        lines.append("  (none)")
    for item in remediation.manual_review:
        lines.append(f"  - [{item.finding.column_name}] {item.finding.issue_type}: {item.reason}")

    return "\n".join(lines)


def _rescore_cleaned_dataframe(cleaned_dataframe: pd.DataFrame, table_name: str):
    """
    Re-runs the FULL diagnostic pipeline on the cleaned output, purely so
    the UI can show a "before vs after" score -- this is a completely
    separate, second scoring pass over the cleaned data, never a
    modification of the original score (see render_remediation_section's
    own banner, and core/remediation.py's module docstring: cleaning can
    never feed back into the diagnostic score it was computed from).
    Round-trips through an actual CSV encode/decode (not a direct
    DatasetProfile construction) so this re-score sees EXACTLY what a
    user re-uploading the downloaded file would see -- same code path,
    no shortcuts. Returns None if the cleaned data is empty or otherwise
    unscoreable (e.g. every row was dropped) rather than raising.
    """
    if cleaned_dataframe.empty:
        return None
    try:
        csv_bytes = cleaned_dataframe.to_csv(index=False).encode("utf-8")
        buffer = io.BytesIO(csv_bytes)
        buffer.name = f"{_safe_filename_stub(table_name)}_cleaned.csv"
        reloaded_profile = load_dataset(buffer, buffer.name)
        return run_pipeline_for_table(reloaded_profile, table_name, use_ai_phrasing=False)
    except Exception:
        # Re-scoring is a nice-to-have UI comparison, not a guarantee --
        # if the cleaned data genuinely can't be re-scored for some
        # reason, the rest of this section (audit log, download) must
        # still work; silently skipping just the comparison is the right
        # failure mode here, not surfacing a raw traceback.
        return None


def render_remediation_section(table_result) -> None:
    """
    Tier 1's whole UI: cleaning-intensity selector, cleaning summary
    (audit log), before/after score comparison, cleaned-data download,
    and the per-column dashboard -- rendered inside the SAME per-table
    expander as the diagnostic score (render_table_section), but
    visually separated by a divider and its own banner making explicit
    that the score above is always about the ORIGINAL data. Cleaning is
    a separate, optional output path -- see core/remediation.py's module
    docstring for why that separation is a hard project requirement, not
    a UI choice.
    """
    st.divider()
    render_html('<div class="mdq-section-title" style="margin-top:0;">🧹 Cleaned Data & Dashboard</div>')

    # Defensive guard: table_result.dataframe / .duplication_result are
    # only populated by the multi-table pipeline path (see
    # core/pipeline.py's TableResult) -- if this function is ever reached
    # from a code path that skipped that (a future refactor, a partially-
    # constructed test double), fail with a clear, contained message
    # instead of an AttributeError crashing the whole page.
    if table_result.dataframe is None or table_result.duplication_result is None:
        render_html(
            '<div class="mdq-banner mdq-banner-error">⚠️ Cleaning is unavailable for this table -- '
            "its original data wasn't carried through to this step. The diagnostic score above is "
            "still fully valid.</div>"
        )
        return

    render_html(
        '<div class="mdq-banner mdq-banner-warn" style="margin-top:0;">'
        "This is a separate, optional output. The score above is always computed from the "
        "<b>original</b> data -- nothing below it ever changes that score.</div>"
    )

    # -- Cleaning intensity --------------------------------------------------
    intensity_key = f"cleaning-intensity-{table_result.table_name}"
    intensity_choice = st.selectbox(
        "Cleaning intensity", options=list(INTENSITY_PRESETS.keys()), index=0,
        key=intensity_key, help=_CLEANING_INTENSITY_HELP,
    )
    if intensity_choice != "Conservative":
        render_html(
            f'<div class="mdq-banner mdq-banner-warn">⚠️ "{_html(intensity_choice)}" trusts more '
            "automatic fixes than the conservative default -- always check the audit log below.</div>"
        )

    remediation = remediate_table(
        table_result.dataframe, table_result.findings, table_result.duplication_result.exact_duplicate_row_indexes,
        config=INTENSITY_PRESETS[intensity_choice],
    )

    # -- Cleaning summary (audit log) ----------------------------------------
    render_html('<div class="mdq-table-sub" style="margin-top:.2rem;">Fixes actually applied</div>')
    if remediation.actions:
        render_html(_remediation_actions_table_html(remediation.actions))
    else:
        render_html('<div class="mdq-banner mdq-banner-warn">No safe automatic fix applied to this table.</div>')

    if remediation.manual_review:
        with st.expander(f"🙋 Needs manual review · {len(remediation.manual_review)} item(s)", expanded=False):
            for index, item in enumerate(remediation.manual_review):
                with st.container(border=True):
                    render_html(
                        f'<div class="mdq-finding-text">'
                        f'<b>{_html(item.finding.column_name)}</b> '
                        f'<span class="mdq-tag-table">{_html(item.finding.issue_type)}</span>'
                        f"<br/>{_html(item.reason)}</div>"
                    )

    # -- Before vs after score ------------------------------------------------
    rescored = _rescore_cleaned_dataframe(remediation.cleaned_dataframe, table_result.table_name)
    if rescored is not None:
        before_score = table_result.scorecard.overall_score
        after_score = rescored.scorecard.overall_score
        delta = round(after_score - before_score, 1)
        delta_text = f"+{delta}" if delta > 0 else str(delta)
        delta_tier = "minor" if delta > 0 else ("critical" if delta < 0 else "moderate")
        render_html(
            f"""
            <div class="mdq-table-sub" style="margin-top:1rem;">Score if you re-ran diagnosis on this cleaned data</div>
            <div class="mdq-count-row">
              <div class="mdq-count-chip"><span class="n">{before_score:.1f}</span><span class="lbl">Original score</span></div>
              <div class="mdq-count-chip"><span class="n">{after_score:.1f}</span><span class="lbl">Cleaned score</span></div>
              <div class="mdq-count-chip"><span class="n" style="color:var(--{delta_tier});">{delta_text}</span><span class="lbl">Change</span></div>
            </div>
            """
        )

    # -- Download -------------------------------------------------------------
    download_cols = st.columns(2)
    with download_cols[0]:
        csv_bytes = remediation.cleaned_dataframe.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download cleaned dataset (CSV)",
            data=csv_bytes,
            file_name=f"{_safe_filename_stub(table_result.table_name)}_cleaned.csv",
            mime="text/csv",
            key=f"download-cleaned-{table_result.table_name}",
        )
    with download_cols[1]:
        st.download_button(
            "📋 Download audit report (TXT)",
            data=_audit_report_text(table_result, remediation, intensity_choice),
            file_name=f"{_safe_filename_stub(table_result.table_name)}_audit_report.txt",
            mime="text/plain",
            key=f"download-audit-{table_result.table_name}",
        )

    # -- Dashboard ------------------------------------------------------------
    with st.expander("📊 Dashboard", expanded=False):
        tokens = _theme_tokens()
        charts = generate_charts_for_table(table_result.dataframe, table_result.column_types, table_result.findings)
        for chart in charts:
            annotation_html = (
                f' <span style="color:var(--moderate); font-size:.82rem; font-weight:600;">{_html(chart.annotation)}</span>'
                if chart.annotation else ""
            )
            render_html(f'<div class="mdq-finding-text" style="margin:.9rem 0 .3rem;"><b>{_html(chart.column_name)}</b>{annotation_html}</div>')

            if chart.chart_data is None:
                render_html('<div class="mdq-table-sub">Not enough data in this column to chart.</div>')
                continue

            low_cardinality_note = _low_cardinality_note(table_result.dataframe, chart.column_name)
            if low_cardinality_note:
                render_html(f'<div class="mdq-table-sub">{_html(low_cardinality_note)}</div>')
                continue

            x_field = chart.chart_data.index.name or "value"
            if chart.chart_type == "date_over_time":
                st.altair_chart(_altair_line_chart(chart.chart_data, x_field, tokens), use_container_width=True)
            else:
                st.altair_chart(_altair_bar_chart(chart.chart_data, x_field, tokens), use_container_width=True)


# ---------------------------------------------------------------------------
# AI-touched content: a consistent, clickable "see for yourself" pattern
# used for both AI-phrased findings and calibration badges (requirement 4).
# ---------------------------------------------------------------------------

def _render_ai_provenance_popover(finding, popover_key: str) -> None:
    """
    The click-to-reveal pattern for an AI-phrased sentence: the trigger
    is a small pill labeled "🤖 AI-enhanced"; clicking it opens a popover
    showing the exact rule-based fact the phrasing came from, so the tag
    means "see for yourself" rather than "trust me". Every AI-phrased
    finding in the whole app renders through this one function, so the
    treatment (wording, layout, icon) is identical everywhere it appears.
    """
    with st.popover("🤖 AI-enhanced", key=popover_key):
        st.caption("This sentence was rephrased by a local AI model. It was only ever given the facts below -- never your raw data.")
        st.markdown(f"**Guaranteed rule-based fact it was derived from:**\n\n> {_html(finding.rule_based_description)}")
        st.markdown(
            f"- Column: `{_html(finding.column_name)}`\n"
            f"- Affected: `{finding.percentage_affected}%`\n"
            f"- Severity: `{finding.severity}`"
        )


def _render_calibration_badge(label: str, value_text: str, was_calibrated: bool, method_note: str, popover_key: str) -> None:
    """
    The same click-to-reveal pattern as the AI-phrasing tag, applied to a
    calibrated threshold (requirement 4 asks for consistent treatment
    across both). The trigger always shows the value; the popover always
    explains whether it was calibrated for this table or is the fixed
    default, and why -- same layout, same icon, every place this appears.
    """
    status_icon = "🎯" if was_calibrated else "🔒"
    with st.popover(f"{status_icon} {label}: {value_text}", key=popover_key):
        if was_calibrated:
            st.markdown(f"**{_html(label)}: {_html(value_text)}** — 🎯 auto-calibrated")
            st.write(method_note)
        else:
            st.markdown(f"**{_html(label)}: {_html(value_text)}** — 🔒 fixed default")
            st.write(
                "This table didn't have enough data to calibrate safely, so a "
                "well-tested flat cutoff was used instead of guessing from too "
                "little evidence."
            )


_DUPLICATE_THRESHOLD_METHOD_NOTE = (
    "Computed by clustering every pairwise name-similarity score for this table "
    "(k-means, k=2: \"similar\" vs. \"not similar\") and taking the midpoint "
    "between the two groups -- see core/calibration.py."
)
_SEVERITY_CUTOFF_METHOD_NOTE = (
    "Computed by clustering every issue's \"% affected\" value for this table "
    "(k-means, k=3: low/medium/high impact) and taking the midpoint between "
    "neighbouring groups -- see core/calibration.py."
)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def render_header():
    """
    st.set_page_config MUST be the very first Streamlit call in the whole
    script (Streamlit's own requirement, to avoid a flash of default
    styling before it takes effect) -- so this function has to run before
    anything else touches st.*, including session_state-based theme setup.
    """
    st.set_page_config(
        page_title="MSME Data Quality Scorecard",
        page_icon=f"data:image/svg+xml,{quote(_FAVICON_SVG)}",
        layout="wide",
    )

    if "theme" not in st.session_state:
        st.session_state.theme = "light"  # light-first default, per this redesign's own brief

    render_html(_build_theme_css())
    render_html(
        f"""
        <div class="mdq-hero">
          <div class="mdq-hero-title-row">
            {_MARK_SVG}
            <h1>AI-Ready Data Quality Scorecard for MSMEs</h1>
          </div>
          <div class="mdq-tagline">Know your data before your AI does.</div>
          <p>Upload raw business data — spreadsheets, POS exports, Tally dumps, CRM lists — and get an
          instant, explainable score on completeness, consistency, duplication, and structure, plus a
          plain-language list of exactly what to fix before it's used in an AI or BI pipeline.</p>
        </div>
        """
    )
    # Workflow stepper: real st.columns(3) (not a single CSS flex row) so
    # each step is its own layout block, with a thin connector glyph
    # between columns standing in for a joining line.
    steps = ["Upload File", "Auto-Scoring", "Prioritized Fix List"]
    step_cols = st.columns([10, 1, 10, 1, 10])
    for i, label in enumerate(steps):
        with step_cols[i * 2]:
            render_html(
                f'<div class="mdq-step-card"><span class="mdq-step-num">{i + 1}</span>'
                f'<span class="mdq-step-label">{_html(label)}</span></div>'
            )
        if i < len(steps) - 1:
            with step_cols[i * 2 + 1]:
                render_html('<div class="mdq-step-connector">&#8250;</div>')


def _on_theme_toggle_change():
    """
    Runs BEFORE the script reruns top-to-bottom (Streamlit callback
    semantics), so by the time render_header() re-injects the CSS on
    this same rerun, st.session_state.theme already reflects the new
    choice -- no "one click behind" lag, no page reload, and nothing
    about the uploaded file or results is lost, since none of that lives
    in this key.
    """
    st.session_state.theme = "dark" if st.session_state.theme_toggle_widget else "light"


def render_theme_toggle():
    """
    A clearly visible, always-accessible dark-mode switch -- top of the
    sidebar, not buried in a menu (the default Streamlit menu is hidden
    entirely, see .streamlit/config.toml's toolbarMode). Switching is
    instant and session-only (the brief's own scope note: persisting the
    choice across a full app restart isn't required here).
    """
    with st.sidebar:
        st.toggle(
            "🌙 Dark mode",
            value=(st.session_state.theme == "dark"),
            key="theme_toggle_widget",
            on_change=_on_theme_toggle_change,
        )
        st.divider()


def render_sidebar_explainer():
    """
    A short, always-visible explanation of what's deterministic and what
    AI touches -- this matters for the project's own hard constraint
    that AI-touched output is never visually indistinguishable from the
    guaranteed core, and it's a handy on-screen cheat sheet for
    explaining the tool live.
    """
    with st.sidebar:
        render_html('<div class="mdq-sidebar-spacer"></div>')  # top buffer, clear of the collapse toggle
        render_html('<h2 style="font-size:1.1rem;margin:0 0 .6rem;">⚙️ How this tool works</h2>')
        kw = lambda s: f'<span class="mdq-kw">{s}</span>'  # noqa: E731 -- tiny local helper, not worth a def
        badges = [
            f"<b>Every score is 100% deterministic</b> — {kw('pandas')} + {kw('RapidFuzz')} "
            "rule-based checks, no AI in the pass/fail logic or the score formula.",
            f"<b>Two thresholds are auto-calibrated per file</b> using {kw('scikit-learn')} "
            "(k-means): the duplicate-name similarity cutoff, and the "
            "critical/moderate severity split. Click any 🎯 badge to see which.",
            "<b>Fix-list sentences are always guaranteed</b> by templates first. "
            "A small local LLM (Qwen2.5-0.5B, offline, GGUF) may rephrase the "
            "most important ones — click any 🤖 tag to see what it was based on.",
        ]
        for text in badges:
            render_html(f'<div class="mdq-info-badge">{text}</div>')
        st.divider()
        st.caption("Runs fully offline after one-time model setup. No cloud AI calls at runtime.")


def render_empty_state():
    """
    The first-run screen, designed on purpose rather than left blank --
    this is most people's actual first impression of the tool. Explains
    what it does and what the five dimensions mean before anyone uploads
    anything, plus a pointer to a sample file to try.
    """
    render_html('<div class="mdq-section-title">What this tool checks</div>')
    # Real st.columns (not a single CSS grid div) -- each dimension is
    # its own layout block, styled via .mdq-feature-card for the elevated,
    # shadow-hover "dashboard widget" look.
    feature_cols = st.columns(len(DIMENSION_META))
    for col, (_attr, label, icon_key, description) in zip(feature_cols, DIMENSION_META):
        with col:
            render_html(
                f"""
                <div class="mdq-feature-card">
                  <div class="icon">{_DIMENSION_ICONS[icon_key]}</div>
                  <h4>{_html(label)}</h4>
                  <p>{_html(description)}</p>
                </div>
                """
            )
    render_html(
        f"""
        <div class="mdq-empty-illustration">{_UPLOAD_ILLUSTRATION_SVG}</div>
        <div class="mdq-empty-hint">
          💡 New here? Try it on <code>sample_data/messy_msme_sample.csv</code> — a small,
          deliberately messy sample file bundled with this tool — or drop a multi-sheet
          Excel workbook straight in; every sheet gets scored as its own table.
        </div>
        """
    )


def _dimension_bar_row_html(label: str, value: Optional[float]) -> str:
    """One labelled horizontal bar for the report's per-table dimension
    breakdown -- value is None for Validity on data scored before that
    dimension existed (build_scorecard's own backward-compatible
    default), skipped entirely rather than drawn as a fake zero."""
    if value is None:
        return ""
    tier_alias = _TIER_TOKEN_ALIAS[_score_tier(value)]
    color = _LIGHT_TOKENS[tier_alias]
    width = max(0.0, min(100.0, value))
    return f"""
    <div class="dim-row">
      <span class="dim-label">{_html(label)}</span>
      <div class="dim-track"><div class="dim-fill" style="width:{width:.1f}%;background:{color};"></div></div>
      <span class="dim-value">{value:.1f}</span>
    </div>
    """


def _findings_table_html(findings) -> str:
    """A plain, print-friendly table of every Finding for one table,
    most-severe first -- the actual substance of the report (every
    problem this tool found, in one place), not a screenshot of the
    on-screen collapsible severity groups."""
    if not findings:
        return '<p class="no-issues">No issues found in this table.</p>'
    order = {"critical": 0, "moderate": 1, "minor": 2}
    ordered = sorted(findings, key=lambda f: order.get(f.severity, 3))
    rows = []
    for finding in ordered:
        meta = SEVERITY_META.get(finding.severity, {"label": finding.severity.title(), "emoji": ""})
        rows.append(f"""
        <tr>
          <td><span class="sev-pill sev-{_html(finding.severity)}">{meta['emoji']} {_html(meta['label'])}</span></td>
          <td>{_html(finding.column_name)}</td>
          <td>{_html(finding.display_description)}</td>
        </tr>
        """)
    return f"""
    <table class="findings-table">
      <thead><tr><th style="width:110px;">Severity</th><th style="width:160px;">Column</th><th>What was found</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _build_scorecard_report_html(summary) -> str:
    """
    The single, professional, self-contained scorecard report -- built
    ONCE from the exact same MultiTableSummary the on-screen results
    render from, and used for BOTH the Print button and the Download
    button (see render_overall_summary), so what a user prints and what
    they download can never disagree with each other or with what's on
    screen. This is a genuine report document (title, generated-on date,
    an overall-score hero, every table's own dimension breakdown, and
    its full findings table) -- not a screenshot or a plain-text dump of
    the dashboard chrome.

    Always rendered in the LIGHT palette regardless of the app's current
    dark-mode toggle -- a printed/downloaded report is meant to be read
    (and possibly re-printed) outside this session entirely, so it uses
    literal light-theme hex values (_LIGHT_TOKENS), never a CSS
    var(--...) reference tied to the live page's current theme, and
    never depends on st.session_state at all.
    """
    t = _LIGHT_TOKENS
    generated_at = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    table_names = ", ".join(_html(tr.table_name) for tr in summary.table_results) or "—"

    overall_section = ""
    per_table_sections = ""
    if summary.table_results:
        tier = _score_tier(summary.overall_score)
        tier_alias = _TIER_TOKEN_ALIAS[tier]
        emoji, tier_label = _TIER_LABELS[tier]

        counts = {sev: 0 for sev in SEVERITY_META}
        for finding in summary.combined_findings:
            counts[finding.severity] += 1
        chips = "".join(
            f'<div class="chip chip-{sev}"><span class="chip-num">{counts[sev]}</span>'
            f'<span class="chip-label">{meta["emoji"]} {_html(meta["label"])}</span></div>'
            for sev, meta in SEVERITY_META.items()
        )

        overall_section = f"""
        <section class="hero">
          <div class="hero-score" style="color:{t[tier_alias]};">{summary.overall_score:.1f}<span class="hero-of100">/100</span></div>
          <div class="hero-meta">
            <div class="tier-pill" style="background:{t[f'{tier_alias}-bg']};color:{t[tier_alias]};border:1px solid {t[f'{tier_alias}-border']};">{emoji} {tier_label}</div>
            <div class="hero-caption">Average across {len(summary.table_results)} successfully-processed table(s).</div>
          </div>
        </section>
        <section class="chip-row">{chips}</section>
        """

        for table_result in summary.table_results:
            scorecard = table_result.scorecard
            table_tier_alias = _TIER_TOKEN_ALIAS[_score_tier(scorecard.overall_score)]
            dim_rows = "".join(
                _dimension_bar_row_html(label, getattr(scorecard, attr_name, None))
                for attr_name, label, _icon_key, _description in DIMENSION_META
            )
            per_table_sections += f"""
            <section class="table-card">
              <div class="table-card-header">
                <div>
                  <div class="table-card-title">{_html(table_result.table_name)}</div>
                  <div class="table-card-sub">{table_result.row_count:,} rows &times; {table_result.column_count} columns</div>
                </div>
                <div class="table-card-score" style="color:{t[table_tier_alias]};">{scorecard.overall_score:.1f}<span class="hero-of100">/100</span></div>
              </div>
              <div class="dim-grid">{dim_rows}</div>
              {_findings_table_html(table_result.findings)}
            </section>
            """

    failures_section = ""
    if summary.table_failures:
        failure_rows = "".join(
            f"<li><b>{_html(f.table_name)}</b>: {_html(f.error_message)}</li>" for f in summary.table_failures
        )
        failures_section = f"""
        <section class="failures">
          <h2>Tables that could not be processed ({len(summary.table_failures)})</h2>
          <ul>{failure_rows}</ul>
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MSME Data Quality Scorecard</title>
<style>
  @page {{ size: A4; margin: 16mm 14mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: {t['text-primary']}; background: #FFFFFF; margin: 0; padding: 0 0 2rem;
    font-size: 13px; line-height: 1.5;
  }}
  .report-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 1.4rem; border-bottom: 3px solid {t['accent']}; margin-bottom: 1.4rem;
  }}
  .report-title {{ font-size: 1.4rem; font-weight: 800; margin: 0; }}
  .report-subtitle {{ font-size: 0.85rem; color: {t['text-secondary']}; margin-top: .15rem; }}
  .report-meta {{ text-align: right; font-size: 0.8rem; color: {t['text-secondary']}; }}
  .report-body {{ padding: 0 1.4rem; }}
  .report-files {{ font-size: 0.85rem; color: {t['text-secondary']}; margin: -0.6rem 0 1.2rem; }}

  .hero {{ display: flex; align-items: baseline; gap: 1.5rem; margin-bottom: .9rem; }}
  .hero-score {{ font-size: 3rem; font-weight: 800; line-height: 1; }}
  .hero-of100 {{ font-size: 1rem; font-weight: 500; opacity: .55; margin-left: .15rem; }}
  .hero-meta {{ display: flex; flex-direction: column; gap: .3rem; }}
  .tier-pill {{ display: inline-flex; width: fit-content; align-items: center; gap: .35rem; font-weight: 700; font-size: .85rem; padding: .25rem .7rem; border-radius: 999px; }}
  .hero-caption {{ font-size: 0.8rem; color: {t['text-secondary']}; }}

  .chip-row {{ display: flex; gap: .6rem; margin-bottom: 1.6rem; }}
  .chip {{ flex: 1; border-radius: 10px; padding: .55rem .8rem; display: flex; flex-direction: column; gap: .1rem; border: 1px solid {t['border']}; }}
  .chip-critical {{ background: {t['critical-bg']}; border-color: {t['critical-border']}; }}
  .chip-moderate {{ background: {t['moderate-bg']}; border-color: {t['moderate-border']}; }}
  .chip-minor {{ background: {t['minor-bg']}; border-color: {t['minor-border']}; }}
  .chip-num {{ font-size: 1.3rem; font-weight: 800; }}
  .chip-label {{ font-size: 0.75rem; color: {t['text-secondary']}; }}

  .table-card {{
    border: 1px solid {t['border']}; border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 1.2rem;
    page-break-inside: avoid;
  }}
  .table-card-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: .7rem; }}
  .table-card-title {{ font-size: 1.05rem; font-weight: 700; }}
  .table-card-sub {{ font-size: 0.78rem; color: {t['text-secondary']}; margin-top: .1rem; }}
  .table-card-score {{ font-size: 1.5rem; font-weight: 800; }}

  .dim-grid {{ display: flex; flex-direction: column; gap: .35rem; margin-bottom: .9rem; }}
  .dim-row {{ display: flex; align-items: center; gap: .6rem; }}
  .dim-label {{ width: 110px; font-size: 0.78rem; color: {t['text-secondary']}; flex-shrink: 0; }}
  .dim-track {{ flex: 1; height: 7px; border-radius: 4px; background: {t['bg-muted']}; overflow: hidden; }}
  .dim-fill {{ height: 100%; border-radius: 4px; }}
  .dim-value {{ width: 34px; text-align: right; font-size: 0.78rem; font-weight: 600; flex-shrink: 0; }}

  .findings-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  .findings-table th {{
    text-align: left; padding: .45rem .5rem; background: {t['bg-subtle']};
    border-bottom: 1px solid {t['border']}; font-size: .72rem; text-transform: uppercase; letter-spacing: .03em; color: {t['text-secondary']};
  }}
  .findings-table td {{ padding: .5rem; border-bottom: 1px solid {t['border-subtle']}; vertical-align: top; }}
  .sev-pill {{ display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .72rem; font-weight: 700; white-space: nowrap; }}
  .sev-critical {{ background: {t['critical-bg']}; color: {t['critical']}; }}
  .sev-moderate {{ background: {t['moderate-bg']}; color: {t['moderate']}; }}
  .sev-minor {{ background: {t['minor-bg']}; color: {t['minor']}; }}
  .no-issues {{ font-size: 0.85rem; color: {t['text-secondary']}; font-style: italic; }}

  .failures h2 {{ font-size: 1rem; }}
  .failures li {{ font-size: 0.85rem; margin-bottom: .3rem; }}

  .report-footer {{
    margin-top: 1.6rem; padding-top: .9rem; border-top: 1px solid {t['border']};
    font-size: 0.72rem; color: {t['text-tertiary']}; display: flex; justify-content: space-between;
  }}

  @media print {{
    body {{ font-size: 11.5px; }}
    .table-card {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <div class="report-header">
    <div>
      <div class="report-title">MSME Data Quality Scorecard</div>
      <div class="report-subtitle">Diagnostic report — completeness, consistency, duplication, structure &amp; validity</div>
    </div>
    <div class="report-meta">Generated {generated_at}</div>
  </div>
  <div class="report-body">
    <div class="report-files">Table(s): {table_names}</div>
    {overall_section}
    {per_table_sections}
    {failures_section}
    <div class="report-footer">
      <span>100% deterministic scoring — pandas + RapidFuzz rule-based checks, no AI in the score formula. Runs fully offline.</span>
      <span>MSME Data Quality Scorecard</span>
    </div>
  </div>
</body>
</html>"""


def _render_print_button(report_html: str) -> None:
    """
    A real, standalone HTML document rendered via components.html (in
    its own same-origin iframe), NOT a raw onclick attribute injected
    through st.markdown's unsafe_allow_html -- content injected that way
    can have its inline event-handler attributes silently stripped or
    blocked depending on the browser/host's content-security policy,
    which is exactly why an earlier version of this button didn't
    actually respond to clicks. A components.html iframe is parsed as an
    ordinary HTML page, so its own onclick works like it would on any
    normal website.

    Prints the EXACT same report_html the Download button saves (see
    _build_scorecard_report_html) -- opened in a new window and printed
    from there, not window.print() on the app itself, which would print
    the dashboard's on-screen chrome (sidebar, upload widget, buttons)
    instead of a clean report. window.open() is called synchronously
    inside the real click handler (not after an await/setTimeout), which
    is what keeps popup blockers from silently swallowing it.
    """
    report_json = json.dumps(report_html)
    components.html(
        f"""
        <div style="margin:0;padding:0;">
          <button id="mdq-print-btn"
            title="Opens the full scorecard report and the print dialog -- choose &quot;Save as PDF&quot; there to download a PDF"
            style="
              width:100%; height:2.5rem; padding:0 0.6rem; border-radius:8px; box-sizing:border-box;
              border:1px solid {_LIGHT_TOKENS['border']}; background:{_LIGHT_TOKENS['bg-elevated']}; color:{_LIGHT_TOKENS['text-primary']};
              font-size:0.95rem; font-weight:500; cursor:pointer; white-space:nowrap; font-family:sans-serif;
            "
            onmouseover="this.style.borderColor='{_LIGHT_TOKENS['accent']}';"
            onmouseout="this.style.borderColor='{_LIGHT_TOKENS['border']}';"
          >🖨️ Print</button>
        </div>
        <script>
          document.getElementById('mdq-print-btn').addEventListener('click', function () {{
            var reportHtml = {report_json};
            var reportWindow = window.open('', '_blank');
            if (!reportWindow) {{ return; }}
            reportWindow.document.open();
            reportWindow.document.write(reportHtml);
            reportWindow.document.close();
            reportWindow.focus();
            setTimeout(function () {{ reportWindow.print(); }}, 300);
          }});
        </script>
        """,
        height=42,
    )


def render_overall_summary(summary):
    if not summary.table_results and not summary.table_failures:
        render_html('<div class="mdq-section-title">📈 Overall Summary</div>')
        st.info("No tables processed yet.")
        return

    # Top-right corner toolbar: Print and Download both output the exact
    # SAME professional scorecard report (see _build_scorecard_report_html)
    # -- built once, here, and handed to both. Print opens it in a new
    # window and triggers the browser's print dialog there (save as PDF
    # from that dialog); Download saves the identical HTML report
    # directly. Neither is a screenshot of this dashboard -- both are the
    # same standalone report document. All three cells share the SAME
    # fixed height (2.5rem) and zero top margin so they line up on one
    # row regardless of each widget's own default spacing (a native
    # st.download_button and a plain text heading don't share the same
    # intrinsic box height otherwise).
    report_html = _build_scorecard_report_html(summary)
    title_col, print_col, download_col = st.columns([5, 1.3, 1.5])
    with title_col:
        render_html(
            '<div style="display:flex;align-items:center;height:2.5rem;">'
            '<div class="mdq-section-title" style="margin:0;">📈 Overall Summary</div>'
            "</div>"
        )
    with print_col:
        _render_print_button(report_html)
    with download_col:
        st.download_button(
            "⬇️ Download",
            data=report_html,
            file_name="msme_data_quality_scorecard.html",
            mime="text/html",
            key="download_scorecard_report",
            help="Download the full scorecard report (overall score, per-table breakdown, every issue found) as an HTML file -- open it in any browser and print or save as PDF from there.",
            use_container_width=True,
        )

    if summary.table_results:
        tier = _score_tier(summary.overall_score)
        tier_alias = _TIER_TOKEN_ALIAS[tier]
        emoji, tier_label = _TIER_LABELS[tier]
        # The overall score is this tool's "north star metric" -- the
        # single most visually prominent thing on the results screen,
        # rendered first, largest, before anything else can compete
        # with it (requirement 3: lead with the score).
        render_html(
            f"""
            <div class="mdq-score-hero">
              <div>
                <span class="mdq-score-number" style="color:var(--{tier_alias});">{summary.overall_score:.1f}<span class="mdq-score-of100">/100</span></span>
              </div>
              <div>
                <div class="mdq-score-tier-pill" style="background:var(--{tier_alias}-bg);color:var(--{tier_alias});border:1px solid var(--{tier_alias}-border);">{emoji} {tier_label}</div>
                <div class="mdq-score-caption">Average across {len(summary.table_results)} successfully-processed table(s).</div>
                <div class="mdq-score-legend">
                  <span>🟢 {SCORE_GOOD_THRESHOLD}+ good</span>
                  <span>🟡 {SCORE_NEEDS_ATTENTION_THRESHOLD}–{SCORE_GOOD_THRESHOLD - 1} needs attention</span>
                  <span>🔴 below {SCORE_NEEDS_ATTENTION_THRESHOLD} poor</span>
                </div>
              </div>
            </div>
            """
        )

        # Executive-summary strip: total issue counts by severity across
        # the whole upload, directly backing up the headline number with
        # "and here's specifically how much needs fixing" -- still part
        # of the same at-a-glance moment, before any drill-in.
        counts = {sev: 0 for sev in SEVERITY_META}
        for finding in summary.combined_findings:
            counts[finding.severity] += 1
        chips = "".join(
            _count_chip_html(counts[sev], f"{meta['label']} issues", sev)
            for sev, meta in SEVERITY_META.items()
        )
        render_html(f'<div class="mdq-count-row">{chips}</div>')

    if summary.table_failures:
        failure_lines = "".join(
            f"<li><b>{_html(f.table_name)}</b>: {_html(f.error_message)}</li>" for f in summary.table_failures
        )
        render_html(
            f"""
            <div class="mdq-banner mdq-banner-warn">
              ⚠️ <b>{len(summary.table_failures)} table(s)</b> could not be processed and were skipped
              (the rest of the upload was still scored normally):
              <ul style="margin:.4rem 0 0;">{failure_lines}</ul>
            </div>
            """
        )

    if summary.combined_findings:
        render_html('<div class="mdq-section-title">🛠️ Combined Fix List — all tables, most critical first</div>')
        render_finding_list(summary.combined_findings, show_table_name=True, key_prefix="combined")


def render_finding_list(findings, show_table_name: bool, key_prefix: str):
    """
    Shared renderer used for both the combined (all-tables) fix list and
    each individual table's own fix list -- severity groups collapsed by
    default except Critical (progressive disclosure, requirement 3), each
    Finding shown with its severity badge (color + icon + label together)
    and, when applicable, a clickable AI-provenance popover.
    """
    for severity, meta in SEVERITY_META.items():
        items_in_tier = [f for f in findings if f.severity == severity]
        if not items_in_tier:
            continue
        label = f"{meta['emoji']} {meta['label']} · {len(items_in_tier)} issue(s)"
        with st.expander(label, expanded=(severity == "critical")):
            for index, finding in enumerate(items_in_tier):
                render_finding_card(finding, show_table_name, key_prefix=f"{key_prefix}-{severity}-{index}")


def render_finding_card(finding, show_table_name: bool, key_prefix: str) -> None:
    """
    One finding, as a neutral bordered card (st.container -- the
    "everything else" neutral palette from requirement 2) containing a
    severity badge (the ONLY colored element in the card, carrying the
    only meaning that needs color) and, when this finding was AI-phrased,
    a clickable provenance popover.
    """
    with st.container(border=True):
        badge_and_tags = _severity_badge_html(finding.severity)
        if show_table_name:
            badge_and_tags += f' <span class="mdq-tag-table">{_html(finding.table_name)}</span>'
        render_html(badge_and_tags)
        render_html(f'<div class="mdq-finding-text">{_html(finding.display_description)}</div>')
        if finding.is_ai_phrased:
            _render_ai_provenance_popover(finding, popover_key=f"ai-{key_prefix}")


def render_ingestion_notices(table_result) -> None:
    """
    Surfaces anything core/ingestion.py had to silently-would-be-wrong-
    to-hide: a detected-and-skipped preamble/title row (core/ingestion.py
    Priority 1), or a non-UTF-8 encoding fallback (Priority 2). Both are
    "never silent" by the project's own rule -- this is the one place
    that rule is actually enforced in the UI, so a skipped row or a
    guessed encoding is always visible right where the rest of that
    table's detail is, not buried in a log no one reads.
    """
    if table_result.skipped_preamble_rows > 0:
        row_word = "row" if table_result.skipped_preamble_rows == 1 else "rows"
        render_html(
            f'<div class="mdq-banner mdq-banner-warn">📄 Detected and skipped '
            f'{table_result.skipped_preamble_rows} header {row_word} before the actual '
            f"data (e.g. a report title or company letterhead line) — the real column "
            f"headers were found further down and used instead.</div>"
        )
    if table_result.encoding_warning:
        render_html(f'<div class="mdq-banner mdq-banner-warn">🔤 {_html(table_result.encoding_warning)}</div>')


def render_table_section(table_result):
    """One table's own detail: score breakdown, calibrated thresholds
    used, per-column table, and its own fix list."""
    header = f"{score_indicator(table_result.scorecard.overall_score)} — {table_result.table_name} ({table_result.scorecard.overall_score:.1f}/100)"
    with st.expander(header, expanded=False):
        render_html(f'<div class="mdq-table-sub">{table_result.row_count} rows × {table_result.column_count} columns</div>')
        render_ingestion_notices(table_result)

        render_html(_dimension_grid_html(table_result.scorecard))

        # Calibration badges: same click-to-reveal treatment as the AI
        # tag (requirement 4), laid out as a row of three.
        dup, crit, mod = table_result.duplicate_threshold, table_result.critical_cutoff, table_result.moderate_cutoff
        badge_columns = st.columns(3)
        with badge_columns[0]:
            _render_calibration_badge(
                "Duplicate threshold", f"{dup.value:.1f}%", dup.was_calibrated,
                _DUPLICATE_THRESHOLD_METHOD_NOTE, popover_key=f"cal-dup-{table_result.table_name}",
            )
        with badge_columns[1]:
            _render_calibration_badge(
                "Critical cutoff", f">{crit.value:.1f}%", crit.was_calibrated,
                _SEVERITY_CUTOFF_METHOD_NOTE, popover_key=f"cal-crit-{table_result.table_name}",
            )
        with badge_columns[2]:
            _render_calibration_badge(
                "Moderate cutoff", f"≥{mod.value:.1f}%", mod.was_calibrated,
                _SEVERITY_CUTOFF_METHOD_NOTE, popover_key=f"cal-mod-{table_result.table_name}",
            )

        with st.expander("📋 Per-Column Score Detail — click any header to sort", expanded=False):
            column_count = len(table_result.scorecard.per_column_scores)
            table_height = min(520, 46 + 38 * max(column_count, 1))
            components.html(
                _sortable_column_table_html(table_result.scorecard, _theme_tokens()),
                height=table_height, scrolling=True,
            )

        render_html('<div class="mdq-section-title" style="margin-top:1.1rem;">🛠️ Fix List for this table</div>')
        if table_result.findings:
            render_finding_list(table_result.findings, show_table_name=False, key_prefix=table_result.table_name)
        else:
            render_html('<div class="mdq-banner mdq-banner-good">✅ No issues found — this table looks clean!</div>')

        render_remediation_section(table_result)


def _make_stage_reporter(status):
    """
    Requirement 2: turns core/pipeline.py's on_stage callback into a live
    st.status() label update, so the ~20-second first-run wait shows real,
    sequential stages as the pipeline actually runs them -- not a static
    "please wait" message. Because _STAGE_LABELS' keys are exactly the
    stage names pipeline.py reports (see that module's docstring), this
    can never drift into showing a stage that isn't really happening;
    an unrecognized stage code falls back to showing the raw code rather
    than a wrong label.
    """
    def on_stage(stage: str, table_name: str) -> None:
        status.update(label=_STAGE_LABELS.get(stage, stage))
    return on_stage


def main():
    render_header()
    render_theme_toggle()
    render_sidebar_explainer()

    render_html(
        '<div class="mdq-section-title" style="margin-top:.4rem;">Upload your data</div>'
        '<div class="mdq-table-sub">Accepts .csv, .xlsx, .xls — one or more files at once, and/or a '
        "multi-sheet Excel workbook (every sheet is scored as its own table automatically).</div>"
    )
    uploaded_files = st.file_uploader(
        "Upload one or more business data files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="CSV or Excel. Select multiple files at once, or drop in a multi-sheet workbook -- every sheet becomes its own table.",
        label_visibility="collapsed",
    )
    if not uploaded_files:
        render_empty_state()
        return

    # Section C: a skeleton matching the real score-hero + dimension-grid
    # shape, shown the instant a file is dropped -- before st.status even
    # starts -- so the moment right after clicking upload has structure
    # to look at instead of a blank pause. Held in an st.empty() slot so
    # it can be cleared cleanly once real results are ready to render.
    skeleton_slot = st.empty()
    with skeleton_slot.container():
        render_html(
            '<div class="mdq-skeleton mdq-skel-hero"></div>'
            '<div class="mdq-skel-dimgrid">'
            '<div class="mdq-skeleton mdq-skel-dim"></div><div class="mdq-skeleton mdq-skel-dim"></div>'
            '<div class="mdq-skeleton mdq-skel-dim"></div><div class="mdq-skeleton mdq-skel-dim"></div>'
            '<div class="mdq-skeleton mdq-skel-dim"></div>'
            "</div>"
        )

    with st.status(_STAGE_LABELS["loading"], expanded=True) as status:
        summary = run_multi_table_pipeline(uploaded_files, on_stage=_make_stage_reporter(status))
        status.update(label="Done — scorecard ready.", state="complete", expanded=False)

    skeleton_slot.empty()

    if not summary.table_results:
        # Every single table in the whole upload failed -- this is the
        # multi-table equivalent of the old single-file "couldn't
        # process this file" error. Still no raw traceback, ever.
        failure_lines = "".join(
            f"<li><b>{_html(f.table_name)}</b>: {_html(f.error_message)}</li>" for f in summary.table_failures
        )
        render_html(
            f"""
            <div class="mdq-banner mdq-banner-error">
              ⚠️ <b>None of the uploaded file(s)/sheet(s) could be processed.</b>
              <ul style="margin:.4rem 0 0;">{failure_lines}</ul>
            </div>
            """
        )
        st.stop()

    render_overall_summary(summary)

    render_html('<div class="mdq-section-title">📂 Per-Table Detail</div>')
    for table_result in summary.table_results:
        render_table_section(table_result)


if __name__ == "__main__":
    main()
