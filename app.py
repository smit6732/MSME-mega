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

import html

import streamlit as st

from core.pipeline import run_multi_table_pipeline


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

DIMENSION_META = [
    ("completeness_score", "Completeness", "📋", "How much of the data is actually filled in."),
    ("consistency_score", "Consistency", "🔤", "Whether values are formatted the same way throughout."),
    ("duplication_score", "Duplication", "🧬", "Exact and near-duplicate records."),
    ("structure_score", "Structure", "🏗️", "Whether the file itself is sound -- headers, columns, IDs."),
]


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

# Each value below was chosen and contrast-checked against ITS OWN
# background (white for light, the dark bg for dark) using the WCAG
# relative-luminance formula, targeting >=4.5:1 for text -- e.g. light
# mode's moderate/amber is a darker shade (#B45309) than you'd reach for
# by eye, specifically because the lighter, more "amber-looking" shade
# fails AA contrast as small text on white (~3.4:1). Dark mode's shades
# are the standard Tailwind "400" family, which is deliberately designed
# for this exact light-text-on-dark-background use case.
_LIGHT_TOKENS = {
    "bg": "#FFFFFF",
    "bg-elevated": "#FFFFFF",
    "bg-subtle": "#F9FAFB",
    "bg-muted": "#F3F4F6",
    "border": "#E5E7EB",
    "border-subtle": "#F0F1F3",
    "text-primary": "#111827",
    "text-secondary": "#4B5563",
    "text-tertiary": "#6B7280",
    "accent": "#2563EB",
    "accent-bg": "#EFF6FF",
    "accent-border": "#DBEAFE",
    "accent-contrast": "#FFFFFF",
    "critical": "#DC2626", "critical-bg": "#FEF2F2", "critical-border": "#FECACA",
    "moderate": "#B45309", "moderate-bg": "#FFFBEB", "moderate-border": "#FDE68A",
    "minor": "#15803D", "minor-bg": "#F0FDF4", "minor-border": "#BBF7D0",
    "shadow-sm": "0 1px 2px rgba(16,24,40,0.05)",
    "shadow-md": "0 4px 12px rgba(16,24,40,0.06)",
}

_DARK_TOKENS = {
    "bg": "#0F1420",
    "bg-elevated": "#161C2C",
    "bg-subtle": "#1B2233",
    "bg-muted": "#212940",
    "border": "#2A3348",
    "border-subtle": "#232B3D",
    "text-primary": "#F3F4F6",
    "text-secondary": "#B4BAC7",
    "text-tertiary": "#838EA3",
    "accent": "#60A5FA",
    "accent-bg": "#17273F",
    "accent-border": "#264160",
    "accent-contrast": "#0B1220",
    "critical": "#F87171", "critical-bg": "#3A1518", "critical-border": "#5B2226",
    "moderate": "#FBBF24", "moderate-bg": "#3A2C0C", "moderate-border": "#5C4315",
    "minor": "#4ADE80", "minor-bg": "#0F2E1C", "minor-border": "#1D4D31",
    "shadow-sm": "0 1px 2px rgba(0,0,0,0.35)",
    "shadow-md": "0 4px 16px rgba(0,0,0,0.4)",
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
    <style>
    :root {{ {root_vars} }}

    /* ---- Native Streamlit chrome, re-themed to match ------------------- */
    [data-testid="stAppViewContainer"], .stApp {{ background: var(--bg); }}
    [data-testid="stHeader"] {{ background: var(--bg); }}
    [data-testid="stSidebar"] {{ background: var(--bg-subtle); border-right: 1px solid var(--border); }}
    [data-testid="stSidebarContent"] {{ color: var(--text-primary); }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }}
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
        background: var(--bg-subtle); border: 1.5px dashed var(--border); border-radius: 14px;
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

    /* ---- Score hero card --------------------------------------------------- */
    .mdq-score-hero {{ display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;
                       border-radius: 18px; padding: 1.75rem 2rem; border: 1px solid var(--border);
                       background: var(--bg-elevated); box-shadow: var(--shadow-md); }}
    .mdq-score-number {{ font-size: 3.4rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }}
    .mdq-score-of100 {{ font-size: 1.15rem; font-weight: 500; opacity: .55; margin-left: .2rem; }}
    .mdq-score-tier-pill {{ display: inline-flex; align-items: center; gap: .35rem; font-weight: 700; font-size: .95rem;
                            padding: .3rem .8rem; border-radius: 999px; margin-bottom: .5rem; }}
    .mdq-score-caption {{ font-size: .87rem; color: var(--text-secondary); max-width: 420px; }}
    .mdq-score-legend {{ display:flex; gap: .9rem; font-size: .78rem; color: var(--text-tertiary); margin-top: .6rem; flex-wrap: wrap; }}

    /* ---- Issue-count chips (executive summary strip) ---------------------- */
    .mdq-count-row {{ display: flex; gap: .7rem; flex-wrap: wrap; margin-top: 1.1rem; }}
    .mdq-count-chip {{ display: flex; align-items: center; gap: .55rem; border-radius: 12px; padding: .7rem 1.05rem;
                       border: 1px solid var(--border); background: var(--bg-elevated); min-width: 140px; }}
    .mdq-count-chip .n {{ font-size: 1.5rem; font-weight: 800; line-height: 1; }}
    .mdq-count-chip .lbl {{ font-size: .78rem; color: var(--text-secondary); font-weight: 600; }}

    /* ---- Dimension score grid ---------------------------------------------- */
    .mdq-dim-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin: .6rem 0 1rem; }}
    @media (max-width: 700px) {{ .mdq-dim-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .mdq-dim-card {{ background: var(--bg-muted); border: 1px solid var(--border-subtle); border-radius: 12px; padding: .9rem 1.05rem; }}
    .mdq-dim-label {{ font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--text-tertiary); }}
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

    /* ---- Status banners ------------------------------------------------------ */
    .mdq-banner {{ border-radius: 12px; padding: .9rem 1.15rem; border: 1px solid; font-size: .9rem; margin: .6rem 0 1rem; }}
    .mdq-banner-warn {{ background: var(--moderate-bg); border-color: var(--moderate-border); color: var(--moderate); }}
    .mdq-banner-good {{ background: var(--minor-bg); border-color: var(--minor-border); color: var(--minor); }}
    .mdq-banner-error {{ background: var(--critical-bg); border-color: var(--critical-border); color: var(--critical); }}

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
    .mdq-empty-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin: 1.4rem 0; }}
    @media (max-width: 700px) {{ .mdq-empty-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .mdq-empty-card {{ background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.2rem; }}
    .mdq-empty-card .icon {{ font-size: 1.5rem; }}
    .mdq-empty-card h4 {{ font-size: .95rem; font-weight: 700; margin: .5rem 0 .3rem; color: var(--text-primary); }}
    .mdq-empty-card p {{ font-size: .82rem; color: var(--text-secondary); line-height: 1.5; margin: 0; }}
    .mdq-empty-hint {{ background: var(--accent-bg); border: 1px solid var(--accent-border); border-radius: 12px;
                       padding: .8rem 1.1rem; font-size: .85rem; color: var(--text-secondary); margin-top: 1rem; }}
    .mdq-empty-hint code {{ background: var(--bg-muted); padding: .1rem .4rem; border-radius: 4px; color: var(--text-primary); }}
    </style>
    """


# ---------------------------------------------------------------------------
# Small HTML component builders -- each returns a ready-to-render string.
# ---------------------------------------------------------------------------

def _dimension_grid_html(scorecard) -> str:
    cards = []
    for attr_name, label, icon, _description in DIMENSION_META:
        score = getattr(scorecard, attr_name)
        tier_alias = _TIER_TOKEN_ALIAS[_score_tier(score)]
        width = max(0.0, min(100.0, score))
        cards.append(f"""
            <div class="mdq-dim-card">
              <div class="mdq-dim-label">{icon} {_html(label)}</div>
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


def _per_column_table_html(scorecard) -> str:
    rows = []
    for column_score in scorecard.per_column_scores.values():
        rows.append(f"""
            <tr>
              <td>{_html(column_score.column_name)}</td>
              <td class="num">{column_score.completeness_score:.1f}</td>
              <td class="num">{column_score.consistency_score:.1f}</td>
              <td class="num combined">{column_score.combined_score:.1f}</td>
            </tr>
        """)
    return f"""
        <div class="mdq-table-wrap">
          <table class="mdq-table">
            <thead>
              <tr><th>Column</th><th class="num">Completeness</th><th class="num">Consistency</th><th class="num">Combined</th></tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>
    """


def _count_chip_html(count: int, label: str, tier_alias: str) -> str:
    return f"""
        <div class="mdq-count-chip">
          <span class="n" style="color:var(--{tier_alias});">{count}</span>
          <span class="lbl">{_html(label)}</span>
        </div>
    """


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
    st.set_page_config(page_title="MSME Data Quality Scorecard", page_icon="📊", layout="wide")

    if "theme" not in st.session_state:
        st.session_state.theme = "light"  # light-first default, per this redesign's own brief

    render_html(_build_theme_css())
    render_html(
        """
        <div class="mdq-hero">
          <h1>📊 AI-Ready Data Quality Scorecard for MSMEs</h1>
          <p>Upload raw business data — spreadsheets, POS exports, Tally dumps, CRM lists — and get an
          instant, explainable score on completeness, consistency, duplication, and structure, plus a
          plain-language list of exactly what to fix before it's used in an AI or BI pipeline.</p>
          <div class="mdq-steps">
            <div class="mdq-step"><span class="mdq-step-num">1</span>Upload CSV / Excel (multi-file &amp; multi-sheet supported)</div>
            <div class="mdq-step"><span class="mdq-step-num">2</span>Automatic scoring &amp; calibration per table</div>
            <div class="mdq-step"><span class="mdq-step-num">3</span>Prioritized, plain-language fix list</div>
          </div>
        </div>
        """
    )


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
        st.subheader("⚙️ How this tool works")
        st.markdown(
            "- **Every score is 100% deterministic** — pandas + RapidFuzz "
            "rule-based checks, no AI in the pass/fail logic or the score formula.\n"
            "- **Two thresholds are auto-calibrated per file** using scikit-learn "
            "(k-means): the duplicate-name similarity cutoff, and the "
            "critical/moderate severity split. Click any 🎯 badge to see which.\n"
            "- **Fix-list sentences are always guaranteed** by templates first. "
            "A small local LLM (Qwen2.5-0.5B, offline, GGUF) may rephrase the "
            "most important ones — click any 🤖 tag to see what it was based on."
        )
        st.divider()
        st.caption("Runs fully offline after one-time model setup. No cloud AI calls at runtime.")


def render_empty_state():
    """
    The first-run screen, designed on purpose rather than left blank --
    this is most people's actual first impression of the tool. Explains
    what it does and what the four dimensions mean before anyone uploads
    anything, plus a pointer to a sample file to try.
    """
    render_html('<div class="mdq-section-title">What this tool checks</div>')
    cards = "".join(
        f"""
        <div class="mdq-empty-card">
          <div class="icon">{icon}</div>
          <h4>{_html(label)}</h4>
          <p>{_html(description)}</p>
        </div>
        """
        for _attr, label, icon, description in DIMENSION_META
    )
    render_html(f'<div class="mdq-empty-grid">{cards}</div>')
    render_html(
        """
        <div class="mdq-empty-hint">
          💡 New here? Try it on <code>sample_data/messy_msme_sample.csv</code> — a small,
          deliberately messy sample file bundled with this tool — or drop a multi-sheet
          Excel workbook straight in; every sheet gets scored as its own table.
        </div>
        """
    )


def render_overall_summary(summary):
    render_html('<div class="mdq-section-title">📈 Overall Summary</div>')

    if not summary.table_results and not summary.table_failures:
        st.info("No tables processed yet.")
        return

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

        with st.expander("📋 Per-Column Score Detail", expanded=False):
            render_html(_per_column_table_html(table_result.scorecard))

        render_html('<div class="mdq-section-title" style="margin-top:1.1rem;">🛠️ Fix List for this table</div>')
        if table_result.findings:
            render_finding_list(table_result.findings, show_table_name=False, key_prefix=table_result.table_name)
        else:
            render_html('<div class="mdq-banner mdq-banner-good">✅ No issues found — this table looks clean!</div>')


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

    with st.status("Reading file(s) and running data quality checks…", expanded=True) as status:
        st.write("First run on this machine may take about 20 seconds while the local AI model loads — every run after that is fast.")
        summary = run_multi_table_pipeline(uploaded_files)
        status.update(label="Done — scorecard ready.", state="complete", expanded=False)

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
