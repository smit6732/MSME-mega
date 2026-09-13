# Upgrade notes: Validity dimension + real cleaning capability

This upgrade extends the existing Data Quality Scorecard / Tier 1 cleaning
pipeline with a 5th scorecard dimension (**Validity**) and a much more
capable, still-conservative-by-default `core/remediation.py`. It is an
**extension** of the existing codebase, not a rewrite -- every existing
module, function signature, and test keeps working. Nothing here changes
how the four original dimensions are computed, and cleaning is still a
strictly separate, optional output path that can never feed back into the
diagnostic score (see `core/remediation.py`'s module docstring, unchanged).

## What's new

### 1. Detection: `core/validity.py` (new module, Dimension 5)

Five new `Finding.issue_type`s, all tagged `check_type="validity"`:

| issue_type | What it catches |
|---|---|
| `invalid_type_in_numeric` | A string that isn't a number at all ("high", "ask", "??") sitting in a numeric column -- not just badly *formatted*, genuinely not numeric. |
| `invalid_type_in_date` | Same idea for date columns. |
| `categorical_inconsistency` | Spelling variants of the same category (`Diesel`/`Diesal`/`Dizel`, `BMW`/`B.M.W`) -- via punctuation-stripped exact grouping *plus* Jaro-Winkler fuzzy matching (threshold 85, see below) against the column's own established spellings. |
| `domain_outlier` | A value that parses fine but is unrealistic: negative mileage/price/age/..., a year outside `[1900, current_year+1]`, `Doors` outside `{2,3,4,5}`, etc. Named rules first (real-world semantics), a conservative 3×IQR statistical fallback only when no named rule applies. |
| `missing_value_representation` | A cell holding a placeholder ("N/A", "Unknown", "null", "?", "TBD", ...) instead of being truly blank. |

`core/completeness.py`'s missing-value definition (`MISSING_VALUE_TOKENS`)
now also treats these placeholder tokens as missing, so completeness
scoring reflects them too -- `core/validity.py` imports that same constant
so a cell is never double-flagged under two different issue types.

**Deviation from the original brief:** the spec suggested extending
`core/consistency.py` with the categorical-typo rapidfuzz pass. It lives in
`core/validity.py` instead -- consistency.py's own scope (`Ahmedabad` vs
`ahmedabad`: same normalized form, different exact spelling) is a genuinely
different question from categorical_inconsistency's (`Diesel` vs `Diesal`:
different normalized forms, same real-world category). Keeping them in
separate, purpose-built modules is more explainable than tangling both into
one function.

**Similarity metric:** categorical matching uses Jaro-Winkler
(`rapidfuzz.distance.JaroWinkler`), not `fuzz.WRatio` — empirically, WRatio
scores short category-word typos (`Diesal` vs `Diesel`, both 6 characters)
around 83, well under any safe "never guess" floor, because it's tuned for
longer free-text comparison. Jaro-Winkler's extra weight on a shared prefix
fits this exact failure mode much better. The floor is 85 on Jaro-Winkler's
scale (not literally the brief's "≥88", which was WRatio-scale). A
single-character brand-code swap on a short acronym (`BMV` vs `BMW`) still
won't clear this floor by design -- that's indistinguishable from a
genuinely different code without domain knowledge, i.e. exactly what a
synonym dictionary is for, not a generic distance metric. No such
dictionary is wired in yet; `CleaningConfig` has room to grow one.

### 2. Remediation: `core/remediation.py` dispatch table extended

| issue_type | Behavior |
|---|---|
| `missing_value_representation` | Always auto-fixed: placeholder token → true blank. Unconditional (normalizing a representation is never a guess). |
| `invalid_type_in_numeric` / `invalid_type_in_date` | Auto-fixed: cleared to blank. **Never imputed.** |
| `categorical_inconsistency` | Auto-fixed **only** for the specific mappings `core/validity.py` already computed, filtered by `config.categorical_fuzzy_threshold` -- remediation never re-derives a mapping, only applies an already-computed one. |
| `domain_outlier` | Set to blank (default) or clipped to the nearest already-computed bound (`config.outlier_action`), reusing `Finding.details`' bounds directly. |
| `missing_data` | **Unchanged default**: manual review. Only acts when `config.missing_strategy` is explicitly set to `drop_row` / `constant` / `median` / `mode`. |
| everything else | Unchanged from before this upgrade. |

### 3. `core/cleaning_config.py` (new module)

A `CleaningConfig` dataclass plus three presets app.py's new "Cleaning
intensity" selector uses: `CONSERVATIVE` (default -- reproduces this
project's exact original behavior, plus the two unconditionally-safe new
fixes above), `STANDARD` (also trusts dominant-case normalization and
very-confident categorical matches), `AGGRESSIVE` (clips outliers instead
of blanking them, trusts every match validity.py's own floor allows).
Detection (`core/validity.py`) never reads this config -- only remediation
does; see that module's own docstring for why.

### 4. Scoring: `core/scoring.py`

`build_scorecard()` gained an **optional** `validity_result` parameter.
Passed → uses the new 5-weight formula (Completeness 27% / Consistency 22%
/ Duplication 20% / Structure 16% / Validity 15%). Omitted (every
pre-existing caller) → the *exact* original 4-weight formula, byte-for-byte
unchanged. `core/pipeline.py` always passes a real one now.

**Known, accepted consequence:** a file with real completeness/
consistency/duplication problems but clean validity can now score a few
points higher than it would have under the old 4-dimension formula --
adding any new, currently-passing dimension necessarily does this for
files that happen to have no problem on that specific axis. One existing
test (`test_single_csv_still_loads_and_scores`) had its threshold widened
from `<80` to `<90` for exactly this reason, with a comment explaining why.

### 5. `app.py` (Tier 1 UI, MSME-mega only)

- 5th dimension card (Validity) in the per-table breakdown and the
  empty-state feature grid; new SVG icon.
- "Cleaning intensity" selector (Conservative / Standard / Aggressive)
  per table, feeding a `CleaningConfig` into `remediate_table()`.
- Before/after score comparison: the cleaned output is round-tripped
  through an actual CSV encode/decode and re-scored via the same
  `run_pipeline_for_table()` path a re-upload would use -- a second,
  fully separate scoring pass, never a mutation of the original score.
- Audit log and manual-review list already existed and needed no
  structural change; new `action_type`s got plain-language labels.

## Backward compatibility

- `run_pipeline_for_table` / `run_multi_table_pipeline`: same public
  signature plus optional params already added in a prior pass
  (`on_stage`); now also silently computes and returns validity data.
- `generate_fix_list(..., validity_result=None)`: optional, defaults to
  producing zero validity Findings.
- `build_scorecard(..., validity_result=None)`: optional, defaults to the
  original 4-dimension formula.
- `remediate_table(..., config=CONSERVATIVE)`: optional, defaults to
  reproducing this module's exact pre-upgrade behavior.
- `Finding` gained one new optional field (`details`, default `None`).
- `ColumnScore` / `ScorecardResult` gained one new optional field
  (`validity_score`, default `None`).

All 45 pre-existing tests pass unmodified except the one noted above;
`tests/test_pipeline.py` gained 16 new tests covering every new issue
type, every new remediation path, `CleaningConfig` intensity differences,
and one full end-to-end run against a messy car-price-style dataset.

## A note on `missing_value_representation` and real CSV uploads

`core/ingestion.py` reads every CSV via `pd.read_csv(..., keep_default_na=True)`
(unchanged by this upgrade). Pandas' own default NA-token list already
includes several of the exact tokens this feature also treats as
"disguised missing" -- `N/A`, `NA`, `null`, `NULL`, `None`, `nan`, `NaN` --
so on a real uploaded CSV, those specific tokens are silently converted to
a genuine null *by pandas itself*, before `core/validity.py` ever sees a
literal string to flag. They're still correctly counted as `missing_data`
(completeness never under-counts them), but the more specific
`missing_value_representation` diagnostic -- and its "normalize this
placeholder to blank" auto-fix -- has nothing left to act on for THOSE
particular tokens on CSV-sourced data. It still works exactly as designed
for the tokens pandas does *not* auto-recognize (`?`, `Unknown`, `TBD`,
`n.a.`, ...), and for any of these tokens on data constructed directly as
a DataFrame (as several of this upgrade's own unit tests do, deliberately,
to exercise the check in isolation). This is a property of composing with
pandas' existing, desirable CSV-parsing behavior, not a defect -- but it's
worth knowing if a `missing_value_representation` Finding for "N/A"
doesn't show up on a real upload where you'd expect it.

## Known limitations / good next steps

- No synonym dictionary yet for short-code brand typos (`BMV`→`BMW`) --
  `CleaningConfig` is where one would plug in.
- A punctuation-only variant that shares its dominant *key* but isn't
  flagged as a *minority* key (e.g. `B.M.W` among mostly `BMW`) is now
  corrected via a full-confidence (score 100) same-key mapping -- but this
  logic lives only in `core/validity.py`'s categorical check, not in
  `core/consistency.py`'s own (punctuation-sensitive) dominant-spelling
  check, which still treats them as separate groups for *scoring* purposes.
- The Validity weight (15%) is a reasoned default, not empirically tuned
  against a labeled dataset -- easy to revisit in `core/scoring.py`.
