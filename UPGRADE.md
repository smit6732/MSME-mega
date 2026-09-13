# Upgrade notes: Validity dimension + real cleaning capability

## Pass 4: final polish -- residual typos, rating/quantity/amount bounds, dates

- **`core/validity.py`**: the fuzzy-typo cardinality guard was gated on
  a distinct/row-count RATIO, which wrongly declined genuinely
  categorical columns (City, with many real cities in a small file) --
  e.g. `Chennnai` next to a handful of real distinct cities was never
  reached. Replaced with the real distinguishing signal: fuzzy matching
  is eligible whenever at least one key is genuinely REPEATED (an
  established "house style" exists to match against), regardless of how
  many total distinct categories the column has. Free text (every value
  unique, nothing repeats -- e.g. Business_Name) is still correctly
  declined, re-verified by its existing test.
- **`core/validity.py`**: added a `customer_rating` named rule (1-5),
  checked before the generic `rating` rule (0-10) so a specifically-
  named Customer_Rating column gets the tighter, more accurate bound.
- **`core/validity.py` / `core/remediation.py`**: Quantity/Qty now
  reject exactly 0 as well as negative (`min_exclusive`) -- an order
  line can't have zero items. The flag is carried end-to-end through
  `Finding.details` and reused verbatim by remediation, never
  re-derived. Price/Amount/Total_Amount stay at plain "can't be
  negative" (a zero price/amount can be legitimately valid).
- **`core/validity.py`**: Total_Amount/Price now also get the
  statistical IQR extreme-outlier check (same reasoning as Quantity --
  these are transaction-scoped within one file, unlike Revenue/Salary/
  Cost which legitimately vary widely across businesses and stay
  IQR-free to avoid false positives).
- **`core/remediation.py`**: date normalization now also splits on `,`
  (`"Jan 20, 2020"`) in addition to the existing separators.
- 4 new tests (74 total).
- Live-verified: `Chennnai`->`Chennai`, `Customer_Rating=9` and
  `Quantity=0`/`-1` neutralized, `Total_Amount=999999` neutralized,
  `Jan 20, 2020` / `2020.04.12` normalized, genuinely ambiguous dates
  (`10-03-2020`, `03-04-2020`) correctly left on manual review.

## Pass 3: curated alias dictionary, quantity outliers, month-name dates

Makes categorical standardization "genuinely strong" per the Priority 1
gap, plus two smaller outlier/date fixes:

- **`core/validity.py`**: added a curated, hand-authored alias
  dictionary (`_KNOWN_CATEGORICAL_ALIASES` / `_CONDITIONAL_CATEGORICAL_ALIASES`)
  for abbreviations and city-name synonyms a similarity score can never
  safely catch on its own (`DC`->`Debit Card`, `NB`->`Net Banking`,
  `CC`->`Credit Card`, `COD`->`Cash on Delivery`, `Dilli`->`Delhi`,
  `Madras`->`Chennai`, `Bengaluru`->`Bangalore`, `Bombay`->`Mumbai`,
  `Calcutta`->`Kolkata`, `Deliverd`->`Delivered`). `New Delhi`->`Delhi`
  is conditional: it only fires when the column's own dominant spelling
  is already `Delhi`, so a dataset that deliberately distinguishes the
  two isn't silently merged. The existing cardinality guard (which stops
  fuzzy matching from misfiring on genuine free-text columns) now only
  gates the STATISTICAL half of categorical detection -- the curated
  alias lookup runs regardless, so a column like City (naturally many
  distinct real values even in a small file) still gets its known
  aliases fixed.
- **`core/validity.py`**: `Quantity`/`Qty`/`Count`-shaped columns now
  also get the statistical IQR extreme-outlier check on top of their
  non-negative bound (catches a Quantity of 999/5000 among mostly
  single-digit orders). Deliberately NOT extended to
  Price/Amount/Revenue/Salary/Cost/... -- those vary legitimately over a
  wide, business-dependent range, so flagging them by statistical
  extremity alone would be a real false-positive risk (verified via a
  new test: a genuinely high but legitimate Annual_Revenue value is
  NOT flagged).
- **`core/remediation.py`**: unambiguous month-name dates (`15-Jan-2020`,
  `3 Feb 2021`) now normalize to `YYYY-MM-DD` -- previously only plain
  all-numeric dates were handled. A month NAME removes the day/month
  ambiguity entirely (unlike an all-numeric date), so there's no
  ambiguous case to decline here.
- 4 new tests (70 total): known-alias mapping, the conditional
  `New Delhi` guard, month-name date normalization, and
  Quantity-extreme-vs-Revenue-not-flagged.
- Live-verified against a synthetic messy orders file covering every
  named success criterion: `Deliverd`->`Delivered`, `DC`/`NB`/`CC`->full
  payment-method names, `Dilli`/`Madras`/`Bengaluru`->`Delhi`/`Chennai`/
  `Bangalore`, `ask` in a date column and `unknown` in a numeric column
  cleared to blank, `Rating=150` and `Total_Amount=-200` neutralized,
  exact duplicate row removed, mixed DD-Mon-YYYY/YYYY-MM-DD dates
  normalized, no crash, no auto-push.

CATEGORICAL_FUZZY_THRESHOLD stays at 85 (Jaro-Winkler scale), not 88 --
see that constant's docstring: 88 on this scale would drop the
project's own named "Dizel"->"Diesel" example (scores 85.8) below the
floor. 88 is correct on rapidfuzz's WRatio scale, which this project
deliberately does not use (see Pass 2/1 notes below for why).

## Pass 2: transactional-data false positives, defensive UI, audit report

Fixes two real false-positive bugs found on order/transaction-style data
(e.g. a `Customer_ID`/`Customer_Name` that legitimately repeats across
many different orders), plus Phase 5/6 polish:

- **`core/structure.py`**: `IDENTIFIER_COLUMN_NAME_HINTS` used to match a
  bare `_id` substring, so `Customer_ID` was flagged as "should be a
  unique identifier" even in normal order data. Now a column only counts
  as a per-row identifier if it names an actual transaction key
  (`Order_ID`, `Invoice_ID`, `GST`, ...) or a bare `_id`/`code` hint that
  does NOT also look like a dimension reference (`customer`, `vendor`,
  `product`, `employee`, ...). `Order_ID` repeating is still caught;
  `Customer_ID` repeating no longer is.
- **`core/duplication.py`**: a pair whose identity-column value is an
  EXACT match (not a fuzzy near-miss) is now only counted as a
  near-duplicate RECORD if enough of the *rest* of the row also matches
  (`_confirm_exact_identity_match`, floor 50%) -- otherwise it's just a
  repeat customer on a different order, not a duplicated entry. A
  genuine spelling variant (`Shree Ganesh` vs `Shri Ganesh`) is
  unaffected -- that check only applies to essentially-identical values.
- **`core/validity.py`**: added named domain rules for `Rating` (0-10)
  and `Discount`/`Discount_Percent` (0-100, documented as a percentage
  assumption).
- **`app.py`**: `render_remediation_section` now guards against
  `table_result.dataframe`/`.duplication_result` being `None` (shows a
  contained message instead of an `AttributeError`). Added a "Download
  audit report (TXT)" button alongside the cleaned-CSV download, built
  from the same `RemediationResult` the on-screen log renders from.
- 5 new tests (66 total): repeating `Customer_ID`/`Customer_Name` no
  longer false-positives, `Order_ID` duplication still caught, a real
  near-duplicate business-name pair (matching row) still caught,
  Rating/Discount domain-outlier detection.
- Live-verified against a synthetic orders file
  (`Order_ID`/`Customer_ID`/`Customer_Name`/`Rating`/`Discount_Percent`):
  0 false criticals, Duplication/Structure both 100.0, real outliers
  (`Rating=150`, `Amount=-200`) correctly caught and cleared.


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
