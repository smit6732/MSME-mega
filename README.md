# MSME Data Quality Scorecard

An AI-ready data quality scorecard for MSMEs. Most MSME data — spreadsheets, POS exports, Tally dumps, CRM exports — is messy, inconsistent, and unstructured, which is why most AI/BI pilots stall before they deliver value. This tool ingests raw business data (one or more files, including multi-sheet Excel workbooks) and automatically scores each table on **completeness**, **consistency**, **duplication**, and **structure**, then generates a plain-language list of exactly what to fix before the data can be trusted in an AI or BI pipeline.

## How it works (pipeline overview)

```
Upload (1+ CSV/Excel files, each Excel sheet = its own table)
    -> core/ingestion.py    (load file(s)/sheet(s), infer each column's type)
    -> core/completeness.py (% missing per column)
    -> core/consistency.py  (formatting problems per column)
    -> core/duplication.py  (exact + fuzzy duplicate rows, threshold auto-calibrated)
    -> core/structure.py    (headers, ragged rows, duplicate IDs)
    -> core/scoring.py      (weighted overall score per table)
    -> core/fixlist.py      (Finding objects: severity auto-calibrated, template sentence)
    -> core/llm_phrasing.py (optional: local LLM rephrases the most important findings)
    -> core/pipeline.py     (runs the above per table, combines into one summary)
    -> app.py                (Streamlit dashboard: overall summary + per-table detail)
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Windows note on `llama-cpp-python`:** this package compiles a C++ library, and on a plain Windows machine (no Visual Studio Build Tools installed) `pip install` will fail to build it from source. If that happens, install it from the project's prebuilt CPU wheels instead:

   ```bash
   pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```

   If `llama-cpp-python` still can't be installed at all, that's fine — **the app runs completely normally without it.** See "Local AI model" below; this is exactly the fallback path the tool is designed around, not a broken state.

3. **One-time step** — download the local AI model (see "Local AI model" below for exactly what this does and why it's safe):

   ```bash
   python scripts/download_model.py
   ```

4. Run the dashboard:

   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually `http://localhost:8501`). Upload `sample_data/messy_msme_sample.csv`, or `sample_data/multi_sheet_msme_workbook.xlsx` (a 4-sheet workbook, one sheet deliberately empty to demonstrate partial-failure handling), or select several files at once.

## Running the tests

```bash
pytest
```

Covers: single-file backward compatibility, multi-sheet/multi-file batches, one bad sheet not crashing the rest of a batch, calibration's real behavior (both the calibrated case and its fallback), full-pipeline determinism (the same file run three times produces byte-identical scores and fix-list text), and — explicitly, per the project's own requirement — that the AI phrasing layer falls back cleanly to the guaranteed template sentence when the model is unavailable.

## No external AI dependency — and what "local AI" actually means here

**No cloud/external LLM API is called anywhere in the running app** — no OpenAI, Claude, Gemini, or any other network-based AI service, ever, at runtime. Every score, severity classification, and threshold decision is produced by deterministic rule-based logic (pandas, RapidFuzz) plus one **local**, scikit-learn-based calibration step. The one place AI genuinely touches the tool's output is described in detail below, and it is boxed in tightly.

### Two calibrated thresholds (scikit-learn, `core/calibration.py`)

Instead of one flat cutoff for every file, two thresholds are calibrated fresh, per table, every time the pipeline runs:

1. **Duplicate-match threshold.** Every pairwise name-similarity score in the table's identity column (e.g. `Business_Name`) is clustered with k-means (k=2: "similar" vs. "not similar"). The midpoint between the two cluster centers becomes this table's threshold, replacing the fixed 90%.
2. **Severity thresholds.** Every "% affected" value across all four checks run on a table is clustered with k-means (k=3: low/medium/high impact). The midpoints between neighbouring centers become this table's critical/moderate cutoffs, replacing the fixed 30%/10% split.

Both are:
- **Fit fresh on each table, every time** — nothing is saved, cached, or reused between tables or between runs. No model file, no pretrained weights.
- **Deterministic** — k-means' random seed is fixed, so the same input always produces the same threshold.
- **Guarded twice** before being trusted: (a) there must be enough distinct data points to cluster meaningfully, and (b) the clusters k-means finds must actually be separated by a real gap — k-means always returns exactly *k* clusters even when the data is one undifferentiated blob, so a "split" with no real gap between the groups is rejected and the tool falls back to the fixed default rather than trust a meaningless split. Both guard conditions and the fixed fallback values are visible in `core/calibration.py`.
- **Shown in the UI, always** — every table's detail section displays the exact threshold used and whether it was calibrated or fell back to the default, so nothing calibrated is silently invisible.

This calibration step is the *only* place anything resembling "ML" appears in this project, and it never changes what a check measures or how the four dimension scores combine into the overall score (`core/scoring.py`'s weights are fixed constants, untouched by any of this).

### Local LLM fix-list phrasing (`core/llm_phrasing.py`)

**Model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), GGUF format, `Q4_K_M` quantization (~490MB on disk).

**Why this model:** it's instruction-tuned (reliably follows a short, narrow instruction instead of rambling), small enough to run fast on CPU with no GPU (typically well under a second per finding once loaded), and its GGUF file is a genuinely reasonable one-time download rather than a multi-gigabyte one.

**"Bundled, no download" — what that actually means:** a model this size cannot ship as plain source code in this repository. What *can* and does ship as source code is `scripts/download_model.py` — a one-time setup step (see "Setup" above) that fetches the GGUF file once and caches it in `models/` (git-ignored). **After that one-time step, the running app never makes a network request.** `core/llm_phrasing.py` only ever reads the already-downloaded file from local disk. This is fundamentally different from "downloading it every time the tool is used" — it's a one-time, explicit, documented install step, exactly like `pip install` itself.

**What the model is (and is never) given:** it receives exactly four already-computed facts about one `Finding` — column name, issue type, percentage affected, severity — and is instructed to phrase those facts in one plain sentence. It is **never** given the raw dataset, never given other findings, and never asked to reason freely about the data. There is nothing for it to hallucinate a new finding from, because it is physically never shown the data a finding is even about. See the full prompt template in `core/llm_phrasing.py`.

**Guaranteed fallback:** if the model file hasn't been downloaded, `llama-cpp-python` isn't installed, loading fails, generation errors out, or the output doesn't look like a usable sentence, `phrase_finding()` returns `None` and the caller (`core/fixlist.py`) keeps the guaranteed template-based sentence — the same one that's *always* computed first, regardless of AI availability. `tests/test_pipeline.py::test_ai_phrasing_falls_back_cleanly_when_model_unavailable` verifies this exact path. **The tool works completely normally with zero AI available** — this is a hard requirement, not a nice-to-have, and it's exactly what that test proves.

**Labeled, not hidden:** every AI-phrased sentence in the UI carries a `🤖 AI-enhanced phrasing` tag, so AI-touched text is never visually indistinguishable from the deterministic core.

### Determinism

The scoring logic, the calibration, and (with greedy/zero-temperature decoding) the AI phrasing layer are all deterministic: **the same file, processed twice, always produces the same score and the same fix list**, verified by `tests/test_pipeline.py::test_same_file_processed_twice_gives_identical_results`. AI is only ever allowed to touch two things — calibrating thresholds, and rephrasing already-computed findings — never the pass/fail logic of a check, and never the final weighted score formula.

## Project structure

```
core/
  ingestion.py     # load CSV/Excel (single table, or every sheet in a workbook), infer column types
  completeness.py  # missing-value % per column + overall completeness score
  consistency.py   # formatting inconsistencies per column + overall consistency score
  duplication.py   # exact duplicate rows + RapidFuzz near-duplicate detection (calibrated threshold)
  structure.py     # header/ragged-row/duplicate-identifier checks
  scoring.py       # combines the four dimension scores into one weighted score (fixed constants)
  findings.py      # the shared Finding dataclass every check's output becomes
  calibration.py   # scikit-learn k-means threshold calibration, with fallback + gap-validation
  fixlist.py       # raw issues -> calibrated severity -> templated Finding objects
  llm_phrasing.py  # optional local-LLM rephrasing layer, strictly boxed in (see above)
  pipeline.py       # orchestrates one table, and multiple tables/files, into one summary
templates/
  fix_templates.py # sentence templates used by fixlist.py (no AI, pure Python)
scripts/
  download_model.py # ONE-TIME setup step: fetches the local LLM's weights
models/
  (git-ignored; created by scripts/download_model.py)
sample_data/
  messy_msme_sample.csv          # 20-row single-table sample with realistic data issues
  realistic_msme_directory.csv   # 168-row larger single-table sample
  multi_sheet_msme_workbook.xlsx # 4-sheet workbook: 3 messy differently, 1 deliberately empty
  preamble_test.csv              # a report-title line before the real header (see "Ingestion robustness")
  regional_language_sample.csv   # genuine Hindi/Gujarati business names (see "Ingestion robustness")
tests/
  test_pipeline.py # end-to-end smoke tests (see "Running the tests" above)
app.py              # Streamlit dashboard
```

## Scoring weights

Each table's overall score is a weighted average of its four dimensions (see constants at the top of `core/scoring.py` — these are fixed, never touched by calibration or AI):

| Dimension     | Weight | Why |
|---------------|-------:|-----|
| Completeness  | 30%    | Blank data is the most fundamental blocker — nothing else can be fixed on data that isn't there. |
| Consistency   | 25%    | Messy-but-present data is still usable after cleanup — a real problem, but a fixable one. |
| Duplication   | 25%    | Duplicate/near-duplicate records directly bias any aggregation or model. |
| Structure     | 20%    | Structural issues (bad headers, ragged rows, duplicate IDs) matter but tend to affect fewer rows/columns. |

The **overall summary score** across a multi-table upload is a plain average of every successfully-processed table's own overall score (see `core/pipeline.py::_compute_overall_score`) — every table counts equally, same philosophy as every column counting equally within `core/completeness.py`.

## Multi-table behavior

- One CSV file = one table. One Excel file = one table **per sheet**.
- Uploading several files at once processes all of them as one combined batch.
- Each table runs the full pipeline independently — its own scorecard, its own calibrated thresholds, its own fix list.
- The UI shows one **overall summary** first (aggregate score + a combined fix list tagged by table, sorted by severity across the whole upload), then an expandable section per table underneath.
- **One bad table never crashes the batch.** A sheet that fails to parse, or is empty, is reported clearly (which file/sheet, and why) while every other table in the same upload is still scored normally. See `sample_data/multi_sheet_msme_workbook.xlsx`'s `Draft_Notes` sheet (deliberately empty) for a live example of this.

## Performance on large files

The near-duplicate check (`core/duplication.py`) compares every candidate row's identity value against every other row's — an *O(n²)* comparison. The original implementation did that comparison with a manual Python loop, which is fine at hundreds of rows but scales badly (a 3,000-row file has ~4.5 million candidate pairs). It was fixed two ways, neither of which changes a single similarity number — both verified to produce bit-identical results to the original pair-by-pair computation:

1. **`rapidfuzz.process.cdist`** runs the whole batch of comparisons in one C call instead of millions of individual Python-to-C calls. Everything downstream (the flat list of scores calibration needs, which pairs clear the threshold) stays in numpy array operations rather than a Python loop, so the fix doesn't just move the O(n²) Python cost somewhere else.
2. **Blocking** (`core/duplication.py`'s `_blocking_key`, only active above `BLOCKING_ROW_COUNT_THRESHOLD` rows): rows are grouped by a cheap signature (their sorted words' first few characters) before comparison, so only rows with a real chance of being near-duplicates get compared — a deliberate recall/speed trade-off, same idea real deduplication pipelines use.

Measured on this machine: a 3,000-row file's duplication check went from **37.2s → 0.49s** (~76x), and total pipeline time for that file went from **39.6s → 2.9s**. Reproduce this with:

```bash
python scripts/generate_large_sample.py 3000 sample_data/large_msme_directory.csv
python scripts/profile_pipeline.py sample_data/messy_msme_sample.csv sample_data/large_msme_directory.csv
```

The local-LLM fix-list phrasing (`core/fixlist.py`'s `MAX_FINDINGS_TO_AI_PHRASE`) was already capped at the 15 most severe findings per table before this investigation — measurement confirmed it does *not* scale with row count (the number of findings is bounded by column count, not row count), so no change was needed there.

## Ingestion robustness (`core/ingestion.py`)

Real Tally/POS exports (the exact source named in this project's own problem statement) aren't always a clean table starting at row 0. Two guarded fallbacks, both following the same philosophy as `core/calibration.py` — take confident action, or fall back to the safe default, but never silently guess and never hide that a fallback happened:

- **Preamble/title-row detection.** A report title or company letterhead line above the real header is common in these exports. `_detect_header_row_index` scans the first ~10 rows for a clear width jump — a real header (and the rows after it) is consistently much wider than a title line with only 1-2 populated cells. It only acts when that jump is unambiguous; anything else (a genuinely narrow table, no clear gap, multiple similarly-wide candidate rows) falls back to "row 0 is the header", exactly the previous behavior. Works for both CSV and Excel. Never silent: `DatasetProfile.skipped_preamble_rows` reports the count, and the UI shows *"Detected and skipped N header row(s) before the actual data"* whenever it's non-zero. See `sample_data/preamble_test.csv` for a live example.
- **Encoding fallback.** CSV loading tries UTF-8 first, then asks `charset-normalizer` to detect the actual encoding, then falls back to Latin-1 (which never raises) as a last resort. Worth knowing if you're ever asked to justify this live: testing found that charset-normalizer's own confidence score (`chaos == 0.0`) does **not** reliably tell apart similar single-byte Western codepages (it detected `cp1250` instead of the actual `cp1252` on realistic MSME-style test content, even at ~1KB) — so `DatasetProfile.encoding_warning` is set, and shown in the UI, whenever a file needed *any* fallback from clean UTF-8, not only when the detector's own score looked uncertain. Overclaiming confidence the detector doesn't actually have would contradict this project's own honesty-about-uncertainty standard everywhere else (calibration, AI phrasing).

**Regional-language data (Hindi/Gujarati script):** tested directly against `sample_data/regional_language_sample.csv` (genuine Devanagari and Gujarati business names, not transliterated) rather than guess-fixed. Result: works correctly with no changes needed. Ingestion, type inference (correctly "text", never misclassified), the consistency check's casing/whitespace logic (a correct no-op — these scripts have no case distinction), and RapidFuzz duplicate matching (correctly scored a genuine single-character Devanagari near-duplicate at 97.87% similarity) all behaved exactly as they do for Latin-script data.

**Known limitation, by design (not automated):** merged cells and formula-heavy Excel exports aren't specially handled. An automated "correction" here risks misinterpreting real data as a merge artifact (or the reverse) — a wrong automated guess would be worse than an honest gap. If a sheet like this produces a confusing score, that's this limitation, not a bug to silently work around.
