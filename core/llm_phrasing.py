"""
core/llm_phrasing.py

Optional enhancement layer: asks a small, local, offline language model
to rephrase an already-computed Finding into a slightly more natural
sentence. This is the ONLY place in the whole tool that touches an LLM,
and it is deliberately boxed in:

- It NEVER sees the raw dataset -- only four already-computed facts
  about one Finding (column name, issue type, percentage affected,
  severity). There is nothing here for the model to "discover" or
  hallucinate, because it is never shown the data a finding is even
  about. See _PROMPT_TEMPLATE below -- that is the entire input.
- It is NEVER used to decide anything -- not whether an issue exists,
  not its severity, not the score. Those are 100% deterministic (see
  core/completeness.py etc., core/calibration.py, core/scoring.py).
  This module's only job is turning one already-decided fact into one
  sentence, after the fact.
- If the model isn't installed, hasn't been downloaded, fails to load,
  errors during generation, or produces obviously broken output, this
  module returns None and the caller (core/fixlist.py) falls back to
  the guaranteed template-based sentence. The tool must work completely
  normally with zero AI available -- whether this file's model is
  present or absent changes nothing else about how the tool runs.

Model choice: Qwen2.5-0.5B-Instruct, GGUF format, Q4_K_M quantization
(~380MB on disk). Why this one:
  - It's instruction-tuned, so it reliably follows a short, narrow
    "rephrase this one fact" instruction instead of rambling.
  - 0.5B parameters is small enough to run fast on CPU with no GPU --
    important since this has to work on a laptop during a live demo.
  - The GGUF build at this quantization is a few hundred MB, which is a
    genuinely reasonable "one-time download" rather than a multi-GB one.
See README.md ("Local AI model") for the one-time setup step
(scripts/download_model.py) that fetches this file -- this module only
ever reads it from a local path on disk; it never makes a network
request itself.
"""

import os
import re
from typing import Optional

# Must match scripts/download_model.py's destination path exactly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIRECTORY = os.path.join(_REPO_ROOT, "models")
MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIRECTORY, MODEL_FILENAME)

# Hard caps so one slow generation can never noticeably stall the UI --
# rephrasing a Finding is a "nice to have" enhancement, never worth
# making the user wait more than a beat for. 90 tokens is enough room
# for one full sentence (including restating the percentage figure);
# we still truncate to the first sentence afterward (see
# _extract_first_sentence) because a 0.5B model doesn't reliably stop
# itself at "exactly one sentence" just because the prompt asked it to.
MAX_NEW_TOKENS = 90
LLAMA_CONTEXT_SIZE = 512  # small on purpose -- our prompts are tiny

# Matches a sentence-ending ". " / "! " / "? " -- but NOT a period that
# sits between two digits, so we never split "77.78%" in half. This is
# the standard "protect decimal points" trick for naive sentence
# splitting: (?<!\d) and (?!\d) are lookaround assertions meaning "not
# immediately preceded/followed by a digit".
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<!\d)([.!?])(?!\d)(\s|$)")

# Maps each internal issue_type code (see core/fixlist.py's
# _ISSUE_TYPE_TO_CHECK_TYPE, the source of truth for which codes exist)
# to a genuine human-readable phrase, used in the prompt instead of the
# raw snake_case string. Without this, a small model has no way to know
# "fuzzy_duplicate_rows" is a technical label rather than language to
# repeat -- and it sometimes DOES repeat it verbatim, producing
# sentences like "...contains a fuzzy_duplicate_rows issue..." in what's
# supposed to be plain language. tests/test_pipeline.py has a test that
# fails loudly if fixlist.py ever gains a new issue_type without a
# matching entry here, so this can't silently drift out of sync.
_ISSUE_TYPE_HUMANIZED = {
    "missing_data": "missing data",
    "inconsistent_format_numeric": "inconsistent number formatting",
    "inconsistent_format_date": "inconsistent date formatting",
    "inconsistent_format_text": "inconsistent text formatting (casing or spacing)",
    "exact_duplicate_rows": "exact duplicate rows",
    "fuzzy_duplicate_rows": "near-duplicate records",
    "structural_issue": "a structural data issue",
}


def _humanize_issue_type(issue_type: str) -> str:
    """
    Looks up the human-readable phrase for an issue_type code. Falls
    back to a de-snake-cased version of the raw code (underscores ->
    spaces) if it's ever missing from the mapping above -- strictly
    better than leaking raw_snake_case into a sentence, though this
    should never actually happen in practice (see the mapping-
    completeness test mentioned above). Either way, if a raw snake_case
    token still ends up in the model's output somehow,
    _contains_leaked_issue_type_code below catches it as a last resort.
    """
    if issue_type in _ISSUE_TYPE_HUMANIZED:
        return _ISSUE_TYPE_HUMANIZED[issue_type]
    return issue_type.replace("_", " ")


# Zero temperature = greedy decoding: the model always picks its single
# highest-probability next token, so the same prompt always produces the
# same output. Two reasons this matters here, not just style: (1) we
# want a faithful rephrasing of already-decided facts, not a creative
# one, and (2) the project's determinism requirement ("the same file
# processed twice must always produce the same score and the same fix
# list") is easiest to honor completely -- including the optional AI
# phrasing overlay -- if the model itself never samples randomly.
GENERATION_TEMPERATURE = 0.0

# Lazy-loaded singleton so the (relatively slow, ~1-2s) model load only
# happens once per running process, not once per Finding. Three module-
# level flags instead of one: we need to remember "did we already try
# loading" separately from "did that attempt fail", so a failed load
# doesn't get retried on every single call (each retry would also be
# slow, for no benefit -- if it failed once with the same file on disk,
# it will fail again).
_model_instance = None
_model_load_attempted = False


def _get_model():
    """
    Loads the GGUF model once per process and reuses it after that.
    Returns None (without raising) if llama-cpp-python isn't installed,
    the model file hasn't been downloaded yet, or loading fails for any
    other reason -- this is the first of several fallback points in this
    file, all leading to the same place: "just return None."
    """
    global _model_instance, _model_load_attempted

    if _model_load_attempted:
        return _model_instance  # None if the earlier attempt failed

    _model_load_attempted = True

    if not os.path.isfile(MODEL_PATH):
        return None

    try:
        from llama_cpp import Llama
        _model_instance = Llama(
            model_path=MODEL_PATH,
            n_ctx=LLAMA_CONTEXT_SIZE,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
    except Exception:
        # Covers: llama-cpp-python not installed, a corrupted/partial
        # model file, an incompatible build for this machine, out of
        # memory, or anything else we didn't anticipate. Whatever the
        # cause, the tool falls back to templates -- see module docstring.
        _model_instance = None

    return _model_instance


# The ENTIRE prompt. Notice what is deliberately absent: no raw data,
# no other findings, no column values, no instruction to "analyze" or
# "look for" anything -- just already-computed facts and an instruction
# to phrase them, one sentence, nothing invented.
#
# Two variants, because not every Finding has a concrete example (see
# core/findings.py's Finding.example) -- a plain missing-data finding
# doesn't have one, a fuzzy-duplicate finding does. Both variants
# explicitly instruct the model to restate the percentage (and the
# example, when there is one) VERBATIM rather than paraphrasing it away.
# This instruction earns its place in the prompt for a concrete reason:
# without it, a small 0.5B model tends to summarize "45.0%, e.g. X vs Y"
# down to vague language like "a high percentage" or "a specific case"
# -- technically not wrong, but it throws away the exact information
# that makes a fix list actionable. See _preserves_key_facts below,
# which actually verifies this instruction was followed rather than
# just hoping it was.
_PROMPT_TEMPLATE_WITH_EXAMPLE = (
    "You are rephrasing one data-quality finding in plain, professional English.\n"
    "Use ONLY the facts given below. Do not invent any new numbers, columns, or issues.\n"
    "You MUST restate the exact percentage and the exact named example given below "
    "verbatim in your sentence -- never replace them with vague phrases like "
    "'a high percentage' or 'a specific case'.\n"
    "Reply with exactly one sentence. No preamble, no markdown, no quotation marks.\n\n"
    "Column: {column_name}\n"
    "Problem type: {issue_type}\n"
    "Percentage of rows affected: {percentage_affected}%\n"
    "Named example: {example}\n"
    "Severity: {severity}\n\n"
    "One-sentence plain-language description:"
)

_PROMPT_TEMPLATE_NO_EXAMPLE = (
    "You are rephrasing one data-quality finding in plain, professional English.\n"
    "Use ONLY the facts given below. Do not invent any new numbers, columns, or issues.\n"
    "You MUST restate the exact percentage given below verbatim in your sentence -- "
    "never replace it with a vague phrase like 'a high percentage' or 'a significant portion'.\n"
    "Reply with exactly one sentence. No preamble, no markdown, no quotation marks.\n\n"
    "Column: {column_name}\n"
    "Problem type: {issue_type}\n"
    "Percentage of rows affected: {percentage_affected}%\n"
    "Severity: {severity}\n\n"
    "One-sentence plain-language description:"
)


def phrase_finding(finding) -> Optional[str]:
    """
    Ask the local model to rephrase a single Finding's already-computed
    facts into one sentence. Returns None (never raises) on ANY failure
    -- model unavailable, generation error, output that doesn't look
    like a usable sentence, or output that drops the specific numbers/
    example it was told to keep -- so the caller can silently fall back
    to finding.rule_based_description.

    `finding` is a core.findings.Finding. We only read its already-
    computed facts (column_name, issue_type, percentage_affected,
    severity, example) -- never rule_based_description itself (no reason
    to feed the model our own template output) and never anything about
    the underlying dataframe, because Finding doesn't carry that at all.
    """
    model = _get_model()
    if model is None:
        return None

    # Humanized, not the raw internal code -- see _humanize_issue_type's
    # docstring for why passing the raw "fuzzy_duplicate_rows"-style
    # string to the model was the root cause of it leaking into output.
    humanized_issue_type = _humanize_issue_type(finding.issue_type)

    if finding.example:
        prompt = _PROMPT_TEMPLATE_WITH_EXAMPLE.format(
            column_name=finding.column_name,
            issue_type=humanized_issue_type,
            percentage_affected=finding.percentage_affected,
            example=finding.example,
            severity=finding.severity,
        )
    else:
        prompt = _PROMPT_TEMPLATE_NO_EXAMPLE.format(
            column_name=finding.column_name,
            issue_type=humanized_issue_type,
            percentage_affected=finding.percentage_affected,
            severity=finding.severity,
        )

    try:
        response = model(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            stop=["\n"],
        )
        generated_text = response["choices"][0]["text"].strip()
    except Exception:
        return None

    # A 0.5B model reliably follows "use only these facts" but not
    # always "exactly one sentence" -- it sometimes keeps going into a
    # second or third sentence, or gets cut off mid-word once it hits
    # MAX_NEW_TOKENS. We enforce the one-sentence rule ourselves rather
    # than trust the model's own restraint.
    first_sentence = _extract_first_sentence(generated_text)

    if not _looks_like_a_usable_sentence(first_sentence):
        return None
    if _contains_leaked_issue_type_code(first_sentence):
        # Humanization (see _humanize_issue_type) should already keep
        # every raw issue_type code out of the prompt entirely -- this
        # is the safety net for if that mapping ever misses a future
        # case, or the model invents a snake_case-looking token some
        # other way. Either way: reject and fall back, don't show it.
        return None
    if not _preserves_key_facts(first_sentence, finding):
        # The prompt asked for the percentage/example to be kept
        # verbatim, but the model dropped it anyway -- rather than show
        # a vague sentence in a UI that specifically promises AI-phrased
        # findings stay factual, we fall back to the guaranteed
        # rule_based_description, exactly like any other AI failure.
        return None
    return first_sentence


def _extract_first_sentence(text: str) -> str:
    """
    Returns everything up to and including the first sentence-ending
    punctuation, ignoring decimal points (see _SENTENCE_BOUNDARY_PATTERN).
    If no sentence boundary is found at all (e.g. generation got cut off
    before finishing even one sentence), returns the text as-is --
    _looks_like_a_usable_sentence will reject it if it's too short/long
    to be usable anyway.
    """
    match = _SENTENCE_BOUNDARY_PATTERN.search(text)
    if match is None:
        return text.strip()
    return text[: match.end(1)].strip()


def _looks_like_a_usable_sentence(text: str) -> bool:
    """
    A cheap sanity filter on the model's raw output. This doesn't need
    to be clever -- it just needs to reject the obviously-broken cases
    (empty output, the model echoing the prompt back instead of
    answering it, a wall of repeated characters) rather than showing
    garbage to the user. When in doubt, we reject and fall back to the
    template sentence: a boring guaranteed-correct sentence beats a
    fancy broken one, every time.
    """
    if not text or len(text) < 10 or len(text) > 400:
        return False
    if "Column:" in text or "Severity:" in text or "Problem type:" in text or "Named example:" in text:
        return False  # the model echoed the prompt instead of answering it
    return True


# Matches a snake_case-style token: two or more lowercase word-parts
# joined by underscores (e.g. "fuzzy_duplicate_rows",
# "inconsistent_format_numeric"). Genuine English prose never contains
# underscores, so a match here is an unambiguous signal of a leaked
# internal code, not a plausible false positive on real language.
_SNAKE_CASE_TOKEN_PATTERN = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")


def _contains_leaked_issue_type_code(text: str) -> bool:
    """
    Safety net matching this file's existing defensive style (see
    _looks_like_a_usable_sentence, _preserves_key_facts): even with the
    humanization mapping in place, this catches a raw internal code if
    it ever leaks into the model's output anyway -- whether because a
    future issue_type is added to fixlist.py without a matching entry in
    _ISSUE_TYPE_HUMANIZED, or the model invents its own snake_case-
    looking token some other way. Deliberately a general pattern match
    rather than a lookup against a specific list of known codes, so it
    stays correct even as new issue_types are added later without this
    file needing to change too.
    """
    return bool(_SNAKE_CASE_TOKEN_PATTERN.search(text))


def _preserves_key_facts(text: str, finding) -> bool:
    """
    Verifies the model actually followed the "restate this verbatim"
    instruction, rather than just hoping the prompt wording was enough
    -- a small model can and does ignore instructions sometimes. Two
    checks, both deliberately lenient (a false REJECTION just means we
    fall back to the guaranteed template sentence, which is always
    fine; a false ACCEPTANCE would let a vague sentence slip through,
    which is the exact bug this function exists to catch):

    1. The percentage's whole-number part must appear somewhere in the
       text. Checking the whole-number part (not the full "45.0")
       tolerates the model writing "45%" or "45.2%" instead of matching
       our own rounding exactly, while still catching the real failure
       case: the number being dropped entirely in favor of words like
       "a high percentage".
    2. If the finding has a named example, at least ONE of the two
       compared values (an example is always "X vs Y") must appear in
       the text. Requiring only one side (not both, not the exact
       "X vs Y" phrasing) tolerates the model rewording "X vs Y" into
       "X and Y" or mentioning just one of the two names, while still
       catching the real failure case: neither name surviving at all.
    """
    whole_number_part = str(int(finding.percentage_affected))
    if whole_number_part not in text:
        return False

    if finding.example:
        example_parts = [part.strip() for part in finding.example.split(" vs ") if part.strip()]
        if example_parts and not any(part in text for part in example_parts):
            return False

    return True


def is_model_available() -> bool:
    """
    Lets the UI show whether AI phrasing is active for this run, without
    triggering a load attempt just to check -- a cheap file-existence
    check only. (The actual load is still lazy and only happens the
    first time phrase_finding() is called.)
    """
    return os.path.isfile(MODEL_PATH)
