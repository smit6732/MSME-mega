"""
scripts/profile_pipeline.py

Dev tool: times each pipeline stage separately (ingestion, completeness,
consistency, duplication, structure, scoring, fix-list/LLM phrasing) for
one or more files, and prints a side-by-side breakdown. This exists to
answer "which stage is actually slow?" with real numbers before changing
any logic -- see the "Performance Fix" investigation this script was
built for.

This deliberately does NOT reuse core/pipeline.py's run_pipeline_for_table
as one opaque call -- it calls each stage separately with a timer around
it, mirroring that function's sequence exactly, so the timing breakdown
is trustworthy (each number really is just that one stage).

Usage:
    python scripts/profile_pipeline.py path/to/file1.csv path/to/file2.csv ...
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingestion import load_dataset
from core.completeness import check_completeness
from core.consistency import check_consistency
from core.duplication import check_duplication
from core.structure import check_structure
from core.scoring import build_scorecard
from core.fixlist import generate_fix_list


STAGE_NAMES = ["ingestion", "completeness", "consistency", "duplication", "structure", "scoring", "fixlist_and_ai_phrasing"]


def profile_file(file_path: str, use_ai_phrasing: bool = True) -> dict:
    """Runs the full pipeline on one file, stage by stage, with a timer
    around each stage. Returns {stage_name: seconds}."""
    file_name = os.path.basename(file_path)
    timings = {}

    start = time.perf_counter()
    dataset_profile = load_dataset(file_path, file_name)
    timings["ingestion"] = time.perf_counter() - start

    start = time.perf_counter()
    completeness_result = check_completeness(dataset_profile.dataframe)
    timings["completeness"] = time.perf_counter() - start

    start = time.perf_counter()
    consistency_result = check_consistency(dataset_profile.dataframe, dataset_profile.column_types)
    timings["consistency"] = time.perf_counter() - start

    start = time.perf_counter()
    duplication_result = check_duplication(dataset_profile.dataframe, dataset_profile.column_types)
    timings["duplication"] = time.perf_counter() - start

    start = time.perf_counter()
    structure_result = check_structure(dataset_profile.dataframe, dataset_profile.raw_text)
    timings["structure"] = time.perf_counter() - start

    start = time.perf_counter()
    build_scorecard(completeness_result, consistency_result, duplication_result, structure_result)
    timings["scoring"] = time.perf_counter() - start

    start = time.perf_counter()
    generate_fix_list(
        completeness_result, consistency_result, duplication_result, structure_result,
        dataset_profile.row_count, table_name=file_name, use_ai_phrasing=use_ai_phrasing,
    )
    timings["fixlist_and_ai_phrasing"] = time.perf_counter() - start

    timings["_row_count"] = dataset_profile.row_count
    timings["_total"] = sum(timings[s] for s in STAGE_NAMES)
    return timings


def print_report(file_path: str, timings: dict) -> None:
    print(f"\n=== {os.path.basename(file_path)} ({timings['_row_count']} rows) ===")
    for stage in STAGE_NAMES:
        seconds = timings[stage]
        bar = "#" * min(60, int(seconds * 4))
        print(f"  {stage:26s} {seconds:8.3f}s  {bar}")
    print(f"  {'TOTAL':26s} {timings['_total']:8.3f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/profile_pipeline.py file1.csv [file2.csv ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        timings = profile_file(path)
        print_report(path, timings)
