#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step3_verify_exports.py
@brief   Step 3 — Sanity check for raw CSV files exported by the MT5 EA.

@description
    Reads '_found_files.json' written by Step 2 and runs a lightweight
    sanity check on each exported CSV file.

    Note: MT5 raw exports are NOT sorted chronologically.
    Sorting and full quality checks happen in Step 5 (merge) and Step 6 (verify).

    Checks performed per file:
      1. File exists on disk
      2. File size > 0 bytes
      3. File can be read as CSV
      4. File has at least one data row
      5. All required columns are present

    Exit codes:
      0 — all timeframes passed
      2 — partial success (some timeframes OK)
      1 — critical failure (no valid files)
"""

import sys
import os
import json

import pandas as pd

from config import TIMEFRAMES
from logger import log, sep

FOUND_FILES_JSON = os.path.join(os.path.dirname(__file__), "_found_files.json")

# Required columns in every exported MT5 CSV
REQUIRED_COLUMNS = ["Datum", "Uhrzeit", "Open", "High", "Low", "Close", "Volumen"]


def check_file(tf: str, path: str) -> tuple[bool, list]:
    """
    @brief  Runs a lightweight sanity check on a single exported CSV file.

    @param  tf    Timeframe string, e.g. 'M15'
    @param  path  Absolute path to the CSV file.
    @return Tuple of (passed: bool, issues: list of issue strings).
    """
    issues = []
    passed = True

    # ── 1. File exists
    if not os.path.exists(path):
        return False, [f"File not found: {path}"]

    # ── 2. File size
    size = os.path.getsize(path)
    if size == 0:
        return False, ["File is empty (0 bytes)"]

    log(f"{tf}: {size // 1024} KB", "INFO")

    # ── 3. Load first 10 rows (fast check)
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, nrows=10)
    except Exception as e:
        return False, [f"Cannot read CSV: {e}"]

    if len(df) == 0:
        return False, ["CSV has no data rows"]

    # ── 4. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        passed = False
        log(f"{tf}: Missing columns: {missing_cols}", "ERROR")
    else:
        log(f"{tf}: All required columns present", "OK")

    return passed, issues


def verify_all() -> tuple[list, list, list]:
    """
    @brief  Loads the export manifest and sanity-checks all timeframe CSV files.

    @return Tuple of (valid, failed, missing) lists of timeframe strings.
    """
    if not os.path.exists(FOUND_FILES_JSON):
        log(f"Export manifest not found: {FOUND_FILES_JSON}", "ERROR")
        log("Did Step 2 complete successfully?", "INFO")
        return [], [], TIMEFRAMES[:]

    with open(FOUND_FILES_JSON, "r", encoding="utf-8") as f:
        found: dict = json.load(f)

    log(f"Loaded manifest: {len(found)} entries", "INFO")

    valid   = []
    failed  = []
    missing = []

    for tf in TIMEFRAMES:
        sep()
        log(f"Checking: {tf}", "STEP")
        path = found.get(tf)

        if not path:
            log(f"{tf}: not in export manifest", "WARN")
            missing.append(tf)
            continue

        passed, issues = check_file(tf, path)

        if passed:
            log(f"{tf}: OK", "OK")
            valid.append(tf)
        else:
            log(f"{tf}: FAILED — {'; '.join(issues)}", "ERROR")
            failed.append(tf)

    return valid, failed, missing


if __name__ == "__main__":
    sep("STEP 3: Sanity Check Exported CSV Files")

    valid, failed, missing = verify_all()
    total = len(TIMEFRAMES)

    sep("VERIFICATION SUMMARY")
    log(f"Passed  : {len(valid)}/{total}  {valid}",   "OK"   if len(valid) == total else "INFO")
    log(f"Failed  : {len(failed)}/{total}  {failed}",  "ERROR" if failed  else "INFO")
    log(f"Missing : {len(missing)}/{total}  {missing}", "WARN"  if missing else "INFO")

    if len(valid) == total:
        log("All exported files passed sanity check!", "OK")
        sys.exit(0)
    elif len(valid) > 0:
        log(f"Partial: {len(valid)}/{total} timeframes passed", "WARN")
        sys.exit(2)
    else:
        log("No files passed — pipeline cannot continue", "ERROR")
        sys.exit(1)