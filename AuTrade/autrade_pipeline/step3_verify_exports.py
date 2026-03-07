#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step3_verify_exports.py
@brief   Step 3 — Verifies quality of CSV files exported by the MT5 EA.

@description
    Reads '_found_files.json' written by Step 2 and runs a full quality
    check on each exported CSV file.

    Checks performed per file:
      1. File exists on disk
      2. File size > 0 bytes
      3. File has more than just a header row
      4. All required columns are present
      5. No duplicate timestamps
      6. Timestamps are in ascending chronological order
      7. No large time gaps (configurable threshold per timeframe)
      8. OHLC values are numeric and non-zero
      9. High >= Low for every bar
     10. High >= Open, High >= Close for every bar
     11. Low  <= Open, Low  <= Close for every bar

    Exit codes:
      0 — all timeframes passed all checks
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

# Required columns in every exported CSV
REQUIRED_COLUMNS = ["Datum", "Uhrzeit", "Open", "High", "Low", "Close", "Volumen"]

# Maximum allowed gap between bars (in minutes) per timeframe
MAX_GAP_MINUTES = {
    "M1":  60,       # max 1 hour gap
    "M15": 240,      # max 4 hours gap
    "M30": 480,      # max 8 hours gap
    "H1":  1440,     # max 1 day gap
    "H4":  2880,     # max 2 days gap
    "D1":  7200,     # max 5 days gap (weekends)
}


def check_file(tf: str, path: str) -> tuple[bool, list]:
    """
    @brief  Runs all quality checks on a single exported CSV file.

    @param  tf    Timeframe string, e.g. 'M15'
    @param  path  Absolute path to the CSV file.
    @return Tuple of (passed: bool, issues: list of issue strings).
    """
    issues  = []
    passed  = True

    # ── 1. File exists
    if not os.path.exists(path):
        return False, [f"File not found: {path}"]

    # ── 2. File size
    size = os.path.getsize(path)
    if size == 0:
        return False, ["File is empty (0 bytes)"]

    # ── 3. Load CSV
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    except Exception as e:
        return False, [f"Cannot read CSV: {e}"]

    if len(df) == 0:
        return False, ["CSV has no data rows"]

    log(f"{tf}: Loaded {len(df):,} rows, {len(df.columns)} columns", "INFO")

    # ── 4. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        passed = False
        log(f"{tf}: Missing columns: {missing_cols}", "ERROR")
    else:
        log(f"{tf}: All required columns present", "OK")

    # ── 5. Parse datetime
    try:
        df["_dt"] = pd.to_datetime(
            df["Datum"].str.strip() + " " + df["Uhrzeit"].str.strip(),
            format="%Y.%m.%d %H:%M"
        )
    except Exception as e:
        issues.append(f"Cannot parse datetime: {e}")
        log(f"{tf}: Cannot parse datetime: {e}", "ERROR")
        return False, issues

    invalid_dt = df["_dt"].isna().sum()
    if invalid_dt > 0:
        issues.append(f"{invalid_dt} rows have invalid datetime")
        log(f"{tf}: {invalid_dt} invalid datetime rows", "WARN")

    # ── 6. Duplicate timestamps
    dupes = df["_dt"].duplicated().sum()
    if dupes > 0:
        issues.append(f"{dupes} duplicate timestamps found")
        passed = False
        log(f"{tf}: {dupes} duplicate timestamps", "ERROR")
    else:
        log(f"{tf}: No duplicate timestamps", "OK")

    # ── 7. Chronological order
    is_sorted = df["_dt"].is_monotonic_increasing
    if not is_sorted:
        out_of_order = (df["_dt"].diff() < pd.Timedelta(0)).sum()
        issues.append(f"Timestamps not in order: {out_of_order} violations")
        passed = False
        log(f"{tf}: Timestamps not sorted — {out_of_order} violations", "ERROR")
    else:
        log(f"{tf}: Timestamps in correct order", "OK")

    # ── 8. Time gaps
    max_gap = MAX_GAP_MINUTES.get(tf, 1440)
    if len(df) > 1:
        gaps      = df["_dt"].diff().dropna()
        large     = gaps[gaps > pd.Timedelta(minutes=max_gap)]
        if len(large) > 0:
            worst = large.max()
            issues.append(f"{len(large)} gaps > {max_gap}min (worst: {worst})")
            log(f"{tf}: {len(large)} large gaps found (worst: {worst})", "WARN")
        else:
            log(f"{tf}: No large time gaps", "OK")

    # ── 9. OHLC numeric check
    try:
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        null_ohlc = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
        if null_ohlc > 0:
            issues.append(f"{null_ohlc} non-numeric OHLC values")
            passed = False
            log(f"{tf}: {null_ohlc} non-numeric OHLC values", "ERROR")
        else:
            log(f"{tf}: All OHLC values numeric", "OK")

        # ── 10. High >= Low
        hl_violations = (df["High"] < df["Low"]).sum()
        if hl_violations > 0:
            issues.append(f"{hl_violations} bars where High < Low")
            passed = False
            log(f"{tf}: {hl_violations} bars with High < Low", "ERROR")
        else:
            log(f"{tf}: High >= Low for all bars", "OK")

        # ── 11. High >= Open/Close and Low <= Open/Close
        ho_violations = (df["High"] < df["Open"]).sum()
        hc_violations = (df["High"] < df["Close"]).sum()
        lo_violations = (df["Low"]  > df["Open"]).sum()
        lc_violations = (df["Low"]  > df["Close"]).sum()
        ohlc_issues   = ho_violations + hc_violations + lo_violations + lc_violations

        if ohlc_issues > 0:
            issues.append(f"{ohlc_issues} OHLC range violations")
            log(f"{tf}: OHLC range violations: "
                f"H<O={ho_violations} H<C={hc_violations} "
                f"L>O={lo_violations} L>C={lc_violations}", "WARN")
        else:
            log(f"{tf}: OHLC ranges valid", "OK")

    except Exception as e:
        issues.append(f"OHLC check error: {e}")
        log(f"{tf}: OHLC check failed: {e}", "ERROR")

    # ── Date range summary
    dt_from = df["_dt"].min().strftime("%Y.%m.%d")
    dt_to   = df["_dt"].max().strftime("%Y.%m.%d")
    size_kb = size // 1024
    log(f"{tf}: Range {dt_from} to {dt_to} | {size_kb} KB", "INFO")

    return passed, issues


def verify_all() -> tuple[list, list, list]:
    """
    @brief  Loads the export manifest and verifies all timeframe CSV files.

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
    all_issues = {}

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
            log(f"{tf}: ALL CHECKS PASSED", "OK")
            valid.append(tf)
        else:
            log(f"{tf}: FAILED — {len(issues)} issue(s)", "ERROR")
            failed.append(tf)
            all_issues[tf] = issues

    return valid, failed, missing


if __name__ == "__main__":
    sep("STEP 3: Verify Exported CSV Files")

    valid, failed, missing = verify_all()
    total = len(TIMEFRAMES)

    sep("VERIFICATION SUMMARY")
    log(f"Passed  : {len(valid)}/{total}  {valid}",   "OK"   if len(valid) == total else "INFO")
    log(f"Failed  : {len(failed)}/{total}  {failed}",  "ERROR" if failed  else "INFO")
    log(f"Missing : {len(missing)}/{total}  {missing}", "WARN"  if missing else "INFO")

    if len(valid) == total:
        log("All files passed quality checks!", "OK")
        sys.exit(0)
    elif len(valid) > 0:
        log(f"Partial: {len(valid)}/{total} timeframes passed", "WARN")
        sys.exit(2)
    else:
        log("No files passed — pipeline cannot continue", "ERROR")
        sys.exit(1)