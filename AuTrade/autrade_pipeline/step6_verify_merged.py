#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step6_verify_merged.py
@brief   Step 6 — Final quality check for all merged CSV files.

@description
    Validates every merged CSV file in MERGED_DATA_DIR.
    Runs the same quality checks as Step 3 but on the final merged output.

    Checks performed per file:
      1.  File exists and is not empty
      2.  All required columns present
      3.  Datetime parseable
      4.  No duplicate timestamps
      5.  Timestamps in ascending chronological order
      6.  No large time gaps (per timeframe threshold)
      7.  OHLC values are numeric
      8.  High >= Low for every bar
      9.  High >= Open and High >= Close
      10. Low  <= Open and Low  <= Close
      11. Session column present and filled
      12. Summary statistics (row count, date range, file size)

@returns
    exit code 0 — all timeframes passed all checks
    exit code 2 — partial success (some passed, some failed)
    exit code 1 — critical failure (no valid files found)
"""

import sys
import os
import glob

import pandas as pd

from config import SYMBOL, TIMEFRAMES, MERGED_DATA_DIR
from logger import log, sep

# Required columns in every merged CSV
REQUIRED_COLUMNS = ["Datum", "Uhrzeit", "Open", "High", "Low",
                    "Close", "Volumen", "Session"]

# Maximum allowed gap between bars (minutes) per timeframe
MAX_GAP_MINUTES = {
    "M1":  60,
    "M15": 240,
    "M30": 480,
    "H1":  1440,
    "H4":  2880,
    "D1":  7200,
}


def find_merged_file(timeframe: str) -> str | None:
    """
    @brief  Finds the latest merged CSV for a given timeframe.

    @param  timeframe  Timeframe string e.g. 'M15'.
    @return Absolute path to the most recent merged file, or None.
    """
    pattern = os.path.join(MERGED_DATA_DIR, f"{SYMBOL}_{timeframe}_merged*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0]


def check_file(tf: str, path: str) -> tuple[bool, list]:
    """
    @brief  Runs all quality checks on a single merged CSV file.

    @param  tf    Timeframe string e.g. 'M15'.
    @param  path  Absolute path to the merged CSV file.
    @return Tuple of (passed: bool, issues: list of issue strings).
    """
    issues = []
    passed = True

    # ── 1. File exists and size
    if not os.path.exists(path):
        return False, ["File not found"]

    size = os.path.getsize(path)
    if size == 0:
        return False, ["File is empty (0 bytes)"]

    # ── 2. Load CSV
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    except Exception as e:
        return False, [f"Cannot read CSV: {e}"]

    if len(df) == 0:
        return False, ["No data rows"]

    log(f"{tf}: Loaded {len(df):,} rows | {size // 1024} KB", "INFO")

    # ── 3. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        passed = False
        log(f"{tf}: Missing columns: {missing_cols}", "ERROR")
    else:
        log(f"{tf}: All required columns present", "OK")

    # ── 4. Parse datetime
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
        issues.append(f"{invalid_dt} rows with invalid datetime")
        log(f"{tf}: {invalid_dt} invalid datetime rows", "WARN")

    # ── 5. Duplicate timestamps
    dupes = df["_dt"].duplicated().sum()
    if dupes > 0:
        issues.append(f"{dupes} duplicate timestamps")
        passed = False
        log(f"{tf}: {dupes} duplicate timestamps", "ERROR")
    else:
        log(f"{tf}: No duplicate timestamps", "OK")

    # ── 6. Chronological order
    if not df["_dt"].is_monotonic_increasing:
        violations = (df["_dt"].diff() < pd.Timedelta(0)).sum()
        issues.append(f"Timestamps not sorted: {violations} violations")
        passed = False
        log(f"{tf}: Timestamps not in order — {violations} violations", "ERROR")
    else:
        log(f"{tf}: Timestamps in correct order", "OK")

    # ── 7. Time gaps
    max_gap = MAX_GAP_MINUTES.get(tf, 1440)
    if len(df) > 1:
        gaps  = df["_dt"].diff().dropna()
        large = gaps[gaps > pd.Timedelta(minutes=max_gap)]
        if len(large) > 0:
            worst = large.max()
            issues.append(f"{len(large)} gaps > {max_gap}min (worst: {worst})")
            log(f"{tf}: {len(large)} large gaps — worst: {worst}", "WARN")
        else:
            log(f"{tf}: No large time gaps", "OK")

    # ── 8. OHLC numeric
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

        # ── 9. High >= Low
        hl = (df["High"] < df["Low"]).sum()
        if hl > 0:
            issues.append(f"{hl} bars where High < Low")
            passed = False
            log(f"{tf}: {hl} bars with High < Low", "ERROR")
        else:
            log(f"{tf}: High >= Low for all bars", "OK")

        # ── 10. OHLC range
        ho = (df["High"] < df["Open"]).sum()
        hc = (df["High"] < df["Close"]).sum()
        lo = (df["Low"]  > df["Open"]).sum()
        lc = (df["Low"]  > df["Close"]).sum()
        total_violations = ho + hc + lo + lc
        if total_violations > 0:
            issues.append(f"{total_violations} OHLC range violations")
            log(f"{tf}: OHLC violations — H<O:{ho} H<C:{hc} L>O:{lo} L>C:{lc}", "WARN")
        else:
            log(f"{tf}: OHLC ranges valid", "OK")

    except Exception as e:
        issues.append(f"OHLC check error: {e}")
        log(f"{tf}: OHLC check failed: {e}", "ERROR")

    # ── 11. Session column
    if "Session" in df.columns:
        empty_session = df["Session"].isna().sum() + (df["Session"].str.strip() == "").sum()
        if empty_session > 0:
            issues.append(f"{empty_session} rows with empty Session")
            log(f"{tf}: {empty_session} rows missing Session", "WARN")
        else:
            log(f"{tf}: Session column filled", "OK")

    # ── 12. Summary statistics
    dt_from = df["_dt"].min().strftime("%Y.%m.%d")
    dt_to   = df["_dt"].max().strftime("%Y.%m.%d")
    log(f"{tf}: Range {dt_from} to {dt_to} | {len(df):,} rows | {size // 1024} KB", "INFO")

    return passed, issues


def verify_all() -> tuple[list, list, list]:
    """
    @brief  Finds and verifies all merged CSV files in MERGED_DATA_DIR.

    @return Tuple of (valid, failed, missing) timeframe lists.
    """
    if not os.path.exists(MERGED_DATA_DIR):
        log(f"Merged_Data folder not found: {MERGED_DATA_DIR}", "ERROR")
        return [], [], TIMEFRAMES[:]

    valid   = []
    failed  = []
    missing = []

    for tf in TIMEFRAMES:
        sep()
        log(f"Checking: {tf}", "STEP")

        path = find_merged_file(tf)
        if not path:
            log(f"{tf}: No merged file found in Merged_Data", "WARN")
            missing.append(tf)
            continue

        log(f"{tf}: {os.path.basename(path)}", "INFO")
        passed, issues = check_file(tf, path)

        if passed:
            log(f"{tf}: ALL CHECKS PASSED", "OK")
            valid.append(tf)
        else:
            for issue in issues:
                log(f"  Issue: {issue}", "ERROR")
            log(f"{tf}: FAILED — {len(issues)} issue(s)", "ERROR")
            failed.append(tf)

    return valid, failed, missing


if __name__ == "__main__":
    sep("STEP 6: Verify Merged CSV Files")

    valid, failed, missing = verify_all()
    total = len(TIMEFRAMES)

    sep("FINAL VERIFICATION SUMMARY")
    log(f"Passed  : {len(valid)}/{total}  {valid}",   "OK"    if len(valid) == total else "INFO")
    log(f"Failed  : {len(failed)}/{total}  {failed}",  "ERROR" if failed              else "INFO")
    log(f"Missing : {len(missing)}/{total}  {missing}", "WARN"  if missing             else "INFO")

    if len(valid) == total:
        log("All merged files passed quality checks!", "OK")
        sys.exit(0)
    elif len(valid) > 0:
        log(f"Partial: {len(valid)}/{total} timeframes passed", "WARN")
        sys.exit(2)
    else:
        log("No merged files passed — check pipeline output", "ERROR")
        sys.exit(1)