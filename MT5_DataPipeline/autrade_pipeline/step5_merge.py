#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step5_merge.py
@brief   Step 5 — Merges Old_Data and New_Data CSV files for each timeframe.

@description
    For each timeframe this step:
      1. Loads the old merged CSV from Old_Data (full historical dataset).
      2. Loads the new CSV from New_Data (fresh MT5 export).
      3. Determines the cutoff date:
           cutoff = last timestamp in Old_Data - 1 bar (safety overlap)
      4. Filters New_Data to only include rows FROM the cutoff onwards.
         This avoids re-processing years of already-merged data.
      5. Concatenates Old_Data + filtered New_Data.
      6. Removes duplicate timestamps (newer source wins on conflict).
      7. Sorts chronologically oldest to newest.
      8. Calculates trading session label if not present.
      9. Saves merged result to Merged_Data with auto-generated filename:
           SYMBOL_TIMEFRAME_merged_FROMDATE_TODATE.csv

    Why 1 bar safety overlap?
      The last bar of Old_Data may have been incomplete or revised.
      By including 1 bar before the cutoff from New_Data, we ensure
      that any corrected values are captured. Duplicates are then
      removed automatically keeping the newer value.

@returns
    exit code 0 — all timeframes merged successfully
    exit code 2 — partial success
    exit code 1 — all merges failed
"""

import sys
import os
import glob
import json
from datetime import datetime

import pandas as pd

from config import SYMBOL, TIMEFRAMES, OLD_DATA_DIR, MERGED_DATA_DIR
from logger import log, sep

FOUND_FILES_JSON = os.path.join(os.path.dirname(__file__), "_found_files.json")

# Required output columns
OUTPUT_COLUMNS = ["Datum", "Uhrzeit", "Open", "High", "Low",
                  "Close", "Volumen", "Session", "Spread_Punkte"]


# ─────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────

def get_session(dt: datetime) -> str:
    """
    @brief  Returns the trading session name for a UTC datetime.
    @param  dt  Bar open time as datetime (UTC).
    @return Tokyo | London | NewYork | London+NY | Tokyo+London | Off
    """
    mins    = dt.hour * 60 + dt.minute
    tokyo   = 0   <= mins < 540
    london  = 420 <= mins < 960
    newyork = 720 <= mins < 1260
    if london and newyork: return "London+NY"
    if tokyo  and london:  return "Tokyo+London"
    if newyork:            return "NewYork"
    if london:             return "London"
    if tokyo:              return "Tokyo"
    return "Off"


# ─────────────────────────────────────────────────────────────
# CSV LOADING
# ─────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame | None:
    """
    @brief  Auto-detects CSV format and loads it into a normalized DataFrame.

    @description
        Tries semicolon, comma, and tab as separators.
        Normalizes column names to unified schema.
        Parses datetime and adds '_dt' column.

    @param  filepath  Absolute path to the CSV file.
    @return Normalized DataFrame with '_dt' column, or None on failure.
    """
    for sep_char in [';', ',', '\t']:
        try:
            df = pd.read_csv(filepath, sep=sep_char, dtype=str,
                             encoding='utf-8-sig', skipinitialspace=True)
            if len(df.columns) >= 6:
                df = _normalize_columns(df)
                df = _parse_datetime(df)
                if df is not None and '_dt' in df.columns:
                    before = len(df)
                    df     = df.dropna(subset=['_dt'])
                    if before - len(df) > 0:
                        log(f"  Dropped {before - len(df)} rows with invalid datetime", "WARN")
                    return df
        except Exception:
            continue
    log(f"  Cannot load: {filepath}", "ERROR")
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief  Maps raw column names to the unified internal schema.
    @param  df  Raw DataFrame as loaded from CSV.
    @return DataFrame with standardized column names.
    """
    if all(isinstance(c, int) for c in df.columns):
        return df.rename(columns={
            0: 'Datum', 1: 'Uhrzeit', 2: 'Open',
            3: 'High',  4: 'Low',     5: 'Close', 6: 'Volumen'
        })

    col_map = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if   cl in ['datum', 'date']:                                    col_map[col] = 'Datum'
        elif cl in ['uhrzeit', 'time']:                                  col_map[col] = 'Uhrzeit'
        elif cl in ['zeit'] and 'Datum' not in col_map.values():         col_map[col] = '_zeit_combined'
        elif cl in ['open', 'o']:                                        col_map[col] = 'Open'
        elif cl in ['high', 'h']:                                        col_map[col] = 'High'
        elif cl in ['low', 'l']:                                         col_map[col] = 'Low'
        elif cl in ['close', 'c']:                                       col_map[col] = 'Close'
        elif cl in ['volume', 'volumen', 'vol', 'volumen/ticks', 'v']:   col_map[col] = 'Volumen'
        elif cl == 'session':                                             col_map[col] = 'Session'
        elif cl in ['spread_punkte', 'spread']:                          col_map[col] = 'Spread_Punkte'
    return df.rename(columns=col_map)


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    @brief  Parses date/time columns into a unified '_dt' datetime column.
    @param  df  DataFrame after column normalization.
    @return DataFrame with '_dt' column added.
    """
    fmts = [
        "%d.%m.%Y %H:%M", "%Y.%m.%d %H:%M",
        "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M",
        "%d.%m.%Y %H:%M:%S", "%Y.%m.%d %H:%M:%S",
    ]

    # Combined datetime column
    if '_zeit_combined' in df.columns:
        for fmt in fmts:
            try:
                df['_dt']     = pd.to_datetime(df['_zeit_combined'].str.strip(), format=fmt)
                df['Datum']   = df['_dt'].dt.strftime('%Y.%m.%d')
                df['Uhrzeit'] = df['_dt'].dt.strftime('%H:%M')
                return df
            except Exception:
                continue

    # Separate Datum + Uhrzeit
    if 'Datum' in df.columns and 'Uhrzeit' in df.columns:
        combined = df['Datum'].str.strip() + ' ' + df['Uhrzeit'].str.strip()
        for fmt in fmts:
            try:
                df['_dt']     = pd.to_datetime(combined, format=fmt)
                df['Datum']   = df['_dt'].dt.strftime('%Y.%m.%d')
                df['Uhrzeit'] = df['_dt'].dt.strftime('%H:%M')
                return df
            except Exception:
                continue

    # Date only (D1)
    if 'Datum' in df.columns:
        for fmt in ["%Y.%m.%d", "%d.%m.%Y", "%Y-%m-%d"]:
            try:
                df['_dt']     = pd.to_datetime(df['Datum'].str.strip(), format=fmt)
                df['Datum']   = df['_dt'].dt.strftime('%Y.%m.%d')
                df['Uhrzeit'] = '00:00'
                return df
            except Exception:
                continue

    return df


# ─────────────────────────────────────────────────────────────
# OLD FILE SEARCH
# ─────────────────────────────────────────────────────────────

def find_old_file(timeframe: str) -> str | None:
    """
    @brief  Searches Old_Data for a CSV matching symbol and timeframe.
    @param  timeframe  Timeframe string e.g. 'M15'.
    @return Path to the most recently modified matching file, or None.
    """
    if not os.path.exists(OLD_DATA_DIR):
        return None
    for pattern in [
        os.path.join(OLD_DATA_DIR, f"{SYMBOL}_{timeframe}*.csv"),
        os.path.join(OLD_DATA_DIR, f"{SYMBOL}{timeframe}*.csv"),
        os.path.join(OLD_DATA_DIR, f"*{timeframe}*.csv"),
    ]:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None


# ─────────────────────────────────────────────────────────────
# MERGE ONE TIMEFRAME
# ─────────────────────────────────────────────────────────────

def merge_timeframe(timeframe: str, new_path: str) -> str | None:
    """
    @brief  Merges Old_Data and New_Data for one timeframe.

    @description
        Loads old CSV from Old_Data and new CSV from New_Data.
        Calculates cutoff = last timestamp of old data - 1 bar.
        Filters new data to only rows >= cutoff (safety overlap).
        Merges, deduplicates, sorts, adds Session, saves to Merged_Data.

    @param  timeframe  Timeframe label e.g. 'M15'.
    @param  new_path   Path to new CSV in New_Data.
    @return Path to merged output file, or None on failure.
    """
    log(f"New file : {os.path.basename(new_path)}", "INFO")

    # ── Load new data
    df_new = load_csv(new_path)
    if df_new is None or df_new.empty:
        log(f"{timeframe}: Cannot load new file", "ERROR")
        return None
    log(f"New data : {len(df_new):,} rows | "
        f"{df_new['_dt'].min().strftime('%Y.%m.%d')} to "
        f"{df_new['_dt'].max().strftime('%Y.%m.%d')}", "INFO")

    # ── Load old data
    old_path = find_old_file(timeframe)
    df_old   = None

    if old_path:
        log(f"Old file : {os.path.basename(old_path)}", "INFO")
        df_old = load_csv(old_path)

    if df_old is None or df_old.empty:
        log(f"{timeframe}: No old data — using new data only", "WARN")
        merged = df_new.copy()
    else:
        log(f"Old data : {len(df_old):,} rows | "
            f"{df_old['_dt'].min().strftime('%Y.%m.%d')} to "
            f"{df_old['_dt'].max().strftime('%Y.%m.%d')}", "INFO")

        # ── Calculate cutoff: last bar of old data - 1 bar (safety overlap)
        old_last   = df_old['_dt'].max()
        old_sorted = df_old.sort_values('_dt')

        # Get the bar just before the last bar as cutoff
        if len(old_sorted) >= 2:
            cutoff = old_sorted['_dt'].iloc[-2]  # 1 bar before last
        else:
            cutoff = old_last

        log(f"Old data ends  : {old_last.strftime('%Y.%m.%d %H:%M')}", "INFO")
        log(f"Cutoff (1 bar) : {cutoff.strftime('%Y.%m.%d %H:%M')}", "INFO")

        # ── Filter new data from cutoff onwards
        df_new_filtered = df_new[df_new['_dt'] >= cutoff].copy()
        skipped         = len(df_new) - len(df_new_filtered)

        log(f"New data filtered: {len(df_new_filtered):,} rows kept | "
            f"{skipped:,} rows skipped (before cutoff)", "INFO")

        if df_new_filtered.empty:
            log(f"{timeframe}: No new data after cutoff — old data is already up to date", "WARN")
            merged = df_old.copy()
        else:
            # ── Merge old + filtered new
            merged = pd.concat([df_old, df_new_filtered], ignore_index=True)

    # ── Deduplicate and sort
    before = len(merged)
    merged = merged.sort_values('_dt')
    merged = merged.drop_duplicates(subset=['_dt'], keep='last')
    merged = merged.sort_values('_dt').reset_index(drop=True)
    removed = before - len(merged)
    log(f"After merge: {len(merged):,} rows | {removed} duplicates removed", "INFO")

    # ── Add Session if missing
    if 'Session' not in merged.columns or merged['Session'].isna().all():
        merged['Session'] = merged['_dt'].apply(get_session)
        log("Session calculated (UTC)", "INFO")

    # ── Select output columns
    out_cols = [c for c in OUTPUT_COLUMNS if c in merged.columns]

    # ── Generate output filename
    date_from = merged['_dt'].min().strftime('%Y%m%d')
    date_to   = merged['_dt'].max().strftime('%Y%m%d')
    out_name  = f"{SYMBOL}_{timeframe}_merged_{date_from}_{date_to}.csv"
    out_path  = os.path.join(MERGED_DATA_DIR, out_name)

    os.makedirs(MERGED_DATA_DIR, exist_ok=True)
    merged[out_cols].to_csv(out_path, sep=';', index=False, encoding='utf-8-sig')

    log(f"Saved    : {out_name}", "OK")
    log(f"Range    : {merged['_dt'].min().strftime('%Y.%m.%d')} to "
        f"{merged['_dt'].max().strftime('%Y.%m.%d')}", "OK")
    return out_path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep("STEP 5: Merge Old_Data + New_Data")

    if not os.path.exists(FOUND_FILES_JSON):
        log(f"_found_files.json not found — run Step 2 first!", "ERROR")
        sys.exit(1)

    with open(FOUND_FILES_JSON, "r", encoding="utf-8") as f:
        found = json.load(f)

    results = []
    for tf in TIMEFRAMES:
        sep()
        log(f"--- {tf} ---", "STEP")
        if tf not in found:
            log(f"{tf}: not in export manifest — skipped", "WARN")
            results.append((tf, None))
            continue
        out = merge_timeframe(tf, found[tf])
        results.append((tf, out))

    sep("MERGE SUMMARY")
    ok = 0
    for tf, path in results:
        if path:
            log(f"[OK  ] {tf:5s} -> {os.path.basename(path)}", "OK")
            ok += 1
        else:
            log(f"[FAIL] {tf:5s} -> no output", "ERROR")

    log(f"Result: {ok}/{len(results)} timeframes merged successfully",
        "OK" if ok == len(results) else "WARN")

    if ok == 0:       sys.exit(1)
    elif ok < len(results): sys.exit(2)
    else:             sys.exit(0)