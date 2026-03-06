#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step4_merge.py
@brief   Step 4 — Merges old and new CSV files for each timeframe.

@description
    For each timeframe, this step:
      1. Reads the new CSV file exported by MT4 (from step 3).
      2. Searches OLD_DATA_DIR for an existing historical CSV for the same
         symbol and timeframe.
      3. Normalizes both files to a unified column format.
      4. Concatenates the data, removes duplicate timestamps (keeping the
         newer source's value on conflict), and sorts chronologically.
      5. Writes the merged result to OUTPUT_DIR with an auto-generated
         filename: SYMBOL_TIMEFRAME_merged_FROMDATE_TODATE.csv
      6. Calculates the trading session label (Tokyo/London/NewYork/overlap)
         from the bar timestamp if not already present in the data.

    Supported input formats are detected automatically (separator, header,
    date format). No manual configuration is needed per file.

@returns  exit code 0 if all merges succeed, 1 if all fail, 2 if partial.
"""

import sys
import os
import re
import glob
import json
from datetime import datetime

import pandas as pd

from config import SYMBOL, TIMEFRAMES, OLD_DATA_DIR, OUTPUT_DIR
from logger import log, sep

FOUND_FILES_JSON = os.path.join(os.path.dirname(__file__), "_found_files.json")

# ─────────────────────────────────────────────────────────────
# SESSION CALCULATION
# ─────────────────────────────────────────────────────────────

def get_session(dt: datetime) -> str:
    """
    @brief  Determines the trading session name for a given UTC datetime.
    @param  dt  A datetime object representing the bar open time in UTC.
    @return Session label: Tokyo | London | NewYork | London+NY |
                           Tokyo+London | Off
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
# FORMAT DETECTION & LOADING
# ─────────────────────────────────────────────────────────────

def detect_and_load(filepath: str) -> pd.DataFrame | None:
    """
    @brief  Auto-detects CSV format and loads it into a normalized DataFrame.

    @description
        Tries multiple separators (semicolon, comma, tab).
        Detects whether the file has a header row or not.
        Normalizes all column names to: Datum, Uhrzeit, Open, High, Low,
        Close, Volumen, Session, Spread_Punkte.
        Parses datetime from multiple date format variants.

    @param  filepath  Absolute path to the CSV file.
    @return Normalized DataFrame with a '_dt' column, or None on failure.
    """
    if not os.path.exists(filepath):
        log(f"Datei nicht gefunden: {filepath}", "ERROR")
        return None

    df = None
    used_sep = None

    for sep_char in [';', ',', '\t']:
        try:
            tmp = pd.read_csv(filepath, sep=sep_char, dtype=str,
                              encoding='utf-8-sig', skipinitialspace=True,
                              nrows=5)
            if len(tmp.columns) >= 6:
                df       = pd.read_csv(filepath, sep=sep_char, dtype=str,
                                       encoding='utf-8-sig', skipinitialspace=True)
                used_sep = sep_char
                break
        except Exception:
            continue

    if df is None:
        log(f"Konnte nicht laden: {filepath}", "ERROR")
        return None

    log(f"  Geladen: {len(df)} Zeilen | sep='{used_sep}' | cols={list(df.columns)}", "INFO")

    df = _normalize_columns(df)
    df = _parse_datetime(df)

    if df is None or '_dt' not in df.columns:
        log(f"  Datetime konnte nicht geparst werden: {filepath}", "ERROR")
        return None

    invalid = df['_dt'].isna().sum()
    if invalid > 0:
        log(f"  {invalid} Zeilen mit ungueltiger Zeit entfernt", "WARN")
        df = df.dropna(subset=['_dt'])

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief  Renames columns to the unified internal schema.
    @param  df  Raw DataFrame as loaded from CSV.
    @return DataFrame with standardized column names.
    """
    # No-header file: columns are 0,1,2,3,4,5,6
    if all(isinstance(c, int) for c in df.columns):
        rename = {0: 'Datum', 1: 'Uhrzeit', 2: 'Open',
                  3: 'High',  4: 'Low',     5: 'Close', 6: 'Volumen'}
        return df.rename(columns=rename)

    col_map = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl in ['datum', 'date']:                                     col_map[col] = 'Datum'
        elif cl in ['uhrzeit', 'time', 'zeit']:                         col_map[col] = 'Uhrzeit'
        elif cl in ['open', 'eroeffnungskurs', 'eröffnungskurs', 'o']:  col_map[col] = 'Open'
        elif cl in ['high', 'hoch', 'h']:                               col_map[col] = 'High'
        elif cl in ['low', 'tief', 'l']:                                col_map[col] = 'Low'
        elif cl in ['close', 'schlusskurs', 'c']:                       col_map[col] = 'Close'
        elif cl in ['volume', 'volumen', 'vol', 'volumen/ticks', 'v']:  col_map[col] = 'Volumen'
        elif cl == 'session':                                            col_map[col] = 'Session'
        elif cl in ['spread_punkte', 'spread']:                         col_map[col] = 'Spread_Punkte'

    # Single combined datetime column e.g. "Zeit" -> "02.12.2014 07:45"
    for col in df.columns:
        if str(col).strip().lower() == 'zeit':
            col_map[col] = '_zeit_combined'

    return df.rename(columns=col_map)


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief  Creates a '_dt' datetime column from available date/time columns.
    @param  df  DataFrame after column normalization.
    @return DataFrame with '_dt' column added.
    """
    formats_combined = [
        "%d.%m.%Y %H:%M", "%Y.%m.%d %H:%M",
        "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M",
        "%d.%m.%Y %H:%M:%S", "%Y.%m.%d %H:%M:%S",
    ]
    formats_date = [
        "%Y.%m.%d", "%d.%m.%Y", "%Y-%m-%d",
    ]

    # Combined column (e.g. "Zeit")
    if '_zeit_combined' in df.columns:
        for fmt in formats_combined:
            try:
                df['_dt'] = pd.to_datetime(df['_zeit_combined'].str.strip(), format=fmt)
                # Split into Datum / Uhrzeit for output
                df['Datum']   = df['_dt'].dt.strftime('%Y.%m.%d')
                df['Uhrzeit'] = df['_dt'].dt.strftime('%H:%M')
                return df
            except Exception:
                continue

    # Separate Datum + Uhrzeit columns
    if 'Datum' in df.columns and 'Uhrzeit' in df.columns:
        combined = df['Datum'].str.strip() + ' ' + df['Uhrzeit'].str.strip()
        for fmt in formats_combined:
            try:
                df['_dt'] = pd.to_datetime(combined, format=fmt)
                # Normalize Datum to YYYY.MM.DD
                df['Datum']   = df['_dt'].dt.strftime('%Y.%m.%d')
                df['Uhrzeit'] = df['_dt'].dt.strftime('%H:%M')
                return df
            except Exception:
                continue

    # Datum only (e.g. D1 bars)
    if 'Datum' in df.columns:
        for fmt in formats_date:
            try:
                df['_dt']   = pd.to_datetime(df['Datum'].str.strip(), format=fmt)
                df['Datum'] = df['_dt'].dt.strftime('%Y.%m.%d')
                df['Uhrzeit'] = df.get('Uhrzeit', pd.Series(['00:00'] * len(df)))
                return df
            except Exception:
                continue

    return df


# ─────────────────────────────────────────────────────────────
# OLD FILE SEARCH
# ─────────────────────────────────────────────────────────────

def find_old_file(timeframe: str) -> str | None:
    """
    @brief  Searches OLD_DATA_DIR for an existing CSV matching the timeframe.
    @param  timeframe  Timeframe string, e.g. 'M15', 'H1'.
    @return Absolute path to the most recently modified matching file,
            or None if not found.
    """
    if not os.path.exists(OLD_DATA_DIR):
        return None

    patterns = [
        os.path.join(OLD_DATA_DIR, f"{SYMBOL}_{timeframe}*.csv"),
        os.path.join(OLD_DATA_DIR, f"{SYMBOL}{timeframe}*.csv"),
        os.path.join(OLD_DATA_DIR, f"*{timeframe}*.csv"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None


# ─────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────

def merge_timeframe(timeframe: str, new_path: str) -> str | None:
    """
    @brief  Merges old and new CSV data for one timeframe and writes output.

    @param  timeframe  Timeframe label, e.g. 'M15'.
    @param  new_path   Path to the new CSV file exported by MT4.
    @return Path to the merged output file, or None on failure.
    """
    sep(f"Merge: {SYMBOL} {timeframe}")

    old_path = find_old_file(timeframe)
    if old_path:
        log(f"Alte Datei : {os.path.basename(old_path)}", "INFO")
    else:
        log("Keine alte Datei gefunden - nur neue Daten", "WARN")

    log(f"Neue Datei : {os.path.basename(new_path)}", "INFO")

    # Load both files
    df_new = detect_and_load(new_path)
    if df_new is None:
        return None

    frames = [df_new]
    if old_path:
        df_old = detect_and_load(old_path)
        if df_old is not None:
            frames.append(df_old)

    # Merge
    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.sort_values('_dt')
    merged = merged.drop_duplicates(subset=['_dt'], keep='last')
    merged = merged.sort_values('_dt').reset_index(drop=True)
    removed = before - len(merged)

    log(f"Zeilen: {before} -> {len(merged)} (Duplikate entfernt: {removed})", "INFO")

    # Add Session if missing
    if 'Session' not in merged.columns or merged['Session'].isna().all():
        merged['Session'] = merged['_dt'].apply(get_session)
        log("Session berechnet (UTC)", "INFO")

    # Output columns
    out_cols = ['Datum', 'Uhrzeit', 'Open', 'High', 'Low', 'Close', 'Volumen', 'Session']
    if 'Spread_Punkte' in merged.columns:
        out_cols.append('Spread_Punkte')
    out_cols = [c for c in out_cols if c in merged.columns]

    # Date range for filename
    date_from = merged['_dt'].min().strftime('%Y%m%d')
    date_to   = merged['_dt'].max().strftime('%Y%m%d')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = f"{SYMBOL}_{timeframe}_merged_{date_from}_{date_to}.csv"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    merged[out_cols].to_csv(out_path, sep=';', index=False, encoding='utf-8-sig')

    log(f"Gespeichert: {out_name}", "OK")
    log(f"Zeitraum   : {merged['_dt'].min().strftime('%Y.%m.%d')} "
        f"bis {merged['_dt'].max().strftime('%Y.%m.%d')}", "OK")
    return out_path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep("SCHRITT 4: Merge alt + neu")

    # Load found files from step 3
    found = {}
    if os.path.exists(FOUND_FILES_JSON):
        with open(FOUND_FILES_JSON) as f:
            found = json.load(f)
    else:
        log("_found_files.json nicht gefunden - step3 zuerst ausfuehren!", "ERROR")
        sys.exit(1)

    results = []
    for tf in TIMEFRAMES:
        if tf not in found:
            log(f"{tf}: Keine neue Datei - uebersprungen", "WARN")
            results.append((tf, None))
            continue
        out = merge_timeframe(tf, found[tf])
        results.append((tf, out))

    sep("MERGE ERGEBNISSE")
    ok = 0
    for tf, path in results:
        if path:
            print(f"  [OK  ] {tf:6s} -> {os.path.basename(path)}")
            ok += 1
        else:
            print(f"  [FAIL] {tf:6s} -> kein Output")

    log(f"Erfolgreich: {ok}/{len(results)}", "OK" if ok == len(results) else "WARN")

    if ok == 0:
        sys.exit(1)
    elif ok < len(results):
        sys.exit(2)
    else:
        sys.exit(0)
