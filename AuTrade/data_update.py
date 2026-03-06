#!/usr/bin/env python3
"""
merge_csv.py
------------
Vergleicht und fuegt zwei oder mehr CSV-Dateien zusammen.
- Erkennt Symbol und Timeframe automatisch aus dem Dateinamen
- Entfernt Duplikate (gleiche Datum+Uhrzeit)
- Sortiert von aeltestem bis neuesten Datum
- Benennt Output automatisch: SYMBOL_TIMEFRAME_merged_VONDATE_BISDATE.csv

Verwendung:
    python merge_csv.py datei1.csv datei2.csv
    python merge_csv.py datei1.csv datei2.csv datei3.csv
    python merge_csv.py C:/pfad/zu/XAUUSD_M15_alt.csv C:/pfad/zu/XAUUSD_M15_neu.csv

Unterstuetzte Dateinamen-Formate:
    XAUUSD_M15.csv
    XAUUSD_M15_alt.csv
    XAUUSD_M15_2024.csv
    (Symbol und Timeframe werden aus dem ersten Teil erkannt)
"""

import sys
import os
import re
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────
SEPARATOR        = ';'          # Trennzeichen in den CSV-Dateien
DATE_COL         = 'Datum'      # Spaltenname Datum
TIME_COL         = 'Uhrzeit'    # Spaltenname Uhrzeit
ENCODING         = 'utf-8-sig'  # UTF-8 mit BOM (fuer Excel-Kompatibilitaet)
OUTPUT_ENCODING  = 'utf-8-sig'

# ─────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────

def log(msg, level="INFO"):
    prefix = {"INFO": "[INFO ]", "WARN": "[WARN ]", "ERROR": "[ERROR]", "OK": "[OK   ]"}
    print(f"{prefix.get(level, '[INFO ]')} {msg}")

def sep():
    print("-" * 60)

def detect_symbol_timeframe(filepath):
    """
    Erkennt Symbol und Timeframe aus dem Dateinamen.
    Beispiele:
        XAUUSD_M15.csv         -> ('XAUUSD', 'M15')
        XAUUSD_M15_alt.csv     -> ('XAUUSD', 'M15')
        Gold_H1_dukascopy.csv  -> ('Gold', 'H1')
    """
    basename = os.path.basename(filepath)
    name     = os.path.splitext(basename)[0]  # ohne .csv
    parts    = name.split('_')

    symbol    = parts[0] if len(parts) >= 1 else "UNKNOWN"
    timeframe = "UNKNOWN"

    # Timeframe-Pattern: M1, M5, M15, M30, H1, H4, D1, W1, MN
    tf_pattern = re.compile(r'^(M\d+|H\d+|D\d+|W\d+|MN)$', re.IGNORECASE)
    for part in parts[1:]:
        if tf_pattern.match(part):
            timeframe = part.upper()
            break

    return symbol, timeframe

def parse_datetime(date_str, time_str):
    """Kombiniert Datum + Uhrzeit zu datetime. Unterstuetzt mehrere Formate."""
    formats = [
        "%Y.%m.%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%d.%m.%Y %H:%M",
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]
    combined = f"{str(date_str).strip()} {str(time_str).strip()}"
    for fmt in formats:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None

def load_csv(filepath):
    """Laedt eine CSV-Datei und gibt DataFrame zurueck."""
    log(f"Lade: {filepath}")

    if not os.path.exists(filepath):
        log(f"Datei nicht gefunden: {filepath}", "ERROR")
        return None

    # Verschiedene Trennzeichen probieren
    for sep_try in [';', ',', '\t']:
        try:
            df = pd.read_csv(filepath, sep=sep_try, encoding='utf-8-sig',
                             dtype=str, skipinitialspace=True)
            if len(df.columns) >= 6:
                log(f"  -> {len(df)} Zeilen | {len(df.columns)} Spalten | Trennzeichen='{sep_try}'")
                return df, sep_try
        except Exception:
            continue

    log(f"Konnte Datei nicht laden: {filepath}", "ERROR")
    return None, None

def normalize_columns(df):
    """
    Normalisiert Spaltennamen - erkennt Datum/Uhrzeit auch wenn
    sie anders heissen (z.B. 'Date', 'Time', 'date', 'time').
    """
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ['datum', 'date']:
            col_map[col] = 'Datum'
        elif cl in ['uhrzeit', 'time', 'zeit']:
            col_map[col] = 'Uhrzeit'
        elif cl in ['open', 'eroeffnungskurs', 'eröffnungskurs', 'o']:
            col_map[col] = 'Open'
        elif cl in ['high', 'hoch', 'h']:
            col_map[col] = 'High'
        elif cl in ['low', 'tief', 'l']:
            col_map[col] = 'Low'
        elif cl in ['close', 'schlusskurs', 'c']:
            col_map[col] = 'Close'
        elif cl in ['volume', 'volumen', 'vol', 'ticks', 'volumen/ticks', 'v']:
            col_map[col] = 'Volumen'
        elif cl in ['session']:
            col_map[col] = 'Session'
        elif cl in ['spread_punkte', 'spread']:
            col_map[col] = 'Spread_Punkte'
    if col_map:
        df = df.rename(columns=col_map)
    return df

# ─────────────────────────────────────────────
# HAUPTFUNKTION
# ─────────────────────────────────────────────

def merge_files(filepaths):
    sep()
    print("  CSV MERGE - XAUUSD History Zusammenfuehren")
    sep()

    if len(filepaths) < 2:
        log("Mindestens 2 Dateien benoetigt!", "ERROR")
        log("Verwendung: python merge_csv.py datei1.csv datei2.csv")
        sys.exit(1)

    # ── Symbol & Timeframe aus erstem Dateinamen erkennen
    symbol, timeframe = detect_symbol_timeframe(filepaths[0])
    log(f"Symbol erkannt    : {symbol}")
    log(f"Timeframe erkannt : {timeframe}")
    sep()

    all_frames = []
    all_columns = None

    # ── Alle Dateien laden
    for fp in filepaths:
        result = load_csv(fp)
        if result[0] is None:
            continue
        df, detected_sep = result
        df = normalize_columns(df)

        # Datum + Uhrzeit pruefen
        if 'Datum' not in df.columns or 'Uhrzeit' not in df.columns:
            log(f"  -> Spalten 'Datum'/'Uhrzeit' nicht gefunden in: {fp}", "WARN")
            log(f"  -> Vorhandene Spalten: {list(df.columns)}", "WARN")
            continue

        # Datetime-Spalte erstellen fuer Sortierung
        df['_dt'] = df.apply(
            lambda row: parse_datetime(row['Datum'], row['Uhrzeit']), axis=1
        )

        invalid = df['_dt'].isna().sum()
        if invalid > 0:
            log(f"  -> {invalid} Zeilen mit ungueltiger Zeit uebersprungen", "WARN")
        df = df.dropna(subset=['_dt'])

        # Spalten merken (vom ersten gueltigen File)
        if all_columns is None:
            all_columns = [c for c in df.columns if c != '_dt']

        all_frames.append(df)
        log(f"  -> Zeitraum: {df['_dt'].min().strftime('%Y.%m.%d')} "
            f"bis {df['_dt'].max().strftime('%Y.%m.%d')}", "OK")

    if not all_frames:
        log("Keine gueltigen Daten geladen!", "ERROR")
        sys.exit(1)

    sep()
    log("Fuege Daten zusammen ...")

    # ── Zusammenfuehren
    merged = pd.concat(all_frames, ignore_index=True)
    total_before = len(merged)
    log(f"Gesamt Zeilen (vor Deduplizierung) : {total_before}")

    # ── Duplikate entfernen (gleiche Datum + Uhrzeit)
    # Bei Duplikaten: neuere Datei gewinnt (last = letzte im concat = neueste Quelle)
    merged = merged.sort_values('_dt')
    merged = merged.drop_duplicates(subset=['_dt'], keep='last')
    total_after = len(merged)
    removed = total_before - total_after
    log(f"Duplikate entfernt                 : {removed}")
    log(f"Zeilen nach Deduplizierung         : {total_after}")

    # ── Sortieren: aeltestes bis neustes Datum
    merged = merged.sort_values('_dt').reset_index(drop=True)

    date_from = merged['_dt'].min().strftime('%Y%m%d')
    date_to   = merged['_dt'].max().strftime('%Y%m%d')

    log(f"Zeitraum merged   : {merged['_dt'].min().strftime('%Y.%m.%d')} "
        f"bis {merged['_dt'].max().strftime('%Y.%m.%d')}")

    # ── Ausgabe-Datei benennen
    # Format: SYMBOL_TIMEFRAME_merged_VONDATE_BISDATE.csv
    output_dir  = os.path.dirname(os.path.abspath(filepaths[0]))
    output_name = f"{symbol}_{timeframe}_merged_{date_from}_{date_to}.csv"
    output_path = os.path.join(output_dir, output_name)

    # ── Spalten fuer Output vorbereiten
    # _dt Hilfsspalte entfernen, nur echte Spalten behalten
    output_cols = [c for c in all_columns if c in merged.columns]
    merged_out  = merged[output_cols]

    # ── Speichern
    merged_out.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')

    sep()
    log(f"OUTPUT Dateiname  : {output_name}", "OK")
    log(f"OUTPUT Pfad       : {output_path}", "OK")
    log(f"Zeilen gesamt     : {total_after}", "OK")
    log(f"Zeitraum          : {merged['_dt'].min().strftime('%Y.%m.%d')} "
        f"bis {merged['_dt'].max().strftime('%Y.%m.%d')}", "OK")
    sep()

    # ── Statistik pro Quelldatei
    print("\n  QUELL-VERGLEICH:")
    sep()
    for i, (fp, df) in enumerate(zip(filepaths, all_frames)):
        name = os.path.basename(fp)
        print(f"  Datei {i+1}: {name}")
        print(f"    Zeilen   : {len(df)}")
        print(f"    Von      : {df['_dt'].min().strftime('%Y.%m.%d %H:%M')}")
        print(f"    Bis      : {df['_dt'].max().strftime('%Y.%m.%d %H:%M')}")
    sep()
    print(f"\n  MERGED: {output_name}")
    print(f"  Zeilen : {total_after} | Von: {merged['_dt'].min().strftime('%Y.%m.%d')} "
          f"bis {merged['_dt'].max().strftime('%Y.%m.%d')}")
    sep()

    return output_path

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print()
        print("  Verwendung:")
        print("    python merge_csv.py DATEI1.csv DATEI2.csv")
        print("    python merge_csv.py DATEI1.csv DATEI2.csv DATEI3.csv")
        print()
        print("  Beispiele:")
        print("    python merge_csv.py XAUUSD_M15_alt.csv XAUUSD_M15_neu.csv")
        print("    python merge_csv.py C:/Users/Wael/Desktop/XAUUSD_H1_old.csv XAUUSD_H1_new.csv")
        print()
        sys.exit(0)

    input_files = sys.argv[1:]
    merge_files(input_files)