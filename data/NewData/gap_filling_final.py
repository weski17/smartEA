"""
===================================================================================
GAP FILLING - Final Version für Gold Trading ML
===================================================================================
Input:  XAUUSD_M15_full_merged.csv
Output: XAUUSD_M15_gapfree.csv (mit gefüllten Gaps)

Methode: Forward Fill für kleine Gaps, Trennung bei großen Gaps
===================================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def fill_gaps_intelligent():
    """
    Intelligentes Gap Filling für Gold Trading Daten
    """

    # ========================================================================
    # KONFIGURATION - HIER DEINE PFADE EINTRAGEN
    # ========================================================================

    DATA_DIR = ""  # Leer = aktuelles Verzeichnis, oder z.B. "C:/Users/Wael/Desktop/Projekts/smartEA/data"
    INPUT_FILE = os.path.join(DATA_DIR, "XAUUSD_M15_full_merged.csv") if DATA_DIR else "XAUUSD_M15_full_merged.csv"
    OUTPUT_FILE = os.path.join(DATA_DIR, "XAUUSD_M15_gapfree.csv") if DATA_DIR else "XAUUSD_M15_gapfree.csv"

    # Gap-Behandlung Schwellenwerte (in Stunden)
    FILL_GAPS_UNDER = 2      # Gaps kleiner 2h werden gefüllt (tägliche Pausen)
    REMOVE_GAPS_OVER = 48    # Gaps größer 48h trennen Segmente (Wochenenden)
    MIN_SEGMENT_SIZE = 50    # Minimale Segment-Größe in Zeilen (12.5h)

    # ========================================================================
    # START PROCESSING
    # ========================================================================

    print("\n" + "=" * 80)
    print("  GAP FILLING - GOLD TRADING ML")
    print("=" * 80)

    # Datei prüfen
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ FEHLER: Input-Datei nicht gefunden!")
        print(f"   Gesucht: {INPUT_FILE}")
        print(f"\n💡 Tipp: Setze DATA_DIR im Script oder lege die Datei ins aktuelle Verzeichnis")
        return

    print(f"\n✓ Input-Datei gefunden: {os.path.basename(INPUT_FILE)}")

    # ========================================================================
    # SCHRITT 1: DATEN LADEN
    # ========================================================================

    print(f"\n{'─' * 80}")
    print("► SCHRITT 1: DATEN LADEN")
    print(f"{'─' * 80}")

    df = pd.read_csv(INPUT_FILE, sep=';')

    print(f"\n✓ Datei geladen")
    print(f"  Zeilen: {len(df):,}")
    print(f"  Spalten: {list(df.columns)}")

    # Datum parsen
    df['DateTime'] = pd.to_datetime(df['Zeit'], format='%d.%m.%Y %H:%M')
    df = df.sort_values('DateTime').reset_index(drop=True)

    print(f"  Zeitraum: {df['Zeit'].iloc[0]} bis {df['Zeit'].iloc[-1]}")

    initial_rows = len(df)

    # ========================================================================
    # SCHRITT 2: GAPS ANALYSIEREN
    # ========================================================================

    print(f"\n{'─' * 80}")
    print("► SCHRITT 2: GAP-ANALYSE")
    print(f"{'─' * 80}")

    df['time_diff'] = df['DateTime'].diff()
    df['gap_hours'] = df['time_diff'].dt.total_seconds() / 3600

    # Gap kategorisieren
    df['gap_type'] = 'OK'
    df.loc[df['gap_hours'] > 0.25, 'gap_type'] = 'Small'  # > 15 Min
    df.loc[df['gap_hours'] > FILL_GAPS_UNDER, 'gap_type'] = 'Medium'
    df.loc[df['gap_hours'] > REMOVE_GAPS_OVER, 'gap_type'] = 'Large'

    gap_summary = df['gap_type'].value_counts()

    print(f"\n✓ Gap-Kategorien:")
    print(f"  OK (≤15 Min):              {gap_summary.get('OK', 0):,}")
    print(f"  Small (15m-{FILL_GAPS_UNDER}h):        {gap_summary.get('Small', 0):,} → Werden gefüllt")
    print(f"  Medium ({FILL_GAPS_UNDER}h-{REMOVE_GAPS_OVER}h):       {gap_summary.get('Medium', 0):,} → Werden gefüllt")
    print(f"  Large (>{REMOVE_GAPS_OVER}h):            {gap_summary.get('Large', 0):,} → Trennen Segmente")

    total_gaps = gap_summary.get('Small', 0) + gap_summary.get('Medium', 0) + gap_summary.get('Large', 0)

    # ========================================================================
    # SCHRITT 3: SEGMENTIERUNG (bei großen Gaps)
    # ========================================================================

    print(f"\n{'─' * 80}")
    print("► SCHRITT 3: SEGMENTIERUNG")
    print(f"{'─' * 80}")

    # Segment-ID: Erhöht sich bei jedem großen Gap
    df['segment_id'] = (df['gap_type'] == 'Large').cumsum()

    segments = df.groupby('segment_id').agg({
        'DateTime': ['min', 'max', 'count']
    })

    segments.columns = ['start', 'end', 'rows']
    segments = segments.reset_index()

    print(f"\n✓ Segmente durch große Gaps: {len(segments)}")
    print(f"  Durchschnittliche Größe: {segments['rows'].mean():.0f} Zeilen")
    print(f"  Größtes Segment: {segments['rows'].max():,} Zeilen")
    print(f"  Kleinstes Segment: {segments['rows'].min():,} Zeilen")

    # Top 5 größte Segmente zeigen
    print(f"\n  Top 5 größte Segmente:")
    top5 = segments.nlargest(5, 'rows')
    for idx, row in top5.iterrows():
        duration = (row['end'] - row['start']).total_seconds() / 3600
        print(f"    Segment {row['segment_id']:3d}: {row['rows']:6,} Zeilen ({duration:6.1f}h)")

    # Kleine Segmente filtern
    print(f"\n  Filter: Segmente >= {MIN_SEGMENT_SIZE} Zeilen behalten")
    valid_segments = segments[segments['rows'] >= MIN_SEGMENT_SIZE]['segment_id'].tolist()

    print(f"  Behalten: {len(valid_segments)} Segmente")
    print(f"  Verworfen: {len(segments) - len(valid_segments)} Segmente (zu klein)")

    df_filtered = df[df['segment_id'].isin(valid_segments)].copy()

    removed_by_filter = len(df) - len(df_filtered)
    print(f"\n✓ Durch Filter entfernt: {removed_by_filter:,} Zeilen ({removed_by_filter/len(df)*100:.2f}%)")

    # ========================================================================
    # SCHRITT 4: GAPS FÜLLEN
    # ========================================================================

    print(f"\n{'─' * 80}")
    print("► SCHRITT 4: GAPS FÜLLEN (Forward Fill)")
    print(f"{'─' * 80}")

    print(f"\nMethode: Forward Fill")
    print(f"  - Preis: Letzter bekannter Wert wird kopiert")
    print(f"  - Volume: Wird auf 0 gesetzt (Markierung!)")

    all_filled_segments = []
    total_filled_rows = 0

    for seg_id in valid_segments:
        seg_data = df_filtered[df_filtered['segment_id'] == seg_id].copy()

        # Zeitreihe von Start bis Ende in 15-Min-Schritten erstellen
        start_time = seg_data['DateTime'].min()
        end_time = seg_data['DateTime'].max()

        # Erwartete kontinuierliche Zeitreihe
        expected_times = pd.date_range(start=start_time, end=end_time, freq='15min')

        # Reindex: Fügt fehlende Zeitpunkte hinzu
        seg_data = seg_data.set_index('DateTime')
        seg_data = seg_data.reindex(expected_times)

        # Zähle gefüllte Zeilen (vor dem Füllen)
        filled_in_segment = seg_data[['Open', 'High', 'Low', 'Close']].isna().any(axis=1).sum()
        total_filled_rows += filled_in_segment

        # Forward Fill für Preise
        seg_data[['Open', 'High', 'Low', 'Close']] = seg_data[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')

        # Volume auf 0 setzen für gefüllte Zeilen (wichtig!)
        seg_data['Volume'] = seg_data['Volume'].fillna(0)

        # Zeit-Spalte neu erstellen im deutschen Format
        seg_data['Zeit'] = seg_data.index.strftime('%d.%m.%Y %H:%M')
        seg_data = seg_data.reset_index(drop=True)

        all_filled_segments.append(seg_data)

        if filled_in_segment > 0:
            print(f"  Segment {seg_id:3d}: {filled_in_segment:5,} Gaps gefüllt ({len(seg_data):6,} Zeilen total)")

    print(f"\n✓ Gesamt gefüllte Gaps: {total_filled_rows:,}")

    # Alle Segmente kombinieren
    df_final = pd.concat(all_filled_segments, ignore_index=True)

    # ========================================================================
    # SCHRITT 5: FINALE PRÜFUNG
    # ========================================================================

    print(f"\n{'─' * 80}")
    print("► SCHRITT 5: KONTINUITÄTSPRÜFUNG")
    print(f"{'─' * 80}")

    # DateTime für Prüfung
    check_times = pd.to_datetime(df_final['Zeit'], format='%d.%m.%Y %H:%M')
    time_diffs = check_times.diff()

    # Gaps innerhalb von Segmenten prüfen (sollte 0 sein)
    gaps_within_segments = (time_diffs > pd.Timedelta(minutes=16)).sum()

    # Gaps zwischen Segmenten (erwartet = Anzahl Segmente - 1)
    expected_gaps = len(valid_segments) - 1

    print(f"\n✓ Kontinuitätsprüfung:")
    print(f"  Gaps INNERHALB Segmente: {gaps_within_segments}")
    print(f"  Gaps ZWISCHEN Segmenten: {expected_gaps} (OK, das sind die großen Gaps)")

    if gaps_within_segments == 0:
        print(f"  ✅ PERFEKT - Alle Segmente sind kontinuierlich!")
    else:
        print(f"  ⚠️  {gaps_within_segments} unerwartete Gaps gefunden")

    # Statistik über gefüllte vs. echte Daten
    filled_count = (df_final['Volume'] == 0).sum()
    real_count = (df_final['Volume'] > 0).sum()

    print(f"\n✓ Daten-Zusammensetzung:")
    print(f"  Echte Daten (Volume > 0):   {real_count:,} Zeilen ({real_count/len(df_final)*100:.1f}%)")
    print(f"  Gefüllte Daten (Volume = 0): {filled_count:,} Zeilen ({filled_count/len(df_final)*100:.1f}%)")

    # ========================================================================
    # SCHRITT 6: SPEICHERN
    # ========================================================================

    print(f"\n{'─' * 80}")
    print("► SCHRITT 6: SPEICHERN")
    print(f"{'─' * 80}")

    # Nur benötigte Spalten
    df_output = df_final[['Zeit', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Speichern
    df_output.to_csv(OUTPUT_FILE, index=False, sep=';')

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    print(f"\n✓ Datei gespeichert:")
    print(f"  Pfad: {OUTPUT_FILE}")
    print(f"  Name: {os.path.basename(OUTPUT_FILE)}")
    print(f"  Größe: {file_size:.2f} MB")
    print(f"  Zeilen: {len(df_output):,}")

    print(f"\n  Zeitraum:")
    print(f"  Von: {df_output['Zeit'].iloc[0]}")
    print(f"  Bis: {df_output['Zeit'].iloc[-1]}")

    # ========================================================================
    # ZUSAMMENFASSUNG
    # ========================================================================

    print(f"\n{'=' * 80}")
    print("  ZUSAMMENFASSUNG")
    print(f"{'=' * 80}")

    print(f"\n  📊 VORHER:")
    print(f"     Datei: {os.path.basename(INPUT_FILE)}")
    print(f"     Zeilen: {initial_rows:,}")
    print(f"     Gaps: {total_gaps:,}")

    print(f"\n  ✅ NACHHER:")
    print(f"     Datei: {os.path.basename(OUTPUT_FILE)}")
    print(f"     Zeilen: {len(df_output):,}")
    print(f"     Gaps innerhalb Segmente: {gaps_within_segments}")
    print(f"     Kontinuierliche Segmente: {len(valid_segments)}")

    change_percent = ((len(df_output) - initial_rows) / initial_rows) * 100
    print(f"\n  🔍 ÄNDERUNGEN:")
    print(f"     Datenänderung: {change_percent:+.1f}%")
    print(f"     Gefüllte Gaps: {total_filled_rows:,}")
    print(f"     Entfernte Zeilen (Filter): {removed_by_filter:,}")

    print(f"\n  💡 HINWEISE:")
    print(f"     • Gefüllte Zeilen haben Volume = 0")
    print(f"     • Preise wurden mit Forward Fill interpoliert")
    print(f"     • Große Gaps (>{REMOVE_GAPS_OVER}h) trennen Segmente")
    print(f"     • Kleine Segmente (<{MIN_SEGMENT_SIZE} Zeilen) wurden entfernt")

    print(f"\n{'=' * 80}")

    if gaps_within_segments == 0:
        print("✅ PERFEKT: Alle Segmente sind 100% kontinuierlich!")
    else:
        print(f"⚠️  {gaps_within_segments} Gaps verblieben - prüfe Konfiguration")

    print(f"{'=' * 80}")

    print(f"\n🎯 BEREIT FÜR ML TRAINING!")
    print(f"   Nutze die Datei: {OUTPUT_FILE}")
    print(f"\n📝 Nächste Schritte:")
    print(f"   1. Prüfe Output-Datei")
    print(f"   2. Feature Engineering starten")
    print(f"   3. ML Model trainieren")

    # Detaillierte Statistik-Datei speichern (optional)
    stats_file = OUTPUT_FILE.replace('.csv', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GAP FILLING - DETAILLIERTE STATISTIK\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Datum: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n\n")
        f.write(f"Input:  {INPUT_FILE}\n")
        f.write(f"Output: {OUTPUT_FILE}\n\n")
        f.write(f"Konfiguration:\n")
        f.write(f"  FILL_GAPS_UNDER:  {FILL_GAPS_UNDER}h\n")
        f.write(f"  REMOVE_GAPS_OVER: {REMOVE_GAPS_OVER}h\n")
        f.write(f"  MIN_SEGMENT_SIZE: {MIN_SEGMENT_SIZE} Zeilen\n\n")
        f.write(f"Resultat:\n")
        f.write(f"  Input Zeilen:     {initial_rows:,}\n")
        f.write(f"  Output Zeilen:    {len(df_output):,}\n")
        f.write(f"  Gefüllte Gaps:    {total_filled_rows:,}\n")
        f.write(f"  Segmente:         {len(valid_segments)}\n")
        f.write(f"  Datenänderung:    {change_percent:+.1f}%\n\n")
        f.write("Segmente im Detail:\n")
        for idx, row in segments[segments['segment_id'].isin(valid_segments)].iterrows():
            duration_h = (row['end'] - row['start']).total_seconds() / 3600
            duration_d = duration_h / 24
            f.write(f"  Segment {row['segment_id']:3d}: {row['rows']:6,} Zeilen ")
            f.write(f"({duration_h:6.1f}h = {duration_d:4.1f} Tage)\n")

    print(f"\n💾 Detaillierte Statistik gespeichert: {os.path.basename(stats_file)}")


if __name__ == "__main__":
    try:
        fill_gaps_intelligent()
    except Exception as e:
        print(f"\n\n❌ FEHLER:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 Prüfe:")
        print(f"   1. Ist die Input-Datei vorhanden?")
        print(f"   2. Ist DATA_DIR korrekt gesetzt?")
        print(f"   3. Hast du Schreibrechte im Verzeichnis?")