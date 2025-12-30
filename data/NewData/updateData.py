import pandas as pd
import shutil
import os
from datetime import datetime


def update_csv_data():
    """
    Kombiniert alte und neue XAUUSD Daten mit einem Cutoff-Datum.
    Behält alte Daten vor dem Cutoff und fügt neue Daten ab dem Cutoff hinzu.
    """

    # Dateipfade - KORRIGIERT
    data_dir = r"/data"
    old_file = os.path.join(data_dir, "XAUUSD_M15_full_backup.csv")  # Alte Daten (behalten)
    new_file = r"C:\Users\Wael\Desktop\XAUUSD15.csv"  # Neue Daten (hinzufügen) - absoluter Pfad
    output_file = os.path.join(data_dir, "XAUUSD_M15_full_merged.csv")  # Ergebnis

    # Cutoff-Datum: Alles VOR diesem Datum bleibt alt, AB diesem Datum wird neu
    cutoff_date = "17.09.2025 03:00"

    try:
        print("=" * 60)
        print("CSV DATEN-MERGER FÜR XAUUSD M15")
        print("=" * 60)

        # Dateien prüfen
        print("\n=== SCHRITT 1: DATEIEN PRÜFEN ===")
        if not os.path.exists(old_file):
            print(f"❌ Alte Datei nicht gefunden: {old_file}")
            return
        if not os.path.exists(new_file):
            print(f"❌ Neue Datei nicht gefunden: {new_file}")
            return

        print(f"✓ Alte Datei gefunden: XAUUSD_M15_full_backup.csv")
        print(f"✓ Neue Datei gefunden: XAUUSD15.csv")

        # Alte Datei laden
        print("\n=== SCHRITT 2: ALTE DATEN LADEN ===")
        df_old = pd.read_csv(old_file, sep=';')
        print(f"✓ Alte Datei geladen: {len(df_old):,} Zeilen")
        print(f"  Spalten: {list(df_old.columns)}")
        print(f"  Erste Zeile: {df_old['Zeit'].iloc[0]}")
        print(f"  Letzte Zeile: {df_old['Zeit'].iloc[-1]}")

        # Neue Datei laden - erst mal analysieren
        print("\n=== SCHRITT 3: NEUE DATEN LADEN UND ANALYSIEREN ===")
        # Lade erste paar Zeilen ohne Header
        df_test = pd.read_csv(new_file, sep=',', header=None, nrows=5)
        print(f"Erste 5 Zeilen der neuen Datei (Rohformat):")
        print(df_test)
        print(f"\nAnzahl Spalten: {len(df_test.columns)}")

        # Jetzt richtig laden mit korrekter Spaltenzuordnung
        df_new = pd.read_csv(
            new_file,
            sep=',',
            header=None,
            names=['Datum', 'Zeit', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        print(f"\n✓ Neue Datei geladen: {len(df_new):,} Zeilen")
        print(f"  Spalten: {list(df_new.columns)}")
        print(f"  Erste Zeile - Datum: {df_new['Datum'].iloc[0]}, Zeit: {df_new['Zeit'].iloc[0]}")
        print(f"  Letzte Zeile - Datum: {df_new['Datum'].iloc[-1]}, Zeit: {df_new['Zeit'].iloc[-1]}")

        # Datum konvertieren
        print("\n=== SCHRITT 4: DATUMSKONVERTIERUNG ===")
        print(f"Cutoff-Datum: {cutoff_date}")

        # Alte Datei: Deutsches Format (dd.mm.yyyy HH:MM)
        df_old['Zeit_parsed'] = pd.to_datetime(df_old['Zeit'], format='%d.%m.%Y %H:%M')

        # Neue Datei: Datum und Zeit kombinieren
        df_new['DateTime_combined'] = df_new['Datum'] + ',' + df_new['Zeit']
        df_new['Zeit_parsed'] = pd.to_datetime(df_new['DateTime_combined'], format='%Y.%m.%d,%H:%M')

        # Zeit-Spalte für Ausgabe erstellen (deutsches Format)
        df_new['Zeit'] = df_new['Zeit_parsed'].dt.strftime('%d.%m.%Y %H:%M')

        # Cutoff: Deutsches Format
        cutoff = pd.to_datetime(cutoff_date, format='%d.%m.%Y %H:%M')

        print(f"✓ Alte Daten konvertiert")
        print(f"  Zeitraum: {df_old['Zeit_parsed'].min()} bis {df_old['Zeit_parsed'].max()}")
        print(f"✓ Neue Daten konvertiert")
        print(f"  Zeitraum: {df_new['Zeit_parsed'].min()} bis {df_new['Zeit_parsed'].max()}")

        # Daten splitten
        print("\n=== SCHRITT 5: DATEN NACH CUTOFF SPLITTEN ===")

        # Alle alten Daten VOR dem Cutoff behalten
        df_old_keep = df_old[df_old['Zeit_parsed'] < cutoff].copy()
        print(f"✓ Behalte {len(df_old_keep):,} alte Datensätze VOR {cutoff_date}")
        if len(df_old_keep) > 0:
            print(f"  Zeitraum: {df_old_keep['Zeit_parsed'].min()} bis {df_old_keep['Zeit_parsed'].max()}")

        # Alle neuen Daten AB dem Cutoff hinzufügen
        df_new_add = df_new[df_new['Zeit_parsed'] >= cutoff].copy()
        print(f"✓ Füge {len(df_new_add):,} neue Datensätze AB {cutoff_date} hinzu")
        if len(df_new_add) > 0:
            print(f"  Zeitraum: {df_new_add['Zeit_parsed'].min()} bis {df_new_add['Zeit_parsed'].max()}")
        else:
            print(f"  ⚠ WARNUNG: Keine neuen Daten ab {cutoff_date} gefunden!")

        # Datentypen angleichen
        print("\n=== SCHRITT 6: DATEN VORBEREITEN ===")
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_old_keep.columns:
                df_old_keep[col] = pd.to_numeric(df_old_keep[col], errors='coerce')
            if col in df_new_add.columns:
                df_new_add[col] = pd.to_numeric(df_new_add[col], errors='coerce')

        # Kombinieren
        print("\n=== SCHRITT 7: DATEN KOMBINIEREN ===")
        df_result = pd.concat([df_old_keep, df_new_add], ignore_index=True)
        print(f"✓ Daten kombiniert: {len(df_result):,} Zeilen")

        # Nach Zeit sortieren
        df_result = df_result.sort_values(by='Zeit_parsed').reset_index(drop=True)

        # Nur die benötigten Spalten behalten
        df_result = df_result[['Zeit', 'Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✓ Daten sortiert")

        # Duplikate entfernen (falls vorhanden)
        initial_count = len(df_result)
        df_result = df_result.drop_duplicates(subset=['Zeit'], keep='first')
        duplicates_removed = initial_count - len(df_result)
        if duplicates_removed > 0:
            print(f"✓ {duplicates_removed} Duplikate entfernt")

        # Speichern
        print("\n=== SCHRITT 8: SPEICHERN ===")
        df_result.to_csv(output_file, index=False, sep=';')
        print(f"✓ Datei gespeichert: XAUUSD_M15_full_merged.csv")
        print(f"✓ Gesamte Datensätze: {len(df_result):,}")

        if len(df_result) > 0:
            print(f"✓ Zeitraum: {df_result['Zeit'].iloc[0]} bis {df_result['Zeit'].iloc[-1]}")

        # Dateigrößen
        old_size = os.path.getsize(old_file) / (1024 * 1024)
        new_size = os.path.getsize(new_file) / (1024 * 1024)
        output_size = os.path.getsize(output_file) / (1024 * 1024)

        print(f"\n=== DATEIGRÖSSEN ===")
        print(f"Alte Datei: {old_size:.2f} MB")
        print(f"Neue Datei: {new_size:.2f} MB")
        print(f"Ergebnis: {output_size:.2f} MB")

        # Abschluss-Statistik
        print(f"\n{'=' * 60}")
        print(f"✅ ERFOLGREICH ABGESCHLOSSEN")
        print(f"{'=' * 60}")
        print(f"Alte Daten behalten:     {len(df_old_keep):,} Zeilen")
        print(f"Neue Daten hinzugefügt:  {len(df_new_add):,} Zeilen")
        print(f"Duplikate entfernt:      {duplicates_removed} Zeilen")
        print(f"{'─' * 60}")
        print(f"GESAMT:                  {len(df_result):,} Zeilen")
        print(f"{'=' * 60}")

        # Prüfung der Kontinuität
        print(f"\n=== DATENQUALITÄT PRÜFEN ===")
        df_result['Zeit_check'] = pd.to_datetime(df_result['Zeit'], format='%d.%m.%Y %H:%M')
        time_diff = df_result['Zeit_check'].diff()
        expected_diff = pd.Timedelta(minutes=15)
        gaps = time_diff[time_diff > expected_diff * 1.5]

        if len(gaps) > 0:
            print(f"⚠ {len(gaps)} Zeitlücken gefunden (größer als 15 Minuten):")
            for idx in gaps.index[:5]:  # Zeige erste 5 Lücken
                print(f"  - Nach {df_result['Zeit'].iloc[idx - 1]} → Lücke von {time_diff.iloc[idx]}")
            if len(gaps) > 5:
                print(f"  ... und {len(gaps) - 5} weitere Lücken")
        else:
            print(f"✓ Keine Zeitlücken gefunden - Daten sind kontinuierlich!")

        print(f"\n💡 Die zusammengeführte Datei wurde gespeichert als:")
        print(f"   {output_file}")
        print(f"\n📝 Wenn alles korrekt ist, kannst du die Datei umbenennen zu:")
        print(f"   XAUUSD_M15_full.csv")

    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    update_csv_data()