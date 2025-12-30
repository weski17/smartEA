import pandas as pd
import os
from datetime import datetime


def create_intraday_dataset():
    """
    Erstellt ein sauberes Intraday-Dataset (nur Mo-Fr) ohne Wochenend-Gaps
    """

    # Dateipfade
    data_dir = r"C:\Users\Wael\Desktop\Projekts\smartEA\data"
    input_file = os.path.join(data_dir, "XAUUSD_M15_full_merged.csv")  # Quelle (mit Gaps)
    output_file = os.path.join(data_dir, "XAUUSD_M15_intraday.csv")  # Ziel (ohne Gaps)

    try:
        print("=" * 60)
        print("INTRADAY-DATASET ERSTELLEN (NUR MO-FR)")
        print("=" * 60)

        # Datei laden
        print("\n=== SCHRITT 1: DATEN LADEN ===")
        if not os.path.exists(input_file):
            print(f"❌ Datei nicht gefunden: {input_file}")
            return

        df = pd.read_csv(input_file, sep=';')
        print(f"✓ Datei geladen: {len(df):,} Zeilen")
        print(f"  Zeitraum: {df['Zeit'].iloc[0]} bis {df['Zeit'].iloc[-1]}")

        # Datum parsen
        print("\n=== SCHRITT 2: DATUM KONVERTIEREN ===")
        df['Zeit_parsed'] = pd.to_datetime(df['Zeit'], format='%d.%m.%Y %H:%M')
        df['weekday'] = df['Zeit_parsed'].dt.weekday  # 0=Montag, 6=Sonntag

        print(f"✓ Datum konvertiert")

        # Nur Wochentage (Mo-Fr) behalten
        print("\n=== SCHRITT 3: WOCHENENDEN ENTFERNEN ===")
        df_intraday = df[df['weekday'] < 5].copy()  # 0-4 = Mo-Fr

        removed = len(df) - len(df_intraday)
        print(f"✓ Wochenend-Daten entfernt: {removed:,} Zeilen")
        print(f"✓ Verbleibend: {len(df_intraday):,} Zeilen (nur Mo-Fr)")
        print(f"  Anteil behalten: {len(df_intraday) / len(df) * 100:.1f}%")

        # Optional: Erste/letzte Stunden entfernen
        print("\n=== SCHRITT 4: RANDZEITEN PRÜFEN (OPTIONAL) ===")
        df_intraday['hour'] = df_intraday['Zeit_parsed'].dt.hour

        # Zeige Statistik
        monday_morning = len(df_intraday[(df_intraday['weekday'] == 0) & (df_intraday['hour'] < 10)])
        friday_evening = len(df_intraday[(df_intraday['weekday'] == 4) & (df_intraday['hour'] >= 20)])

        print(f"  Montag 08:00-10:00: {monday_morning:,} Zeilen")
        print(f"  Freitag 20:00-22:00: {friday_evening:,} Zeilen")
        print(f"  💡 Diese könnten optional auch entfernt werden (volatil)")

        # Aufräumen
        df_intraday = df_intraday[['Zeit', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Gap-Prüfung
        print("\n=== SCHRITT 5: DATENQUALITÄT PRÜFEN ===")
        df_intraday['Zeit_check'] = pd.to_datetime(df_intraday['Zeit'], format='%d.%m.%Y %H:%M')
        time_diff = df_intraday['Zeit_check'].diff()
        expected_diff = pd.Timedelta(minutes=15)

        # Große Lücken (>4h = wahrscheinlich Wochenende oder Feiertag)
        large_gaps = time_diff[time_diff > pd.Timedelta(hours=4)]

        if len(large_gaps) > 0:
            print(f"⚠ {len(large_gaps)} große Lücken gefunden (>4h):")
            for idx in large_gaps.index[:5]:
                gap_hours = time_diff.iloc[idx].total_seconds() / 3600
                print(f"  - Nach {df_intraday['Zeit'].iloc[idx - 1]} → {gap_hours:.1f}h Lücke")
            if len(large_gaps) > 5:
                print(f"  ... und {len(large_gaps) - 5} weitere")
            print(f"  💡 Das sind wahrscheinlich Feiertage (normal)")
        else:
            print(f"✓ Keine großen Lücken - Daten sind kontinuierlich!")

        # Speichern
        print("\n=== SCHRITT 6: SPEICHERN ===")
        df_intraday_save = df_intraday[['Zeit', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df_intraday_save.to_csv(output_file, index=False, sep=';')

        print(f"✓ Datei gespeichert: XAUUSD_M15_intraday.csv")
        print(f"✓ Zeilen: {len(df_intraday_save):,}")

        # Dateigröße
        input_size = os.path.getsize(input_file) / (1024 * 1024)
        output_size = os.path.getsize(output_file) / (1024 * 1024)

        print(f"\n=== DATEIGRÖSSEN ===")
        print(f"Original (mit Gaps):    {input_size:.2f} MB")
        print(f"Intraday (ohne Gaps):   {output_size:.2f} MB")
        print(f"Reduzierung:            {(1 - output_size / input_size) * 100:.1f}%")

        # Zusammenfassung
        print(f"\n{'=' * 60}")
        print(f"✅ ERFOLGREICH ABGESCHLOSSEN")
        print(f"{'=' * 60}")
        print(f"Original-Daten:          {len(df):,} Zeilen")
        print(f"Wochenenden entfernt:    {removed:,} Zeilen")
        print(f"{'─' * 60}")
        print(f"INTRADAY-DATASET:        {len(df_intraday_save):,} Zeilen")
        print(f"{'=' * 60}")

        print(f"\n💡 Das Intraday-Dataset wurde gespeichert als:")
        print(f"   {output_file}")
        print(f"\n📝 Nutze diese Datei für Training (ohne Wochenend-Gaps)")

    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_intraday_dataset()