import pandas as pd
import shutil



def update_csv_data():
    # Dateipfade
    old_file = r"C:\Users\Wael\Desktop\Projekts\smartEA\data\XAUUSD_M15_full.csv"
    new_file = r"C:\Users\Wael\Desktop\Projekts\smartEA\data\XAUUSD15.csv"
    backup_file = r"C:\Users\Wael\Desktop\Projekts\smartEA\data\XAUUSD_M15_full_backup.csv"

    # Cutoff-Datum
    cutoff_date = "2025-08-25 19:00:00"

    try:
        # 1. Backup erstellen
        print("Erstelle Backup...")
        shutil.copy2(old_file, backup_file)
        print(f"✓ Backup erstellt: {backup_file}")

        # 2. Neue Datei manuell verarbeiten (da sie ein spezielles Format hat)
        print("\n=== NEUE DATEI VERARBEITEN ===")

        print("Lade neue Datei und kombiniere Datum+Zeit...")

        # Neue Datei Zeile für Zeile lesen und Datum+Zeit kombinieren
        new_data = []
        with open(new_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) >= 6:
                    # Format: 2015.01.05,12:00,1195.69,1197.40,1195.41,1196.02,1490
                    datum = parts[0]  # 2015.01.05
                    zeit = parts[1]  # 12:00
                    open_price = parts[2]
                    high_price = parts[3]
                    low_price = parts[4]
                    close_price = parts[5]
                    volume = parts[6] if len(parts) > 6 else '0'

                    # Datum und Zeit kombinieren
                    datetime_str = f"{datum} {zeit}"

                    new_data.append([datetime_str, open_price, high_price, low_price, close_price, volume])

                if line_num <= 3:
                    print(
                        f"  Zeile {line_num}: {datetime_str} -> {open_price},{high_price},{low_price},{close_price},{volume}")

        # DataFrame aus neuen Daten erstellen
        df_new = pd.DataFrame(new_data, columns=['Zeit', 'Open', 'High', 'Low', 'Close', 'Volume'])
        print(f"✓ Neue Datei geladen: {len(df_new)} Zeilen")
        print(f"✓ Beispiel neues Datum: '{df_new['Zeit'].iloc[0]}'")

        # 3. Alte Datei laden
        print("\n=== ALTE DATEI LADEN ===")
        df_old = pd.read_csv(old_file, sep=';')
        print(f"✓ Alte Datei geladen: {len(df_old)} Zeilen")
        print(f"✓ Beispiel altes Datum: '{df_old['Zeit'].iloc[0]}'")

        # 4. Datumskonvertierung
        print("\n=== DATUM-KONVERTIERUNG ===")

        # Alte Daten: Format 02.12.2014 07:45
        df_old['Zeit'] = pd.to_datetime(df_old['Zeit'], format='%d.%m.%Y %H:%M')
        print("✓ Alte Daten konvertiert")

        # Neue Daten: Format 2015.01.05 12:00
        df_new['Zeit'] = pd.to_datetime(df_new['Zeit'], format='%Y.%m.%d %H:%M')
        print("✓ Neue Daten konvertiert")

        # 5. Daten kombinieren
        print("\n=== DATEN KOMBINIEREN ===")

        cutoff = pd.to_datetime(cutoff_date)
        print(f"Cutoff-Datum: {cutoff}")

        # Alte Daten vor Cutoff behalten
        df_old_keep = df_old[df_old['Zeit'] < cutoff].copy()
        print(f"Behalte {len(df_old_keep)} alte Datensätze vor {cutoff_date}")

        # Neue Daten ab Cutoff hinzufügen
        df_new_add = df_new[df_new['Zeit'] >= cutoff].copy()
        print(f"Füge {len(df_new_add)} neue Datensätze ab {cutoff_date} hinzu")

        # Prüfen ob neue Daten im richtigen Zeitraum sind
        if len(df_new_add) > 0:
            earliest_new = df_new_add['Zeit'].min()
            latest_new = df_new_add['Zeit'].max()
            print(f"Neue Daten Zeitraum: {earliest_new} bis {latest_new}")
        else:
            print("⚠ Keine neuen Daten im Zeitraum ab 2025-08-25 19:00 gefunden!")
            print(f"Neue Daten Zeitraum: {df_new['Zeit'].min()} bis {df_new['Zeit'].max()}")

        # Datentypen angleichen
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_old_keep[col] = pd.to_numeric(df_old_keep[col], errors='coerce')
            df_new_add[col] = pd.to_numeric(df_new_add[col], errors='coerce')

        # Kombinieren
        df_result = pd.concat([df_old_keep, df_new_add], ignore_index=True)
        df_result = df_result.sort_values(by='Zeit').reset_index(drop=True)

        # Datum zurück ins ursprüngliche Format
        df_result['Zeit'] = df_result['Zeit'].dt.strftime('%d.%m.%Y %H:%M')

        # 6. Speichern
        print("\n=== SPEICHERN ===")

        df_result.to_csv(old_file, index=False, sep=';')

        print(f"✓ Datei gespeichert: {old_file}")
        print(f"✓ Gesamte Datensätze: {len(df_result)}")

        if len(df_result) > 0:
            print(f"✓ Zeitraum: {df_result['Zeit'].iloc[0]} bis {df_result['Zeit'].iloc[-1]}")

        # Statistik
        print(f"\n=== STATISTIK ===")
        print(f"Alte Daten behalten: {len(df_old_keep)}")
        print(f"Neue Daten hinzugefügt: {len(df_new_add)}")
        print(f"Gesamt: {len(df_result)}")

        print("\n=== FERTIG ===")

    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    update_csv_data()