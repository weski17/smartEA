import pandas as pd
import numpy as np
import talib
import os
from pathlib import Path


def calculate_indicators(df, rsi_period=14, atr_period=14, sma_period=20, volume_ratio_period=20):
    """
    Berechnet technische Indikatoren für Trading-Daten

    Parameters:
    df: DataFrame mit OHLCV Daten
    rsi_period: RSI Periode (default: 14)
    atr_period: ATR Periode (default: 14)
    sma_period: SMA Periode (default: 20)
    volume_ratio_period: Volume Ratio Periode (default: 20)

    Returns:
    DataFrame mit ursprünglichen Daten + Indikatoren
    """

    # Kopie erstellen um Original nicht zu verändern
    result_df = df.copy()

    # Spalten normalisieren (deutsche und englische Namen unterstützen)
    column_mapping = {
        'Zeit': 'datetime',
        'Time': 'datetime',
        'Datum': 'datetime',
        'Eröffnungskurs': 'open',
        'Open': 'open',
        'Hoch': 'high',
        'High': 'high',
        'Tief': 'low',
        'Low': 'low',
        'Schlusskurs': 'close',
        'Close': 'close',
        'Volumen/Ticks': 'volume',
        'Volume': 'volume',
        'Volumen': 'volume'
    }

    # Finde die korrekten Spaltennamen
    actual_columns = {}
    for col in df.columns:
        if col in column_mapping:
            actual_columns[column_mapping[col]] = col

    print(f"Gefundene Spalten: {actual_columns}")

    # Extrahiere die benötigten Daten
    try:
        high = df[actual_columns['high']].astype(float)
        low = df[actual_columns['low']].astype(float)
        close = df[actual_columns['close']].astype(float)

        # Volume (falls nicht vorhanden, mit 1 füllen)
        if 'volume' in actual_columns:
            volume = df[actual_columns['volume']].astype(float)
        else:
            volume = pd.Series([1] * len(df), index=df.index)
            print("Warning: Volume-Spalte nicht gefunden, verwende Standardwerte")

    except KeyError as e:
        raise ValueError(f"Benötigte Spalte nicht gefunden: {e}")
    except Exception as e:
        raise ValueError(f"Fehler beim Konvertieren der Daten: {e}")

    print(f"Verarbeite {len(df)} Datensätze...")

    # 1. RSI berechnen (talib)
    print(f"Berechne RSI({rsi_period})...")
    result_df[f'RSI{rsi_period}'] = talib.RSI(close.values, timeperiod=rsi_period)

    # 2. ATR berechnen (talib)
    print(f"Berechne ATR({atr_period})...")
    result_df[f'ATR{atr_period}'] = talib.ATR(high.values, low.values, close.values, timeperiod=atr_period)

    # 3. SMA berechnen (pandas rolling)
    print(f"Berechne SMA({sma_period})...")
    result_df[f'SMA{sma_period}'] = close.rolling(window=sma_period).mean()

    # 4. Volume Ratio berechnen
    print(f"Berechne Volume Ratio({volume_ratio_period})...")
    volume_sma = volume.rolling(window=volume_ratio_period).mean()
    result_df[f'VolumeRatio{volume_ratio_period}'] = volume / volume_sma

    # 5. Returns berechnen (prozentuale Änderung)
    print("Berechne Returns(1)...")
    result_df['Returns1'] = close.pct_change()

    return result_df


def clean_data_for_quality(df, max_lookback=20):
    """
    Entfernt Datensätze ohne vollständige Indikatorenwerte

    Parameters:
    df: DataFrame mit Indikatoren
    max_lookback: Maximale Lookback-Periode (default: 20)

    Returns:
    Bereinigte DataFrame
    """

    # Entferne die ersten max_lookback Zeilen
    cleaned_df = df.iloc[max_lookback:].copy()

    # Optional: Entferne Zeilen mit NaN-Werten in Indikatoren
    indicator_columns = [col for col in cleaned_df.columns if
                         any(ind in col for ind in ['RSI', 'ATR', 'SMA', 'VolumeRatio', 'Returns'])]

    # Zähle NaN-Werte vor Bereinigung
    nan_counts_before = cleaned_df[indicator_columns].isna().sum()

    # Entferne Zeilen mit NaN in Indikatoren
    cleaned_df = cleaned_df.dropna(subset=indicator_columns)

    print(f"\nDatenbereinigung:")
    print(f"Original: {len(df)} Datensätze")
    print(f"Nach Entfernung der ersten {max_lookback} Zeilen: {len(df) - max_lookback} Datensätze")
    print(f"Nach NaN-Bereinigung: {len(cleaned_df)} Datensätze")
    print(f"Entfernte Datensätze insgesamt: {len(df) - len(cleaned_df)}")

    if not nan_counts_before.empty:
        print(f"\nNaN-Werte vor Bereinigung:")
        for col, count in nan_counts_before.items():
            if count > 0:
                print(f"  {col}: {count}")

    return cleaned_df


def process_xauusd_data(input_file, output_file=None, rsi_period=14, atr_period=14, sma_period=20,
                        volume_ratio_period=20):
    """
    Hauptfunktion zur Verarbeitung der XAUUSD CSV-Datei

    Parameters:
    input_file: Pfad zur Input CSV-Datei
    output_file: Pfad zur Output CSV-Datei (optional)
    rsi_period, atr_period, sma_period, volume_ratio_period: Indikator-Parameter

    Returns:
    DataFrame mit berechneten Indikatoren
    """

    try:
        # CSV-Datei laden
        print(f"Lade CSV-Datei: {input_file}")

        # Verschiedene Trennzeichen ausprobieren
        separators = [',', ';', '\t', '|']
        df = None

        for sep in separators:
            try:
                df = pd.read_csv(input_file, sep=sep, encoding='utf-8')
                if len(df.columns) > 1:  # Erfolgreicher Parse
                    print(f"Erfolgreich geladen mit Trennzeichen: '{sep}'")
                    break
            except:
                continue

        if df is None or len(df.columns) <= 1:
            # Fallback zu encoding-detection
            try:
                df = pd.read_csv(input_file, sep=None, engine='python', encoding='latin-1')
                print("Geladen mit automatischer Trennzeichen-Erkennung")
            except:
                raise ValueError("Konnte CSV-Datei nicht laden")

        print(f"Daten geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
        print(f"Spalten: {list(df.columns)}")
        print(f"\nErste 3 Zeilen:")
        print(df.head(3))

        # Indikatoren berechnen
        print(f"\n{'=' * 50}")
        print("BERECHNE INDIKATOREN")
        print(f"{'=' * 50}")

        df_with_indicators = calculate_indicators(
            df,
            rsi_period=rsi_period,
            atr_period=atr_period,
            sma_period=sma_period,
            volume_ratio_period=volume_ratio_period
        )

        # Daten qualitativ bereinigen
        print(f"\n{'=' * 50}")
        print("DATENBEREINIGUNG")
        print(f"{'=' * 50}")

        max_lookback = max(rsi_period, atr_period, sma_period, volume_ratio_period)
        cleaned_df = clean_data_for_quality(df_with_indicators, max_lookback)

        # Statistiken anzeigen
        print(f"\n{'=' * 50}")
        print("STATISTIKEN")
        print(f"{'=' * 50}")

        indicator_columns = [col for col in cleaned_df.columns if
                             any(ind in col for ind in ['RSI', 'ATR', 'SMA', 'VolumeRatio', 'Returns'])]

        for col in indicator_columns:
            values = cleaned_df[col].dropna()
            if len(values) > 0:
                print(f"{col}:")
                print(f"  Min: {values.min():.6f}")
                print(f"  Max: {values.max():.6f}")
                print(f"  Mean: {values.mean():.6f}")
                print(f"  Letzte 5 Werte: {values.tail().round(6).tolist()}")

        # Sample der letzten 5 Zeilen mit Indikatoren
        print(f"\n{'=' * 50}")
        print("SAMPLE DER LETZTEN 5 ZEILEN")
        print(f"{'=' * 50}")
        print(cleaned_df[indicator_columns].tail())

        # Backup-Datei speichern
        if output_file is None:
            # Erstelle automatischen Dateinamen
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_with_indicators_backup.csv"

        print(f"\nSpeichere Backup-Datei: {output_file}")
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Backup erfolgreich gespeichert!")

        return cleaned_df

    except Exception as e:
        print(f"❌ Fehler bei der Verarbeitung: {e}")
        raise


# Beispiel-Verwendung:
if __name__ == "__main__":
    # Pfad zur CSV-Datei anpassen
    input_file = r"C:\Users\Wael\Desktop\Projekts\smartEA\data\XAUUSD_M15_full_backup.csv"

    # Überprüfen ob Datei existiert
    if os.path.exists(input_file):
        print(f"✅ Datei gefunden: {input_file}")

        # Verarbeitung starten
        try:
            result_df = process_xauusd_data(
                input_file=input_file,
                rsi_period=14,
                atr_period=14,
                sma_period=20,
                volume_ratio_period=20
            )

            print(f"\n🎉 Verarbeitung abgeschlossen!")
            print(f"Finale Datensätze: {len(result_df)}")

        except Exception as e:
            print(f"❌ Fehler: {e}")
    else:
        print(f"❌ Datei nicht gefunden: {input_file}")
        print("Bitte passen Sie den Pfad entsprechend an.")


# Zusätzliche Funktionen für erweiterte Analyse
def validate_indicators(df):
    """Validiert die berechneten Indikatoren"""

    print(f"\n{'=' * 50}")
    print("INDIKATOR VALIDIERUNG")
    print(f"{'=' * 50}")

    indicator_columns = [col for col in df.columns if
                         any(ind in col for ind in ['RSI', 'ATR', 'SMA', 'VolumeRatio', 'Returns'])]

    for col in indicator_columns:
        values = df[col].dropna()

        # RSI sollte zwischen 0 und 100 liegen
        if 'RSI' in col:
            invalid_rsi = values[(values < 0) | (values > 100)]
            if len(invalid_rsi) > 0:
                print(f"⚠️ {col}: {len(invalid_rsi)} Werte außerhalb 0-100 Bereich")
            else:
                print(f"✅ {col}: Alle Werte im gültigen Bereich (0-100)")

        # ATR sollte positiv sein
        elif 'ATR' in col:
            invalid_atr = values[values < 0]
            if len(invalid_atr) > 0:
                print(f"⚠️ {col}: {len(invalid_atr)} negative Werte")
            else:
                print(f"✅ {col}: Alle Werte positiv")

        # Volume Ratio sollte positiv sein
        elif 'VolumeRatio' in col:
            invalid_vol = values[values <= 0]
            if len(invalid_vol) > 0:
                print(f"⚠️ {col}: {len(invalid_vol)} nicht-positive Werte")
            else:
                print(f"✅ {col}: Alle Werte positiv")

        # Returns: extreme Werte prüfen
        elif 'Returns' in col:
            extreme_returns = values[(values > 1) | (values < -1)]  # >100% oder <-100%
            if len(extreme_returns) > 0:
                print(f"⚠️ {col}: {len(extreme_returns)} extreme Werte (>100% oder <-100%)")
                print(f"   Extreme Werte: {extreme_returns.tolist()}")
            else:
                print(f"✅ {col}: Keine extremen Werte")


def export_summary_statistics(df, output_dir=None, output_file="indicator_statistics.csv"):
    """Exportiert zusammenfassende Statistiken der Indikatoren"""

    if output_dir is None:
        output_dir = r"C:\Users\Wael\Desktop\Projekts\smartEA\data\indicators"

    # Stelle sicher, dass das Verzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)

    full_output_path = Path(output_dir) / output_file

    indicator_columns = [col for col in df.columns if
                         any(ind in col for ind in ['RSI', 'ATR', 'SMA', 'VolumeRatio', 'Returns'])]

    stats_df = df[indicator_columns].describe()
    stats_df.to_csv(full_output_path)
    print(f"📊 Statistiken exportiert nach: {full_output_path}")

    return stats_df