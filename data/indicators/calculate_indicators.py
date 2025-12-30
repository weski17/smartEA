import pandas as pd
import numpy as np
import talib
import os
from pathlib import Path


def calculate_indicators(df, rsi_period=14, atr_period=14, ema_fast=20, ema_slow=50,
                         macd_fast=12, macd_slow=26, macd_signal=9):
    """
    Berechnet die 6 ausgewählten technischen Indikatoren für 15-Min Trading

    Indikatoren:
    1. MACD (Trend)
    2. EMA 20/50 (Trend-Filter)
    3. RSI (Momentum)
    4. OBV (Volumen)
    5. ATR (Volatilität)
    6. Pivot Points (Preis)

    Parameters:
    df: DataFrame mit OHLCV Daten
    rsi_period: RSI Periode (default: 14)
    atr_period: ATR Periode (default: 14)
    ema_fast: Schnelle EMA (default: 20)
    ema_slow: Langsame EMA (default: 50)
    macd_fast: MACD Fast Period (default: 12)
    macd_slow: MACD Slow Period (default: 26)
    macd_signal: MACD Signal Period (default: 9)

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
        open_price = df[actual_columns['open']].astype(float)

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

    # ========================================
    # 1. RSI (MOMENTUM)
    # ========================================
    print(f"Berechne RSI({rsi_period})...")
    result_df[f'RSI_{rsi_period}'] = talib.RSI(close.values, timeperiod=rsi_period)

    # ========================================
    # 2. ATR (VOLATILITÄT)
    # ========================================
    print(f"Berechne ATR({atr_period})...")
    result_df[f'ATR_{atr_period}'] = talib.ATR(high.values, low.values, close.values, timeperiod=atr_period)

    # ========================================
    # 3. MACD (TREND)
    # ========================================
    print(f"Berechne MACD({macd_fast},{macd_slow},{macd_signal})...")
    macd, macd_signal_line, macd_hist = talib.MACD(
        close.values,
        fastperiod=macd_fast,
        slowperiod=macd_slow,
        signalperiod=macd_signal
    )
    result_df['MACD_Line'] = macd
    result_df['MACD_Signal'] = macd_signal_line
    result_df['MACD_Histogram'] = macd_hist

    # ========================================
    # 4. EMA 20/50 (TREND-FILTER)
    # ========================================
    print(f"Berechne EMA({ema_fast}) und EMA({ema_slow})...")
    result_df[f'EMA_{ema_fast}'] = talib.EMA(close.values, timeperiod=ema_fast)
    result_df[f'EMA_{ema_slow}'] = talib.EMA(close.values, timeperiod=ema_slow)

    # EMA Crossover Signal (1 = bullish, -1 = bearish, 0 = neutral)
    result_df['EMA_CrossOver'] = 0
    result_df.loc[result_df[f'EMA_{ema_fast}'] > result_df[f'EMA_{ema_slow}'], 'EMA_CrossOver'] = 1
    result_df.loc[result_df[f'EMA_{ema_fast}'] < result_df[f'EMA_{ema_slow}'], 'EMA_CrossOver'] = -1

    # ========================================
    # 5. OBV (VOLUMEN)
    # ========================================
    print("Berechne OBV (On-Balance Volume)...")
    result_df['OBV'] = talib.OBV(close.values, volume.values)

    # OBV Momentum (prozentuale Änderung über 5 Perioden)
    result_df['OBV_Momentum'] = result_df['OBV'].pct_change(periods=5)

    # ========================================
    # 6. PIVOT POINTS (PREIS)
    # ========================================
    print("Berechne Pivot Points...")
    # Standard Pivot Points (täglich)
    # PP = (High + Low + Close) / 3
    # R1 = 2*PP - Low
    # S1 = 2*PP - High
    # R2 = PP + (High - Low)
    # S2 = PP - (High - Low)

    result_df['Pivot_Point'] = (high + low + close) / 3
    result_df['Resistance_1'] = 2 * result_df['Pivot_Point'] - low
    result_df['Support_1'] = 2 * result_df['Pivot_Point'] - high
    result_df['Resistance_2'] = result_df['Pivot_Point'] + (high - low)
    result_df['Support_2'] = result_df['Pivot_Point'] - (high - low)

    # Preisposition relativ zu Pivot Points (-1 = unter S1, 0 = neutral, 1 = über R1)
    result_df['Price_Position'] = 0
    result_df.loc[close > result_df['Resistance_1'], 'Price_Position'] = 1
    result_df.loc[close < result_df['Support_1'], 'Price_Position'] = -1

    # ========================================
    # ZUSÄTZLICHE HILFSINDIKATOREN
    # ========================================

    # Returns (für ML-Training)
    print("Berechne Returns...")
    result_df['Returns_1'] = close.pct_change()

    return result_df


def clean_data_for_quality(df, max_lookback=50):
    """
    Entfernt die ersten Datensätze, damit alle Indikatoren korrekt berechnet sind

    Begründung:
    - EMA_50 benötigt mindestens 50 Perioden
    - MACD (26,12,9) benötigt ca. 35 Perioden
    - RSI_14 benötigt 14 Perioden
    - ATR_14 benötigt 14 Perioden

    Daher: Die ersten 50 Zeilen werden entfernt für korrekte Daten!

    Parameters:
    df: DataFrame mit Indikatoren
    max_lookback: Maximale Lookback-Periode (default: 50 für EMA_50)

    Returns:
    Bereinigte DataFrame nur mit vollständig berechneten Indikatoren
    """

    print(f"\n{'=' * 50}")
    print(f"BEREINIGUNG: Entferne erste {max_lookback} Zeilen")
    print(f"{'=' * 50}")

    # Entferne die ersten max_lookback Zeilen für korrekte Indikatorenwerte
    cleaned_df = df.iloc[max_lookback:].copy()

    # Liste der Indikator-Spalten
    indicator_columns = [col for col in cleaned_df.columns if any(ind in col for ind in
                                                                  ['RSI', 'ATR', 'MACD', 'EMA', 'OBV', 'Pivot',
                                                                   'Resistance', 'Support',
                                                                   'Returns', 'Price_Position', 'EMA_CrossOver',
                                                                   'OBV_Momentum'])]

    # Zähle NaN-Werte NACH der ersten Bereinigung
    nan_counts_after = cleaned_df[indicator_columns].isna().sum()

    # Entferne zusätzliche Zeilen mit NaN in kritischen Indikatoren
    critical_indicators = [col for col in indicator_columns if any(ind in col for ind in
                                                                   ['RSI', 'ATR', 'MACD_Line', 'EMA_50', 'OBV'])]

    rows_before_nan_drop = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=critical_indicators)
    rows_after_nan_drop = len(cleaned_df)

    print(f"\nBereinigungs-Statistik:")
    print(f"  Original Datensätze: {len(df)}")
    print(f"  Nach Entfernung erste {max_lookback} Zeilen: {len(df) - max_lookback}")
    print(f"  Nach NaN-Bereinigung: {rows_after_nan_drop}")
    print(f"  ➜ Gesamt entfernt: {len(df) - rows_after_nan_drop} Zeilen")
    print(f"  ➜ Verbleibend für Training: {rows_after_nan_drop} Zeilen ✅")

    if nan_counts_after.sum() > 0:
        print(f"\n⚠️ Verbleibende NaN-Werte nach Bereinigung:")
        for col, count in nan_counts_after.items():
            if count > 0:
                print(f"  {col}: {count}")
    else:
        print(f"\n✅ Keine NaN-Werte mehr in kritischen Indikatoren!")

    return cleaned_df


def validate_indicators(df):
    """Validiert die berechneten Indikatoren"""

    print(f"\n{'=' * 50}")
    print("INDIKATOR VALIDIERUNG")
    print(f"{'=' * 50}")

    # RSI sollte zwischen 0 und 100 liegen
    if 'RSI_14' in df.columns:
        rsi_values = df['RSI_14'].dropna()
        invalid_rsi = rsi_values[(rsi_values < 0) | (rsi_values > 100)]
        if len(invalid_rsi) > 0:
            print(f"⚠️ RSI: {len(invalid_rsi)} Werte außerhalb 0-100 Bereich")
        else:
            print(f"✅ RSI: Alle Werte im gültigen Bereich (0-100)")
            print(f"   Min: {rsi_values.min():.2f}, Max: {rsi_values.max():.2f}, Mean: {rsi_values.mean():.2f}")

    # ATR sollte positiv sein
    if 'ATR_14' in df.columns:
        atr_values = df['ATR_14'].dropna()
        invalid_atr = atr_values[atr_values < 0]
        if len(invalid_atr) > 0:
            print(f"⚠️ ATR: {len(invalid_atr)} negative Werte")
        else:
            print(f"✅ ATR: Alle Werte positiv")
            print(f"   Min: {atr_values.min():.6f}, Max: {atr_values.max():.6f}, Mean: {atr_values.mean():.6f}")

    # MACD Prüfung
    if 'MACD_Line' in df.columns:
        macd_values = df['MACD_Line'].dropna()
        print(f"✅ MACD Line berechnet: {len(macd_values)} Werte")
        print(f"   Min: {macd_values.min():.6f}, Max: {macd_values.max():.6f}, Mean: {macd_values.mean():.6f}")

    # EMA Crossover Check
    if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
        ema_20 = df['EMA_20'].dropna()
        ema_50 = df['EMA_50'].dropna()
        print(f"✅ EMA 20/50 berechnet")
        print(f"   EMA 20: Min: {ema_20.min():.2f}, Max: {ema_20.max():.2f}")
        print(f"   EMA 50: Min: {ema_50.min():.2f}, Max: {ema_50.max():.2f}")

        # Crossover Statistik
        if 'EMA_CrossOver' in df.columns:
            crossover = df['EMA_CrossOver'].value_counts()
            print(f"   Crossover Status:")
            print(f"     Bullish (1): {crossover.get(1, 0)}")
            print(f"     Bearish (-1): {crossover.get(-1, 0)}")
            print(f"     Neutral (0): {crossover.get(0, 0)}")

    # OBV Prüfung
    if 'OBV' in df.columns:
        obv_values = df['OBV'].dropna()
        print(f"✅ OBV berechnet: {len(obv_values)} Werte")
        print(f"   Min: {obv_values.min():.0f}, Max: {obv_values.max():.0f}")

    # Pivot Points Prüfung
    if 'Pivot_Point' in df.columns:
        pivot = df['Pivot_Point'].dropna()
        print(f"✅ Pivot Points berechnet: {len(pivot)} Werte")
        print(f"   Pivot: Min: {pivot.min():.2f}, Max: {pivot.max():.2f}")

        if 'Price_Position' in df.columns:
            pos = df['Price_Position'].value_counts()
            print(f"   Price Position:")
            print(f"     Über R1 (1): {pos.get(1, 0)}")
            print(f"     Unter S1 (-1): {pos.get(-1, 0)}")
            print(f"     Neutral (0): {pos.get(0, 0)}")


def process_xauusd_data(input_file, output_file=None, rsi_period=14, atr_period=14,
                        ema_fast=20, ema_slow=50, macd_fast=12, macd_slow=26, macd_signal=9):
    """
    Hauptfunktion zur Verarbeitung der XAUUSD CSV-Datei

    Parameters:
    input_file: Pfad zur Input CSV-Datei
    output_file: Pfad zur Output CSV-Datei (optional)
    rsi_period, atr_period, ema_fast, ema_slow: Indikator-Parameter
    macd_fast, macd_slow, macd_signal: MACD Parameter

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
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal
        )

        # Daten qualitativ bereinigen
        print(f"\n{'=' * 50}")
        print("DATENBEREINIGUNG")
        print(f"{'=' * 50}")

        max_lookback = max(rsi_period, atr_period, ema_slow, macd_slow)
        cleaned_df = clean_data_for_quality(df_with_indicators, max_lookback)

        # Validierung
        validate_indicators(cleaned_df)

        # Statistiken anzeigen
        print(f"\n{'=' * 50}")
        print("STATISTIKEN DER HAUPTINDIKATOREN")
        print(f"{'=' * 50}")

        main_indicators = ['RSI_14', 'ATR_14', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
                           'EMA_20', 'EMA_50', 'OBV', 'Pivot_Point']

        for col in main_indicators:
            if col in cleaned_df.columns:
                values = cleaned_df[col].dropna()
                if len(values) > 0:
                    print(f"\n{col}:")
                    print(f"  Count: {len(values)}")
                    print(f"  Min: {values.min():.6f}")
                    print(f"  Max: {values.max():.6f}")
                    print(f"  Mean: {values.mean():.6f}")
                    print(f"  Std: {values.std():.6f}")
                    print(f"  Letzte 3 Werte: {values.tail(3).round(6).tolist()}")

        # Sample der letzten 5 Zeilen
        print(f"\n{'=' * 50}")
        print("SAMPLE DER LETZTEN 5 ZEILEN (Hauptindikatoren)")
        print(f"{'=' * 50}")
        display_cols = [col for col in main_indicators if col in cleaned_df.columns]
        print(cleaned_df[display_cols].tail())

        # Output-Datei speichern
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_with_6indicators.csv"

        print(f"\nSpeichere Datei: {output_file}")
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.8f')
        print(f"✅ Erfolgreich gespeichert!")

        # CSV-Datei wieder einlesen und prüfen
        print("\n=== DEBUGGING CSV INHALT ===")
        test_df = pd.read_csv(output_file)
        print(f"Spalten in CSV: {list(test_df.columns)}")
        print(f"Anzahl Zeilen: {len(test_df)}")
        print(f"RSI_14 (letzte 3): {test_df['RSI_14'].tail(3).values}")
        print(f"MACD_Line (letzte 3): {test_df['MACD_Line'].tail(3).values}")

        return cleaned_df

    except Exception as e:
        print(f"❌ Fehler bei der Verarbeitung: {e}")
        import traceback
        traceback.print_exc()
        raise


def export_summary_statistics(df, output_dir=None, output_file="indicator_statistics.csv"):
    """Exportiert zusammenfassende Statistiken der Indikatoren"""

    if output_dir is None:
        output_dir = r"C:\Users\Wael\Desktop\Projekts\smartEA\data\indicators"

    # Stelle sicher, dass das Verzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)

    full_output_path = Path(output_dir) / output_file

    indicator_columns = ['RSI_14', 'ATR_14', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
                         'EMA_20', 'EMA_50', 'OBV', 'Pivot_Point', 'Resistance_1',
                         'Support_1', 'Returns_1']

    available_indicators = [col for col in indicator_columns if col in df.columns]

    stats_df = df[available_indicators].describe()
    stats_df.to_csv(full_output_path)
    print(f"📊 Statistiken exportiert nach: {full_output_path}")

    return stats_df


# Beispiel-Verwendung:
if __name__ == "__main__":
    # Pfad zur CSV-Datei anpassen
    input_file = r"/data/NewData/XAUUSD_M15_full_merged.csv"

    # Überprüfen ob Datei existiert
    if os.path.exists(input_file):
        print(f"✅ Datei gefunden: {input_file}")

        # Verarbeitung starten mit den 6 ausgewählten Indikatoren
        try:
            result_df = process_xauusd_data(
                input_file=input_file,
                rsi_period=14,
                atr_period=14,
                ema_fast=20,
                ema_slow=50,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9
            )

            print(f"\n🎉 Verarbeitung abgeschlossen!")
            print(f"Finale Datensätze: {len(result_df)}")

            # Optional: Statistiken exportieren
            export_summary_statistics(result_df)

        except Exception as e:
            print(f"❌ Fehler: {e}")
    else:
        print(f"❌ Datei nicht gefunden: {input_file}")
        print("Bitte passen Sie den Pfad entsprechend an.")