import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
import json


class XAUUSDProcessor:
    """
    Komplette XAUUSD Datenverarbeitung in einer Klasse
    - Laden, Bereinigen, Feature Engineering, ML-Vorbereitung
    - Update-Funktion für neue Daten
    - Backup und Versionierung
    """

    def __init__(self, input_path="data/XAUUSD_M15_full_backup.csv", output_dir="data/prepared"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.scaler = None
        self.metadata = {}

        # Verzeichnisse erstellen
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        print(f"🔧 XAUUSD Processor initialisiert")
        print(f"📂 Input: {input_path}")
        print(f"💾 Output: {output_dir}")

    def process(self, create_backup=True):
        """
        🚀 HAUPTFUNKTION: Komplette Datenverarbeitung
        """
        print(f"\n{'=' * 60}")
        print(f"🚀 XAUUSD DATENVERARBEITUNG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        try:
            # Backup erstellen
            if create_backup:
                self._create_backup()

            # Pipeline ausführen
            df_raw = self._load_data()
            df_clean = self._clean_data(df_raw)
            df_features = self._engineer_features(df_clean)
            df_final = self._create_targets(df_features)
            X_scaled, y = self._prepare_ml_data(df_final)
            file_paths = self._save_all(df_final, X_scaled, y)

            # Erfolgsmeldung
            self._print_success_summary()
            return file_paths

        except Exception as e:
            print(f"\n❌ FEHLER: {str(e)}")
            self._print_troubleshooting()
            raise

    def update(self, new_input_path=None):
        """
        🔄 UPDATE: Neue Daten verarbeiten
        """
        if new_input_path:
            self.input_path = new_input_path

        print(f"🔄 UPDATE-MODUS: Verarbeite {self.input_path}")
        return self.process(create_backup=True)

    def load_processed(self):
        """
        📂 Lädt bereits verarbeitete ML-Daten
        """
        try:
            files = {
                'features': os.path.join(self.output_dir, 'features_scaled.csv'),
                'targets': os.path.join(self.output_dir, 'targets.csv'),
                'scaler': os.path.join(self.output_dir, 'scaler.joblib'),
                'metadata': os.path.join(self.output_dir, 'metadata.json')
            }

            # Prüfen ob Dateien existieren
            missing = [name for name, path in files.items() if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(f"Fehlende Dateien: {missing}. Führe zuerst process() aus.")

            # Laden
            X = pd.read_csv(files['features'], index_col=0, parse_dates=True)
            y = pd.read_csv(files['targets'], index_col=0, parse_dates=True)
            scaler = joblib.load(files['scaler'])

            with open(files['metadata'], 'r') as f:
                metadata = json.load(f)

            print(f"✅ ML-Daten geladen: {X.shape[0]} Samples, {X.shape[1]} Features")
            return X, y, scaler, metadata

        except Exception as e:
            print(f"❌ Laden fehlgeschlagen: {str(e)}")
            raise

    # =================== PRIVATE METHODEN ===================

    def _load_data(self):
        """Daten laden"""
        print(f"\n📂 Lade Daten...")

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {self.input_path}")

        df = pd.read_csv(self.input_path, header=None,
                         names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

        print(f"✅ {len(df):,} Zeilen geladen")
        return df

    def _clean_data(self, df):
        """Datenbereinigung"""
        print(f"🧹 Bereinige Daten...")

        original_len = len(df)

        # DateTime erstellen
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.set_index('DateTime', inplace=True)
        df.drop(['Date', 'Time'], axis=1, inplace=True)

        # Ungültige Werte entfernen
        df = df[(df['High'] >= df['Low']) & (df['Open'] > 0) &
                (df['Close'] > 0) & (df['Volume'] > 0)]

        # Outliers entfernen
        for col in ['Open', 'High', 'Low', 'Close']:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 3 * IQR) & (df[col] <= Q3 + 3 * IQR)]

        removed = original_len - len(df)
        print(f"✅ {removed:,} ungültige Zeilen entfernt")
        return df

    def _engineer_features(self, df):
        """Feature Engineering"""
        print(f"🔧 Erstelle Features...")

        df = df.copy()

        # Candlestick Features
        df['Range'] = df['High'] - df['Low']
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

        # Returns & Momentum
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_14'] = df['Close'] / df['Close'].shift(14) - 1

        # Volatilität
        df['Volatility_14'] = df['Returns'].rolling(14).std()
        df['Volatility_28'] = df['Returns'].rolling(28).std()

        # Moving Averages
        for period in [5, 14, 28]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'Price_vs_SMA_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']

        df['EMA_14'] = df['Close'].ewm(span=14).mean()

        # RSI
        for period in [14, 28]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Volume Features
        df['Volume_SMA'] = df['Volume'].rolling(14).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Price_Trend'] = df['Volume'] * np.sign(df['Returns'])

        # Support/Resistance
        df['High_14'] = df['High'].rolling(14).max()
        df['Low_14'] = df['Low'].rolling(14).min()
        df['Range_Position'] = (df['Close'] - df['Low_14']) / (df['High_14'] - df['Low_14'])

        # NaN entfernen
        before_dropna = len(df)
        df = df.dropna()

        print(f"✅ {len(df.columns)} Features erstellt, {before_dropna - len(df)} NaN-Zeilen entfernt")
        return df

    def _create_targets(self, df):
        """ML-Targets erstellen"""
        print(f"🎯 Erstelle Targets...")

        df = df.copy()

        # Haupttargets
        df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target_Range'] = (df['High'].shift(-1) - df['Low'].shift(-1)) / df['Close']

        # Multi-Period Targets
        for periods in [3, 5]:
            df[f'Target_Direction_{periods}'] = (df['Close'].shift(-periods) > df['Close']).astype(int)
            df[f'Target_Return_{periods}'] = df['Close'].shift(-periods) / df['Close'] - 1

        print(f"✅ 8 Targets erstellt")
        return df

    def _prepare_ml_data(self, df):
        """ML-Daten vorbereiten"""
        print(f"🤖 Bereite ML-Daten vor...")

        # Features vs Targets trennen
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        feature_cols = [col for col in df.columns if col not in target_cols]

        X = df[feature_cols].copy()
        y = df[target_cols].copy()

        # Zeilen ohne Target entfernen
        valid_idx = ~y['Target_Direction'].isna()
        X, y = X[valid_idx], y[valid_idx]

        # Skalierung
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        print(f"✅ {X_scaled.shape[0]:,} Samples, {X_scaled.shape[1]} Features")
        return X_scaled, y

    def _save_all(self, df_final, X_scaled, y):
        """Alle Daten speichern"""
        print(f"💾 Speichere Daten...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Dateipfade
        files = {
            'cleaned': os.path.join(self.output_dir, 'cleaned.csv'),
            'features': os.path.join(self.output_dir, 'features_scaled.csv'),
            'targets': os.path.join(self.output_dir, 'targets.csv'),
            'scaler': os.path.join(self.output_dir, 'scaler.joblib'),
            'metadata': os.path.join(self.output_dir, 'metadata.json')
        }

        # Speichern
        df_final.to_csv(files['cleaned'])
        X_scaled.to_csv(files['features'])
        y.to_csv(files['targets'])
        joblib.dump(self.scaler, files['scaler'])

        # Metadata
        feature_cols = [col for col in df_final.columns if not col.startswith('Target_')]
        target_cols = [col for col in df_final.columns if col.startswith('Target_')]

        self.metadata = {
            'timestamp': timestamp,
            'input_file': self.input_path,
            'samples': len(X_scaled),
            'features': len(feature_cols),
            'targets': len(target_cols),
            'date_range': {
                'start': str(X_scaled.index.min()),
                'end': str(X_scaled.index.max())
            },
            'target_stats': {
                'up_samples': int(y['Target_Direction'].sum()),
                'down_samples': int(len(y) - y['Target_Direction'].sum()),
                'up_ratio': float(y['Target_Direction'].mean())
            },
            'feature_names': feature_cols,
            'target_names': target_cols
        }

        with open(files['metadata'], 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✅ 5 Dateien gespeichert in {self.output_dir}")
        return files

    def _create_backup(self):
        """Backup erstellen"""
        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{self.output_dir}_backup_{timestamp}"
            shutil.copytree(self.output_dir, backup_dir)
            print(f"📦 Backup erstellt: {backup_dir}")

    def _print_success_summary(self):
        """Erfolgszusammenfassung"""
        print(f"\n🎉 VERARBEITUNG ERFOLGREICH!")
        print(f"{'=' * 50}")
        print(f"📊 Samples: {self.metadata['samples']:,}")
        print(f"🔧 Features: {self.metadata['features']}")
        print(f"🎯 Targets: {self.metadata['targets']}")
        print(f"📅 Zeitraum: {self.metadata['date_range']['start']} - {self.metadata['date_range']['end']}")
        print(f"⚖️ Target-Balance: {self.metadata['target_stats']['up_ratio']:.1%} UP")
        print(f"📁 Output: {self.output_dir}")
        print(f"✅ Daten sind ML-ready!")

    def _print_troubleshooting(self):
        """Troubleshooting-Tipps"""
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   📁 Prüfe ob {self.input_path} existiert")
        print(f"   📊 Prüfe CSV-Format: Date,Time,Open,High,Low,Close,Volume")
        print(f"   🔒 Prüfe Schreibrechte in {self.output_dir}")
        print(f"   📦 Installiere: pip install pandas scikit-learn joblib")


# =================== VERWENDUNG ===================

def main():
    """Hauptfunktion für direkten Aufruf"""
    print("🚀 XAUUSD Data Processor")
    print("=" * 40)

    # Automatische Pfad-Erkennung
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Eine Ebene höher

    input_path = os.path.join(project_root, "data", "orginal", "XAUUSD15.csv")
    output_dir = os.path.join(project_root, "data", "prepared")

    print(f"📁 Script-Verzeichnis: {script_dir}")
    print(f"📁 Projekt-Root: {project_root}")
    print(f"📂 Suche Input: {input_path}")

    # Prüfen ob Datei existiert
    if not os.path.exists(input_path):
        print(f"\n❌ Datei nicht gefunden: {input_path}")
        print(f"💡 Verfügbare Dateien in data/orginal/:")

        orginal_dir = os.path.join(project_root, "data", "orginal")
        if os.path.exists(orginal_dir):
            files = os.listdir(orginal_dir)
            for file in files:
                print(f"   📄 {file}")
        else:
            print(f"   📁 Verzeichnis {orginal_dir} existiert nicht")

        return None

    # Processor erstellen mit korrekten Pfaden
    processor = XAUUSDProcessor(
        input_path=input_path,
        output_dir=output_dir
    )

    # Verarbeitung starten
    file_paths = processor.process()

    # Test: Daten laden
    print("\n🔄 Test: Lade verarbeitete Daten...")
    X, y, scaler, metadata = processor.load_processed()

    print(f"\n✨ FERTIG! Du kannst jetzt ML-Training starten.")
    return processor


if __name__ == "__main__":
    processor = main()

    # Beispiel Update (für später)
    # processor.update("data/orginal/XAUUSD15_new.csv")