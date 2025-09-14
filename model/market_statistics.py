import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class UpdatedGoldDataStatistics:
    """
    Statistische Analyse für die aktualisierte XAUUSD M15 Datei
    """

    def __init__(self, csv_path):
        """
        Initialisiert die Statistik-Klasse für die aktualisierte Datei

        Args:
            csv_path (str): Pfad zur aktualisierten CSV-Datei
        """
        self.csv_path = csv_path
        self.df = None
        self.statistics_dict = {}
        self.load_data()

    def load_data(self):
        """
        Lädt die aktualisierte CSV-Datei
        """
        print("📊 Lade aktualisierte XAUUSD M15 Daten...")

        # Lade mit Semikolon-Separator
        self.df = pd.read_csv(self.csv_path, sep=';')

        # Konvertiere Zeit-Spalte
        self.df['Zeit'] = pd.to_datetime(self.df['Zeit'], format='%d.%m.%Y %H:%M')

        # Konvertiere numerische Spalten
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Sortiere nach Zeit
        self.df = self.df.sort_values('Zeit').reset_index(drop=True)

        print(f"✅ Aktualisierte Daten geladen: {len(self.df):,} Datensätze")
        print(f"📅 Neuer Zeitraum: {self.df['Zeit'].min()} bis {self.df['Zeit'].max()}")

        # Zeige Update-Info
        cutoff_date = pd.to_datetime("2025-08-25 19:00:00")
        old_data = self.df[self.df['Zeit'] < cutoff_date]
        new_data = self.df[self.df['Zeit'] >= cutoff_date]

        print(f"📈 Alte Daten (vor 25.08.2025 19:00): {len(old_data):,} Datensätze")
        print(f"🆕 Neue Daten (ab 25.08.2025 19:00): {len(new_data):,} Datensätze")

    def dataset_overview(self):
        """
        Übersicht über das aktualisierte Dataset
        """
        print("\n" + "=" * 60)
        print("📋 AKTUALISIERTE DATEN - ÜBERSICHT")
        print("=" * 60)

        print(f"Gesamte Datensätze: {len(self.df):,}")
        print(f"Zeitraum: {self.df['Zeit'].min()} bis {self.df['Zeit'].max()}")
        print(f"Gesamte Tage: {(self.df['Zeit'].max() - self.df['Zeit'].min()).days:,}")

        # Update-Analyse
        cutoff_date = pd.to_datetime("2025-08-25 19:00:00")
        old_data = self.df[self.df['Zeit'] < cutoff_date]
        new_data = self.df[self.df['Zeit'] >= cutoff_date]

        print(f"\n🔄 UPDATE-ANALYSE:")
        print(f"   Cutoff-Datum: 25.08.2025 19:00")
        print(f"   Alte Daten: {len(old_data):,} Datensätze ({len(old_data) / len(self.df) * 100:.1f}%)")
        print(f"   Neue Daten: {len(new_data):,} Datensätze ({len(new_data) / len(self.df) * 100:.1f}%)")

        if len(new_data) > 0:
            print(f"   Neue Daten Zeitraum: {new_data['Zeit'].min()} bis {new_data['Zeit'].max()}")
            print(f"   Neue Daten Dauer: {(new_data['Zeit'].max() - new_data['Zeit'].min()).days} Tage")

        # Datenqualität
        print(f"\n🔍 DATENQUALITÄT:")
        print(f"   Fehlende Werte: {self.df.isnull().sum().sum()}")
        print(f"   Duplikate: {self.df.duplicated().sum()}")

        # Speichere Dataset-Info
        self.statistics_dict['dataset_info'] = {
            'total_records': int(len(self.df)),
            'start_date': str(self.df['Zeit'].min()),
            'end_date': str(self.df['Zeit'].max()),
            'total_days': int((self.df['Zeit'].max() - self.df['Zeit'].min()).days),
            'old_data_count': int(len(old_data)),
            'new_data_count': int(len(new_data)),
            'new_data_start': str(new_data['Zeit'].min()) if len(new_data) > 0 else None,
            'new_data_end': str(new_data['Zeit'].max()) if len(new_data) > 0 else None,
            'missing_values': int(self.df.isnull().sum().sum()),
            'duplicates': int(self.df.duplicated().sum()),
            'update_cutoff': "2025-08-25 19:00:00"
        }

    def price_analysis(self):
        """
        Analyse der Preisentwicklung
        """
        print("\n" + "=" * 60)
        print("💰 PREIS-ANALYSE")
        print("=" * 60)

        # Gesamte Preisspanne
        print(f"🎯 GESAMTE PREISSPANNE:")
        print(f"   Niedrigster Preis: ${self.df['Low'].min():.2f}")
        print(f"   Höchster Preis: ${self.df['High'].max():.2f}")
        print(f"   Aktuelle Preisspanne: ${self.df['High'].max() - self.df['Low'].min():.2f}")

        # Aktuelle vs. historische Preise
        cutoff_date = pd.to_datetime("2025-08-25 19:00:00")
        old_data = self.df[self.df['Zeit'] < cutoff_date]
        new_data = self.df[self.df['Zeit'] >= cutoff_date]

        print(f"\n📊 PREISVERGLEICH ALT vs. NEU:")
        if len(old_data) > 0 and len(new_data) > 0:
            print(f"   Alte Daten - Durchschnitt: ${old_data['Close'].mean():.2f}")
            print(f"   Neue Daten - Durchschnitt: ${new_data['Close'].mean():.2f}")
            print(f"   Preisänderung: ${new_data['Close'].mean() - old_data['Close'].mean():.2f}")
            print(f"   Relative Änderung: {((new_data['Close'].mean() / old_data['Close'].mean()) - 1) * 100:.2f}%")

            print(f"\n   Alte Daten - Max: ${old_data['High'].max():.2f}")
            print(f"   Neue Daten - Max: ${new_data['High'].max():.2f}")
            print(f"   Neue Daten - Min: ${new_data['Low'].min():.2f}")

        # Spread-Analyse
        self.df['Spread'] = self.df['High'] - self.df['Low']
        print(f"\n📏 SPREAD-ANALYSE:")
        print(f"   Durchschnittlicher Spread: ${self.df['Spread'].mean():.2f}")
        print(f"   Median Spread: ${self.df['Spread'].median():.2f}")
        print(f"   Max Spread: ${self.df['Spread'].max():.2f}")
        print(f"   Min Spread: ${self.df['Spread'].min():.2f}")

        # Speichere Preis-Statistiken
        self.statistics_dict['price_analysis'] = {
            'overall': {
                'min_price': float(self.df['Low'].min()),
                'max_price': float(self.df['High'].max()),
                'price_range': float(self.df['High'].max() - self.df['Low'].min()),
                'avg_close': float(self.df['Close'].mean()),
                'avg_spread': float(self.df['Spread'].mean()),
                'max_spread': float(self.df['Spread'].max())
            }
        }

    def _get_period_stats(self, days):
        """
        Berechnet Statistiken für die letzten N Tage
        """
        end_date = self.df['Zeit'].max()
        start_date = end_date - pd.Timedelta(days=days)
        period_data = self.df[self.df['Zeit'] >= start_date]

        if len(period_data) == 0:
            return None

        return {
            'record_count': len(period_data),
            'avg_price': float(period_data['Close'].mean()),
            'min_price': float(period_data['Close'].min()),
            'max_price': float(period_data['Close'].max()),
            'price_change': float(period_data['Close'].iloc[-1] - period_data['Close'].iloc[0]) if len(
                period_data) > 1 else 0.0,
            'price_change_pct': float(
                ((period_data['Close'].iloc[-1] / period_data['Close'].iloc[0]) - 1) * 100) if len(
                period_data) > 1 else 0.0,
            'avg_volume': float(period_data['Volume'].mean()),
            'volatility': float(period_data['Close'].std()),
            'start_date': str(start_date.date()),
            'end_date': str(end_date.date())
        }

        if len(old_data) > 0 and len(new_data) > 0:
            self.statistics_dict['price_analysis']['comparison'] = {
                'old_data_avg': float(old_data['Close'].mean()),
                'new_data_avg': float(new_data['Close'].mean()),
                'price_change': float(new_data['Close'].mean() - old_data['Close'].mean()),
                'relative_change_pct': float(((new_data['Close'].mean() / old_data['Close'].mean()) - 1) * 100),
                'new_data_max': float(new_data['High'].max()),
                'new_data_min': float(new_data['Low'].min())
            }

    def volume_analysis(self):
        """
        Volume-Analyse
        """
        print("\n" + "=" * 60)
        print("📊 VOLUME-ANALYSE")
        print("=" * 60)

        vol = self.df['Volume']
        print(f"📈 VOLUME-STATISTIKEN:")
        print(f"   Gesamt-Volume: {vol.sum():,}")
        print(f"   Durchschn. Volume: {vol.mean():.0f}")
        print(f"   Median Volume: {vol.median():.0f}")
        print(f"   Max Volume: {vol.max():,}")
        print(f"   Min Volume: {vol.min():,}")

        # Volume-Vergleich alt vs. neu
        cutoff_date = pd.to_datetime("2025-08-25 19:00:00")
        old_data = self.df[self.df['Zeit'] < cutoff_date]
        new_data = self.df[self.df['Zeit'] >= cutoff_date]

        if len(old_data) > 0 and len(new_data) > 0:
            print(f"\n🔄 VOLUME-VERGLEICH:")
            print(f"   Alte Daten - Ø Volume: {old_data['Volume'].mean():.0f}")
            print(f"   Neue Daten - Ø Volume: {new_data['Volume'].mean():.0f}")
            print(f"   Volume-Änderung: {new_data['Volume'].mean() - old_data['Volume'].mean():.0f}")
            print(f"   Relative Änderung: {((new_data['Volume'].mean() / old_data['Volume'].mean()) - 1) * 100:.1f}%")

        # Volume Quantile
        print(f"\n📊 VOLUME-VERTEILUNG:")
        for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            print(f"   {q * 100:5.1f}%-Quantil: {vol.quantile(q):8,.0f}")

        # Speichere Volume-Statistiken
        self.statistics_dict['volume_analysis'] = {
            'total_volume': int(vol.sum()),
            'avg_volume': float(vol.mean()),
            'median_volume': float(vol.median()),
            'max_volume': int(vol.max()),
            'min_volume': int(vol.min()),
            'quantiles': {f'q{int(q * 100)}': float(vol.quantile(q)) for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]}
        }

        if len(old_data) > 0 and len(new_data) > 0:
            self.statistics_dict['volume_analysis']['comparison'] = {
                'old_data_avg': float(old_data['Volume'].mean()),
                'new_data_avg': float(new_data['Volume'].mean()),
                'volume_change': float(new_data['Volume'].mean() - old_data['Volume'].mean()),
                'relative_change_pct': float(((new_data['Volume'].mean() / old_data['Volume'].mean()) - 1) * 100)
            }

    def returns_analysis(self):
        """
        Rendite-Analyse der aktualisierten Daten
        """
        print("\n" + "=" * 60)
        print("📈 RENDITE-ANALYSE")
        print("=" * 60)

        # Berechne Returns
        self.df['Returns'] = self.df['Close'].pct_change()
        returns = self.df['Returns'].dropna()

        print(f"📊 GESAMT-RETURNS:")
        print(f"   Durchschn. Return: {returns.mean() * 100:.6f}%")
        print(f"   Volatilität: {returns.std() * 100:.4f}%")
        print(f"   Min Return: {returns.min() * 100:.4f}%")
        print(f"   Max Return: {returns.max() * 100:.4f}%")
        print(f"   Skewness: {returns.skew():.4f}")
        print(f"   Kurtosis: {returns.kurtosis():.4f}")

        # Sharpe Ratio
        annual_return = returns.mean() * 252 * 24 * 4  # M15 data
        annual_vol = returns.std() * np.sqrt(252 * 24 * 4)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        print(f"   Sharpe Ratio: {sharpe:.4f}")

        # Returns für neue Daten
        cutoff_date = pd.to_datetime("2025-08-25 19:00:00")
        new_data = self.df[self.df['Zeit'] >= cutoff_date]

        if len(new_data) > 1:
            new_returns = new_data['Returns'].dropna()
            if len(new_returns) > 0:
                print(f"\n🆕 NEUE DATEN - RETURNS:")
                print(f"   Durchschn. Return: {new_returns.mean() * 100:.6f}%")
                print(f"   Volatilität: {new_returns.std() * 100:.4f}%")
                print(f"   Min Return: {new_returns.min() * 100:.4f}%")
                print(f"   Max Return: {new_returns.max() * 100:.4f}%")

                # Performance seit Update
                if len(new_data) > 0:
                    start_price = new_data['Close'].iloc[0]
                    end_price = new_data['Close'].iloc[-1]
                    total_return = (end_price / start_price - 1) * 100
                    print(f"   Gesamtperformance seit Update: {total_return:.2f}%")

        # Positive vs. Negative Returns
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        print(f"\n📊 RETURN-VERTEILUNG:")
        print(f"   Positive Returns: {len(positive_returns):,} ({len(positive_returns) / len(returns) * 100:.1f}%)")
        print(f"   Negative Returns: {len(negative_returns):,} ({len(negative_returns) / len(returns) * 100:.1f}%)")
        print(f"   Neutrale Returns: {len(returns) - len(positive_returns) - len(negative_returns):,}")

        # Speichere Returns-Statistiken
        self.statistics_dict['returns_analysis'] = {
            'overall': {
                'mean_return': float(returns.mean()),
                'volatility': float(returns.std()),
                'min_return': float(returns.min()),
                'max_return': float(returns.max()),
                'skewness': float(returns.skew()),
                'kurtosis': float(returns.kurtosis()),
                'sharpe_ratio': float(sharpe),
                'positive_returns_pct': float(len(positive_returns) / len(returns) * 100),
                'negative_returns_pct': float(len(negative_returns) / len(returns) * 100)
            }
        }

        if len(new_data) > 1 and len(new_returns) > 0:
            start_price = new_data['Close'].iloc[0]
            end_price = new_data['Close'].iloc[-1]
            total_return = (end_price / start_price - 1) * 100

            self.statistics_dict['returns_analysis']['new_data'] = {
                'mean_return': float(new_returns.mean()),
                'volatility': float(new_returns.std()),
                'min_return': float(new_returns.min()),
                'max_return': float(new_returns.max()),
                'total_performance_pct': float(total_return),
                'start_price': float(start_price),
                'end_price': float(end_price)
            }

    def time_analysis(self):
        """
        Zeitbasierte Analyse der aktualisierten Daten
        """
        print("\n" + "=" * 60)
        print("⏰ ZEIT-ANALYSE")
        print("=" * 60)

        # Extrahiere Zeitkomponenten
        self.df['Hour'] = self.df['Zeit'].dt.hour
        self.df['Weekday'] = self.df['Zeit'].dt.dayofweek
        self.df['Year'] = self.df['Zeit'].dt.year
        self.df['Month'] = self.df['Zeit'].dt.month
        self.df['Date'] = self.df['Zeit'].dt.date
        self.df['Week'] = self.df['Zeit'].dt.isocalendar().week
        self.df['Year_Week'] = self.df['Zeit'].dt.strftime('%Y-W%U')

        # Stündliche Aktivität
        hourly_vol = self.df.groupby('Hour')['Volume'].mean().sort_values(ascending=False)
        print(f"📊 TOP 10 AKTIVSTE STUNDEN (Volume):")
        for hour, vol in hourly_vol.head(10).items():
            print(f"   {hour:2d}:00 Uhr: {vol:8,.0f}")

        # Wochentags-Analyse
        weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
        print(f"\n📅 WOCHENTAGS-ANALYSE:")
        for day_num, day_name in enumerate(weekdays):
            day_data = self.df[self.df['Weekday'] == day_num]
            if len(day_data) > 0:
                avg_vol = day_data['Volume'].mean()
                avg_spread = day_data['Spread'].mean()
                print(f"   {day_name}: Volume {avg_vol:6,.0f}, Spread ${avg_spread:.2f}")

        # Jährliche Entwicklung
        print(f"\n📈 JÄHRLICHE ENTWICKLUNG:")
        yearly_stats = self.df.groupby('Year').agg({
            'Close': ['min', 'max', 'mean'],
            'Volume': 'mean'
        }).round(2)

        for year in sorted(yearly_stats.index):
            min_price = yearly_stats.loc[year, ('Close', 'min')]
            max_price = yearly_stats.loc[year, ('Close', 'max')]
            avg_price = yearly_stats.loc[year, ('Close', 'mean')]
            avg_vol = yearly_stats.loc[year, ('Volume', 'mean')]
            print(f"   {year}: ${min_price:.0f} - ${max_price:.0f} (Ø ${avg_price:.0f}), Vol: {avg_vol:,.0f}")

        print(f"\n📊 Detaillierte Analysen werden in JSON gespeichert...")

        # Speichere Zeit-Statistiken mit erweiterten Analysen
        self.statistics_dict['time_analysis'] = {
            'hourly_stats': {
                str(hour): {
                    'avg_volume': float(group['Volume'].mean()),
                    'avg_price': float(group['Close'].mean()),
                    'avg_spread': float(group['Spread'].mean()),
                    'record_count': len(group),
                    'price_volatility': float(group['Close'].std())
                } for hour, group in self.df.groupby('Hour')
            },
            'weekday_stats': {
                weekdays[day]: {
                    'avg_volume': float(day_data['Volume'].mean()),
                    'avg_spread': float(day_data['Spread'].mean()),
                    'avg_price': float(day_data['Close'].mean()),
                    'record_count': len(day_data),
                    'price_volatility': float(day_data['Close'].std()),
                    'min_price': float(day_data['Close'].min()),
                    'max_price': float(day_data['Close'].max())
                } for day in range(7)
                for day_data in [self.df[self.df['Weekday'] == day]]
                if len(day_data) > 0
            },
            'weekly_stats': {
                week: {
                    'start_date': str(group['Zeit'].min().date()),
                    'end_date': str(group['Zeit'].max().date()),
                    'avg_price': float(group['Close'].mean()),
                    'min_price': float(group['Close'].min()),
                    'max_price': float(group['Close'].max()),
                    'open_price': float(group['Open'].iloc[0]),
                    'close_price': float(group['Close'].iloc[-1]),
                    'weekly_return': float((group['Close'].iloc[-1] / group['Open'].iloc[0]) - 1),
                    'avg_volume': float(group['Volume'].mean()),
                    'total_volume': int(group['Volume'].sum()),
                    'avg_spread': float(group['Spread'].mean()),
                    'record_count': len(group),
                    'volatility': float(group['Close'].std())
                } for week, group in self.df.groupby('Year_Week')
                if len(group) > 1
            },
            'daily_stats': {
                str(date): {
                    'avg_price': float(group['Close'].mean()),
                    'min_price': float(group['Close'].min()),
                    'max_price': float(group['Close'].max()),
                    'open_price': float(group['Open'].iloc[0]),
                    'close_price': float(group['Close'].iloc[-1]),
                    'daily_return': float((group['Close'].iloc[-1] / group['Open'].iloc[0]) - 1) if len(
                        group) > 1 else 0.0,
                    'avg_volume': float(group['Volume'].mean()),
                    'total_volume': int(group['Volume'].sum()),
                    'avg_spread': float(group['Spread'].mean()),
                    'record_count': len(group),
                    'intraday_volatility': float(group['Close'].std()),
                    'price_range': float(group['High'].max() - group['Low'].min()),
                    'weekday': int(group['Weekday'].iloc[0])
                } for date, group in self.df.groupby('Date')
            },
            'yearly_stats': {
                str(year): {
                    'min_price': float(group['Close'].min()),
                    'max_price': float(group['Close'].max()),
                    'avg_price': float(group['Close'].mean()),
                    'open_price': float(group['Open'].iloc[0]),
                    'close_price': float(group['Close'].iloc[-1]),
                    'yearly_return': float((group['Close'].iloc[-1] / group['Open'].iloc[0]) - 1) if len(
                        group) > 1 else 0.0,
                    'avg_volume': float(group['Volume'].mean()),
                    'total_volume': int(group['Volume'].sum()),
                    'avg_spread': float(group['Spread'].mean()),
                    'record_count': len(group),
                    'volatility': float(group['Close'].std()),
                    'price_range': float(group['High'].max() - group['Low'].min())
                } for year, group in self.df.groupby('Year')
            }
        }

    def generate_visualizations(self):
        """
        Erstellt Visualisierungen der aktualisierten Daten
        """
        print("\n" + "=" * 60)
        print("📊 ERSTELLE VISUALISIERUNGEN")
        print("=" * 60)

        # Setup
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('XAUUSD M15 - Aktualisierte Daten Analyse', fontsize=16, fontweight='bold')

        # 1. Preisverlauf mit Update-Markierung
        axes[0, 0].plot(self.df['Zeit'], self.df['Close'], linewidth=0.8, alpha=0.8, color='gold')
        cutoff_date = pd.to_datetime("2025-08-25 19:00:00")
        axes[0, 0].axvline(x=cutoff_date, color='red', linestyle='--', alpha=0.7, label='Update-Punkt')
        axes[0, 0].set_title('Gold Preisverlauf (mit Update)')
        axes[0, 0].set_ylabel('Preis ($)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Returns Histogram
        returns = self.df['Returns'].dropna()
        axes[0, 1].hist(returns, bins=100, alpha=0.7, edgecolor='black', color='lightblue')
        axes[0, 1].set_title('Returns Verteilung')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].set_ylabel('Häufigkeit')

        # 3. Volume über Zeit
        axes[0, 2].plot(self.df['Zeit'], self.df['Volume'], linewidth=0.5, alpha=0.6, color='green')
        axes[0, 2].axvline(x=cutoff_date, color='red', linestyle='--', alpha=0.7)
        axes[0, 2].set_title('Volume über Zeit')
        axes[0, 2].set_ylabel('Volume')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # 4. Stündliches Volume
        hourly_vol = self.df.groupby('Hour')['Volume'].mean()
        axes[1, 0].bar(hourly_vol.index, hourly_vol.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('Durchschnittliches Volume pro Stunde')
        axes[1, 0].set_xlabel('Stunde')
        axes[1, 0].set_ylabel('Volume')

        # 5. Jährliche Preisentwicklung
        yearly_avg = self.df.groupby('Year')['Close'].mean()
        axes[1, 1].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Jährliche Durchschnittspreise')
        axes[1, 1].set_xlabel('Jahr')
        axes[1, 1].set_ylabel('Durchschnittspreis ($)')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Spread Verteilung
        axes[1, 2].hist(self.df['Spread'], bins=50, alpha=0.7, edgecolor='black', color='purple')
        axes[1, 2].set_title('Spread Verteilung')
        axes[1, 2].set_xlabel('Spread ($)')
        axes[1, 2].set_ylabel('Häufigkeit')

        plt.tight_layout()

        # Speichern
        output_path = 'c:\\Users\\Wael\\Desktop\\Projekts\\smartEA\\images\\updated_gold_statistics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualisierungen gespeichert: {output_path}")

        plt.show()

    def save_statistics_to_json(self, output_path=None):
        """
        Speichert alle Statistiken der aktualisierten Daten als JSON
        """
        if output_path is None:
            output_path = "c:\\Users\\Wael\\Desktop\\Projekts\\smartEA\\data\\updated_gold_statistics.json"

        # Kombiniere alle Statistiken
        all_statistics = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_source': self.csv_path,
            'description': 'Statistische Analyse der aktualisierten XAUUSD M15 Daten',
            **self.statistics_dict
        }

        # Speichere als JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_statistics, f, indent=2, ensure_ascii=False)

        print(f"\n💾 AKTUALISIERTE STATISTIKEN GESPEICHERT:")
        print(f"   📄 Datei: {output_path}")
        print(f"   📊 Kategorien: {len(self.statistics_dict)} Hauptkategorien")
        print(f"   🔢 Größe: {len(json.dumps(all_statistics, indent=2))} Zeichen")

        return output_path

    def run_complete_analysis(self):
        """
        Führt die komplette Analyse der aktualisierten Daten durch
        """
        print("🚀 STARTE ANALYSE DER AKTUALISIERTEN DATEN")
        print("=" * 60)

        self.dataset_overview()
        self.price_analysis()
        self.volume_analysis()
        self.returns_analysis()
        self.time_analysis()
        self.generate_visualizations()

        # Speichere Statistiken
        json_path = self.save_statistics_to_json()

        print("\n" + "=" * 60)
        print("✅ ANALYSE DER AKTUALISIERTEN DATEN ABGESCHLOSSEN!")
        print("=" * 60)

        # Zusammenfassung
        cutoff_date = pd.to_datetime("2025-08-25 19:00:00")
        new_data = self.df[self.df['Zeit'] >= cutoff_date]

        print(f"\n📋 ZUSAMMENFASSUNG:")
        print(f"   📊 Gesamte Datensätze: {len(self.df):,}")
        print(f"   🆕 Neue Datensätze: {len(new_data):,}")
        print(f"   💰 Aktuelle Preisspanne: ${self.df['Low'].min():.2f} - ${self.df['High'].max():.2f}")
        print(f"   📈 Aktueller Durchschnittspreis: ${self.df['Close'].mean():.2f}")
        print(f"   📅 Gesamter Zeitraum: {(self.df['Zeit'].max() - self.df['Zeit'].min()).days:,} Tage")
        print(f"   💾 JSON-Export: {json_path}")


if __name__ == "__main__":
    # Pfad zur aktualisierten Datendatei
    data_path = "c:\\Users\\Wael\\Desktop\\Projekts\\smartEA\\data\\XAUUSD_M15_full.csv"

    # Erstelle Statistik-Objekt und führe Analyse durch
    updated_stats = UpdatedGoldDataStatistics(data_path)
    updated_stats.run_complete_analysis()