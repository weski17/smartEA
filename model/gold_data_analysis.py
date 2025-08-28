import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class GoldDataStatistics:
    """
    Umfassende statistische Analyse für XAUUSD M15 Daten
    """
    
    def __init__(self, csv_path):
        """
        Initialisiert die Statistik-Klasse und lädt die Daten
        
        Args:
            csv_path (str): Pfad zur CSV-Datei
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """
        Lädt und bereinigt die CSV-Daten
        """
        print("📊 Lade XAUUSD M15 Daten...")
        
        # Lade Daten mit Semikolon als Separator
        self.df = pd.read_csv(self.csv_path, sep=';', decimal=',')
        
        # Konvertiere Zeit-Spalte zu datetime
        self.df['Zeit'] = pd.to_datetime(self.df['Zeit'], format='%d.%m.%Y %H:%M')
        
        # Konvertiere numerische Spalten
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if self.df[col].dtype == 'object':
                # Falls Komma als Dezimaltrennzeichen verwendet wird
                self.df[col] = self.df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Sortiere nach Zeit
        self.df = self.df.sort_values('Zeit').reset_index(drop=True)
        
        print(f"✅ Daten geladen: {len(self.df):,} Datensätze")
        print(f"📅 Zeitraum: {self.df['Zeit'].min()} bis {self.df['Zeit'].max()}")
        
    def basic_info(self):
        """
        Grundlegende Informationen über den Datensatz
        """
        print("\n" + "="*60)
        print("📋 GRUNDLEGENDE DATEN-INFORMATIONEN")
        print("="*60)
        
        print(f"Anzahl Datensätze: {len(self.df):,}")
        print(f"Anzahl Spalten: {len(self.df.columns)}")
        print(f"Zeitraum: {self.df['Zeit'].min()} bis {self.df['Zeit'].max()}")
        print(f"Anzahl Tage: {(self.df['Zeit'].max() - self.df['Zeit'].min()).days:,}")
        
        # Datentypen
        print("\n📊 Datentypen:")
        print(self.df.dtypes)
        
        # Fehlende Werte
        print("\n❌ Fehlende Werte:")
        missing = self.df.isnull().sum()
        print(missing)
        
        # Duplikate
        duplicates = self.df.duplicated().sum()
        print(f"\n🔄 Duplikate: {duplicates}")
        
    def price_statistics(self):
        """
        Detaillierte Preis-Statistiken
        """
        print("\n" + "="*60)
        print("💰 PREIS-STATISTIKEN")
        print("="*60)
        
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        for col in price_cols:
            print(f"\n📈 {col}-Preis Statistiken:")
            print(f"   Min:      ${self.df[col].min():.2f}")
            print(f"   Max:      ${self.df[col].max():.2f}")
            print(f"   Mittel:   ${self.df[col].mean():.2f}")
            print(f"   Median:   ${self.df[col].median():.2f}")
            print(f"   Std:      ${self.df[col].std():.2f}")
            print(f"   Var:      ${self.df[col].var():.2f}")
            print(f"   25%-ile:  ${self.df[col].quantile(0.25):.2f}")
            print(f"   75%-ile:  ${self.df[col].quantile(0.75):.2f}")
        
        # Spread Statistiken
        self.df['Spread'] = self.df['High'] - self.df['Low']
        print(f"\n📊 Spread (High-Low) Statistiken:")
        print(f"   Min:      ${self.df['Spread'].min():.2f}")
        print(f"   Max:      ${self.df['Spread'].max():.2f}")
        print(f"   Mittel:   ${self.df['Spread'].mean():.2f}")
        print(f"   Median:   ${self.df['Spread'].median():.2f}")
        print(f"   Std:      ${self.df['Spread'].std():.2f}")
        
    def volume_statistics(self):
        """
        Volume-Statistiken
        """
        print("\n" + "="*60)
        print("📊 VOLUME-STATISTIKEN")
        print("="*60)
        
        vol = self.df['Volume']
        print(f"Min Volume:        {vol.min():,}")
        print(f"Max Volume:        {vol.max():,}")
        print(f"Durchschn. Volume: {vol.mean():.0f}")
        print(f"Median Volume:     {vol.median():.0f}")
        print(f"Std Volume:        {vol.std():.0f}")
        print(f"Gesamt Volume:     {vol.sum():,}")
        
        # Volume Quantile
        print(f"\nVolume Quantile:")
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            print(f"   {q*100:5.1f}%-ile: {vol.quantile(q):8,.0f}")
            
    def returns_analysis(self):
        """
        Rendite-Analyse
        """
        print("\n" + "="*60)
        print("📈 RENDITE-ANALYSE")
        print("="*60)
        
        # Berechne Returns
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        returns = self.df['Returns'].dropna()
        log_returns = self.df['Log_Returns'].dropna()
        
        print(f"🔢 Lineare Returns:")
        print(f"   Mittel:       {returns.mean()*100:.6f}%")
        print(f"   Std:          {returns.std()*100:.4f}%")
        print(f"   Min:          {returns.min()*100:.4f}%")
        print(f"   Max:          {returns.max()*100:.4f}%")
        print(f"   Skewness:     {returns.skew():.4f}")
        print(f"   Kurtosis:     {returns.kurtosis():.4f}")
        
        print(f"\n📊 Log Returns:")
        print(f"   Mittel:       {log_returns.mean()*100:.6f}%")
        print(f"   Std:          {log_returns.std()*100:.4f}%")
        print(f"   Min:          {log_returns.min()*100:.4f}%")
        print(f"   Max:          {log_returns.max()*100:.4f}%")
        print(f"   Skewness:     {log_returns.skew():.4f}")
        print(f"   Kurtosis:     {log_returns.kurtosis():.4f}")
        
        # Sharpe Ratio (annualisiert)
        annual_return = returns.mean() * 252 * 24 * 4  # M15 data
        annual_vol = returns.std() * np.sqrt(252 * 24 * 4)
        sharpe = annual_return / annual_vol
        print(f"\n📊 Sharpe Ratio (annualisiert): {sharpe:.4f}")
        
        # Normalitätstest
        stat, p_value = stats.jarque_bera(returns.dropna())
        print(f"\n🧪 Jarque-Bera Normalitätstest:")
        print(f"   Statistik: {stat:.4f}")
        print(f"   p-value:   {p_value:.6f}")
        print(f"   Normal?:   {'Nein' if p_value < 0.05 else 'Ja'}")
        
    def time_analysis(self):
        """
        Zeitbasierte Analyse
        """
        print("\n" + "="*60)
        print("⏰ ZEITBASIERTE ANALYSE")
        print("="*60)
        
        # Extrahiere Zeitkomponenten
        self.df['Hour'] = self.df['Zeit'].dt.hour
        self.df['Weekday'] = self.df['Zeit'].dt.dayofweek
        self.df['Month'] = self.df['Zeit'].dt.month
        self.df['Year'] = self.df['Zeit'].dt.year
        
        # Durchschnittliches Volume pro Stunde
        print("📊 Durchschnittliches Volume pro Stunde:")
        hourly_vol = self.df.groupby('Hour')['Volume'].mean().sort_values(ascending=False)
        for hour, vol in hourly_vol.head(10).items():
            print(f"   {hour:2d}:00 Uhr: {vol:8,.0f}")
            
        # Durchschnittliche Volatilität pro Wochentag
        weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
        print(f"\n📊 Durchschnittliche Spread pro Wochentag:")
        for day_num, day_name in enumerate(weekdays):
            day_data = self.df[self.df['Weekday'] == day_num]
            if len(day_data) > 0:
                avg_spread = day_data['Spread'].mean()
                print(f"   {day_name}: ${avg_spread:.3f}")
                
        # Jährliche Statistiken
        print(f"\n📊 Jährliche Statistiken:")
        yearly_stats = self.df.groupby('Year').agg({
            'Close': ['min', 'max', 'mean'],
            'Volume': 'mean',
            'Spread': 'mean'
        }).round(2)
        
        for year in yearly_stats.index:
            print(f"   {year}: Min=${yearly_stats.loc[year, ('Close', 'min')]:.2f}, "
                  f"Max=${yearly_stats.loc[year, ('Close', 'max')]:.2f}, "
                  f"Avg=${yearly_stats.loc[year, ('Close', 'mean')]:.2f}")
                  
    def correlation_analysis(self):
        """
        Korrelationsanalyse
        """
        print("\n" + "="*60)
        print("🔗 KORRELATIONSANALYSE")
        print("="*60)
        
        # Korrelationsmatrix für Preisdaten
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
        corr_matrix = self.df[price_cols].corr()
        
        print("📊 Korrelationsmatrix:")
        print(corr_matrix.round(4))
        
        # Stärkste Korrelationen
        print(f"\n🔝 Stärkste Korrelationen:")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, corr in corr_pairs[:5]:
            print(f"   {col1} vs {col2}: {corr:.4f}")
            
    def generate_visualizations(self):
        """
        Erstellt wichtige Visualisierungen
        """
        print("\n" + "="*60)
        print("📊 ERSTELLE VISUALISIERUNGEN")
        print("="*60)
        
        # Setup für Plots
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('XAUUSD M15 - Statistische Analyse', fontsize=16, fontweight='bold')
        
        # 1. Preisverlauf
        axes[0,0].plot(self.df['Zeit'], self.df['Close'], linewidth=0.5, alpha=0.8)
        axes[0,0].set_title('Gold Preisverlauf')
        axes[0,0].set_ylabel('Preis ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Returns Histogram
        returns = self.df['Returns'].dropna()
        axes[0,1].hist(returns, bins=100, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Returns Verteilung')
        axes[0,1].set_xlabel('Returns')
        axes[0,1].set_ylabel('Häufigkeit')
        
        # 3. Volume über Zeit
        axes[0,2].plot(self.df['Zeit'], self.df['Volume'], linewidth=0.5, alpha=0.6)
        axes[0,2].set_title('Volume über Zeit')
        axes[0,2].set_ylabel('Volume')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Spread Histogram
        axes[1,0].hist(self.df['Spread'], bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Spread Verteilung')
        axes[1,0].set_xlabel('Spread ($)')
        axes[1,0].set_ylabel('Häufigkeit')
        
        # 5. Volume pro Stunde
        hourly_vol = self.df.groupby('Hour')['Volume'].mean()
        axes[1,1].bar(hourly_vol.index, hourly_vol.values)
        axes[1,1].set_title('Durchschnittliches Volume pro Stunde')
        axes[1,1].set_xlabel('Stunde')
        axes[1,1].set_ylabel('Durchschn. Volume')
        
        # 6. Price vs Volume Scatter
        sample_df = self.df.sample(n=min(10000, len(self.df)))  # Sample für Performance
        axes[1,2].scatter(sample_df['Volume'], sample_df['Close'], alpha=0.5, s=1)
        axes[1,2].set_title('Preis vs Volume')
        axes[1,2].set_xlabel('Volume')
        axes[1,2].set_ylabel('Close Preis ($)')
        
        plt.tight_layout()
        
        # Speichere Plot
        output_path = 'c:\\Users\\Wael\\Desktop\\Projekts\\smartEA\\images\\gold_statistics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualisierungen gespeichert: {output_path}")
        
        plt.show()
        
    def run_complete_analysis(self):
        """
        Führt die komplette statistische Analyse durch
        """
        print("🚀 STARTE KOMPLETTE STATISTISCHE ANALYSE")
        print("="*60)
        
        self.basic_info()
        self.price_statistics()
        self.volume_statistics()
        self.returns_analysis()
        self.time_analysis()
        self.correlation_analysis()
        self.generate_visualizations()
        
        print("\n" + "="*60)
        print("✅ ANALYSE ABGESCHLOSSEN!")
        print("="*60)
        
        # Zusammenfassung
        print(f"\n📋 ZUSAMMENFASSUNG:")
        print(f"   📊 Datensätze analysiert: {len(self.df):,}")
        print(f"   💰 Preisspanne: ${self.df['Close'].min():.2f} - ${self.df['Close'].max():.2f}")
        print(f"   📈 Durchschnittspreis: ${self.df['Close'].mean():.2f}")
        print(f"   📊 Durchschn. Volume: {self.df['Volume'].mean():,.0f}")
        print(f"   📅 Zeitraum: {(self.df['Zeit'].max() - self.df['Zeit'].min()).days:,} Tage")

if __name__ == "__main__":
    # Pfad zur Datendatei
    data_path = "c:\\Users\\Wael\\Desktop\\Projekts\\smartEA\\data\\XAUUSD_M15_full.csv"
    
    # Erstelle Statistik-Objekt und führe Analyse durch
    gold_stats = GoldDataStatistics(data_path)
    gold_stats.run_complete_analysis()