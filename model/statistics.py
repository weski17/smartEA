import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
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
        self.statistics_dict = {}  # Dictionary für alle Statistiken
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
        
    def advanced_statistics(self):
        """
        Erweiterte Statistiken und technische Indikatoren
        """
        print("\n" + "="*60)
        print("🔬 ERWEITERTE STATISTIKEN")
        print("="*60)
        
        # Berechne erweiterte Kennzahlen
        close_prices = self.df['Close']
        returns = self.df['Returns'].dropna()
        
        # Volatilitäts-Clustering (GARCH-artige Eigenschaften)
        squared_returns = returns ** 2
        volatility_autocorr = squared_returns.autocorr(lag=1)
        
        # Drawdown Analyse
        rolling_max = close_prices.expanding().max()
        drawdown = (close_prices - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # Upside/Downside Capture
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        # Calmar Ratio
        annual_return = returns.mean() * 252 * 24 * 4
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252 * 24 * 4)
        sortino_ratio = annual_return / downside_deviation if downside_deviation != 0 else 0
        
        # Rolling Volatility (30-period)
        rolling_vol = returns.rolling(window=30).std()
        
        # Hurst Exponent (Mean Reversion/Trending)
        hurst_exp = self._calculate_hurst_exponent(close_prices.values)
        
        print(f"🔍 Volatilitäts-Clustering:")
        print(f"   Autokorrelation squared returns: {volatility_autocorr:.4f}")
        
        print(f"\n📉 Drawdown Analyse:")
        print(f"   Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        print(f"   Max Drawdown Dauer: {max_drawdown_duration} Perioden")
        
        print(f"\n🎯 Risk Metrics:")
        print(f"   VaR 95%: {var_95:.4f} ({var_95*100:.2f}%)")
        print(f"   VaR 99%: {var_99:.4f} ({var_99*100:.2f}%)")
        print(f"   ES 95%:  {es_95:.4f} ({es_95*100:.2f}%)")
        print(f"   ES 99%:  {es_99:.4f} ({es_99*100:.2f}%)")
        
        print(f"\n📊 Performance Ratios:")
        print(f"   Calmar Ratio: {calmar_ratio:.4f}")
        print(f"   Sortino Ratio: {sortino_ratio:.4f}")
        
        print(f"\n🔄 Markt-Eigenschaften:")
        print(f"   Hurst Exponent: {hurst_exp:.4f}")
        market_type = "Trending" if hurst_exp > 0.5 else "Mean Reverting" if hurst_exp < 0.5 else "Random Walk"
        print(f"   Markt-Typ: {market_type}")
        
        print(f"\n📈 Return Eigenschaften:")
        print(f"   % Positive Returns: {(len(positive_returns)/len(returns)*100):.2f}%")
        print(f"   % Negative Returns: {(len(negative_returns)/len(returns)*100):.2f}%")
        print(f"   Avg Positive Return: {positive_returns.mean()*100:.4f}%")
        print(f"   Avg Negative Return: {negative_returns.mean()*100:.4f}%")
        
        # Speichere erweiterte Statistiken
        self.statistics_dict['advanced_statistics'] = {
            'volatility_clustering': float(volatility_autocorr),
            'max_drawdown': float(max_drawdown),
            'max_drawdown_duration': int(max_drawdown_duration),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'expected_shortfall_95': float(es_95),
            'expected_shortfall_99': float(es_99),
            'calmar_ratio': float(calmar_ratio),
            'sortino_ratio': float(sortino_ratio),
            'hurst_exponent': float(hurst_exp),
            'market_type': market_type,
            'positive_returns_pct': float(len(positive_returns)/len(returns)*100),
            'negative_returns_pct': float(len(negative_returns)/len(returns)*100),
            'avg_positive_return': float(positive_returns.mean()),
            'avg_negative_return': float(negative_returns.mean()),
            'rolling_volatility_mean': float(rolling_vol.mean()),
            'rolling_volatility_std': float(rolling_vol.std())
        }
        
    def _calculate_max_drawdown_duration(self, drawdown):
        """Berechnet die maximale Drawdown-Dauer"""
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
                
        if current_period > 0:
            drawdown_periods.append(current_period)
            
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_hurst_exponent(self, prices, max_lag=20):
        """Berechnet den Hurst Exponenten"""
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
        
    def market_regime_analysis(self):
        """
        Marktregime-Analyse
        """
        print("\n" + "="*60)
        print("🏛️ MARKTREGIME-ANALYSE")
        print("="*60)
        
        close_prices = self.df['Close']
        returns = self.df['Returns'].dropna()
        
        # Volatilitätsregime basierend auf rollierender Volatilität
        rolling_vol = returns.rolling(window=96).std()  # 24h window (96 * 15min)
        vol_quantiles = rolling_vol.quantile([0.33, 0.67])
        
        low_vol_threshold = vol_quantiles.iloc[0]
        high_vol_threshold = vol_quantiles.iloc[1]
        
        # Klassifiziere Regime
        self.df['Vol_Regime'] = 'Medium'
        self.df.loc[rolling_vol <= low_vol_threshold, 'Vol_Regime'] = 'Low'
        self.df.loc[rolling_vol >= high_vol_threshold, 'Vol_Regime'] = 'High'
        
        # Trend-Regime basierend auf gleitendem Durchschnitt
        self.df['MA_50'] = close_prices.rolling(window=50).mean()
        self.df['MA_200'] = close_prices.rolling(window=200).mean()
        
        self.df['Trend_Regime'] = 'Sideways'
        self.df.loc[self.df['MA_50'] > self.df['MA_200'], 'Trend_Regime'] = 'Uptrend'
        self.df.loc[self.df['MA_50'] < self.df['MA_200'], 'Trend_Regime'] = 'Downtrend'
        
        # Statistiken pro Regime
        regime_stats = {}
        
        for vol_regime in ['Low', 'Medium', 'High']:
            regime_data = self.df[self.df['Vol_Regime'] == vol_regime]
            if len(regime_data) > 0:
                regime_returns = regime_data['Returns'].dropna()
                regime_stats[f'vol_{vol_regime.lower()}'] = {
                    'count': len(regime_data),
                    'avg_return': float(regime_returns.mean()),
                    'volatility': float(regime_returns.std()),
                    'min_return': float(regime_returns.min()),
                    'max_return': float(regime_returns.max())
                }
                
        for trend_regime in ['Uptrend', 'Downtrend', 'Sideways']:
            regime_data = self.df[self.df['Trend_Regime'] == trend_regime]
            if len(regime_data) > 0:
                regime_returns = regime_data['Returns'].dropna()
                regime_stats[f'trend_{trend_regime.lower()}'] = {
                    'count': len(regime_data),
                    'avg_return': float(regime_returns.mean()),
                    'volatility': float(regime_returns.std()),
                    'avg_volume': float(regime_data['Volume'].mean())
                }
        
        print("📊 Volatilitätsregime:")
        for regime in ['Low', 'Medium', 'High']:
            count = len(self.df[self.df['Vol_Regime'] == regime])
            pct = count / len(self.df) * 100
            print(f"   {regime}: {count:,} Perioden ({pct:.1f}%)")
            
        print("\n📊 Trendregime:")
        for regime in ['Uptrend', 'Downtrend', 'Sideways']:
            count = len(self.df[self.df['Trend_Regime'] == regime])
            pct = count / len(self.df) * 100
            print(f"   {regime}: {count:,} Perioden ({pct:.1f}%)")
            
        self.statistics_dict['market_regimes'] = regime_stats
        
    def save_statistics_to_json(self, output_path=None):
        """
        Speichert alle Statistiken in eine JSON-Datei
        """
        if output_path is None:
            output_path = "c:\\Users\\Wael\\Desktop\\Projekts\\smartEA\\data\\prepared\\gold_statistics.json"
            
        # Sammle alle Basis-Statistiken
        close_prices = self.df['Close']
        returns = self.df['Returns'].dropna() if 'Returns' in self.df.columns else self.df['Close'].pct_change().dropna()
        
        # Grundlegende Statistiken
        basic_stats = {
            'dataset_info': {
                'total_records': int(len(self.df)),
                'start_date': str(self.df['Zeit'].min()),
                'end_date': str(self.df['Zeit'].max()),
                'total_days': int((self.df['Zeit'].max() - self.df['Zeit'].min()).days),
                'missing_values': int(self.df.isnull().sum().sum()),
                'duplicates': int(self.df.duplicated().sum())
            },
            'price_statistics': {
                'open': {
                    'min': float(self.df['Open'].min()),
                    'max': float(self.df['Open'].max()),
                    'mean': float(self.df['Open'].mean()),
                    'median': float(self.df['Open'].median()),
                    'std': float(self.df['Open'].std()),
                    'var': float(self.df['Open'].var()),
                    'q25': float(self.df['Open'].quantile(0.25)),
                    'q75': float(self.df['Open'].quantile(0.75))
                },
                'high': {
                    'min': float(self.df['High'].min()),
                    'max': float(self.df['High'].max()),
                    'mean': float(self.df['High'].mean()),
                    'median': float(self.df['High'].median()),
                    'std': float(self.df['High'].std()),
                    'var': float(self.df['High'].var()),
                    'q25': float(self.df['High'].quantile(0.25)),
                    'q75': float(self.df['High'].quantile(0.75))
                },
                'low': {
                    'min': float(self.df['Low'].min()),
                    'max': float(self.df['Low'].max()),
                    'mean': float(self.df['Low'].mean()),
                    'median': float(self.df['Low'].median()),
                    'std': float(self.df['Low'].std()),
                    'var': float(self.df['Low'].var()),
                    'q25': float(self.df['Low'].quantile(0.25)),
                    'q75': float(self.df['Low'].quantile(0.75))
                },
                'close': {
                    'min': float(self.df['Close'].min()),
                    'max': float(self.df['Close'].max()),
                    'mean': float(self.df['Close'].mean()),
                    'median': float(self.df['Close'].median()),
                    'std': float(self.df['Close'].std()),
                    'var': float(self.df['Close'].var()),
                    'q25': float(self.df['Close'].quantile(0.25)),
                    'q75': float(self.df['Close'].quantile(0.75))
                },
                'spread': {
                    'min': float((self.df['High'] - self.df['Low']).min()),
                    'max': float((self.df['High'] - self.df['Low']).max()),
                    'mean': float((self.df['High'] - self.df['Low']).mean()),
                    'median': float((self.df['High'] - self.df['Low']).median()),
                    'std': float((self.df['High'] - self.df['Low']).std())
                }
            },
            'volume_statistics': {
                'min': int(self.df['Volume'].min()),
                'max': int(self.df['Volume'].max()),
                'mean': float(self.df['Volume'].mean()),
                'median': float(self.df['Volume'].median()),
                'std': float(self.df['Volume'].std()),
                'total': int(self.df['Volume'].sum()),
                'quantiles': {
                    'q10': float(self.df['Volume'].quantile(0.1)),
                    'q25': float(self.df['Volume'].quantile(0.25)),
                    'q75': float(self.df['Volume'].quantile(0.75)),
                    'q90': float(self.df['Volume'].quantile(0.9)),
                    'q95': float(self.df['Volume'].quantile(0.95)),
                    'q99': float(self.df['Volume'].quantile(0.99))
                }
            },
            'returns_analysis': {
                'mean_return': float(returns.mean()),
                'std_return': float(returns.std()),
                'min_return': float(returns.min()),
                'max_return': float(returns.max()),
                'skewness': float(returns.skew()),
                'kurtosis': float(returns.kurtosis()),
                'sharpe_ratio_annualized': float((returns.mean() * 252 * 24 * 4) / (returns.std() * np.sqrt(252 * 24 * 4))),
                'jarque_bera_stat': float(stats.jarque_bera(returns)[0]),
                'jarque_bera_pvalue': float(stats.jarque_bera(returns)[1]),
                'is_normal_distribution': bool(stats.jarque_bera(returns)[1] >= 0.05)
            },
            'time_analysis': {
                'hourly_volume': {
                    str(hour): float(vol) for hour, vol in 
                    self.df.groupby(self.df['Zeit'].dt.hour)['Volume'].mean().items()
                },
                'daily_spread': {
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day]: float(spread)
                    for day, spread in self.df.groupby(self.df['Zeit'].dt.dayofweek)['Spread'].mean().items()
                    if 'Spread' in self.df.columns
                },
                'yearly_stats': {
                    str(year): {
                        'min_price': float(group['Close'].min()),
                        'max_price': float(group['Close'].max()),
                        'avg_price': float(group['Close'].mean()),
                        'avg_volume': float(group['Volume'].mean())
                    }
                    for year, group in self.df.groupby(self.df['Zeit'].dt.year)
                }
            },
            'correlation_matrix': {
                col1: {
                    col2: float(self.df[['Open', 'High', 'Low', 'Close', 'Volume']].corr().loc[col1, col2])
                    for col2 in ['Open', 'High', 'Low', 'Close', 'Volume']
                }
                for col1 in ['Open', 'High', 'Low', 'Close', 'Volume']
            }
        }
        
        # Füge Spread-Spalte hinzu falls nicht vorhanden
        if 'Spread' not in self.df.columns:
            self.df['Spread'] = self.df['High'] - self.df['Low']
            
        # Kombiniere mit bereits gesammelten erweiterten Statistiken
        all_statistics = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_source': self.csv_path,
            **basic_stats,
            **self.statistics_dict
        }
        
        # Speichere als JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_statistics, f, indent=2, ensure_ascii=False)
            
        print(f"\n💾 STATISTIKEN GESPEICHERT:")
        print(f"   Datei: {output_path}")
        print(f"   Größe: {len(json.dumps(all_statistics, indent=2))} Zeichen")
        print(f"   Kategorien: {len(all_statistics)} Hauptkategorien")
        
        return output_path
        
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
        self.advanced_statistics()
        self.market_regime_analysis()
        self.generate_visualizations()
        
        # Speichere alle Statistiken als JSON
        json_path = self.save_statistics_to_json()
        
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
        print(f"   💾 JSON-Export: {json_path}")

if __name__ == "__main__":
    # Pfad zur Datendatei
    data_path = "c:\\Users\\Wael\\Desktop\\Projekts\\smartEA\\data\\XAUUSD_M15_full.csv"
    
    # Erstelle Statistik-Objekt und führe Analyse durch
    gold_stats = GoldDataStatistics(data_path)
    gold_stats.run_complete_analysis()