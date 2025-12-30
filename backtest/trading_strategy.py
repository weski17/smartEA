#!/usr/bin/env python3
"""
Trading Strategy Klasse
Koordiniert SignalGenerator und RiskManager
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from .signal_generator import SignalGenerator
    from .risk_manager import RiskManager
except ImportError:
    # Fallback für direkte Ausführung
    from signal_generator import SignalGenerator
    from risk_manager import RiskManager


class TradingStrategy:
    """
    Hauptklasse für Trading-Strategien
    
    Koordiniert:
    - Signal Generation (Entry/Exit)
    - Risk Management (Stop-Loss/Take-Profit)
    - Portfolio-Erstellung mit VectorBT
    - Strategie-Ausführung
    """
    
    def __init__(self, initial_cash: float = 10000, fees: float = 0.001, freq: str = '15T'):
        """
        Initialisiert die Trading Strategy
        
        Args:
            initial_cash: Startkapital
            fees: Trading-Gebühren (als Dezimalzahl, z.B. 0.001 für 0.1%)
            freq: Frequenz der Daten (z.B. '15T' für 15 Minuten)
        """
        self.initial_cash = initial_cash
        self.fees = fees
        self.freq = freq
        
        # Komponenten
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        
        # Ergebnisse
        self._last_portfolio = None
        self._last_signals = None
        self._last_df = None
    
    def update_signal_parameters(self, rsi_oversold: int = None, rsi_overbought: int = None,
                                volume_threshold: float = None, use_trend_filter: bool = None):
        """
        Aktualisiert Signal-Parameter
        
        Args:
            rsi_oversold: RSI Oversold Level
            rsi_overbought: RSI Overbought Level  
            volume_threshold: Volume Threshold
            use_trend_filter: Trend-Filter aktivieren
        """
        self.signal_generator.update_parameters(
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            volume_threshold=volume_threshold,
            use_trend_filter=use_trend_filter
        )
    
    def update_risk_parameters(self, atr_multiplier: float = None, risk_reward_ratio: float = None):
        """
        Aktualisiert Risk Management Parameter
        
        Args:
            atr_multiplier: ATR Multiplikator für Stop-Loss
            risk_reward_ratio: Risk-Reward Verhältnis
        """
        self.risk_manager.update_parameters(
            atr_multiplier=atr_multiplier,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def update_portfolio_parameters(self, initial_cash: float = None, fees: float = None):
        """
        Aktualisiert Portfolio-Parameter
        
        Args:
            initial_cash: Neues Startkapital
            fees: Neue Trading-Gebühren
        """
        if initial_cash is not None:
            self.initial_cash = initial_cash
        if fees is not None:
            self.fees = fees
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validiert das Input DataFrame
        
        Args:
            df: DataFrame mit OHLC Daten
            
        Returns:
            True wenn DataFrame gültig ist
        """
        required_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Erforderliche Spalte '{col}' fehlt im DataFrame")
        
        if len(df) < 100:
            raise ValueError("DataFrame muss mindestens 100 Zeilen haben für sinnvolle Analyse")
        
        return True
    
    def run_backtest(self, df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """
        Führt kompletten Backtest durch
        
        Args:
            df: DataFrame mit OHLC und Indikatoren
            
        Returns:
            Tuple von (Portfolio-Objekt, Signals-Dictionary)
        """
        # DataFrame validieren
        self._validate_dataframe(df)
        
        # 1. Signale generieren
        signal_results = self.signal_generator.generate_all_signals(df)
        entry_signals = signal_results['entry_signals']
        signal_exit_signals = signal_results['exit_signals']
        
        # 2. Risk Management Signale generieren
        risk_results = self.risk_manager.generate_all_risk_signals(df, entry_signals)
        
        # 3. Alle Exit-Signale kombinieren
        combined_exit_signals = (
            signal_exit_signals |
            risk_results['risk_exit_signals']
        )
        
        # 4. Portfolio mit VectorBT erstellen
        try:
            portfolio = vbt.Portfolio.from_signals(
                close=df['Close'],
                entries=entry_signals,
                exits=combined_exit_signals,
                init_cash=self.initial_cash,
                fees=self.fees,
                freq=self.freq,
                sl_stop=risk_results['stop_loss_levels'],
                tp_stop=risk_results['take_profit_levels']
            )
        except Exception as e:
            raise RuntimeError(f"Fehler bei Portfolio-Erstellung: {e}")
        
        # 5. Alle Ergebnisse zusammenfassen
        all_signals = {
            **signal_results,
            **risk_results,
            'combined_exit_signals': combined_exit_signals
        }
        
        # Ergebnisse speichern
        self._last_portfolio = portfolio
        self._last_signals = all_signals
        self._last_df = df
        
        return portfolio, all_signals
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung der aktuellen Strategie-Konfiguration zurück
        
        Returns:
            Dictionary mit Strategie-Zusammenfassung
        """
        signal_params = self.signal_generator.get_parameter_summary()
        risk_params = self.risk_manager.get_parameter_summary()
        
        return {
            'portfolio_config': {
                'initial_cash': self.initial_cash,
                'fees': self.fees,
                'frequency': self.freq
            },
            'signal_parameters': signal_params,
            'risk_parameters': risk_params
        }
    
    def optimize_parameters(self, df: pd.DataFrame, param_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """
        Einfache Parameter-Optimierung (Grid Search)
        
        Args:
            df: DataFrame mit OHLC Daten
            param_ranges: Dictionary mit Parameter-Bereichen
                z.B. {'rsi_oversold': [20, 25, 30], 'atr_multiplier': [1.0, 1.5, 2.0]}
        
        Returns:
            Dictionary mit besten Parametern und Performance
        """
        best_performance = -np.inf
        best_params = {}
        results = []
        
        # Alle Kombinationen testen (vereinfachte Implementierung)
        rsi_values = param_ranges.get('rsi_oversold', [self.signal_generator.rsi_oversold])
        atr_values = param_ranges.get('atr_multiplier', [self.risk_manager.atr_multiplier])
        
        for rsi in rsi_values:
            for atr in atr_values:
                try:
                    # Parameter setzen
                    self.update_signal_parameters(rsi_oversold=rsi)
                    self.update_risk_parameters(atr_multiplier=atr)
                    
                    # Backtest durchführen
                    portfolio, _ = self.run_backtest(df)
                    
                    # Performance bewerten (Total Return)
                    performance = portfolio.stats()['Total Return [%]']
                    
                    result = {
                        'rsi_oversold': rsi,
                        'atr_multiplier': atr,
                        'performance': performance
                    }
                    results.append(result)
                    
                    # Beste Parameter aktualisieren
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {'rsi_oversold': rsi, 'atr_multiplier': atr}
                
                except Exception as e:
                    # Parameter-Kombination fehlgeschlagen
                    continue
        
        return {
            'best_params': best_params,
            'best_performance': best_performance,
            'all_results': results
        }
    
    def get_trade_analysis(self) -> Dict[str, Any]:
        """
        Analysiert die Trades des letzten Backtests
        
        Returns:
            Dictionary mit Trade-Analyse
        """
        if self._last_portfolio is None:
            return {'error': 'Kein Backtest durchgeführt'}
        
        try:
            trades = self._last_portfolio.trades.records_readable
            
            if len(trades) == 0:
                return {'no_trades': True}
            
            # Risk Management Performance
            risk_analysis = self.risk_manager.analyze_risk_performance(
                self._last_df, self._last_portfolio
            )
            
            # Basic Trade Stats
            total_trades = len(trades)
            winning_trades = len(trades[trades['PnL'] > 0])
            losing_trades = len(trades[trades['PnL'] < 0])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = trades[trades['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades[trades['PnL'] < 0]['PnL'].mean()) if losing_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'risk_analysis': risk_analysis
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def export_signals(self, filepath: str = None) -> pd.DataFrame:
        """
        Exportiert alle Signale als DataFrame
        
        Args:
            filepath: Optional - Pfad zum Speichern als CSV
            
        Returns:
            DataFrame mit allen Signalen
        """
        if self._last_signals is None or self._last_df is None:
            raise ValueError("Kein Backtest durchgeführt")
        
        # DataFrame mit Signalen erstellen
        signals_df = self._last_df.copy()
        
        # Signale hinzufügen
        for signal_name, signal_series in self._last_signals.items():
            if isinstance(signal_series, pd.Series):
                signals_df[f'signal_{signal_name}'] = signal_series
        
        # Optional speichern
        if filepath:
            signals_df.to_csv(filepath)
        
        return signals_df
    
    def get_last_portfolio(self):
        """Gibt das letzte Portfolio-Objekt zurück"""
        return self._last_portfolio
    
    def get_last_signals(self):
        """Gibt die letzten Signale zurück"""
        return self._last_signals
