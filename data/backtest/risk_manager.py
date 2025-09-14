#!/usr/bin/env python3
"""
Risk Management Klasse
Verwaltet Stop-Loss und Take-Profit Logik
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


class RiskManager:
    """
    Klasse für Risk Management
    
    Verwaltet:
    - Stop-Loss Berechnung und Signale
    - Take-Profit Berechnung und Signale
    - ATR-basierte Position Sizing
    - Risk-Reward Ratio Management
    """
    
    def __init__(self, atr_multiplier: float = 1.5, risk_reward_ratio: float = 2.0):
        """
        Initialisiert den Risk Manager
        
        Args:
            atr_multiplier: ATR Multiplikator für Stop-Loss Distanz
            risk_reward_ratio: Risk-Reward Verhältnis (Take-Profit = Stop-Loss * Ratio)
        """
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        
        # Tracking der aktuellen Positionen
        self._current_entry_prices = None
        self._current_sl_levels = None
        self._current_tp_levels = None
    
    def update_parameters(self, atr_multiplier: float = None, risk_reward_ratio: float = None):
        """
        Aktualisiert Risk Management Parameter
        
        Args:
            atr_multiplier: Neuer ATR Multiplikator
            risk_reward_ratio: Neues Risk-Reward Verhältnis
        """
        if atr_multiplier is not None:
            self.atr_multiplier = atr_multiplier
        if risk_reward_ratio is not None:
            self.risk_reward_ratio = risk_reward_ratio
    
    def _get_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Holt ATR aus Daten oder berechnet Default-Werte
        
        Args:
            df: DataFrame mit OHLC Daten
            
        Returns:
            ATR Series
        """
        if 'ATR14' in df.columns:
            return df['ATR14']
        else:
            # Fallback: Einfache ATR Berechnung
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            return true_range.rolling(14).mean()
    
    def calculate_stop_loss_levels(self, df: pd.DataFrame, entry_signals: pd.Series) -> pd.Series:
        """
        Berechnet Stop-Loss Levels für Long-Positionen
        
        Args:
            df: DataFrame mit OHLC Daten
            entry_signals: Boolean Series mit Entry-Signalen
            
        Returns:
            Series mit Stop-Loss Levels
        """
        atr = self._get_atr(df)
        
        # Entry-Preise bestimmen
        entry_prices = df['Close'].where(entry_signals)
        entry_prices = entry_prices.fillna(method='ffill')  # Forward fill für aktive Positionen
        
        # Stop-Loss = Entry-Preis - (ATR * Multiplier)
        stop_loss_levels = entry_prices - (atr * self.atr_multiplier)
        
        # Nur gültige Stop-Loss Levels (wo Entry-Preise existieren)
        stop_loss_levels = stop_loss_levels.where(entry_prices.notna())
        
        self._current_entry_prices = entry_prices
        self._current_sl_levels = stop_loss_levels
        
        return stop_loss_levels
    
    def calculate_take_profit_levels(self, df: pd.DataFrame, entry_signals: pd.Series) -> pd.Series:
        """
        Berechnet Take-Profit Levels für Long-Positionen
        
        Args:
            df: DataFrame mit OHLC Daten
            entry_signals: Boolean Series mit Entry-Signalen
            
        Returns:
            Series mit Take-Profit Levels
        """
        atr = self._get_atr(df)
        
        # Entry-Preise bestimmen
        entry_prices = df['Close'].where(entry_signals)
        entry_prices = entry_prices.fillna(method='ffill')
        
        # Take-Profit = Entry-Preis + (ATR * Multiplier * Risk-Reward-Ratio)
        take_profit_levels = entry_prices + (atr * self.atr_multiplier * self.risk_reward_ratio)
        
        # Nur gültige Take-Profit Levels
        take_profit_levels = take_profit_levels.where(entry_prices.notna())
        
        self._current_tp_levels = take_profit_levels
        
        return take_profit_levels
    
    def generate_stop_loss_signals(self, df: pd.DataFrame, stop_loss_levels: pd.Series) -> pd.Series:
        """
        Generiert Stop-Loss Exit-Signale
        
        Args:
            df: DataFrame mit OHLC Daten
            stop_loss_levels: Series mit Stop-Loss Levels
            
        Returns:
            Boolean Series mit Stop-Loss Signalen (True = Stop-Loss getroffen)
        """
        # Stop-Loss wird getroffen wenn Low <= Stop-Loss Level
        stop_loss_hits = (df['Low'] <= stop_loss_levels) & (stop_loss_levels.notna())
        
        return stop_loss_hits
    
    def generate_take_profit_signals(self, df: pd.DataFrame, take_profit_levels: pd.Series) -> pd.Series:
        """
        Generiert Take-Profit Exit-Signale
        
        Args:
            df: DataFrame mit OHLC Daten
            take_profit_levels: Series mit Take-Profit Levels
            
        Returns:
            Boolean Series mit Take-Profit Signalen (True = Take-Profit getroffen)
        """
        # Take-Profit wird getroffen wenn High >= Take-Profit Level
        take_profit_hits = (df['High'] >= take_profit_levels) & (take_profit_levels.notna())
        
        return take_profit_hits
    
    def calculate_position_size(self, account_balance: float, risk_percent: float,
                               entry_price: float, stop_loss_price: float) -> float:
        """
        Berechnet optimale Positionsgröße basierend auf Risiko
        
        Args:
            account_balance: Aktueller Kontostand
            risk_percent: Risiko pro Trade in Prozent (z.B. 2.0 für 2%)
            entry_price: Entry-Preis
            stop_loss_price: Stop-Loss Preis
            
        Returns:
            Positionsgröße in Einheiten
        """
        risk_amount = account_balance * (risk_percent / 100)
        price_difference = abs(entry_price - stop_loss_price)
        
        if price_difference == 0:
            return 0
        
        position_size = risk_amount / price_difference
        return position_size
    
    def get_risk_metrics(self, df: pd.DataFrame, entry_signals: pd.Series) -> Dict[str, Any]:
        """
        Berechnet Risk-Metriken für die aktuelle Konfiguration
        
        Args:
            df: DataFrame mit OHLC Daten
            entry_signals: Boolean Series mit Entry-Signalen
            
        Returns:
            Dictionary mit Risk-Metriken
        """
        atr = self._get_atr(df)
        
        # Durchschnittliche ATR
        avg_atr = atr.mean()
        
        # Durchschnittliche Stop-Loss Distanz
        avg_sl_distance = avg_atr * self.atr_multiplier
        
        # Durchschnittliche Take-Profit Distanz
        avg_tp_distance = avg_sl_distance * self.risk_reward_ratio
        
        # Risk-Reward Verhältnis
        actual_risk_reward = avg_tp_distance / avg_sl_distance if avg_sl_distance > 0 else 0
        
        return {
            'avg_atr': avg_atr,
            'avg_stop_loss_distance': avg_sl_distance,
            'avg_take_profit_distance': avg_tp_distance,
            'atr_multiplier': self.atr_multiplier,
            'risk_reward_ratio': self.risk_reward_ratio,
            'actual_risk_reward': actual_risk_reward
        }
    
    def generate_all_risk_signals(self, df: pd.DataFrame, entry_signals: pd.Series) -> Dict[str, pd.Series]:
        """
        Generiert alle Risk Management Signale
        
        Args:
            df: DataFrame mit OHLC Daten
            entry_signals: Boolean Series mit Entry-Signalen
            
        Returns:
            Dictionary mit allen Risk Management Signalen und Levels
        """
        # Stop-Loss und Take-Profit Levels berechnen
        stop_loss_levels = self.calculate_stop_loss_levels(df, entry_signals)
        take_profit_levels = self.calculate_take_profit_levels(df, entry_signals)
        
        # Exit-Signale generieren
        stop_loss_signals = self.generate_stop_loss_signals(df, stop_loss_levels)
        take_profit_signals = self.generate_take_profit_signals(df, take_profit_levels)
        
        # Kombinierte Risk-basierte Exits
        risk_exit_signals = stop_loss_signals | take_profit_signals
        
        return {
            'stop_loss_levels': stop_loss_levels,
            'take_profit_levels': take_profit_levels,
            'stop_loss_signals': stop_loss_signals,
            'take_profit_signals': take_profit_signals,
            'risk_exit_signals': risk_exit_signals,
            'entry_prices': self._current_entry_prices
        }
    
    def analyze_risk_performance(self, df: pd.DataFrame, portfolio_results: Any) -> Dict[str, Any]:
        """
        Analysiert Risk Management Performance
        
        Args:
            df: DataFrame mit OHLC Daten
            portfolio_results: Portfolio-Ergebnisse von VectorBT
            
        Returns:
            Dictionary mit Risk Performance Analyse
        """
        try:
            trades = portfolio_results.trades.records_readable
            
            if len(trades) == 0:
                return {'no_trades': True}
            
            # Stop-Loss vs Take-Profit Analyse
            sl_trades = 0
            tp_trades = 0
            signal_trades = 0
            
            # Vereinfachte Analyse basierend auf PnL
            for _, trade in trades.iterrows():
                pnl = trade['PnL']
                # Heuristische Klassifizierung
                if pnl < 0:
                    sl_trades += 1  # Verlust → wahrscheinlich Stop-Loss
                elif pnl > 0:
                    tp_trades += 1  # Gewinn → wahrscheinlich Take-Profit
                else:
                    signal_trades += 1
            
            total_trades = len(trades)
            
            return {
                'total_trades': total_trades,
                'stop_loss_trades': sl_trades,
                'take_profit_trades': tp_trades,
                'signal_exit_trades': signal_trades,
                'sl_percentage': (sl_trades / total_trades * 100) if total_trades > 0 else 0,
                'tp_percentage': (tp_trades / total_trades * 100) if total_trades > 0 else 0,
                'signal_percentage': (signal_trades / total_trades * 100) if total_trades > 0 else 0,
                'avg_sl_pnl': trades[trades['PnL'] < 0]['PnL'].mean() if sl_trades > 0 else 0,
                'avg_tp_pnl': trades[trades['PnL'] > 0]['PnL'].mean() if tp_trades > 0 else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung der aktuellen Risk Management Parameter zurück
        
        Returns:
            Dictionary mit Parameter-Zusammenfassung
        """
        return {
            'atr_multiplier': self.atr_multiplier,
            'risk_reward_ratio': self.risk_reward_ratio,
            'stop_loss_method': 'ATR-basiert',
            'take_profit_method': 'Risk-Reward basiert'
        }
