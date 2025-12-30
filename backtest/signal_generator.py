#!/usr/bin/env python3
"""
Trading Signal Generator Klasse
Verwaltet alle Entry/Exit Signal-Logiken
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class SignalGenerator:
    """
    Klasse für die Generierung von Trading-Signalen
    
    Verwaltet:
    - RSI-basierte Entry/Exit Signale
    - Volume-Filter
    - Trend-Filter (SMA)
    - Trading-Zeiten Filter
    """
    
    def __init__(self, rsi_oversold: int = 30, rsi_overbought: int = 70, 
                 volume_threshold: float = 1.2, use_trend_filter: bool = True):
        """
        Initialisiert den Signal Generator
        
        Args:
            rsi_oversold: RSI Oversold Level für Buy-Signale
            rsi_overbought: RSI Overbought Level für Sell-Signale
            volume_threshold: Mindest-Volume Ratio für Signale
            use_trend_filter: Ob SMA20 Trend-Filter verwendet werden soll
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold = volume_threshold
        self.use_trend_filter = use_trend_filter
        
    def update_parameters(self, rsi_oversold: int = None, rsi_overbought: int = None,
                         volume_threshold: float = None, use_trend_filter: bool = None):
        """Aktualisiert die Signal-Parameter"""
        if rsi_oversold is not None:
            self.rsi_oversold = rsi_oversold
        if rsi_overbought is not None:
            self.rsi_overbought = rsi_overbought
        if volume_threshold is not None:
            self.volume_threshold = volume_threshold
        if use_trend_filter is not None:
            self.use_trend_filter = use_trend_filter
    
    def _get_trading_hours_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Erstellt Trading-Zeiten Filter (8:00 - 21:00)
        
        Args:
            df: DataFrame mit DateTime Index
            
        Returns:
            Boolean Series für Trading-Zeiten
        """
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        return df_copy['hour'].isin(range(8, 21))
    
    def _get_trend_filter(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Erstellt Trend-Filter basierend auf SMA20
        
        Args:
            df: DataFrame mit OHLC Daten
            
        Returns:
            Tuple von (trend_up, trend_down) Boolean Series
        """
        if not self.use_trend_filter:
            # Wenn kein Trend-Filter, dann alle True
            return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
        
        # SMA20 aus Daten oder berechnen
        if 'SMA20' in df.columns:
            sma20 = df['SMA20']
        else:
            sma20 = df['Close'].rolling(20).mean()
        
        trend_up = df['Close'] > sma20
        trend_down = df['Close'] < sma20
        
        return trend_up, trend_down
    
    def _get_rsi_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generiert RSI-basierte Buy/Sell Bedingungen
        
        Args:
            df: DataFrame mit RSI Daten
            
        Returns:
            Tuple von (buy_condition, sell_condition) Boolean Series
        """
        # RSI aus Daten oder Default-Werte
        if 'RSI14' in df.columns:
            rsi = df['RSI14']
        else:
            # Fallback: Neutral RSI
            rsi = pd.Series(50, index=df.index)
        
        buy_condition = rsi < self.rsi_oversold
        sell_condition = rsi > self.rsi_overbought
        
        return buy_condition, sell_condition
    
    def _get_volume_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Erstellt Volume-Filter
        
        Args:
            df: DataFrame mit Volume Ratio Daten
            
        Returns:
            Boolean Series für Volume-Filter
        """
        if 'VolumeRatio20' in df.columns:
            volume_ratio = df['VolumeRatio20']
        else:
            # Fallback: Normale Volume
            volume_ratio = pd.Series(1.0, index=df.index)
        
        return volume_ratio > self.volume_threshold
    
    def generate_entry_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generiert Entry-Signale (Long-Positionen)
        
        Args:
            df: DataFrame mit OHLC und Indikatoren
            
        Returns:
            Boolean Series mit Entry-Signalen (True = Entry)
        """
        # Einzelne Filter/Bedingungen
        trading_hours = self._get_trading_hours_filter(df)
        trend_up, _ = self._get_trend_filter(df)
        rsi_buy, _ = self._get_rsi_signals(df)
        volume_filter = self._get_volume_filter(df)
        
        # Kombinierte Entry-Bedingung
        entry_condition = (
            rsi_buy &           # RSI Oversold
            volume_filter &     # Volume hoch genug
            trading_hours &     # Trading-Zeiten
            trend_up           # Trend-Filter (falls aktiviert)
        )
        
        # Edge Detection: Nur bei Wechsel von False zu True
        entry_signals = entry_condition & (~entry_condition.shift(1).fillna(False))
        
        return entry_signals
    
    def generate_exit_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generiert Exit-Signale (Signal-basierte Exits)
        
        Args:
            df: DataFrame mit OHLC und Indikatoren
            
        Returns:
            Boolean Series mit Exit-Signalen (True = Exit)
        """
        # Einzelne Filter/Bedingungen
        trading_hours = self._get_trading_hours_filter(df)
        _, trend_down = self._get_trend_filter(df)
        _, rsi_sell = self._get_rsi_signals(df)
        volume_filter = self._get_volume_filter(df)
        
        # Kombinierte Exit-Bedingung
        exit_condition = (
            rsi_sell &          # RSI Overbought
            volume_filter &     # Volume hoch genug
            trading_hours &     # Trading-Zeiten
            trend_down         # Trend nach unten (falls Filter aktiv)
        )
        
        # Edge Detection: Nur bei Wechsel von False zu True
        exit_signals = exit_condition & (~exit_condition.shift(1).fillna(False))
        
        return exit_signals
    
    def generate_all_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generiert alle Signale und gibt zusätzliche Informationen zurück
        
        Args:
            df: DataFrame mit OHLC und Indikatoren
            
        Returns:
            Dictionary mit allen Signalen und Indikatoren
        """
        # Signale generieren
        entry_signals = self.generate_entry_signals(df)
        exit_signals = self.generate_exit_signals(df)
        
        # Zusätzliche Informationen
        trading_hours = self._get_trading_hours_filter(df)
        trend_up, trend_down = self._get_trend_filter(df)
        rsi_buy, rsi_sell = self._get_rsi_signals(df)
        volume_filter = self._get_volume_filter(df)
        
        # RSI und Volume Ratio für Anzeige
        rsi = df['RSI14'] if 'RSI14' in df.columns else pd.Series(50, index=df.index)
        volume_ratio = df['VolumeRatio20'] if 'VolumeRatio20' in df.columns else pd.Series(1.0, index=df.index)
        sma20 = df['SMA20'] if 'SMA20' in df.columns else df['Close'].rolling(20).mean()
        
        return {
            'entry_signals': entry_signals,
            'exit_signals': exit_signals,
            'trading_hours': trading_hours,
            'trend_up': trend_up,
            'trend_down': trend_down,
            'rsi_buy_condition': rsi_buy,
            'rsi_sell_condition': rsi_sell,
            'volume_filter': volume_filter,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'sma20': sma20
        }
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung der aktuellen Parameter zurück
        
        Returns:
            Dictionary mit Parameter-Zusammenfassung
        """
        return {
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'volume_threshold': self.volume_threshold,
            'use_trend_filter': self.use_trend_filter,
            'trading_hours': '08:00 - 21:00'
        }
