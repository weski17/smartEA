#!/usr/bin/env python3
"""
Performance Analyzer Klasse
Berechnet erweiterte Trading-Metriken und Statistiken
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """
    Klasse für erweiterte Performance-Analyse
    
    Berechnet:
    - Erweiterte Trade-Metriken
    - Drawdown-Analyse
    - Risk-adjusted Returns
    - Consecutive Win/Loss Streaks
    - Monthly/Weekly Performance
    """
    
    def __init__(self):
        """Initialisiert den Performance Analyzer"""
        self._last_trades = None
        self._last_portfolio_values = None
        
    def calculate_basic_metrics(self, portfolio_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Berechnet Basis-Metriken aus VectorBT Portfolio Stats
        
        Args:
            portfolio_stats: VectorBT Portfolio Stats Dictionary
            
        Returns:
            Dictionary mit Basis-Metriken
        """
        return {
            'total_return_pct': portfolio_stats.get('Total Return [%]', 0),
            'total_return_abs': portfolio_stats.get('Total Return', 0),
            'max_drawdown_pct': portfolio_stats.get('Max Drawdown [%]', 0),
            'max_drawdown_abs': portfolio_stats.get('Max Drawdown', 0),
            'sharpe_ratio': portfolio_stats.get('Sharpe Ratio', 0),
            'calmar_ratio': portfolio_stats.get('Calmar Ratio', 0),
            'max_drawdown_duration': portfolio_stats.get('Max Drawdown Duration', pd.Timedelta(0)),
            'total_trades': portfolio_stats.get('Total Trades', 0)
        }
    
    def calculate_trade_metrics(self, portfolio_obj: Any) -> Dict[str, Any]:
        """
        Berechnet erweiterte Trade-Metriken
        
        Args:
            portfolio_obj: VectorBT Portfolio Objekt
            
        Returns:
            Dictionary mit Trade-Metriken
        """
        try:
            trades = portfolio_obj.trades.records_readable
            self._last_trades = trades
            
            if len(trades) == 0:
                return self._empty_trade_metrics()
            
            # Basis-Metriken
            total_trades = len(trades)
            winning_trades = len(trades[trades['PnL'] > 0])
            losing_trades = len(trades[trades['PnL'] < 0])
            break_even_trades = len(trades[trades['PnL'] == 0])
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            loss_rate = (losing_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Gewinn/Verlust Analyse
            wins = trades[trades['PnL'] > 0]['PnL']
            losses = trades[trades['PnL'] < 0]['PnL']
            
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            
            max_win = wins.max() if len(wins) > 0 else 0
            max_loss = abs(losses.min()) if len(losses) > 0 else 0
            
            # Profit Factor
            total_wins = wins.sum() if len(wins) > 0 else 0
            total_losses = abs(losses.sum()) if len(losses) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Risk-Reward Verhältnis
            risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Consecutive Wins/Losses
            consecutive_metrics = self._calculate_consecutive_metrics(trades)
            
            # Trade Duration Analyse
            duration_metrics = self._calculate_duration_metrics(trades)
            
            # Expectancy
            expectancy = (avg_win * win_rate / 100) - (avg_loss * loss_rate / 100)
            
            # Kelly Criterion
            kelly_percent = self._calculate_kelly_criterion(win_rate / 100, avg_win, avg_loss)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'break_even_trades': break_even_trades,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_win': max_win,
                'max_loss': max_loss,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'profit_factor': profit_factor,
                'risk_reward_ratio': risk_reward_ratio,
                'expectancy': expectancy,
                'kelly_percent': kelly_percent,
                **consecutive_metrics,
                **duration_metrics
            }
            
        except Exception as e:
            return {'error': f"Fehler bei Trade-Metriken: {str(e)}"}
    
    def _empty_trade_metrics(self) -> Dict[str, Any]:
        """Gibt leere Trade-Metriken zurück wenn keine Trades vorhanden"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'break_even_trades': 0,
            'win_rate': 0,
            'loss_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_win': 0,
            'max_loss': 0,
            'total_wins': 0,
            'total_losses': 0,
            'profit_factor': 0,
            'risk_reward_ratio': 0,
            'expectancy': 0,
            'kelly_percent': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak': 0,
            'avg_trade_duration_hours': 0,
            'median_trade_duration_hours': 0,
            'min_trade_duration_hours': 0,
            'max_trade_duration_hours': 0
        }
    
    def _calculate_consecutive_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Berechnet Consecutive Win/Loss Metriken"""
        if len(trades) == 0:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0
            }
        
        # Trade-Ergebnisse als Win(1)/Loss(-1)/BreakEven(0)
        trade_results = []
        for pnl in trades['PnL']:
            if pnl > 0:
                trade_results.append(1)  # Win
            elif pnl < 0:
                trade_results.append(-1)  # Loss
            else:
                trade_results.append(0)  # Break Even
        
        # Consecutive Streaks berechnen
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for result in trade_results:
            if result == 1:  # Win
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif result == -1:  # Loss
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:  # Break Even
                current_wins = 0
                current_losses = 0
        
        # Aktueller Streak
        current_streak = current_wins if current_wins > 0 else -current_losses if current_losses > 0 else 0
        
        return {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'current_streak': current_streak
        }
    
    def _calculate_duration_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Berechnet Trade Duration Metriken"""
        if len(trades) == 0 or 'Entry Timestamp' not in trades.columns or 'Exit Timestamp' not in trades.columns:
            return {
                'avg_trade_duration_hours': 0,
                'median_trade_duration_hours': 0,
                'min_trade_duration_hours': 0,
                'max_trade_duration_hours': 0
            }
        
        # Duration in Stunden berechnen
        durations = (trades['Exit Timestamp'] - trades['Entry Timestamp']).dt.total_seconds() / 3600
        
        return {
            'avg_trade_duration_hours': durations.mean(),
            'median_trade_duration_hours': durations.median(),
            'min_trade_duration_hours': durations.min(),
            'max_trade_duration_hours': durations.max()
        }
    
    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Berechnet Kelly Criterion für optimale Position Size
        
        Args:
            win_rate: Win Rate als Dezimalzahl (0-1)
            avg_win: Durchschnittlicher Gewinn
            avg_loss: Durchschnittlicher Verlust (positiv)
            
        Returns:
            Kelly Percentage (kann negativ sein wenn Strategie unprofitabel)
        """
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        return kelly * 100  # Als Prozent
    
    def calculate_drawdown_analysis(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """
        Detaillierte Drawdown-Analyse
        
        Args:
            portfolio_values: Portfolio Value Series
            
        Returns:
            Dictionary mit Drawdown-Metriken
        """
        self._last_portfolio_values = portfolio_values
        
        if len(portfolio_values) == 0:
            return {
                'max_drawdown_pct': 0,
                'max_drawdown_abs': 0,
                'avg_drawdown_pct': 0,
                'drawdown_duration_days': 0,
                'recovery_time_days': 0,
                'drawdown_periods': 0
            }
        
        # Running Maximum (High Water Mark)
        running_max = portfolio_values.expanding().max()
        
        # Drawdown berechnen
        drawdown_abs = portfolio_values - running_max
        drawdown_pct = (drawdown_abs / running_max) * 100
        
        # Maximum Drawdown
        max_drawdown_pct = drawdown_pct.min()
        max_drawdown_abs = drawdown_abs.min()
        
        # Durchschnittlicher Drawdown (nur negative Werte)
        negative_drawdowns = drawdown_pct[drawdown_pct < 0]
        avg_drawdown_pct = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0
        
        # Drawdown Perioden identifizieren
        in_drawdown = drawdown_pct < -0.01  # Drawdown > 0.01%
        drawdown_periods = self._count_periods(in_drawdown)
        
        # Durchschnittliche Drawdown Duration
        avg_drawdown_duration = 0
        if drawdown_periods > 0:
            drawdown_changes = in_drawdown.astype(int).diff()
            drawdown_starts = drawdown_changes[drawdown_changes == 1].index
            drawdown_ends = drawdown_changes[drawdown_changes == -1].index
            
            # Letzte Periode handhaben
            if len(drawdown_starts) > len(drawdown_ends):
                drawdown_ends = drawdown_ends.append(pd.Index([portfolio_values.index[-1]]))
            
            if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
                durations = []
                for start, end in zip(drawdown_starts, drawdown_ends):
                    duration = (end - start).total_seconds() / (24 * 3600)  # in Tagen
                    durations.append(duration)
                avg_drawdown_duration = np.mean(durations) if durations else 0
        
        return {
            'max_drawdown_pct': abs(max_drawdown_pct),
            'max_drawdown_abs': abs(max_drawdown_abs),
            'avg_drawdown_pct': abs(avg_drawdown_pct),
            'drawdown_duration_days': avg_drawdown_duration,
            'drawdown_periods': drawdown_periods,
            'current_drawdown_pct': abs(drawdown_pct.iloc[-1]) if len(drawdown_pct) > 0 else 0
        }
    
    def _count_periods(self, boolean_series: pd.Series) -> int:
        """Zählt die Anzahl der True-Perioden in einer Boolean Series"""
        if len(boolean_series) == 0:
            return 0
        
        changes = boolean_series.astype(int).diff()
        periods = len(changes[changes == 1])
        return periods
    
    def calculate_monthly_returns(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """
        Berechnet monatliche Returns
        
        Args:
            portfolio_values: Portfolio Value Series
            
        Returns:
            Dictionary mit monatlichen Return-Statistiken
        """
        if len(portfolio_values) == 0:
            return {'error': 'Keine Portfolio-Werte verfügbar'}
        
        try:
            # Monatliche Werte (letzter Wert jedes Monats)
            monthly_values = portfolio_values.resample('M').last()
            
            if len(monthly_values) < 2:
                return {'insufficient_data': True}
            
            # Monatliche Returns berechnen
            monthly_returns = monthly_values.pct_change().dropna() * 100
            
            # Statistiken
            avg_monthly_return = monthly_returns.mean()
            std_monthly_return = monthly_returns.std()
            best_month = monthly_returns.max()
            worst_month = monthly_returns.min()
            positive_months = len(monthly_returns[monthly_returns > 0])
            negative_months = len(monthly_returns[monthly_returns < 0])
            total_months = len(monthly_returns)
            
            return {
                'avg_monthly_return_pct': avg_monthly_return,
                'monthly_volatility_pct': std_monthly_return,
                'best_month_pct': best_month,
                'worst_month_pct': worst_month,
                'positive_months': positive_months,
                'negative_months': negative_months,
                'total_months': total_months,
                'monthly_win_rate': (positive_months / total_months * 100) if total_months > 0 else 0
            }
            
        except Exception as e:
            return {'error': f"Fehler bei monatlichen Returns: {str(e)}"}
    
    def generate_performance_report(self, portfolio_obj: Any) -> Dict[str, Any]:
        """
        Generiert einen umfassenden Performance-Report
        
        Args:
            portfolio_obj: VectorBT Portfolio Objekt
            
        Returns:
            Dictionary mit vollständigem Performance-Report
        """
        try:
            # Portfolio Stats
            stats = portfolio_obj.stats()
            basic_metrics = self.calculate_basic_metrics(stats.to_dict())
            
            # Trade-Metriken
            trade_metrics = self.calculate_trade_metrics(portfolio_obj)
            
            # Portfolio Values
            portfolio_values = portfolio_obj.value()
            
            # Drawdown-Analyse
            drawdown_analysis = self.calculate_drawdown_analysis(portfolio_values)
            
            # Monatliche Returns
            monthly_returns = self.calculate_monthly_returns(portfolio_values)
            
            # Kombinierter Report
            report = {
                'basic_metrics': basic_metrics,
                'trade_metrics': trade_metrics,
                'drawdown_analysis': drawdown_analysis,
                'monthly_returns': monthly_returns,
                'report_timestamp': pd.Timestamp.now(),
                'data_period': {
                    'start': portfolio_values.index[0] if len(portfolio_values) > 0 else None,
                    'end': portfolio_values.index[-1] if len(portfolio_values) > 0 else None,
                    'total_days': (portfolio_values.index[-1] - portfolio_values.index[0]).days if len(portfolio_values) > 0 else 0
                }
            }
            
            return report
            
        except Exception as e:
            return {'error': f"Fehler bei Performance-Report: {str(e)}"}
    
    def compare_strategies(self, reports: List[Dict[str, Any]], 
                          strategy_names: List[str] = None) -> Dict[str, Any]:
        """
        Vergleicht mehrere Strategie-Reports
        
        Args:
            reports: Liste von Performance-Reports
            strategy_names: Optionale Namen für die Strategien
            
        Returns:
            Dictionary mit Strategie-Vergleich
        """
        if not reports:
            return {'error': 'Keine Reports zum Vergleichen'}
        
        if strategy_names is None:
            strategy_names = [f"Strategy_{i+1}" for i in range(len(reports))]
        
        comparison = {}
        
        # Key-Metriken extrahieren
        key_metrics = [
            'total_return_pct', 'max_drawdown_pct', 'sharpe_ratio',
            'win_rate', 'profit_factor', 'expectancy'
        ]
        
        for metric in key_metrics:
            comparison[metric] = {}
            for i, report in enumerate(reports):
                name = strategy_names[i] if i < len(strategy_names) else f"Strategy_{i+1}"
                
                # Metric aus verschiedenen Bereichen des Reports extrahieren
                value = None
                if 'basic_metrics' in report and metric in report['basic_metrics']:
                    value = report['basic_metrics'][metric]
                elif 'trade_metrics' in report and metric in report['trade_metrics']:
                    value = report['trade_metrics'][metric]
                elif 'drawdown_analysis' in report and metric in report['drawdown_analysis']:
                    value = report['drawdown_analysis'][metric]
                
                comparison[metric][name] = value
        
        # Beste Strategie für jede Metrik identifizieren
        best_strategies = {}
        for metric in key_metrics:
            if metric in comparison and comparison[metric]:
                values = {k: v for k, v in comparison[metric].items() if v is not None}
                if values:
                    if metric in ['max_drawdown_pct']:  # Niedrigere Werte sind besser
                        best_strategies[metric] = min(values, key=values.get)
                    else:  # Höhere Werte sind besser
                        best_strategies[metric] = max(values, key=values.get)
        
        return {
            'comparison': comparison,
            'best_strategies': best_strategies,
            'strategy_names': strategy_names
        }
    
    def export_report_to_csv(self, report: Dict[str, Any], filepath: str):
        """
        Exportiert Performance-Report als CSV
        
        Args:
            report: Performance-Report Dictionary
            filepath: Pfad für CSV-Export
        """
        try:
            # Flache Struktur für CSV erstellen
            flat_data = {}
            
            for section, data in report.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        flat_data[f"{section}_{key}"] = value
                else:
                    flat_data[section] = data
            
            # Als DataFrame und CSV speichern
            df = pd.DataFrame([flat_data])
            df.to_csv(filepath, index=False)
            
        except Exception as e:
            raise RuntimeError(f"Fehler beim CSV-Export: {str(e)}")
