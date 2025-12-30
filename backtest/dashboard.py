#!/usr/bin/env python3
"""
Trading Dashboard Klasse
Verwaltet die Streamlit UI-Logik
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, Tuple
import hashlib

try:
    from .trading_strategy import TradingStrategy
    from .performance_analyzer import PerformanceAnalyzer
except ImportError:
    # Fallback für direkte Ausführung
    from trading_strategy import TradingStrategy
    from performance_analyzer import PerformanceAnalyzer


class TradingDashboard:
    """
    Klasse für das Streamlit Trading Dashboard
    
    Verwaltet:
    - UI-Layout und Komponenten
    - Parameter-Input und Validation
    - Chart-Generierung
    - Data Export Funktionen
    - Session State Management
    """
    
    def __init__(self):
        """Initialisiert das Trading Dashboard"""
        self.strategy = TradingStrategy()
        self.analyzer = PerformanceAnalyzer()
        
        # Session State Keys
        self.PORTFOLIO_KEY = 'enhanced_pf'
        self.SIGNALS_KEY = 'enhanced_signals'
        self.DF_KEY = 'enhanced_df'
        self.HASH_KEY = 'param_hash'
    
    def setup_page_config(self):
        """Konfiguriert die Streamlit Seite"""
        st.set_page_config(
            page_title="Enhanced Trading Strategy Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_header(self):
        """Rendert den Header-Bereich"""
        st.title("📈 Enhanced Trading Strategy Dashboard")
        st.markdown("**Live-Dashboard mit modularer Klassen-Struktur**")
        st.markdown("---")
    
    def render_sidebar_parameters(self) -> Tuple[Dict[str, Any], bool, bool]:
        """
        Rendert die Sidebar mit Parametern
        
        Returns:
            Tuple von (parameters_dict, run_backtest, auto_update)
        """
        st.sidebar.header("🔧 Strategy Parameters")
        
        # Daten-Sektion
        st.sidebar.subheader("📂 Data Source")
        use_default_path = st.sidebar.checkbox("Standard-Datenpfad verwenden", value=True)
        
        if use_default_path:
            data_path = r"C:\Users\Wael\Desktop\Projekts\smartEA\data\indicators\XAUUSD_M15_full_backup_with_indicators_backup.csv"
            st.sidebar.success("Standard: XAUUSD_M15_full_backup...")
        else:
            uploaded_file = st.sidebar.file_uploader("CSV-Datei hochladen", type=['csv'])
            if uploaded_file:
                data_path = uploaded_file
            else:
                st.sidebar.warning("Bitte Datei hochladen")
                return {}, False, False
        
        # Signal Parameters
        st.sidebar.subheader("📊 Signal Parameters")
        rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 40, 30, 1)
        rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 90, 70, 1)
        volume_threshold = st.sidebar.slider("Volume Threshold", 1.0, 3.0, 1.2, 0.1)
        use_trend_filter = st.sidebar.checkbox("SMA20 Trend-Filter aktivieren", value=True,
                                              help="Nur Long-Trades über SMA20")
        
        # Risk Management Parameters
        st.sidebar.subheader("🛡️ Risk Management")
        atr_multiplier = st.sidebar.slider("ATR Multiplier (Stop Loss)", 0.5, 3.0, 1.5, 0.1,
                                          help="Stop Loss = Entry - (ATR × Multiplier)")
        risk_reward_ratio = st.sidebar.slider("Risk:Reward Ratio", 1.0, 4.0, 2.0, 0.1,
                                             help="Take Profit = Stop Loss Distance × Ratio")
        
        # Portfolio Parameters
        st.sidebar.subheader("💰 Portfolio Settings")
        initial_cash = st.sidebar.number_input("Startkapital ($)", 1000, 100000, 10000, 1000)
        fees = st.sidebar.slider("Trading Fees (%)", 0.0, 0.5, 0.1, 0.01) / 100
        
        # Controls
        st.sidebar.subheader("⚡ Controls")
        auto_update = st.sidebar.checkbox("Auto-Update bei Parameter-Änderung", value=True)
        run_backtest = st.sidebar.button("🚀 Backtest Durchführen", type="primary")
        
        # Parameter Dictionary
        parameters = {
            'data_path': data_path,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'volume_threshold': volume_threshold,
            'use_trend_filter': use_trend_filter,
            'atr_multiplier': atr_multiplier,
            'risk_reward_ratio': risk_reward_ratio,
            'initial_cash': initial_cash,
            'fees': fees
        }
        
        return parameters, run_backtest, auto_update
    
    def _get_parameter_hash(self, params: Dict[str, Any]) -> str:
        """Erstellt Hash aus Parametern für Change Detection"""
        # Nur relevante Parameter für Hash (ohne data_path)
        hash_params = {k: v for k, v in params.items() if k != 'data_path'}
        param_string = str(sorted(hash_params.items()))
        return hashlib.md5(param_string.encode()).hexdigest()
    
    def _parameters_changed(self, params: Dict[str, Any]) -> bool:
        """Prüft ob sich Parameter geändert haben"""
        current_hash = self._get_parameter_hash(params)
        
        if self.HASH_KEY not in st.session_state:
            st.session_state[self.HASH_KEY] = current_hash
            return True
        
        if st.session_state[self.HASH_KEY] != current_hash:
            st.session_state[self.HASH_KEY] = current_hash
            return True
        
        return False
    
    @st.cache_data
    def load_data(_self, file_path) -> Optional[pd.DataFrame]:
        """
        Lädt Trading-Daten mit Caching
        
        Args:
            file_path: Pfad zur CSV-Datei
            
        Returns:
            DataFrame oder None bei Fehler
        """
        try:
            df = pd.read_csv(file_path)
            
            if 'Zeit' in df.columns:
                df['Zeit'] = pd.to_datetime(df['Zeit'], format='%d.%m.%Y %H:%M', errors='coerce')
                df = df.dropna(subset=['Zeit'])
                df.set_index('Zeit', inplace=True)
            
            return df
        
        except Exception as e:
            st.error(f"Fehler beim Laden der Daten: {e}")
            return None
    
    def run_strategy_backtest(self, df: pd.DataFrame, params: Dict[str, Any]):
        """
        Führt Strategie-Backtest durch und speichert Ergebnisse
        
        Args:
            df: Trading-Daten DataFrame
            params: Parameter Dictionary
        """
        try:
            # Strategy Parameter setzen
            self.strategy.update_signal_parameters(
                rsi_oversold=params['rsi_oversold'],
                rsi_overbought=params['rsi_overbought'],
                volume_threshold=params['volume_threshold'],
                use_trend_filter=params['use_trend_filter']
            )
            
            self.strategy.update_risk_parameters(
                atr_multiplier=params['atr_multiplier'],
                risk_reward_ratio=params['risk_reward_ratio']
            )
            
            self.strategy.update_portfolio_parameters(
                initial_cash=params['initial_cash'],
                fees=params['fees']
            )
            
            # Backtest durchführen
            portfolio, signals = self.strategy.run_backtest(df)
            
            # Ergebnisse in Session State speichern
            st.session_state[self.PORTFOLIO_KEY] = portfolio
            st.session_state[self.SIGNALS_KEY] = signals
            st.session_state[self.DF_KEY] = df
            
            st.sidebar.success("✅ Backtest erfolgreich!")
            
        except Exception as e:
            st.error(f"Fehler beim Backtest: {e}")
            st.sidebar.error("❌ Backtest fehlgeschlagen!")
    
    def render_performance_overview(self, portfolio, analyzer_report: Dict[str, Any]):
        """
        Rendert Performance-Übersicht
        
        Args:
            portfolio: VectorBT Portfolio Objekt
            analyzer_report: Performance Analyzer Report
        """
        st.header("📊 Performance Overview")
        
        # Basis-Metriken
        basic_metrics = analyzer_report.get('basic_metrics', {})
        trade_metrics = analyzer_report.get('trade_metrics', {})
        
        # Erste Zeile - Hauptkennzahlen
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = basic_metrics.get('total_return_pct', 0)
            st.metric("Total Return", f"{total_return:.2f}%",
                     delta=f"{total_return:.2f}%" if total_return > 0 else f"{total_return:.2f}%")
        
        with col2:
            max_dd = basic_metrics.get('max_drawdown_pct', 0)
            st.metric("Max Drawdown", f"{max_dd:.2f}%", delta=f"-{max_dd:.2f}%")
        
        with col3:
            win_rate = trade_metrics.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col4:
            total_trades = trade_metrics.get('total_trades', 0)
            st.metric("Total Trades", f"{total_trades:,}")
        
        # Zweite Zeile - Erweiterte Metriken
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            profit_factor = trade_metrics.get('profit_factor', 0)
            if profit_factor == float('inf'):
                st.metric("Profit Factor", "∞")
            else:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        with col6:
            sharpe = basic_metrics.get('sharpe_ratio', 0)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with col7:
            expectancy = trade_metrics.get('expectancy', 0)
            st.metric("Expectancy", f"${expectancy:.2f}")
        
        with col8:
            avg_duration = trade_metrics.get('avg_trade_duration_hours', 0)
            st.metric("Avg Trade Duration", f"{avg_duration:.1f}h")
    
    def render_detailed_analysis(self, analyzer_report: Dict[str, Any]):
        """
        Rendert detaillierte Analyse-Sektion
        
        Args:
            analyzer_report: Performance Analyzer Report
        """
        st.header("📋 Detailed Analysis")
        
        trade_metrics = analyzer_report.get('trade_metrics', {})
        drawdown_metrics = analyzer_report.get('drawdown_analysis', {})
        monthly_metrics = analyzer_report.get('monthly_returns', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Trade Analysis")
            if 'error' not in trade_metrics:
                st.write(f"**Winning Trades:** {trade_metrics.get('winning_trades', 0):,}")
                st.write(f"**Losing Trades:** {trade_metrics.get('losing_trades', 0):,}")
                st.write(f"**Win Rate:** {trade_metrics.get('win_rate', 0):.1f}%")
                st.write(f"**Avg Win:** ${trade_metrics.get('avg_win', 0):.2f}")
                st.write(f"**Avg Loss:** ${trade_metrics.get('avg_loss', 0):.2f}")
                st.write(f"**Risk:Reward:** 1:{trade_metrics.get('risk_reward_ratio', 0):.2f}")
        
        with col2:
            st.subheader("📉 Drawdown Analysis")
            if 'error' not in drawdown_metrics:
                st.write(f"**Max Drawdown:** {drawdown_metrics.get('max_drawdown_pct', 0):.2f}%")
                st.write(f"**Avg Drawdown:** {drawdown_metrics.get('avg_drawdown_pct', 0):.2f}%")
                st.write(f"**Drawdown Periods:** {drawdown_metrics.get('drawdown_periods', 0)}")
                st.write(f"**Avg Duration:** {drawdown_metrics.get('drawdown_duration_days', 0):.1f} days")
                st.write(f"**Current Drawdown:** {drawdown_metrics.get('current_drawdown_pct', 0):.2f}%")
        
        with col3:
            st.subheader("📅 Monthly Performance")
            if 'error' not in monthly_metrics and not monthly_metrics.get('insufficient_data', False):
                st.write(f"**Avg Monthly Return:** {monthly_metrics.get('avg_monthly_return_pct', 0):.2f}%")
                st.write(f"**Monthly Volatility:** {monthly_metrics.get('monthly_volatility_pct', 0):.2f}%")
                st.write(f"**Best Month:** {monthly_metrics.get('best_month_pct', 0):.2f}%")
                st.write(f"**Worst Month:** {monthly_metrics.get('worst_month_pct', 0):.2f}%")
                st.write(f"**Monthly Win Rate:** {monthly_metrics.get('monthly_win_rate', 0):.1f}%")
            else:
                st.write("Nicht genügend Daten für monatliche Analyse")
    
    def render_portfolio_chart(self, portfolio):
        """
        Rendert Portfolio Value Chart
        
        Args:
            portfolio: VectorBT Portfolio Objekt
        """
        st.subheader("Portfolio Value Over Time")
        
        portfolio_values = portfolio.value()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='gold', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 215, 0, 0.1)'
        ))
        
        fig.update_layout(
            title="Portfolio Development",
            xaxis_title="Zeit",
            yaxis_title="Portfolio Value ($)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_price_chart(self, df: pd.DataFrame, signals: Dict[str, Any]):
        """
        Rendert erweiterten Price Chart mit Signalen
        
        Args:
            df: Trading-Daten DataFrame
            signals: Signals Dictionary
        """
        st.subheader("Price Chart mit Signalen und Risk Levels")
        
        # Sample für Performance
        sample_size = min(5000, len(df))
        df_sample = df.tail(sample_size)
        
        # Signale samplen
        entry_signals = signals['entry_signals'].tail(sample_size)
        exit_signals = signals['combined_exit_signals'].tail(sample_size)
        sl_levels = signals['stop_loss_levels'].tail(sample_size)
        tp_levels = signals['take_profit_levels'].tail(sample_size)
        sma20 = signals['sma20'].tail(sample_size)
        rsi = signals['rsi'].tail(sample_size)
        
        # Subplots erstellen
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price, Signals & Risk Levels', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick Chart
        fig.add_trace(
            go.Candlestick(
                x=df_sample.index,
                open=df_sample['Open'],
                high=df_sample['High'],
                low=df_sample['Low'],
                close=df_sample['Close'],
                name='XAUUSD'
            ),
            row=1, col=1
        )
        
        # SMA20 Trend Line
        fig.add_trace(
            go.Scatter(
                x=df_sample.index,
                y=sma20,
                mode='lines',
                name='SMA20 (Trend)',
                line=dict(color='blue', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        # Entry Signals
        entry_points = df_sample.loc[entry_signals, 'Close']
        if not entry_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=entry_points.index,
                    y=entry_points.values,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='ENTRY'
                ),
                row=1, col=1
            )
        
        # Exit Signals
        exit_points = df_sample.loc[exit_signals, 'Close']
        if not exit_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=exit_points.index,
                    y=exit_points.values,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='EXIT'
                ),
                row=1, col=1
            )
        
        # Stop Loss Levels
        valid_sl = sl_levels.dropna()
        if not valid_sl.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_sl.index,
                    y=valid_sl.values,
                    mode='lines',
                    name='Stop Loss',
                    line=dict(color='red', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Take Profit Levels
        valid_tp = tp_levels.dropna()
        if not valid_tp.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_tp.index,
                    y=valid_tp.values,
                    mode='lines',
                    name='Take Profit',
                    line=dict(color='green', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df_sample.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # RSI Levels
        oversold = self.strategy.signal_generator.rsi_oversold
        overbought = self.strategy.signal_generator.rsi_overbought
        
        fig.add_hline(y=oversold, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=overbought, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_parameter_summary(self):
        """Rendert aktuelle Parameter-Zusammenfassung"""
        st.header("📋 Current Strategy Configuration")
        
        strategy_summary = self.strategy.get_strategy_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Signal Parameters")
            signal_params = strategy_summary['signal_parameters']
            for key, value in signal_params.items():
                if isinstance(value, bool):
                    st.write(f"**{key.replace('_', ' ').title()}:** {'✅ Aktiv' if value else '❌ Inaktiv'}")
                else:
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with col2:
            st.subheader("🛡️ Risk & Portfolio")
            risk_params = strategy_summary['risk_parameters']
            portfolio_params = strategy_summary['portfolio_config']
            
            for key, value in risk_params.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            st.write("---")
            for key, value in portfolio_params.items():
                if key == 'initial_cash':
                    st.write(f"**{key.replace('_', ' ').title()}:** ${value:,}")
                elif key == 'fees':
                    st.write(f"**{key.replace('_', ' ').title()}:** {value*100:.2f}%")
                else:
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    def render_export_section(self, portfolio, analyzer_report: Dict[str, Any]):
        """
        Rendert Export-Sektion
        
        Args:
            portfolio: VectorBT Portfolio Objekt
            analyzer_report: Performance Analyzer Report
        """
        st.header("📁 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Performance Report"):
                report_json = pd.Series(analyzer_report).to_json()
                st.download_button(
                    label="📥 Performance Report herunterladen",
                    data=report_json,
                    file_name=f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Download Trade Details"):
                try:
                    trades_df = portfolio.trades.records_readable
                    trades_csv = trades_df.to_csv()
                    st.download_button(
                        label="📥 Trade Details herunterladen",
                        data=trades_csv,
                        file_name=f"trade_details_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except:
                    st.warning("Keine Trade-Details verfügbar")
        
        with col3:
            if st.button("📈 Download Portfolio Values"):
                portfolio_csv = portfolio.value().to_csv()
                st.download_button(
                    label="📥 Portfolio Values herunterladen",
                    data=portfolio_csv,
                    file_name=f"portfolio_values_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def run(self):
        """Hauptfunktion - startet das Dashboard"""
        # Page Setup
        self.setup_page_config()
        self.render_header()
        
        # Sidebar Parameter
        params, run_backtest, auto_update = self.render_sidebar_parameters()
        
        if not params:
            return
        
        # Daten laden
        df = self.load_data(params['data_path'])
        if df is None:
            st.error("Fehler beim Laden der Daten")
            return
        
        st.sidebar.success(f"✅ {len(df):,} Datenpunkte geladen")
        
        # Verfügbare Indikatoren anzeigen
        available_indicators = [col for col in df.columns if col in ['RSI14', 'ATR14', 'SMA20', 'VolumeRatio20']]
        st.sidebar.info(f"Indikatoren: {', '.join(available_indicators)}")
        
        # Parameter-Änderung prüfen
        params_changed = self._parameters_changed(params)
        should_run_backtest = (
            run_backtest or
            (auto_update and params_changed) or
            self.PORTFOLIO_KEY not in st.session_state
        )
        
        # Backtest ausführen
        if should_run_backtest:
            with st.spinner('Führe Backtest durch...'):
                if params_changed and auto_update:
                    st.sidebar.info("📊 Parameter geändert - Aktualisiere...")
                
                self.run_strategy_backtest(df, params)
        
        # Ergebnisse anzeigen
        if self.PORTFOLIO_KEY in st.session_state:
            portfolio = st.session_state[self.PORTFOLIO_KEY]
            signals = st.session_state[self.SIGNALS_KEY]
            
            # Performance-Analyse
            analyzer_report = self.analyzer.generate_performance_report(portfolio)
            
            # UI Komponenten rendern
            self.render_performance_overview(portfolio, analyzer_report)
            self.render_detailed_analysis(analyzer_report)
            
            # Charts
            st.header("📈 Charts")
            self.render_portfolio_chart(portfolio)
            self.render_price_chart(df, signals)
            
            # Parameter Summary
            self.render_parameter_summary()
            
            # Export
            self.render_export_section(portfolio, analyzer_report)
        
        else:
            st.info("👆 Klicken Sie auf 'Backtest Durchführen' um zu starten")
