#!/usr/bin/env python3
"""
Enhanced Streamlit Trading Dashboard - Refactorisiert
Modulare Klassen-Struktur für bessere Wartung und Erweiterung

Klassen-Struktur:
- SignalGenerator: Entry/Exit Signale, RSI-Logik, Volume-Filter, Trend-Filter
- RiskManager: Stop-Loss, Take-Profit, ATR-basierte Risk-Management Logik  
- TradingStrategy: Koordiniert SignalGenerator und RiskManager
- PerformanceAnalyzer: Erweiterte Trade-Metriken und Statistiken
- TradingDashboard: Streamlit UI-Logik und Charts
"""

import warnings
warnings.filterwarnings('ignore')

# VectorBT Import Check
try:
    import vectorbt as vbt
    print(f"✅ VectorBT {vbt.__version__} geladen")
except ImportError:
    print("❌ VectorBT nicht installiert! Führen Sie aus: pip install vectorbt")
    exit(1)

# Dashboard Import
import sys
import os

# Stelle sicher, dass das aktuelle Verzeichnis im Python-Pfad ist
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from dashboard import TradingDashboard
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("Stelle sicher, dass alle Klassen-Dateien im gleichen Verzeichnis sind:")
    print("- signal_generator.py")
    print("- risk_manager.py") 
    print("- trading_strategy.py")
    print("- performance_analyzer.py")
    print("- dashboard.py")
    exit(1)


def main():
    """
    Hauptfunktion - startet das modulare Trading Dashboard
    """
    print("🚀 Starting Enhanced Trading Dashboard...")
    print("📊 Modulare Klassen-Struktur geladen:")
    print("   • SignalGenerator - Entry/Exit Signale")
    print("   • RiskManager - Stop-Loss/Take-Profit")
    print("   • TradingStrategy - Strategie-Koordination")
    print("   • PerformanceAnalyzer - Erweiterte Metriken")
    print("   • TradingDashboard - Streamlit UI")
    print("-" * 50)
    
    # Dashboard erstellen und starten
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
