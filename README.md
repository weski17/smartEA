# 📊 Machine Learning Quant Trading System - XAUUSD

Ein Machine Learning-basiertes quantitatives Trading-System für den Goldhandel (XAUUSD) auf dem 15-Minuten-Timeframe, entwickelt als Teil einer Bachelorarbeit im Bereich Informatik.

## 🎯 Projektüberblick

Dieses System untersucht die Leistungsfähigkeit moderner Machine Learning-Algorithmen im quantitativen Handel und vergleicht diese mit klassischen technischen Handelsindikatoren. Der Fokus liegt auf dem automatisierten Handel von Gold (XAUUSD) mit präzisen Ein- und Ausstiegssignalen basierend auf historischen Marktdaten und Echtzeitanalyse.

### Kernfunktionalitäten

- **Datenanalyse & Feature Engineering:** Aufbereitung historischer Marktdaten und Generierung relevanter Trading-Features
- **ML-Modell Training:** Implementierung und Training von Random Forest und XGBoost-Modellen
- **Backtesting Framework:** Umfassende historische Performance-Analyse mit Walk-Forward-Validierung
- **Live Trading Integration:** API-basierte Anbindung an MetaTrader 5 für automatisierte Signalübertragung
- **Performance Analytics:** Detaillierte Auswertung mit Sharpe Ratio, Profit Factor, Maximum Drawdown und statistischer Signifikanzanalyse

## 🛠 Technische Architektur

### Core Technologies
- **Python 3.9+** - Hauptprogrammiersprache
- **scikit-learn** - Machine Learning Framework für Random Forest
- **XGBoost** - Gradient Boosting Framework
- **pandas & NumPy** - Datenmanipulation und numerische Berechnungen
- **Backtrader** - Backtesting Engine
- **FastAPI** - RESTful API für Trading-Signale
- **SHAP** - Model Interpretability und Feature Importance

### Trading Infrastructure
- **MetaTrader 5** - Trading Platform Integration
- **MQL5** - Expert Advisor für automatisierte Orderausführung
- **DigitalOcean VPS** - Cloud Deployment für 24/7 Betrieb

## 🔬 Methodologie

### Machine Learning Approach
Das System implementiert einen systematischen ML-basierten Ansatz für Trading-Entscheidungen:

1. **Feature Engineering:** Extraktion von über 50 technischen Indikatoren und Marktfeatures
2. **Model Training:** Supervised Learning mit labelled historischen Trades
3. **Walk-Forward Analysis:** Robuste Validierung durch zeitbasierte Datenaufteilung  
4. **Ensemble Methods:** Kombination mehrerer Modelle für verbesserte Vorhersagegenauigkeit

### Benchmarking
Vergleich der ML-Modelle gegen etablierte Baseline-Strategien:
- Moving Average Crossover
- RSI-basierte Strategien  
- MACD Signal-Strategien
- Buy & Hold Benchmark

## 📊 Performance Metriken

Das System trackt umfassende Leistungskennzahlen:

### Quantitative Metriken
- **Sharpe Ratio** - Risiko-adjustierte Rendite
- **Profit Factor** - Verhältnis Gewinne zu Verlusten  
- **Maximum Drawdown** - Größter Kapitalverlust
- **Win Rate** - Prozentsatz profitabler Trades
- **Average Trade Duration** - Durchschnittliche Haltedauer

### Model Interpretability
- **SHAP Values** - Feature Importance Analyse
- **Permutation Importance** - Feature Impact Assessment
- **Partial Dependence Plots** - Model Behavior Visualization

## 🚀 Deployment & Operations

### Production Environment
- **Cloud Infrastructure:** DigitalOcean Droplet mit automatisiertem Deployment
- **API Architecture:** RESTful Services für Echtzeit-Signalübertragung
- **Monitoring:** Logging und Performance-Tracking für Live-Trading
- **Failsafe Mechanisms:** Risk Management und Position Sizing Controls

### MetaTrader 5 Integration
- Custom Expert Advisor für automatisierte Orderausführung
- Real-time Signalverarbeitung über HTTP API
- Integrierte Risk Management Parameter
- Trade Execution Monitoring und Reporting

## 🔒 Risk Management

### Position Sizing
- Dynamische Positionsgrößenbestimmung basierend auf Volatilität
- Kelly Criterion für optimale Kapitalallokation
- Maximum Risk per Trade Begrenzung

### Model Risk Controls
- Out-of-Sample Performance Monitoring  
- Model Degradation Detection
- Automatic Model Retraining Triggers

## 📈 Forschungszielsetzung

Diese Arbeit adressiert folgende wissenschaftliche Fragestellungen:

1. **Können moderne ML-Algorithmen konsistent Alpha in hochfrequenten Goldmärkten generieren?**
2. **Welche Features haben die höchste prädiktive Kraft für XAUUSD Preisbewegungen?**
3. **Wie robust sind ML-basierte Trading-Strategien gegenüber Marktregime-Wechseln?**
4. **Welcher Ansatz liefert bessere Risk-adjusted Returns: ML oder klassische technische Analyse?**

## ⚠️ Disclaimer

Dieses Repository dient ausschließlich Forschungs- und Bildungszwecken im Rahmen einer wissenschaftlichen Abschlussarbeit. Es stellt **keine Finanzberatung** dar und soll **nicht für Live-Trading ohne entsprechende Due Diligence** verwendet werden.

**Trading beinhaltet erhebliche finanzielle Risiken und kann zum Totalverlust des eingesetzten Kapitals führen.**

---

*Entwickelt als Bachelorarbeit im Studiengang Informatik*  
*Fokus: Machine Learning Applications in Quantitative Finance*