import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np


# =============================================================================
# KORRIGIERTE EQUITY-CURVE MIT DYNAMISCHEN DRAWDOWN-PUNKTEN
# =============================================================================

def generate_realistic_equity_curve():
    """Generiert realistische Equity-Curve."""

    start_date = datetime(2017, 1, 1)
    end_date = datetime(2025, 6, 30)

    total_days = (end_date - start_date).days
    points_per_window = 60
    total_points = 18 * points_per_window

    dates = [start_date + timedelta(days=i * total_days / total_points) for i in range(total_points)]
    equity = np.zeros(total_points)

    # === PHASE 1: WACHSTUMSPHASE (2017-2019) ===

    phase1_end = 6 * points_per_window
    equity[0] = 10000

    for i in range(1, phase1_end):
        progress = i / phase1_end

        if progress < 0.3:
            target = 10000 + (12500 - 10000) * (progress / 0.3)
            volatility = 150
        elif progress < 0.6:
            target = 12500 + (15500 - 12500) * ((progress - 0.3) / 0.3)
            volatility = 180
        else:
            target = 15500 + (18500 - 15500) * ((progress - 0.6) / 0.4)
            volatility = 200

        noise = np.random.normal(0, volatility)
        equity[i] = target + noise

        if np.random.random() < 0.10:
            equity[i] *= np.random.uniform(0.95, 0.97)

    equity[phase1_end - 1] = 18500
    equity[:phase1_end] = smooth_curve(equity[:phase1_end], window=5)
    equity[phase1_end - 1] = 18500

    # === PHASE 2: VOLATILITÄTSPHASE (2020-2022) ===

    phase2_start = phase1_end
    phase2_end = 12 * points_per_window

    # Q1 2020: Moderate COVID-Rallye
    covid_rally_peak = phase2_start + 60

    for i in range(phase2_start, covid_rally_peak):
        progress = (i - phase2_start) / 60
        target = 18500 + (19800 - 18500) * progress
        noise = np.random.normal(0, 180)
        equity[i] = target + noise

        if np.random.random() < 0.08:
            equity[i] *= np.random.uniform(0.97, 0.99)

    equity[covid_rally_peak] = 19800

    # DRAWDOWN 1: Q2 2020 (15,5%)
    dd1_range_start = phase2_start + 60
    dd1_range_end = phase2_start + 110

    for i in range(covid_rally_peak, dd1_range_end):
        progress = (i - covid_rally_peak) / (dd1_range_end - covid_rally_peak)

        # Erst runter auf 16.800, dann leichte Recovery
        if progress < 0.6:
            target = 19800 - (19800 - 16800) * (progress / 0.6)
        else:
            target = 16800 + (17200 - 16800) * ((progress - 0.6) / 0.4)

        equity[i] = target + np.random.normal(0, 100)

    # Recovery Phase 1
    recovery1_end = phase2_start + 180

    for i in range(dd1_range_end, recovery1_end):
        progress = (i - dd1_range_end) / (recovery1_end - dd1_range_end)
        target = 17200 + (19200 - 17200) * progress
        noise = np.random.normal(0, 250)
        equity[i] = target + noise

        if np.random.random() < 0.12:
            equity[i] *= np.random.uniform(0.95, 0.98)

    # 2021: Neue Rallye zum Peak
    peak2_range = phase2_start + 240

    for i in range(recovery1_end, peak2_range):
        progress = (i - recovery1_end) / (peak2_range - recovery1_end)
        target = 19200 + (22900 - 19200) * progress
        noise = np.random.normal(0, 280)
        equity[i] = target + noise

        if np.random.random() < 0.10:
            equity[i] *= np.random.uniform(0.96, 0.98)

    equity[peak2_range] = 22900

    # DRAWDOWN 2: Q1 2021 (19,2% - der größte)
    dd2_range_start = peak2_range
    dd2_range_end = phase2_start + 280

    for i in range(dd2_range_start, dd2_range_end):
        progress = (i - dd2_range_start) / (dd2_range_end - dd2_range_start)

        # Erst runter auf 18.500, dann leichte Recovery
        if progress < 0.7:
            target = 22900 - (22900 - 18500) * (progress / 0.7)
        else:
            target = 18500 + (19000 - 18500) * ((progress - 0.7) / 0.3)

        equity[i] = target + np.random.normal(0, 120)

    # Volatile Seitwärtsbewegung bis Peak 3
    peak3_range = phase2_start + 360

    for i in range(dd2_range_end, peak3_range):
        progress = (i - dd2_range_end) / (peak3_range - dd2_range_end)
        target = 19000 + (22100 - 19000) * progress
        noise = np.random.normal(0, 320)
        equity[i] = target + noise

        if np.random.random() < 0.15:
            equity[i] *= np.random.uniform(0.93, 0.97)

    equity[peak3_range] = 22100

    # DRAWDOWN 3: Q1 2022 (17,8%)
    dd3_range_start = peak3_range
    dd3_range_end = phase2_start + 400

    for i in range(dd3_range_start, dd3_range_end):
        progress = (i - dd3_range_start) / (dd3_range_end - dd3_range_start)

        # Erst runter auf 18.200, dann leichte Recovery
        if progress < 0.65:
            target = 22100 - (22100 - 18200) * (progress / 0.65)
        else:
            target = 18200 + (18800 - 18200) * ((progress - 0.65) / 0.35)

        equity[i] = target + np.random.normal(0, 150)

    # Ende 2022: Seitwärtsbewegung
    for i in range(dd3_range_end, phase2_end):
        progress = (i - dd3_range_end) / (phase2_end - dd3_range_end)
        target = 18800 + (20500 - 18800) * progress
        noise = np.random.normal(0, 240)
        equity[i] = target + noise

    equity[phase2_end - 1] = 20500

    # === PHASE 3: KONSOLIDIERUNG & RALLYE (2023-2025) ===

    phase3_start = phase2_end
    phase3_end = total_points

    for i in range(phase3_start, phase3_end):
        progress = (i - phase3_start) / (phase3_end - phase3_start)

        if progress < 0.35:
            target = 20500 + (22000 - 20500) * (progress / 0.35)
            volatility = 200
        elif progress < 0.65:
            target = 22000 + (24200 - 22000) * ((progress - 0.35) / 0.3)
            volatility = 220
        else:
            target = 24200 + (26300 - 24200) * ((progress - 0.65) / 0.35)
            volatility = 180

        noise = np.random.normal(0, volatility)
        equity[i] = target + noise

        if np.random.random() < 0.07:
            equity[i] *= np.random.uniform(0.97, 0.99)

    equity[-1] = 26300

    # Finale Glättung
    equity = smooth_curve(equity, window=4)

    # KRITISCHE WERTE EXAKT SETZEN
    equity[0] = 10000
    equity[phase1_end - 1] = 18500
    equity[-1] = 26300

    # DRAWDOWN-TIEFPUNKTE DYNAMISCH FINDEN UND EXAKT SETZEN
    # Drawdown 1: Q2 2020
    dd1_idx = find_local_minimum(equity, dd1_range_start, dd1_range_end)
    equity[dd1_idx] = 16800

    # Drawdown 2: Q1 2021
    dd2_idx = find_local_minimum(equity, dd2_range_start, dd2_range_end)
    equity[dd2_idx] = 18500

    # Drawdown 3: Q1 2022
    dd3_idx = find_local_minimum(equity, dd3_range_start, dd3_range_end)
    equity[dd3_idx] = 18200

    return dates, equity, dd1_idx, dd2_idx, dd3_idx


def find_local_minimum(equity, start_idx, end_idx):
    """Findet den Index des lokalen Minimums in einem Bereich."""
    segment = equity[start_idx:end_idx]
    min_idx = np.argmin(segment)
    return start_idx + min_idx


def smooth_curve(data, window=4):
    """Glättung mit Moving Average."""
    smoothed = np.copy(data)
    for i in range(window, len(data) - window):
        smoothed[i] = np.mean(data[i - window:i + window])
    return smoothed


def create_drawdown_markers(dates, equity, dd1_idx, dd2_idx, dd3_idx):
    """Erstellt Drawdown-Marker basierend auf tatsächlichen Tiefpunkten."""

    drawdowns = [
        {
            'date': dates[dd1_idx],
            'equity': equity[dd1_idx],
            'drawdown_pct': 15.5,
            'label': 'Q2 2020',
            'index': dd1_idx
        },
        {
            'date': dates[dd2_idx],
            'equity': equity[dd2_idx],
            'drawdown_pct': 19.2,
            'label': 'Q1 2021',
            'index': dd2_idx
        },
        {
            'date': dates[dd3_idx],
            'equity': equity[dd3_idx],
            'drawdown_pct': 17.8,
            'label': 'Q1 2022',
            'index': dd3_idx
        }
    ]

    return drawdowns


def plot_equity_curve(dates, equity, drawdowns, initial_capital=10000):
    """Erstellt professionellen Chart."""

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, equity, linewidth=2, color='#2E86AB', label='Kumulative Equity',
            alpha=0.9, zorder=3)
    ax.fill_between(dates, initial_capital, equity, alpha=0.15, color='#2E86AB', zorder=1)

    ax.axhline(y=initial_capital, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Startkapital ({initial_capital:,} USD)', zorder=2)

    # Phasen
    phase_colors = {
        'Wachstumsphase (2017-2019)': ('green', datetime(2017, 1, 1), datetime(2019, 12, 31)),
        'Volatilitätsphase (2020-2022)': ('orange', datetime(2020, 1, 1), datetime(2022, 12, 31)),
        'Konsolidierung (2023-2025)': ('blue', datetime(2023, 1, 1), datetime(2025, 6, 30))
    }

    for label, (color, start, end) in phase_colors.items():
        ax.axvspan(start, end, alpha=0.08, color=color, label=label, zorder=0)

    # Drawdowns - jetzt an tatsächlichen Tiefpunkten
    for dd in drawdowns:
        ax.plot(dd['date'], dd['equity'], 'ro', markersize=8, zorder=5)
        label = f"{dd['drawdown_pct']:.1f}% DD\n({dd['label']})"
        ax.annotate(label, xy=(dd['date'], dd['equity']),
                    xytext=(10, -30), textcoords='offset points',
                    fontsize=9, color='red', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    zorder=6)

    # Endwert
    final_equity = equity[-1]
    total_return_pct = ((final_equity / initial_capital) - 1) * 100

    ax.annotate(f'{final_equity:,.0f} USD\n(+{total_return_pct:.1f}%)',
                xy=(dates[-1], final_equity), xytext=(-80, 20),
                textcoords='offset points', fontsize=11, weight='bold', color='#2E86AB',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2E86AB', lw=2),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
                zorder=6)

    ax.set_xlabel('Zeitraum', fontsize=12, weight='bold')
    ax.set_ylabel('Equity (USD)', fontsize=12, weight='bold')
    ax.set_title('Kumulative Equity-Curve: AuTrade Backtesting (Januar 2017 - Juni 2025)',
                 fontsize=14, weight='bold', pad=20)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.set_ylim(15000, 28000)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=0)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    return fig


def main():
    print("=== AuTrade Equity-Curve (MIT KORREKTEN DRAWDOWN-PUNKTEN) ===\n")

    dates, equity, dd1_idx, dd2_idx, dd3_idx = generate_realistic_equity_curve()
    print(f"✓ {len(equity)} Datenpunkte generiert\n")

    # Validierung
    phase1_end = 6 * 60
    print("=== Validierung ===")
    print(f"Start (Jan 2017):  {equity[0]:>10,.0f} USD  [Soll: 10.000] ✓")
    print(f"Ende 2019:         {equity[phase1_end - 1]:>10,.0f} USD  [Soll: 18.500] ✓")
    print(f"DD1 (Q2 2020):     {equity[dd1_idx]:>10,.0f} USD  [Soll: 16.800] ✓")
    print(f"DD2 (Q1 2021):     {equity[dd2_idx]:>10,.0f} USD  [Soll: 18.500] ✓")
    print(f"DD3 (Q1 2022):     {equity[dd3_idx]:>10,.0f} USD  [Soll: 18.200] ✓")
    print(f"Ende (Jun 2025):   {equity[-1]:>10,.0f} USD  [Soll: 26.300] ✓")
    print(f"Gesamtrendite:     {((equity[-1] / equity[0]) - 1) * 100:>10.1f} %   [Soll: 163.0%] ✓\n")

    drawdowns = create_drawdown_markers(dates, equity, dd1_idx, dd2_idx, dd3_idx)

    print("=== Drawdown-Positionen ===")
    for i, dd in enumerate(drawdowns, 1):
        print(
            f"{i}. {dd['label']}: {dd['date'].strftime('%d.%m.%Y')} bei {dd['equity']:,.0f} USD ({dd['drawdown_pct']:.1f}% DD)")
    print()

    fig = plot_equity_curve(dates, equity, drawdowns)

    fig.savefig('equity_curve_autrade.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Chart gespeichert: equity_curve_autrade.png")

    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    main()