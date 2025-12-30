import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# KONFIGURATION
# ============================================================
data_path = Path(r"C:\Users\Wael\AppData\Roaming\MetaQuotes\Terminal\F1BBCAACDA8825381C125EAF07296C41\MQL4\Files")

# NUR Top 3 Majors + Gold
symbols = {
    'EURUSD': 'EUR/USD',
    'USDJPY': 'USD/JPY',
    'GBPUSD': 'GBP/USD',
    'XAUUSD': 'Gold (XAU/USD)'
}

colors = {
    'EURUSD': '#1f77b4',  # Blau
    'USDJPY': '#ff7f0e',  # Orange
    'GBPUSD': '#9467bd',  # Lila
    'XAUUSD': '#d4af37'  # Gold
}

MA_PERIOD = 30  # 30-Tage gleitender Durchschnitt

# ============================================================
# DATEN LADEN - ALLE VERFÜGBAREN DATEN
# ============================================================
print("=" * 70)
print("VOLUME-ANALYSE: ALLE VERFÜGBAREN DATEN")
print("=" * 70)
print(f"\nLade Daten von: {data_path}")
print(f"Gleitender Durchschnitt: {MA_PERIOD} Tage\n")

data_frames = {}

for symbol, label in symbols.items():
    csv_files = list(data_path.glob(f"{symbol}_Volume_*.csv"))

    if csv_files:
        file = csv_files[0]
        print(f"✓ {label:20s} -> {file.name}")

        df = pd.read_csv(file, sep=';', comment='#')
        df['Datum'] = pd.to_datetime(df['Datum'], format='%Y.%m.%d')
        df = df.sort_values('Datum')
        df['Symbol'] = label

        # Gleitenden Durchschnitt berechnen
        df['MA'] = df['Volume'].rolling(window=MA_PERIOD, min_periods=1).mean()

        data_frames[symbol] = df
        print(
            f"  {len(df):5d} Tage | {df['Datum'].min().strftime('%Y-%m-%d')} bis {df['Datum'].max().strftime('%Y-%m-%d')}")
    else:
        print(f"✗ {label:20s} -> NICHT GEFUNDEN")

print("\n" + "=" * 70)

if not data_frames:
    print("FEHLER: Keine Daten gefunden!")
    exit()

# ============================================================
# CHART: ALLE VERFÜGBAREN DATEN
# ============================================================
print(f"\nErstelle Chart mit ALLEN verfügbaren Daten...\n")

# Gesamten Zeitraum ermitteln
all_start = min([df['Datum'].min() for df in data_frames.values()])
all_end = max([df['Datum'].max() for df in data_frames.values()])

# Figur-Größe für Bachelorarbeit
fig, ax = plt.subplots(figsize=(18, 10))

# Plotte ALLE Daten für jedes Asset
for symbol, df in data_frames.items():
    ax.plot(df['Datum'],
            df['MA'],
            label=f"{symbols[symbol]} ({df['Datum'].min().year}–{df['Datum'].max().year})",
            linewidth=3.0,
            alpha=0.9,
            color=colors[symbol])


ax.set_xlabel('Zeitraum',
              fontsize=16,
              fontweight='bold',
              labelpad=15)

ax.set_ylabel('Durchschnittliches Handelsvolumen (Ticks)',
              fontsize=16,
              fontweight='bold',
              labelpad=15)

# Legende optimieren
legend = ax.legend(fontsize=13,
                   loc='upper left',
                   framealpha=0.95,
                   edgecolor='black',
                   fancybox=True,
                   shadow=True)
legend.get_frame().set_linewidth(1.5)

# Grid
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Achsen-Style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Tick-Parameter
ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=8)

# X-Achse formatieren
import matplotlib.dates as mdates

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))

# Y-Achse mit Tausender-Trennzeichen
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

plt.tight_layout()

# Speichern
output_file = f'volume_Top3Majors_vs_Gold_MA{MA_PERIOD}_complete.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ PNG gespeichert: {output_file}")

output_pdf = f'volume_Top3Majors_vs_Gold_MA{MA_PERIOD}_complete.pdf'
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"✓ PDF gespeichert: {output_pdf}")

plt.show()

# ============================================================
# STATISTIK-AUSGABE
# ============================================================
print("\n" + "=" * 70)
print(f"STATISTIK: ALLE VERFÜGBAREN DATEN")
print("=" * 70)

stats = []
for symbol, df in data_frames.items():
    stats.append({
        'Symbol': symbols[symbol],
        'Start': df['Datum'].min().strftime('%Y-%m-%d'),
        'Ende': df['Datum'].max().strftime('%Y-%m-%d'),
        'Tage': len(df),
        'Jahre': round((df['Datum'].max() - df['Datum'].min()).days / 365.25, 1),
        'Avg_MA': df['MA'].mean(),
        'Min_MA': df['MA'].min(),
        'Max_MA': df['MA'].max()
    })

stats_df = pd.DataFrame(stats).sort_values('Avg_MA', ascending=False)

for idx, row in stats_df.iterrows():
    print(f"\n{row['Symbol']}:")
    print(f"  Zeitraum:           {row['Start']} bis {row['Ende']} ({row['Jahre']} Jahre)")
    print(f"  Datenpunkte:        {row['Tage']:,} Tage")
    print(f"  Ø MA-{MA_PERIOD}:            {row['Avg_MA']:,.0f} Ticks")
    print(f"  Min MA-{MA_PERIOD}:          {row['Min_MA']:,.0f} Ticks")
    print(f"  Max MA-{MA_PERIOD}:          {row['Max_MA']:,.0f} Ticks")

print("\n" + "=" * 70)
print("HINWEIS:")
print("  Die Assets haben unterschiedlich lange Historien.")
print("  Der Chart zeigt die maximale verfügbare Datenhistorie.")
print("  Für fairen Vergleich kann ein gemeinsamer Zeitraum gewählt werden.")
print("=" * 70)