# Datei: plot_xauusd_m15.py
# Nutzung: python plot_xauusd_m15.py

import pandas as pd
import matplotlib.pyplot as plt

# === Pfad zu deiner Datei ===
csv_path = r"C:\Users\Wael\Desktop\Projekts\smartEA\data\XAUUSD_M15_full.csv"

# CSV laden
df = pd.read_csv(
    csv_path,
    sep=";",                # Semikolon-getrennt
    parse_dates=["Zeit"],   # Zeitspalte als Datum
    dayfirst=True           # deutsches Datumsformat (DD.MM.YYYY)
)

# Nach Zeit sortieren
df = df.sort_values("Zeit").set_index("Zeit")

# Sicherstellen, dass Daten wirklich im 15-Minuten-Takt sind
df = df.asfreq("15T")   # '15T' = 15 Minuten

# Plot Close-Preis
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Close"], label="Close", linewidth=0.8)
plt.title("XAUUSD – M15 Close Chart")
plt.xlabel("Zeit")
plt.ylabel("Preis (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
