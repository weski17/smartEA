# -*- coding: utf-8 -*-
"""
prepare.py
==========
ML Data Preparation Pipeline for XAUUSD H4.

Transforms raw OHLCV data exported from MetaTrader 5 into a
feature-engineered dataset ready for supervised ML training.

Pipeline steps
--------------
1.  Load CSV
2.  Date + Time -> DatetimeIndex
3.  Log-returns  (1, 3, 6, 18, 24 candles)
4.  Candle structure
5.  Indicators   (MA, Z-Score, RSI, MACD, BB, ATR, ROC)
6.  Volatility   (Rolling + optional GARCH)
7.  Volume features
8.  Session flags + cyclic time encoding
9.  Candlestick patterns
10. Labels
11. Remove warmup + look-ahead rows
12. Drop remaining NaN rows

Output
------
Saved to ``feature_engineering/<YYYY_MM_DD__HH_MM>/XAUUSD_H4_PREPARED.csv``.
``feature_engineering/LATEST.txt`` always contains the path to the newest run.
"""

import argparse
import io
import os
import sys
import warnings
from datetime import datetime

# Force UTF-8 stdout on Windows consoles that default to cp1252.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import zscore as scipy_zscore

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="ML Data Preparation Pipeline")
parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol to process (e.g., XAUUSD, BTCUSD)")
parser.add_argument("--tf", type=str, default="H4", help="Timeframe to process (e.g., H4, D1)")
args, _ = parser.parse_known_args()

SYMBOL = args.symbol.upper()
TIMEFRAME = args.tf.upper()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

_DP_LATEST = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "MT5_DataPipeline", "Backup_Data", SYMBOL, "LATEST.txt"
))

if not os.path.isfile(_DP_LATEST):
    raise FileNotFoundError(
        f"DataPipeline LATEST.txt not found at '{_DP_LATEST}'.\n"
        "Run the DataPipeline first to generate the merged datasets."
    )

with open(_DP_LATEST, encoding="utf-8") as _f:
    _dp_lines = [l.strip() for l in _f.readlines() if l.strip()]

# Find the CSV path in the LATEST.txt lines
_csv_pattern = f"{SYMBOL}_{TIMEFRAME}_"
_h4_csv = next((p for p in _dp_lines if _csv_pattern in p or f"{SYMBOL}_{TIMEFRAME}_ML_MERGED" in p), None)

if not _h4_csv or not os.path.isfile(_h4_csv):
    raise FileNotFoundError(f"{TIMEFRAME} CSV not found in DataPipeline LATEST.txt for {SYMBOL}: {_h4_csv}")

SOURCE_CSV = _h4_csv

# Each run creates its own timestamped subfolder so results are never overwritten.
_RUN_TS      = datetime.now().strftime("%Y_%m_%d__%H_%M")
_FE_BASE_DIR = os.path.join(os.path.dirname(__file__), "feature_engineering", SYMBOL, TIMEFRAME)
_RUN_DIR     = os.path.join(_FE_BASE_DIR, _RUN_TS)
OUTPUT_CSV   = os.path.join(_RUN_DIR, f"{SYMBOL}_{TIMEFRAME}_PREPARED.csv")
LATEST_FILE  = os.path.join(_FE_BASE_DIR, "LATEST.txt")

# -- Label --------------------------------------------------------------------
LABEL_HORIZON   = 3       # candles ahead for the target return
LABEL_THRESHOLD = 0.003   # +- 0.3 % minimum move to classify as Long / Short

# -- Indicators ---------------------------------------------------------------
MA_PERIODS   = [20, 50, 200]
RSI_PERIOD   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
BB_PERIOD    = 20
BB_STD       = 2
ATR_PERIOD   = 14
ROC_PERIODS  = [6, 24]

# -- Volatility ---------------------------------------------------------------
VOL_SHORT    = 20     # lookback for short-term rolling std
VOL_LONG     = 100    # lookback for long-term rolling std (regime normalisation)
USE_GARCH    = True   # set False to skip GARCH and speed up the pipeline
GARCH_P      = 1
GARCH_Q      = 1

# -- Pipeline row management --------------------------------------------------
WARMUP_ROWS    = max(MA_PERIODS) + 10   # rows dropped at the start (indicator warmup)
LOOKAHEAD_ROWS = LABEL_HORIZON          # rows dropped at the end (no future close available)


# ==============================================================================
# HELPERS
# ==============================================================================

def log(msg: str) -> None:
    """Print a formatted progress message to stdout.

    Args:
        msg: The message to display.
    """
    print(f"  [OK] {msg}")


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Wilder RSI using exponential weighted moving averages.

    Args:
        series: Closing price series.
        period: Lookback period. Defaults to 14.

    Returns:
        RSI values in the range [0, 100].
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ==============================================================================
# STEP 1 - Load CSV
# ==============================================================================

def step1_load_csv(path: str) -> pd.DataFrame:
    """Load the raw MT5 CSV export.

    The file uses semicolons as separators and dots as decimal separators.
    Column names are kept as-is (German MT5 export format).

    Args:
        path: Absolute path to the source CSV file.

    Returns:
        Raw DataFrame with original column names.
    """
    log(f"Loading CSV: {path}")
    df = pd.read_csv(
        path,
        sep=";",
        decimal=".",
        dtype={
            "Datum":   str,
            "Uhrzeit": str,
            "Open":    float,
            "High":    float,
            "Low":     float,
            "Close":   float,
            "Volumen": float,
            "Session": str,
        },
    )
    log(f"Loaded: {len(df):,} rows | columns: {list(df.columns)}")
    return df


# ==============================================================================
# STEP 2 - Date + Time -> DatetimeIndex
# ==============================================================================

def step2_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the 'Datum' and 'Uhrzeit' columns into a sorted DatetimeIndex.

    The original format is ``YYYY.MM.DD HH:MM``.
    Both source columns are dropped after the index is built.

    Args:
        df: Raw DataFrame from step 1.

    Returns:
        DataFrame with a ``DateTime`` DatetimeIndex, sorted ascending.
    """
    log("Building DatetimeIndex ...")
    df["DateTime"] = pd.to_datetime(
        df["Datum"].str.strip() + " " + df["Uhrzeit"].str.strip(),
        format="%Y.%m.%d %H:%M",
    )
    df = df.drop(columns=["Datum", "Uhrzeit"])
    df = df.set_index("DateTime").sort_index()
    log(f"Index: {df.index[0]}  ->  {df.index[-1]}")
    return df


# ==============================================================================
# STEP 3 - Log-Returns
# ==============================================================================

def step3_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-returns over multiple candle horizons.

    Features produced
    -----------------
    - ``log_ret_n`` : close-to-close log-return over n candles (n in 1,3,6,18,24).
    - ``oc_return``  : open-to-close log-return (candle body direction and size).
    - ``hl_range``   : log(High/Low) – intrabar volatility proxy.

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        DataFrame with log-return columns appended.
    """
    log("Computing log-returns (1, 3, 6, 18, 24 candles) ...")

    for n in [1, 3, 6, 18, 24]:
        df[f"log_ret_{n}"] = np.log(df["Close"] / df["Close"].shift(n))

    df["oc_return"] = np.log(df["Close"] / df["Open"])
    df["hl_range"]  = np.log(df["High"]  / df["Low"])

    return df


# ==============================================================================
# STEP 4 - Candle Structure
# ==============================================================================

def step4_candle_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Extract normalised geometric properties of each candlestick.

    All ratios are bounded to [0, 1] by dividing by the total High-Low range.

    Features produced
    -----------------
    - ``body_ratio``         : fraction of range occupied by the candle body.
    - ``upper_shadow_ratio`` : fraction of range above the body (upper wick).
    - ``lower_shadow_ratio`` : fraction of range below the body (lower wick).
    - ``close_position``     : where Close sits within the range (0 = Low, 1 = High).
    - ``candle_dir``         : direction: +1 bullish, -1 bearish, 0 doji.

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        DataFrame with candle structure features appended.
    """
    log("Computing candle structure ...")

    rng  = df["High"] - df["Low"] + 1e-9
    body = (df["Close"] - df["Open"]).abs()
    hi   = df[["Open", "Close"]].max(axis=1)
    lo   = df[["Open", "Close"]].min(axis=1)

    df["body_ratio"]         = body / rng
    df["upper_shadow_ratio"] = (df["High"] - hi) / rng
    df["lower_shadow_ratio"] = (lo - df["Low"])  / rng
    df["close_position"]     = (df["Close"] - df["Low"]) / rng
    df["candle_dir"]         = np.sign(df["Close"] - df["Open"])

    return df


# ==============================================================================
# STEP 5 - Technical Indicators
# ==============================================================================

def step5_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a comprehensive set of technical indicators.

    Indicator groups
    ----------------
    Moving Averages + Z-Score
        For each period p in MA_PERIODS:
        - ``ma_p``       : simple moving average.
        - ``ma_p_dist``  : percentage distance of Close from the MA.
        - ``z_p``        : Z-Score = (Close - MA) / rolling std.

    RSI
        - ``rsi`` : Wilder RSI (period = RSI_PERIOD).

    MACD
        - ``macd``            : MACD line (fast EMA - slow EMA).
        - ``macd_signal``     : signal line (EMA of MACD).
        - ``macd_hist``       : histogram (MACD - signal).
        - ``macd_hist_delta`` : first difference of histogram (momentum of momentum).

    Bollinger Bands
        - ``bb_upper`` / ``bb_lower`` : bands at BB_STD standard deviations.
        - ``bb_width`` : normalised band width (volatility measure).
        - ``bb_pct``   : %B – where Close sits within the bands.

    ATR (Wilder)
        - ``atr``     : Average True Range.
        - ``atr_pct`` : ATR divided by Close (normalised for price-level independence).

    Rate of Change
        For each period n in ROC_PERIODS:
        - ``roc_n`` : percentage change over n candles.

    Args:
        df: DataFrame with OHLCV columns and log-returns.

    Returns:
        DataFrame with all indicator columns appended.
    """
    log("Computing indicators (MA, Z-Score, RSI, MACD, BB, ATR, ROC) ...")

    for p in MA_PERIODS:
        ma               = df["Close"].rolling(p).mean()
        std              = df["Close"].rolling(p).std()
        df[f"ma_{p}"]      = ma
        df[f"ma_{p}_dist"] = (df["Close"] - ma) / (ma + 1e-9)
        df[f"z_{p}"]       = (df["Close"] - ma) / (std + 1e-9)

    df["rsi"] = _rsi(df["Close"], RSI_PERIOD)

    ema_fast          = df["Close"].ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow          = df["Close"].ewm(span=MACD_SLOW,   adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_hist_delta"] = df["macd_hist"].diff()

    bb_mid         = df["Close"].rolling(BB_PERIOD).mean()
    bb_std         = df["Close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = bb_mid + BB_STD * bb_std
    df["bb_lower"] = bb_mid - BB_STD * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-9)
    df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["atr"]     = tr.ewm(com=ATR_PERIOD - 1, min_periods=ATR_PERIOD).mean()
    df["atr_pct"] = df["atr"] / (df["Close"] + 1e-9)

    for n in ROC_PERIODS:
        df[f"roc_{n}"] = (df["Close"] - df["Close"].shift(n)) / (df["Close"].shift(n) + 1e-9)

    return df


# ==============================================================================
# STEP 6 - Volatility
# ==============================================================================

def step6_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling volatility features and optionally fit a GARCH model.

    Rolling volatility features
    ---------------------------
    - ``vol_short``  : short-term rolling std of log-returns (VOL_SHORT periods).
    - ``vol_long``   : long-term rolling std of log-returns (VOL_LONG periods).
    - ``vol_ratio``  : vol_short / vol_long – volatility regime indicator.
                       Values > 1 indicate elevated volatility; < 1 calm market.
    - ``vol_trend``  : 5-period change in vol_short – is volatility rising or falling?

    GARCH features (requires the ``arch`` package, only if USE_GARCH=True)
    -----------------------------------------------------------------------
    - ``garch_vol``        : GARCH(1,1) conditional volatility.
    - ``garch_vol_zscore`` : standardised GARCH volatility.

    Args:
        df: DataFrame containing ``log_ret_1``.

    Returns:
        DataFrame with volatility features appended.
    """
    log("Computing volatility (Rolling + optional GARCH) ...")

    ret = df["log_ret_1"]

    df["vol_short"] = ret.rolling(VOL_SHORT).std()
    df["vol_long"]  = ret.rolling(VOL_LONG).std()
    df["vol_ratio"] = df["vol_short"] / (df["vol_long"] + 1e-9)
    df["vol_trend"] = df["vol_short"] - df["vol_short"].shift(5)

    if USE_GARCH:
        try:
            from arch import arch_model
            returns_pct = ret.dropna() * 100
            model  = arch_model(returns_pct, vol="Garch",
                                p=GARCH_P, q=GARCH_Q,
                                dist="normal", rescale=False)
            result   = model.fit(disp="off", show_warning=False)
            cond_vol = result.conditional_volatility / 100

            df["garch_vol"] = np.nan
            df.loc[cond_vol.index, "garch_vol"] = cond_vol.values
            df["garch_vol_zscore"] = scipy_zscore(
                df["garch_vol"].fillna(method="ffill").fillna(0)
            )
            log(f"GARCH done - {result.nobs} observations")
        except Exception as exc:
            print(f"  [WARN] GARCH failed ({exc}) - skipped")
            df["garch_vol"]        = np.nan
            df["garch_vol_zscore"] = np.nan
    else:
        log("GARCH disabled (USE_GARCH=False)")

    return df


# ==============================================================================
# STEP 7 - Volume Features
# ==============================================================================

def step7_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based features.

    Features produced
    -----------------
    - ``vol_ratio_20``   : volume relative to its 20-bar moving average.
    - ``vol_log``        : log1p-normalised volume.
    - ``vol_delta_5``    : net bull-minus-bear volume fraction over the last 5 candles.
                           +1.0 = all buyers; -1.0 = all sellers.
    - ``vol_delta_sign`` : sign of the 1-bar volume change.
    - ``obv``            : On-Balance Volume (cumulative directional volume).
    - ``obv_slope``      : normalised 10-bar slope of OBV.
    - ``force_index``    : volume times candle body (Elder Force Index).
    - ``force_index_ma`` : 13-bar moving average of force_index.

    Args:
        df: DataFrame with OHLCV columns and candle direction.

    Returns:
        DataFrame with volume feature columns appended.
    """
    log("Computing volume features ...")

    vol  = df["Volumen"]
    ma20 = vol.rolling(20).mean()

    df["vol_ratio_20"] = vol / (ma20 + 1e-9)
    df["vol_log"]      = np.log1p(vol)

    bull_vol = vol.where(df["Close"] > df["Open"], 0)
    bear_vol = vol.where(df["Close"] < df["Open"], 0)
    df["vol_delta_5"] = (
        bull_vol.rolling(5).sum() - bear_vol.rolling(5).sum()
    ) / (vol.rolling(5).sum() + 1e-9)

    df["vol_delta_sign"] = np.sign(vol.diff())

    obv            = (np.sign(df["Close"].diff()) * vol).fillna(0).cumsum()
    df["obv"]      = obv
    df["obv_slope"] = obv.diff(10) / (obv.abs().rolling(10).mean() + 1e-9)

    df["force_index"]    = vol * (df["Close"] - df["Open"])
    df["force_index_ma"] = df["force_index"].rolling(13).mean()

    return df


# ==============================================================================
# STEP 8 - Session Flags + Cyclic Time Encoding
# ==============================================================================

def step8_session_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw Session string into binary flags and encode time cyclically.

    Session flags
    -------------
    The raw Session column is normalised (stripped, lowercased, punctuation removed)
    before matching, making the detection robust against variations like
    ``"New York"``, ``"NewYork"``, ``"Tokyo+London"``, etc.

    - ``sess_tokyo``      : 1 if Tokyo session is active.
    - ``sess_london``     : 1 if London session is active.
    - ``sess_ny``         : 1 if New York session is active.
    - ``sess_overlap_tl`` : 1 during Tokyo-London overlap.
    - ``sess_overlap_ln`` : 1 during London-NY overlap (highest-volume period).

    Cyclic time encoding
    --------------------
    Encodes hour, weekday, and month as (sin, cos) pairs so that boundary
    values (e.g. 23:00 and 00:00, or December and January) are numerically close.

    Args:
        df: DataFrame with a ``Session`` column and a DatetimeIndex.

    Returns:
        DataFrame with session flags and time encoding appended.
        The original ``Session`` column is dropped.
    """
    log("Cleaning session + encoding time ...")

    sess = (
        df["Session"]
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("-", "", regex=False)
    )

    df["sess_tokyo"]      = sess.str.contains("tokyo").astype(int)
    df["sess_london"]     = sess.str.contains("london").astype(int)
    df["sess_ny"]         = sess.str.contains("newyork").astype(int)
    df["sess_overlap_tl"] = ((df["sess_tokyo"]  == 1) & (df["sess_london"] == 1)).astype(int)
    df["sess_overlap_ln"] = ((df["sess_london"] == 1) & (df["sess_ny"]     == 1)).astype(int)
    df = df.drop(columns=["Session"])

    hour    = df.index.hour
    weekday = df.index.weekday   # 0 = Monday, 4 = Friday
    month   = df.index.month

    df["hour_sin"]    = np.sin(2 * np.pi * hour    / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * hour    / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 5)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 5)
    df["month_sin"]   = np.sin(2 * np.pi * month   / 12)
    df["month_cos"]   = np.cos(2 * np.pi * month   / 12)

    return df


# ==============================================================================
# STEP 9 - Candlestick Patterns
# ==============================================================================

def step9_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect classical single- and multi-candle reversal / continuation patterns.

    All pattern columns are binary (0 / 1).

    Patterns detected
    -----------------
    - ``pat_doji``           : body < 10% of range (indecision).
    - ``pat_hammer``         : lower wick > 2x body, minimal upper wick (buying pressure).
    - ``pat_shooting_star``  : upper wick > 2x body, minimal lower wick (selling pressure).
    - ``pat_bull_engulfing`` : bullish candle completely engulfs the previous bearish candle.
    - ``pat_bear_engulfing`` : bearish candle completely engulfs the previous bullish candle.
    - ``pat_marubozu_bull``  : bullish candle with virtually no wicks (strong trend).
    - ``pat_marubozu_bear``  : bearish candle with virtually no wicks (strong trend).
    - ``pat_morning_star``   : 3-candle bullish reversal pattern.
    - ``pat_evening_star``   : 3-candle bearish reversal pattern.
    - ``pat_spinning_top``   : small body with both wicks present (equilibrium).

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        DataFrame with candlestick pattern columns appended.
    """
    log("Computing candlestick patterns ...")

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body         = (c - o).abs()
    rng          = h - l + 1e-9
    avg_body     = body.rolling(10).mean()
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l

    df["pat_doji"] = (body < 0.1 * rng).astype(int)

    df["pat_hammer"] = (
        (lower_shadow > 2 * body) &
        (upper_shadow < body) &
        (body > 0)
    ).astype(int)

    df["pat_shooting_star"] = (
        (upper_shadow > 2 * body) &
        (lower_shadow < body) &
        (body > 0)
    ).astype(int)

    df["pat_bull_engulfing"] = (
        (c.shift(1) < o.shift(1)) &
        (c > o) &
        (o < c.shift(1)) &
        (c > o.shift(1))
    ).astype(int)

    df["pat_bear_engulfing"] = (
        (c.shift(1) > o.shift(1)) &
        (c < o) &
        (o > c.shift(1)) &
        (c < o.shift(1))
    ).astype(int)

    df["pat_marubozu_bull"] = (
        (c > o) &
        (upper_shadow < 0.05 * rng) &
        (lower_shadow < 0.05 * rng)
    ).astype(int)

    df["pat_marubozu_bear"] = (
        (c < o) &
        (upper_shadow < 0.05 * rng) &
        (lower_shadow < 0.05 * rng)
    ).astype(int)

    df["pat_morning_star"] = (
        (c.shift(2) < o.shift(2)) &
        (body.shift(1) < 0.5 * avg_body.shift(1)) &
        (c > o) &
        (c > (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)

    df["pat_evening_star"] = (
        (c.shift(2) > o.shift(2)) &
        (body.shift(1) < 0.5 * avg_body.shift(1)) &
        (c < o) &
        (c < (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)

    df["pat_spinning_top"] = (
        (body < 0.3 * rng) &
        (upper_shadow > 0.2 * rng) &
        (lower_shadow > 0.2 * rng)
    ).astype(int)

    return df


# ==============================================================================
# STEP 10 - Calculate Labels
# ==============================================================================

def step10_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the classification target based on future log-return.

    The target looks LABEL_HORIZON candles ahead using the closing price.
    This introduces look-ahead data for those rows, which is why the last
    LOOKAHEAD_ROWS rows are dropped in step 11.

    Label encoding
    --------------
    - ``future_return`` : raw log-return over the horizon (continuous).
    - ``label``         :
        -  1  (Long)    if future_return >  LABEL_THRESHOLD
        - -1  (Short)   if future_return < -LABEL_THRESHOLD
        -  0  (Neutral) otherwise

    Args:
        df: DataFrame with a ``Close`` column.

    Returns:
        DataFrame with ``future_return`` and ``label`` columns appended.
    """
    log(f"Computing labels (horizon={LABEL_HORIZON} candles, threshold=+-{LABEL_THRESHOLD:.1%}) ...")

    future_close  = df["Close"].shift(-LABEL_HORIZON)
    future_return = np.log(future_close / df["Close"])

    df["future_return"] = future_return
    df["label"] = 0
    df.loc[future_return >  LABEL_THRESHOLD, "label"] =  1
    df.loc[future_return < -LABEL_THRESHOLD, "label"] = -1

    dist  = df["label"].value_counts().sort_index()
    total = len(df)
    log(f"Label distribution -> Short:{dist.get(-1,0):,}  Neutral:{dist.get(0,0):,}  Long:{dist.get(1,0):,}")
    for lbl, name in [(-1, "Short"), (0, "Neutral"), (1, "Long")]:
        print(f"    {name}: {dist.get(lbl, 0) / total * 100:.1f}%")

    return df


# ==============================================================================
# STEP 11 - Remove Warmup + Look-Ahead Rows
# ==============================================================================

def step11_trim(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain invalid indicator or label values.

    - First WARMUP_ROWS rows are dropped because long-period indicators
      (e.g. MA-200) are not yet stable.
    - Last LOOKAHEAD_ROWS rows are dropped because no valid future close
      exists for label computation.

    Args:
        df: Full feature DataFrame before trimming.

    Returns:
        Trimmed DataFrame ready for NaN validation.
    """
    log(f"Removing warmup ({WARMUP_ROWS}) + look-ahead ({LOOKAHEAD_ROWS}) rows ...")
    before = len(df)
    df = df.iloc[WARMUP_ROWS: len(df) - LOOKAHEAD_ROWS]
    log(f"Before: {before:,}  ->  After: {len(df):,} rows")
    return df


# ==============================================================================
# STEP 12 - Check and Remove NaN Values
# ==============================================================================

def step12_nan_check(df: pd.DataFrame) -> pd.DataFrame:
    """Report and remove any remaining NaN values.

    NaN values can remain if GARCH fitting failed for some rows or if
    rolling windows were not fully satisfied after trimming.
    All affected rows are dropped with a diagnostic report.

    Args:
        df: Trimmed feature DataFrame.

    Returns:
        DataFrame free of NaN values.
    """
    log("Checking for NaN values ...")
    nan_cols = df.isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]

    if len(nan_cols) > 0:
        print("  Columns with NaN:")
        for col, cnt in nan_cols.items():
            print(f"    {col}: {cnt:,}")
    else:
        log("No NaN values found - all clean")

    before  = len(df)
    df      = df.dropna()
    removed = before - len(df)
    if removed:
        log(f"{removed:,} rows with NaN removed -> {len(df):,} remaining")

    return df


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Execute the full data preparation pipeline end-to-end.

    Runs all 12 steps in sequence, saves the result to a timestamped
    subfolder under ``feature_engineering/``, and prints a full column
    overview with recommended ML input features.
    """
    print("\n" + "=" * 65)
    print(f"  ML Data Preparation - {SYMBOL} {TIMEFRAME}  (Version 2)")
    print("=" * 65)

    df = step1_load_csv(SOURCE_CSV)
    df = step2_datetime_index(df)
    df = step3_log_returns(df)
    df = step4_candle_structure(df)
    df = step5_indicators(df)
    df = step6_volatility(df)
    df = step7_volume_features(df)
    df = step8_session_time(df)
    df = step9_candlestick_patterns(df)
    df = step10_labels(df)
    df = step11_trim(df)
    df = step12_nan_check(df)

    # -- Save -----------------------------------------------------------------
    print("\n" + "-" * 65)
    os.makedirs(_RUN_DIR, exist_ok=True)
    log(f"Folder created: feature_engineering/{SYMBOL}/{TIMEFRAME}/{_RUN_TS}/")
    log(f"Saving: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, sep=";", decimal=".", float_format="%.6f")

    with open(LATEST_FILE, "w", encoding="utf-8") as f:
        f.write(f"{_RUN_TS}\n{_RUN_DIR}\n{OUTPUT_CSV}\n")

    log(f"Done!  {len(df):,} rows x {len(df.columns)} columns")
    log(f"Run folder:  feature_engineering/{SYMBOL}/{TIMEFRAME}/{_RUN_TS}/")
    log(f"LATEST.txt:  feature_engineering/{SYMBOL}/{TIMEFRAME}/LATEST.txt")
    print("=" * 65 + "\n")

    # -- Column overview ------------------------------------------------------
    print("All output columns:\n")

    categories = {
        "OHLCV (raw)":             ["Open", "High", "Low", "Close", "Volumen"],
        "Log-Returns":             [c for c in df.columns if c.startswith("log_ret") or c in ("oc_return", "hl_range")],
        "Candle Structure":        [c for c in df.columns if c in ("body_ratio", "upper_shadow_ratio", "lower_shadow_ratio", "close_position", "candle_dir")],
        "MA + Z-Score":            [c for c in df.columns if c.startswith("ma_") or c.startswith("z_")],
        "Momentum (RSI/MACD/ROC)": [c for c in df.columns if c in ("rsi",) or c.startswith("macd") or c.startswith("roc_")],
        "Bollinger + ATR":         [c for c in df.columns if c.startswith("bb_") or c.startswith("atr")],
        "Volatility":              [c for c in df.columns if c.startswith("vol_short") or c.startswith("vol_long") or c.startswith("vol_ratio") or c.startswith("vol_trend") or c.startswith("garch")],
        "Volume":                  [c for c in df.columns if c.startswith("vol_") and not any(c.startswith(x) for x in ("vol_short", "vol_long", "vol_ratio", "vol_trend"))],
        "OBV + Force Index":       [c for c in df.columns if c.startswith("obv") or c.startswith("force")],
        "Session":                 [c for c in df.columns if c.startswith("sess_")],
        "Time (cyclic)":           [c for c in df.columns if any(c.startswith(p) for p in ("hour_", "weekday_", "month_"))],
        "Candlestick Patterns":    [c for c in df.columns if c.startswith("pat_")],
        "Label":                   [c for c in df.columns if c in ("future_return", "label")],
    }

    total_features = 0
    for cat, cols in categories.items():
        existing = [c for c in cols if c in df.columns]
        if existing:
            print(f"  [{len(existing):2d}] {cat}")
            for c in existing:
                print(f"        {c}")
            total_features += len(existing)
            print()

    print(f"  TOTAL: {total_features} columns\n")

    # -- Recommended ML input features ----------------------------------------
    print("-" * 65)
    print("  RECOMMENDED INPUT FEATURES FOR ML MODEL (Top 28):")
    print("-" * 65)

    ml_features = [
        ("log_ret_1",         "Return last candle"),
        ("log_ret_6",         "Return last 24 h"),
        ("log_ret_24",        "Return last 4 days"),
        ("oc_return",         "Candle body direction"),
        ("body_ratio",        "Body size ratio"),
        ("upper_shadow_ratio","Upper wick ratio"),
        ("lower_shadow_ratio","Lower wick ratio"),
        ("close_position",    "Close position in range"),
        ("z_20",              "Z-Score 20 periods  *"),
        ("z_50",              "Z-Score 50 periods  ** (strongest feature)"),
        ("z_200",             "Z-Score 200 periods *"),
        ("rsi",               "RSI 14"),
        ("vol_short",         "Short-term rolling vol"),
        ("vol_ratio",         "Vol regime (short/long)"),
        ("atr_pct",           "ATR normalised"),
        ("bb_pct",            "Bollinger %B position"),
        ("bb_width",          "Bollinger band width"),
        ("macd_hist",         "MACD histogram"),
        ("macd_hist_delta",   "MACD momentum change"),
        ("roc_6",             "ROC 24 hours"),
        ("roc_24",            "ROC 4 days"),
        ("vol_ratio_20",      "Relative volume"),
        ("vol_delta_5",       "Bull vs Bear volume  *"),
        ("obv_slope",         "OBV slope"),
        ("sess_london",       "London session active"),
        ("sess_overlap_ln",   "London-NY overlap"),
        ("hour_sin",          "Time of day (sin)"),
        ("hour_cos",          "Time of day (cos)"),
    ]

    for feat, desc in ml_features:
        status = "[OK]    " if feat in df.columns else "[MISSING]"
        print(f"  {status}  {feat:<22} {desc}")

    print()
    print(f"  * = especially important for {SYMBOL} {TIMEFRAME}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()