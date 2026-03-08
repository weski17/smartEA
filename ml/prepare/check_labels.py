# -*- coding: utf-8 -*-
"""
check_labels.py
===============
Label verification suite for the XAUUSD H4 prepared dataset.

Automatically loads the CSV path from ``feature_engineering/LATEST.txt``
so it always validates the most recent ``prepare.py`` run without any
manual path updates.

Tests
-----
1. Statistical test   – do mean returns match label direction?
2. Manual test        – spot-check 5 rows by recomputing returns by hand.
3. Look-ahead test    – confirm no future data leaks into the last rows.
4. Distribution test  – are label class ratios within sensible bounds?
5. Visual test        – price chart with Long / Short / Neutral markers.
"""

import argparse
import io
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Force UTF-8 stdout on Windows consoles that default to cp1252.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser(description="Label Verification")
parser.add_argument("--symbol", type=str, default="XAUUSD")
parser.add_argument("--tf", type=str, default="H4")
args, _ = parser.parse_known_args()

SYMBOL = args.symbol.upper()
TIMEFRAME = args.tf.upper()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths -----------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_FE_BASE_DIR = os.path.join(_SCRIPT_DIR, "feature_engineering", SYMBOL, TIMEFRAME)
_LATEST_FILE = os.path.join(_FE_BASE_DIR, "LATEST.txt")


def _resolve_csv_path() -> str:
    """Read the CSV path from LATEST.txt produced by prepare.py.

    LATEST.txt format (three lines)::

        <run_timestamp>
        <absolute path to run folder>
        <absolute path to CSV file>

    Returns:
        Absolute path to the most recently prepared CSV.

    Raises:
        FileNotFoundError: If LATEST.txt does not exist (prepare.py not yet run).
    """
    if not os.path.isfile(_LATEST_FILE):
        raise FileNotFoundError(
            f"LATEST.txt not found at '{_LATEST_FILE}'.\n"
            "Run prepare.py first to generate the feature-engineered dataset."
        )
    with open(_LATEST_FILE, encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Third line is the full CSV path.
    return lines[2]


PREPARED_CSV  = _resolve_csv_path()
LABEL_HORIZON = 3       # must match LABEL_HORIZON in prepare.py
THRESHOLD     = 0.003   # must match LABEL_THRESHOLD in prepare.py

# Plot output — saved next to the CSV in the same run folder.
_RUN_DIR  = os.path.dirname(PREPARED_CSV)
SAVE_PLOT = True
PLOT_PATH = os.path.join(_RUN_DIR, "check_labels_plot.png")


# ==============================================================================
# HELPERS
# ==============================================================================

def log(msg: str) -> None:
    """Print a success / info message.

    Args:
        msg: The message text.
    """
    print(f"  [OK] {msg}")


def warn(msg: str) -> None:
    """Print a warning message.

    Args:
        msg: The warning text.
    """
    print(f"  [!!] {msg}")


def header(title: str) -> None:
    """Print a section separator with a title.

    Args:
        title: Section heading text.
    """
    print()
    print("-" * 60)
    print(f"  {title}")
    print("-" * 60)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data() -> pd.DataFrame:
    """Load the prepared CSV identified by LATEST.txt.

    Returns:
        DataFrame indexed by DateTime with all feature and label columns.
    """
    log(f"Source (from LATEST.txt): {PREPARED_CSV}")
    df = pd.read_csv(PREPARED_CSV, sep=";", index_col=0, parse_dates=True)
    log(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")
    log(f"Period: {df.index[0]}  ->  {df.index[-1]}")
    return df


# ==============================================================================
# TEST 1 — Statistical Test
# ==============================================================================

def test1_statistical(df: pd.DataFrame) -> bool:
    """Verify that each label class has the expected mean future_return sign.

    Checks:
    - Long  (+1) rows should have a positive mean future_return.
    - Short (-1) rows should have a negative mean future_return.
    - Neutral (0) rows should have a mean future_return close to zero.

    Args:
        df: Prepared DataFrame with ``label`` and ``future_return`` columns.

    Returns:
        True if all three classes pass, False otherwise.
    """
    header("TEST 1 — Statistical Test")
    print("  Checks: does each label class have the correct mean return?\n")

    long_ret    = df[df["label"] ==  1]["future_return"]
    short_ret   = df[df["label"] == -1]["future_return"]
    neutral_ret = df[df["label"] ==  0]["future_return"]

    results = {
        "Long   (+1)": (long_ret,    "positive",   lambda x: x > 0),
        "Short  (-1)": (short_ret,   "negative",   lambda x: x < 0),
        "Neutral (0)": (neutral_ret, "near zero",  lambda x: abs(x) < THRESHOLD),
    }

    all_ok = True
    for name, (series, expectation, check) in results.items():
        mean_val = series.mean()
        ok       = check(mean_val)
        status   = "[PASS]" if ok else "[FAIL]"
        print(f"  {status}  {name}:")
        print(f"           Mean return = {mean_val:+.4f}  ({mean_val*100:+.3f}%)")
        print(f"           Expected:     {expectation}")
        print(f"           Count:        {len(series):,}\n")
        if not ok:
            all_ok = False

    if all_ok:
        log("TEST 1 PASSED")
    else:
        warn("TEST 1 FAILED - review label logic in prepare.py")

    return all_ok


# ==============================================================================
# TEST 2 — Manual Spot-Check
# ==============================================================================

def test2_manual(df: pd.DataFrame) -> bool:
    """Spot-check 5 individual rows by recomputing future_return from raw prices.

    For each sampled row at position ``idx``, the expected future log-return is
    computed as ``log(Close[idx + LABEL_HORIZON] / Close[idx])`` and compared
    against the stored ``future_return`` value.  The derived label is also
    verified against the stored ``label`` value.

    Args:
        df: Prepared DataFrame with ``Close``, ``future_return``, and ``label``.

    Returns:
        True if all sampled rows match, False if any discrepancy is found.
    """
    header("TEST 2 — Manual Spot-Check (5 rows)")
    print("  Checks: does future_return match Close(t+3) / Close(t) manually?\n")

    test_indices = [500, 1000, 5000, 15000, 25000]
    all_ok = True

    for idx in test_indices:
        if idx + LABEL_HORIZON >= len(df):
            continue

        row        = df.iloc[idx]
        row_future = df.iloc[idx + LABEL_HORIZON]

        manual_return = np.log(row_future["Close"] / row["Close"])
        label_return  = row["future_return"]
        label         = row["label"]

        diff = abs(manual_return - label_return)
        ok   = diff < 1e-6

        if   manual_return >  THRESHOLD: expected_label =  1
        elif manual_return < -THRESHOLD: expected_label = -1
        else:                            expected_label =  0

        label_ok = (label == expected_label)
        status   = "[PASS]" if (ok and label_ok) else "[FAIL]"

        print(f"  {status}  Index {idx} | {df.index[idx]}")
        print(f"           Close(t)     = {row['Close']:.2f}")
        print(f"           Close(t+{LABEL_HORIZON})   = {row_future['Close']:.2f}")
        print(f"           Manual:      {manual_return:+.5f}  ({manual_return*100:+.3f}%)")
        print(f"           Stored:      {label_return:+.5f}  ({label_return*100:+.3f}%)")
        print(f"           Label:       {int(label):+d}  (expected: {expected_label:+d})\n")

        if not (ok and label_ok):
            all_ok = False

    if all_ok:
        log("TEST 2 PASSED")
    else:
        warn("TEST 2 FAILED - future_return or label computed incorrectly")

    return all_ok


# ==============================================================================
# TEST 3 — Look-Ahead Bias Test
# ==============================================================================

def test3_lookahead(df: pd.DataFrame) -> bool:
    """Confirm that no look-ahead data leaks into the final rows of the dataset.

    ``prepare.py`` removes the last ``LABEL_HORIZON`` rows in step 11 because
    no future close price exists for them.  This test verifies that the trimming
    was applied correctly by checking that no NaN values are present in
    ``future_return`` at the end of the dataset.

    Args:
        df: Prepared and trimmed DataFrame.

    Returns:
        True if the last rows contain valid future_return values, False if NaNs
        are found (indicating trimming was skipped or misconfigured).
    """
    header("TEST 3 — Look-Ahead Bias Test")
    print("  Checks: do the last rows have valid future_return (no data leak)?\n")

    last_rows = df.tail(LABEL_HORIZON + 2)
    nan_count = last_rows["future_return"].isnull().sum()

    print(f"  Last {LABEL_HORIZON + 2} rows:")
    print(last_rows[["Close", "future_return", "label"]].to_string())
    print()

    if nan_count == 0:
        log("No NaN at end - warmup/trim applied correctly")
        log("TEST 3 PASSED")
        return True
    else:
        warn(f"{nan_count} NaN values found at end - check step11_trim in prepare.py")
        return False


# ==============================================================================
# TEST 4 — Distribution Test
# ==============================================================================

def test4_distribution(df: pd.DataFrame) -> bool:
    """Verify that label class ratios are within sensible bounds.

    Heuristic thresholds:
    - Neutral < 20 %  -> threshold probably too large (too few neutral rows).
    - Neutral > 70 %  -> threshold probably too small (too many neutral rows).
    - |Short - Long| > 15 pp -> strong imbalance that may bias the model.

    Args:
        df: Prepared DataFrame with a ``label`` column.

    Returns:
        True if all distribution checks pass, False if any warning fires.
    """
    header("TEST 4 — Distribution Test")
    print("  Checks: are label class ratios within sensible bounds?\n")

    total = len(df)
    dist  = df["label"].value_counts().sort_index()

    for lbl, name in [(-1, "Short  "), (0, "Neutral"), (1, "Long   ")]:
        count = dist.get(lbl, 0)
        pct   = count / total * 100
        bar   = "#" * int(pct / 2)
        print(f"  {name}: {count:6,}  ({pct:5.1f}%)  {bar}")

    print()

    short_pct   = dist.get(-1, 0) / total * 100
    neutral_pct = dist.get( 0, 0) / total * 100
    long_pct    = dist.get( 1, 0) / total * 100

    issues = []
    if neutral_pct < 20:
        issues.append(f"Neutral too low ({neutral_pct:.1f}%) - LABEL_THRESHOLD may be too large")
    if neutral_pct > 70:
        issues.append(f"Neutral too high ({neutral_pct:.1f}%) - LABEL_THRESHOLD may be too small")
    if abs(short_pct - long_pct) > 15:
        issues.append(f"Strong Short/Long imbalance ({short_pct:.1f}% / {long_pct:.1f}%)")

    if issues:
        for i in issues:
            warn(i)
        return False
    else:
        log("Distribution looks good")
        log("TEST 4 PASSED")
        return True


# ==============================================================================
# TEST 5 — Visual Test
# ==============================================================================

def test5_visual(df: pd.DataFrame) -> bool:
    """Generate a price chart overlaid with Long / Short / Neutral label markers.

    Plots three separate 60-candle windows from different parts of the dataset
    to give a broad visual sense of label quality across the full history.
    The chart is saved to ``PLOT_PATH`` if ``SAVE_PLOT`` is True.

    Args:
        df: Prepared DataFrame with ``Close`` and ``label`` columns.

    Returns:
        Always True (visual test cannot fail automatically).
    """
    header("TEST 5 — Visual Test (chart)")
    print("  Generating price chart with label markers for 3 periods ...\n")

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0e1a0f")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)

    periods = [
        (1000,  1060, "Period 1 — 60 candles"),
        (10000, 10060, "Period 2 — 60 candles"),
        (25000, 25060, "Period 3 — 60 candles (recent data)"),
    ]

    for i, (start, end, title) in enumerate(periods):
        ax     = fig.add_subplot(gs[i])
        sample = df.iloc[start:end].copy()

        ax.set_facecolor("#0e1a0f")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#2a3a2a")

        # Price line
        ax.plot(range(len(sample)), sample["Close"],
                color="#c8d4c0", linewidth=1.2, zorder=2)

        # Long signals - green upward triangles
        longs     = sample[sample["label"] == 1]
        long_idx  = [sample.index.get_loc(idx) for idx in longs.index]
        ax.scatter(long_idx, longs["Close"],
                   color="#4a9b6a", marker="^", s=60,
                   zorder=3, label="Long (+1)")

        # Short signals - red downward triangles
        shorts    = sample[sample["label"] == -1]
        short_idx = [sample.index.get_loc(idx) for idx in shorts.index]
        ax.scatter(short_idx, shorts["Close"],
                   color="#c0392b", marker="v", s=60,
                   zorder=3, label="Short (-1)")

        # Neutral - small grey dots
        neutrals    = sample[sample["label"] == 0]
        neutral_idx = [sample.index.get_loc(idx) for idx in neutrals.index]
        ax.scatter(neutral_idx, neutrals["Close"],
                   color="#555555", marker=".", s=15,
                   zorder=1, label="Neutral (0)")

        ax.set_title(
            f"{title}  |  {sample.index[0].date()} -> {sample.index[-1].date()}",
            color="#e8f0eb", fontsize=10, pad=8,
        )
        ax.set_ylabel("Close", color="#aaaaaa", fontsize=8)
        ax.legend(loc="upper left", fontsize=8,
                  facecolor="#1a2218", labelcolor="#e8f0eb",
                  edgecolor="#2a3a2a")

        tick_positions = range(0, len(sample), 10)
        tick_labels    = [sample.index[j].strftime("%m-%d %H:%M")
                          for j in tick_positions if j < len(sample)]
        ax.set_xticks(list(tick_positions)[:len(tick_labels)])
        ax.set_xticklabels(tick_labels, rotation=30, fontsize=7, color="#aaaaaa")

    fig.suptitle(
        f"{SYMBOL} {TIMEFRAME} — Label Verification\n"
        "^ Long (+1)   v Short (-1)   . Neutral (0)",
        color="#e8f0eb", fontsize=13, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if SAVE_PLOT:
        plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        log(f"Chart saved: {PLOT_PATH}")

    # plt.show() removed to allow the pipeline to complete automatically
    log("TEST 5 PASSED - chart generated")
    return True


# ==============================================================================
# SUMMARY
# ==============================================================================

def summary(results: list) -> None:
    """Print a final pass / fail summary for all tests.

    Args:
        results: List of five booleans, one per test, in order 1-5.
    """
    header("SUMMARY")

    tests = [
        ("Test 1 — Statistical Test",  results[0]),
        ("Test 2 — Manual Spot-Check", results[1]),
        ("Test 3 — Look-Ahead Test",   results[2]),
        ("Test 4 — Distribution Test", results[3]),
        ("Test 5 — Visual Test",       results[4]),
    ]

    all_ok = True
    for name, ok in tests:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  " + "=" * 39)
        print("  ALL TESTS PASSED")
        print("  Labels are correct - proceed to train.py")
        print("  " + "=" * 39)
    else:
        print("  " + "=" * 39)
        print("  ERRORS FOUND")
        print("  Fix prepare.py before starting training")
        print("  " + "=" * 39)

    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run all five label verification tests against the latest prepared CSV.

    The CSV path is resolved automatically from
    ``feature_engineering/LATEST.txt``.
    """
    print("\n" + "=" * 60)
    print(f"  Label Check — {SYMBOL} {TIMEFRAME}")
    print("=" * 60)

    df = load_data()

    r1 = test1_statistical(df)
    r2 = test2_manual(df)
    r3 = test3_lookahead(df)
    r4 = test4_distribution(df)
    r5 = test5_visual(df)

    summary([r1, r2, r3, r4, r5])


if __name__ == "__main__":
    main()
