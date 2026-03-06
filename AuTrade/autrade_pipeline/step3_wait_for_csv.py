#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step3_wait_for_csv.py
@brief   Step 3 — Polls the MT4 Common/Files folder until all CSV exports appear.

@description
    Repeatedly checks for the presence and stability of each expected CSV file.
    A file is considered complete when its size remains unchanged across two
    consecutive polls, indicating MT4 has finished writing it.

    Prints a live status line every POLL_INTERVAL_SEC seconds showing which
    files are ready and which are still pending.

    On timeout, returns whatever files are available rather than failing hard,
    so the merge step can still process partial results.

@returns  exit code 0 if all files found, 2 if partial, 1 if none found.
"""

import sys
import os
import time
import json

from config import (
    MT4_COMMON_FILES, SYMBOL, TIMEFRAMES,
    EXPORT_TIMEOUT_SEC, POLL_INTERVAL_SEC
)
from logger import log, sep

# Temp file to pass found file paths to next step
FOUND_FILES_JSON = os.path.join(os.path.dirname(__file__), "_found_files.json")


def wait_for_export() -> dict:
    """
    @brief  Waits until all expected CSV files appear and stabilize in size.
    @return Dictionary mapping timeframe string to absolute file path.
            May be partial if timeout is reached before all files appear.
    """
    expected = {
        tf: os.path.join(MT4_COMMON_FILES, f"{SYMBOL}_{tf}.csv")
        for tf in TIMEFRAMES
    }

    log(f"Erwarte {len(TIMEFRAMES)} Dateien in:", "INFO")
    log(MT4_COMMON_FILES, "INFO")
    sep()

    start      = time.time()
    prev_sizes = {}
    found      = {}

    while time.time() - start < EXPORT_TIMEOUT_SEC:
        elapsed = int(time.time() - start)
        ready   = []
        pending = []

        for tf, path in expected.items():
            if tf in found:
                ready.append(tf)
                continue
            if os.path.exists(path):
                size = os.path.getsize(path)
                if prev_sizes.get(tf) == size and size > 0:
                    found[tf] = path
                    ready.append(tf)
                    log(f"{tf}: fertig ({size // 1024} KB)", "OK")
                else:
                    prev_sizes[tf] = size
                    pending.append(f"{tf}(~{size // 1024}KB)")
            else:
                pending.append(f"{tf}(fehlt)")

        if pending:
            log(f"[{elapsed}s/{EXPORT_TIMEOUT_SEC}s] "
                f"Fertig: {ready} | Ausstehend: {pending}", "INFO")

        if len(found) == len(TIMEFRAMES):
            log("Alle CSV-Dateien empfangen!", "OK")
            break

        time.sleep(POLL_INTERVAL_SEC)
    else:
        log(f"Timeout nach {EXPORT_TIMEOUT_SEC}s", "WARN")
        # Collect whatever exists
        for tf, path in expected.items():
            if tf not in found and os.path.exists(path):
                found[tf] = path

    return found


if __name__ == "__main__":
    sep("SCHRITT 3: Warten auf CSV-Export")
    found = wait_for_export()

    # Save found files for next step
    with open(FOUND_FILES_JSON, "w") as f:
        json.dump(found, f, indent=2)
    log(f"Gefundene Dateien gespeichert: {FOUND_FILES_JSON}", "INFO")

    if not found:
        log("Keine Dateien gefunden - abgebrochen", "ERROR")
        sys.exit(1)
    elif len(found) < len(TIMEFRAMES):
        log(f"Nur {len(found)}/{len(TIMEFRAMES)} Dateien gefunden - teilweise fortfahren", "WARN")
        sys.exit(2)
    else:
        sys.exit(0)
