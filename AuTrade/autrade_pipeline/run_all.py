#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    run_all.py
@brief   Master runner — executes all pipeline steps sequentially.

@description
    Runs each step as a subprocess in order:
      1. step1_start_mt5.py   — start MetaTrader 4 if not running
      2. step2_trigger_export.py — copy MQ4 script to MT4 Scripts folder
      3. step3_wait_for_csv.py   — wait until all CSV exports appear
      4. step4_merge.py          — merge old + new data per timeframe

    Each step's stdout is streamed live to the console.
    If a critical step fails (exit code 1), the pipeline stops immediately.
    A partial result (exit code 2) logs a warning but continues.

    Usage:
        python run_all.py

@author   Custom
@version  1.0.0
"""

import sys
import os
import subprocess
from datetime import datetime

from logger import log, sep

# Pipeline steps in execution order
STEPS = [
    ("SCHRITT 1", "step1_start_mt5.py",      "critical"),
    ("SCHRITT 2", "step2_trigger_export.py",  "critical"),
    ("SCHRITT 3", "step3_wait_for_csv.py",    "partial"),
    ("SCHRITT 4", "step4_merge.py",           "partial"),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(label: str, script: str, mode: str) -> bool:
    """
    @brief  Executes a single pipeline step as a subprocess.

    @param  label   Display name for logging, e.g. 'SCHRITT 1'.
    @param  script  Filename of the step script, e.g. 'step1_start_mt5.py'.
    @param  mode    'critical' = stop pipeline on failure |
                    'partial'  = log warning but continue.
    @return True if step succeeded or was non-critical, False if critical failure.
    """
    script_path = os.path.join(SCRIPT_DIR, script)
    sep(label)

    if not os.path.exists(script_path):
        log(f"Script nicht gefunden: {script_path}", "ERROR")
        return mode != "critical"

    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=SCRIPT_DIR
    )

    # Stream output live
    for line in process.stdout:
        print(line, end="")
    process.wait()

    code = process.returncode

    if code == 0:
        log(f"{label} abgeschlossen (OK)", "OK")
        return True
    elif code == 2:
        log(f"{label} abgeschlossen (teilweise)", "WARN")
        return True  # partial is acceptable
    else:
        log(f"{label} FEHLGESCHLAGEN (exit={code})", "ERROR")
        if mode == "critical":
            log("Pipeline gestoppt - kritischer Fehler", "ERROR")
            return False
        else:
            log("Nicht kritisch - fahre fort", "WARN")
            return True


def main():
    """
    @brief  Entry point — runs all pipeline steps and prints final summary.
    """
    start = datetime.now()

    sep("AUTRADE PIPELINE")
    log(f"Start     : {start.strftime('%Y.%m.%d %H:%M:%S')}")
    log(f"Schritte  : {len(STEPS)}")
    sep()

    results = []
    for label, script, mode in STEPS:
        success = run_step(label, script, mode)
        results.append((label, script, success))
        if not success:
            break  # Critical failure — stop

    # Final summary
    elapsed = int((datetime.now() - start).total_seconds())
    sep("PIPELINE ZUSAMMENFASSUNG")
    for label, script, success in results:
        icon = "OK  " if success else "FAIL"
        print(f"  [{icon}] {label} — {script}")

    total_ok = sum(1 for _, _, s in results if s)
    log(f"Ergebnis  : {total_ok}/{len(results)} Schritte erfolgreich", 
        "OK" if total_ok == len(results) else "WARN")
    log(f"Dauer     : {elapsed}s")
    sep()

    sys.exit(0 if total_ok == len(results) else 1)


if __name__ == "__main__":
    main()
