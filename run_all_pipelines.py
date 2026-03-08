#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_pipelines.py
====================
Master Orchestrator for the entire smartEA data workflow.

This script executes the two main pipelines in sequence:
1. Data Pipeline (Exports history from MT5 and merges it)
2. ML Prepare Pipeline (Feature engineering and label verification)

Usage:
    python run_all_pipelines.py
    python run_all_pipelines.py --symbol BTCUSD --tf D1
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths to the individual pipeline runner scripts
ROOT_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_PIPELINE_DIR = os.path.join(ROOT_DIR, "MT5_DataPipeline", "autrade_pipeline")
ML_PIPELINE_DIR   = os.path.join(ROOT_DIR, "ml", "prepare")

DATA_PIPELINE_CMD = [sys.executable, "run_all.py"]
ML_PIPELINE_CMD   = [sys.executable, "run_all.py"]


# ==============================================================================
# HELPERS
# ==============================================================================

def _sep(title: str = "") -> None:
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
    else:
        print("=" * 70)

def _log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{level:<5}] {ts}  {msg}")

def run_pipeline(name: str, cmd: list, cwd: str) -> int:
    """Run a pipeline script as a subprocess and stream output."""
    _sep(name)
    start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    elapsed = int(time.time() - start)

    if proc.returncode == 0:
        _log(f"{name} completed EXCELLENTLY in {elapsed}s", "OK")
    elif proc.returncode == 2:
        _log(f"{name} finished with minor warnings in {elapsed}s", "WARN")
    else:
        _log(f"{name} FAILED in {elapsed}s (exit {proc.returncode})", "ERROR")

    return proc.returncode

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Auto-detect default symbol from DataPipeline config
    if DATA_PIPELINE_DIR not in sys.path:
        sys.path.insert(0, DATA_PIPELINE_DIR)
    try:
        import config as dp_config
        default_symbol = dp_config.SYMBOL
    except ImportError:
        default_symbol = "XAUUSD"

    parser = argparse.ArgumentParser(description="Run Entire smartEA Workflow")
    parser.add_argument("--symbol", type=str, default=default_symbol, help="Symbol to process")
    parser.add_argument("--tf", type=str, default="H4", help="Timeframe for the ML pipeline")
    args, unknown_args = parser.parse_known_args()

    symbol = args.symbol.upper()
    tf = args.tf.upper()

    overall_start = datetime.now()

    _sep(f"SMARTEA MASTER ORCHESTRATOR — {symbol}")
    _log(f"Started   : {overall_start.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    _log(f"Symbol    : {symbol}", "INFO")
    _log(f"Timeframe : {tf} (ML Pipeline)", "INFO")

    # 1. RUN DATA PIPELINE
    dp_cmd = DATA_PIPELINE_CMD.copy()
    code = run_pipeline("PHASE 1: MT5 DATA PIPELINE", dp_cmd, DATA_PIPELINE_DIR)
    
    if code not in (0, 2):
        _log("Data pipeline failed critically. Aborting ML pipeline.", "ERROR")
        sys.exit(1)

    # 2. RUN ML PREPARE PIPELINE
    ml_cmd = ML_PIPELINE_CMD + ["--symbol", symbol, "--tf", tf]
    code = run_pipeline(f"PHASE 2: ML DATA PREPARATION ({symbol} {tf})", ml_cmd, ML_PIPELINE_DIR)

    if code not in (0, 2):
        _log("ML prep pipeline failed critically. Aborting.", "ERROR")
        sys.exit(1)

    # SUCCESS SUMMARY
    elapsed_total = int((datetime.now() - overall_start).total_seconds())
    _sep("SMARTEA WORKFLOW COMPLETE")
    _log(f"Total Duration : {elapsed_total}s", "INFO")
    _log("Both pipelines finished successfully! Ready for Model Training.", "OK")
    print()


if __name__ == "__main__":
    main()
