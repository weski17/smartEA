#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step1_start_mt5.py
@brief   Step 1 — Checks if MetaTrader 5 is running and starts it if not.

@description
    Scans all running Windows processes for terminal64.exe (MetaTrader 5).
    If MT5 is not found, it is launched using the path defined in config.py
    and the script waits MT5_STARTUP_WAIT_SEC seconds for it to fully load
    before returning control to the pipeline.

    If MT5 is already running, this step completes immediately.

@returns
    exit code 0 — MT5 is running and ready
    exit code 1 — MT5 could not be started
"""

import sys
import os
import time
import subprocess

from config import MT5_EXE, MT5_STARTUP_WAIT_SEC
from logger import log, sep


def is_mt5_running() -> bool:
    """
    @brief  Checks whether MetaTrader 5 (terminal64.exe) is currently running.

    @description
        Uses psutil if available for reliable process detection.
        Falls back to Windows tasklist command if psutil is not installed.

    @return True if terminal64.exe process is found, False otherwise.
    """
    try:
        import psutil
        for proc in psutil.process_iter(['name']):
            name = proc.info.get('name') or ""
            if 'terminal64' in name.lower():
                return True
        return False
    except ImportError:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq terminal64.exe'],
            capture_output=True, text=True
        )
        return 'terminal64.exe' in result.stdout.lower()


def start_mt5() -> bool:
    """
    @brief  Starts MetaTrader 5 if it is not already running.

    @description
        Validates that the MT5_EXE path from config.py exists on disk.
        Launches the process via subprocess.Popen (non-blocking).
        Waits MT5_STARTUP_WAIT_SEC seconds for MT5 to fully initialize
        before verifying the process is running.

    @return True if MT5 is running after this call, False on error.
    """
    if is_mt5_running():
        log("MT5 is already running", "OK")
        return True

    if not MT5_EXE:
        log("MT5_EXE is empty — set it in config.py", "ERROR")
        return False

    if not os.path.exists(MT5_EXE):
        log(f"MT5 executable not found: {MT5_EXE}", "ERROR")
        log("Update MT5_EXE in config.py to the correct path", "ERROR")
        return False

    log(f"Starting MT5: {MT5_EXE}", "INFO")
    try:
        subprocess.Popen([MT5_EXE])
    except Exception as e:
        log(f"Failed to launch MT5: {e}", "ERROR")
        return False

    log(f"Waiting {MT5_STARTUP_WAIT_SEC}s for MT5 to initialize...", "INFO")
    time.sleep(MT5_STARTUP_WAIT_SEC)

    if is_mt5_running():
        log("MT5 started successfully", "OK")
        return True

    log("MT5 process not found after startup wait", "ERROR")
    log("Try increasing MT5_STARTUP_WAIT_SEC in config.py", "WARN")
    return False


if __name__ == "__main__":
    sep("STEP 1: Check / Start MT5")
    success = start_mt5()
    sys.exit(0 if success else 1)