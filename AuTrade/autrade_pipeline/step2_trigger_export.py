#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step2_trigger_export.py
@brief   Step 2 — Triggers the MT4 EA export via a file-based signal.

@description
    Communicates with the ExportHistoryEA.mq4 Expert Advisor running in MT4
    using a shared trigger file in the MT4 Common/Files folder.

    Trigger file protocol:
      Python writes  "START"   -> EA picks up the signal and begins export
      EA writes      "RUNNING" -> export is in progress
      EA writes      "DONE"    -> export completed successfully
      EA writes      "ERROR"   -> export failed inside MT4
      EA writes      "IDLE"    -> EA is waiting for next trigger

    This step:
      1. Resets trigger to IDLE
      2. Writes "START" to trigger.txt with a timestamp
      3. Polls every POLL_INTERVAL_SEC seconds for EA response
      4. Prints progress every 30 seconds while waiting
      5. Returns success when EA writes "DONE"
      6. Times out after EXPORT_TIMEOUT_SEC seconds

@returns
    exit code 0 — EA responded with DONE, export successful
    exit code 1 — ERROR response, timeout, or trigger file not writable
"""

import sys
import os
import time
from datetime import datetime

from config import MT5_COMMON_FILES, EXPORT_TIMEOUT_SEC, POLL_INTERVAL_SEC
from logger import log, sep

# Path to the shared trigger file in MT4 Common/Files
TRIGGER_FILE    = os.path.join(MT5_COMMON_FILES, "trigger.txt")

# How often to print a progress line while waiting (seconds)
PROGRESS_EVERY  = 30


def write_trigger(status: str) -> bool:
    """
    @brief  Writes a status string to the shared trigger file.

    @param  status  The status to write:
                      "START" — signals the EA to begin export
                      "IDLE"  — resets trigger to waiting state
    @return True if written successfully, False on I/O error.
    """
    try:
        os.makedirs(MT5_COMMON_FILES, exist_ok=True)
        with open(TRIGGER_FILE, "w", encoding="utf-8") as f:
            f.write(status)
        return True
    except Exception as e:
        log(f"Cannot write trigger file: {e}", "ERROR")
        log(f"Path: {TRIGGER_FILE}", "ERROR")
        return False


def read_trigger() -> str:
    """
    @brief  Reads the current status from the shared trigger file.

    @return Status string: START | RUNNING | DONE | ERROR | IDLE
            Returns empty string if file does not exist or is unreadable.
    """
    try:
        with open(TRIGGER_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def trigger_and_wait() -> bool:
    """
    @brief  Sends START signal to the MT4 EA and waits for DONE response.

    @description
        Resets any leftover state, sends the START trigger, then polls
        the trigger file until the EA responds or the timeout is reached.

        Terminal states (stop polling):
          DONE  — export completed successfully -> return True
          ERROR — export failed inside MT4      -> return False

        In-progress states (keep polling):
          START   — EA has not yet picked up the signal
          RUNNING — EA is actively exporting data

        Progress is printed every PROGRESS_EVERY seconds to show the
        pipeline is alive during long exports (M1 can take minutes).

    @return True if EA responded with DONE, False on ERROR or timeout.
    """
    log(f"Trigger file  : {TRIGGER_FILE}", "INFO")
    log(f"Timeout       : {EXPORT_TIMEOUT_SEC}s", "INFO")
    log(f"Poll interval : {POLL_INTERVAL_SEC}s", "INFO")

    # ── Reset leftover state from previous run
    log("Resetting trigger to IDLE...", "INFO")
    if not write_trigger("IDLE"):
        return False
    time.sleep(1)

    # ── Send START signal
    sent_at = datetime.now().strftime("%H:%M:%S")
    if not write_trigger("START"):
        return False

    log(f"START signal sent at {sent_at}", "OK")
    log("Waiting for ExportHistoryEA to respond...", "INFO")
    log("Reminder: EA must be running on a XAUUSD chart in MT5!", "WARN")

    start         = time.time()
    timeout       = EXPORT_TIMEOUT_SEC
    last_progress = time.time()
    last_status   = ""

    while time.time() - start < timeout:
        elapsed = int(time.time() - start)
        status  = read_trigger()

        # ── Terminal states
        if status == "DONE":
            log(f"EA responded: DONE — export complete ({elapsed}s)", "OK")
            return True

        if status == "ERROR":
            log(f"EA responded: ERROR after {elapsed}s", "ERROR")
            log("Open MT4 -> Experts tab to see the error details", "ERROR")
            return False

        # ── Log status change immediately
        if status != last_status:
            if status == "RUNNING":
                log(f"EA status changed: RUNNING — export in progress", "INFO")
            elif status == "START":
                log(f"EA status: START — waiting for EA to pick up signal", "INFO")
            last_status = status

        # ── Print progress every PROGRESS_EVERY seconds
        if time.time() - last_progress >= PROGRESS_EVERY:
            remaining = timeout - elapsed
            log(f"Still waiting... {elapsed}s elapsed | {remaining}s remaining | status={status}", "INFO")
            last_progress = time.time()

        time.sleep(POLL_INTERVAL_SEC)

    # ── Timeout reached
    log(f"Timeout after {timeout}s — EA did not respond with DONE", "ERROR")
    log("Possible causes:", "WARN")
    log("  1. ExportHistoryEA.mq4 is not running on any chart", "WARN")
    log("  2. MT4 automated trading is disabled (check top toolbar)", "WARN")
    log("  3. Export is taking longer than expected", "WARN")
    log(f"  4. Increase EXPORT_TIMEOUT_SEC in config.py (current: {timeout}s)", "WARN")
    return False


if __name__ == "__main__":
    sep("STEP 2: Trigger MT4 Export")
    success = trigger_and_wait()
    sys.exit(0 if success else 1)