#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    logger.py
@brief   Shared logging utilities for the Autrade pipeline.

@description
    Provides consistent timestamped log output across all pipeline steps.
    Every step imports log() and sep() from this module.
    No configuration needed — works out of the box.
"""

from datetime import datetime


def log(msg: str, level: str = "INFO") -> None:
    """
    @brief  Prints a timestamped, leveled log message to stdout.

    @param  msg    The message text to display.
    @param  level  Severity level:
                     INFO  — normal progress message
                     OK    — step completed successfully
                     WARN  — non-critical issue, pipeline continues
                     ERROR — critical issue, pipeline may stop
                     STEP  — sub-step label inside a larger step
    
    @example
        log("Exporting M15 data")
        log("File saved successfully", "OK")
        log("Old file not found, skipping", "WARN")
        log("MT5 could not be started", "ERROR")
    """
    ts     = datetime.now().strftime("%H:%M:%S")
    icons  = {
        "INFO":  "[INFO ]",
        "OK":    "[OK   ]",
        "WARN":  "[WARN ]",
        "ERROR": "[ERROR]",
        "STEP":  "[STEP ]",
    }
    prefix = icons.get(level, "[INFO ]")
    print(f"{ts} {prefix} {msg}")


def sep(title: str = "") -> None:
    """
    @brief  Prints a visual separator line with an optional section title.

    @param  title  If provided, prints a double-line banner with the title.
                   If empty, prints a single dashed divider line.

    @example
        sep()                    # prints dashed line
        sep("STEP 1: Start MT5") # prints banner with title
    """
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print("-" * 60)