#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    config.py
@brief   Central configuration for the Autrade pipeline.

@description
    Single source of truth for all paths, symbols, and settings.
    All other pipeline steps import from this file.
    This file is configured once and never needs to be changed again.

    Credentials (login, password) are stored in .env — never in this file.
"""

import os
from dotenv import load_dotenv

# Load credentials and machine-specific paths from .env
load_dotenv()

# ─────────────────────────────────────────────────────────────
# BASE DIRECTORY
# ─────────────────────────────────────────────────────────────

BASE_DIR = r"C:\Users\Wael\Desktop\Projekts\smartEA\MT5_DataPipeline"

# ─────────────────────────────────────────────────────────────
# EXPORT SETTINGS
# ─────────────────────────────────────────────────────────────

# Change this to switch symbol — all folders adjust automatically
SYMBOL     = "XAUUSD"
# SYMBOL   = "BTCUSD"   # ← uncomment to switch to Bitcoin
TIMEFRAMES = ["M1", "M15", "M30", "H1", "H4", "D1"]

# ─────────────────────────────────────────────────────────────
# DATA FOLDERS  (symbol-aware — one set of folders per symbol)
# ─────────────────────────────────────────────────────────────

NEW_DATA_DIR    = os.path.join(BASE_DIR, "New_Data",    SYMBOL)
OLD_DATA_DIR    = os.path.join(BASE_DIR, "Old_Data",    SYMBOL)
MERGED_DATA_DIR = os.path.join(BASE_DIR, "Merged_Data", SYMBOL)
BACKUP_DATA_DIR = os.path.join(BASE_DIR, "Backup_Data", SYMBOL)

# ─────────────────────────────────────────────────────────────
# MT5 PATHS (Loaded from .env for security and privacy)
# ─────────────────────────────────────────────────────────────

MT5_EXE      = r"C:\Program Files\Vantage International MT5\terminal64.exe"

# These are specific to your machine and hidden in .env
_TERMINAL_ID = os.getenv("MT5_TERMINAL_ID", "AE2CC2E013FDE1E3CDF010AA51C60400")
_APPDATA     = os.getenv("MT5_APPDATA",     r"C:\Users\Wael\AppData\Roaming\MetaQuotes\Terminal")

MT5_EXPERTS_DIR  = os.path.join(_APPDATA, _TERMINAL_ID, "MQL5", "Experts")
MT5_COMMON_FILES = os.path.join(_APPDATA, "Common", "Files")

EA_SOURCE = os.path.join(BASE_DIR, "autrade_pipeline", "ExportHistoryEA.mq5")

# ─────────────────────────────────────────────────────────────
# TIMING SETTINGS
# ─────────────────────────────────────────────────────────────

MT5_STARTUP_WAIT_SEC = 15
EXPORT_TIMEOUT_SEC   = 600
POLL_INTERVAL_SEC    = 10