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

# ─────────────────────────────────────────────────────────────
# BASE DIRECTORY
# ─────────────────────────────────────────────────────────────

BASE_DIR = r"C:\Users\Wael\Desktop\Projekts\smartEA\AuTrade"

# ─────────────────────────────────────────────────────────────
# DATA FOLDERS
# ─────────────────────────────────────────────────────────────

NEW_DATA_DIR    = os.path.join(BASE_DIR, "New_Data")
OLD_DATA_DIR    = os.path.join(BASE_DIR, "Old_Data")
MERGED_DATA_DIR = os.path.join(BASE_DIR, "Merged_Data")
BACKUP_DATA_DIR = os.path.join(BASE_DIR, "Backup_Data")

# ─────────────────────────────────────────────────────────────
# MT5 PATHS
# ─────────────────────────────────────────────────────────────

MT5_EXE      = r"C:\Program Files\Vantage International MT5\terminal64.exe"

_TERMINAL_ID = "AE2CC2E013FDE1E3CDF010AA51C60400"
_APPDATA     = r"C:\Users\Wael\AppData\Roaming\MetaQuotes\Terminal"

# Experts folder — EA goes here (runs permanently in MT5)
MT5_EXPERTS_DIR  = os.path.join(_APPDATA, _TERMINAL_ID, "MQL5", "Experts")

# Common Files — shared folder for trigger.txt and exported CSVs
MT5_COMMON_FILES = os.path.join(_APPDATA, "Common", "Files")

# ─────────────────────────────────────────────────────────────
# EA PATH
# ─────────────────────────────────────────────────────────────

# Expert Advisor source — copy to MT5_EXPERTS_DIR and compile with F7
# Then drag onto XAUUSD chart once — runs permanently after that
EA_SOURCE = os.path.join(BASE_DIR, "ExportHistoryEA.mq5")

# ─────────────────────────────────────────────────────────────
# EXPORT SETTINGS
# ─────────────────────────────────────────────────────────────

SYMBOL     = "XAUUSD"
TIMEFRAMES = ["M1", "M15", "M30", "H1", "H4", "D1"]

# ─────────────────────────────────────────────────────────────
# TIMING SETTINGS
# ─────────────────────────────────────────────────────────────

MT5_STARTUP_WAIT_SEC = 15
EXPORT_TIMEOUT_SEC   = 600
POLL_INTERVAL_SEC    = 10