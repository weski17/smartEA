#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    step4_rotate_data.py
@brief   Step 4 — Rotates data folders to prepare for a fresh merge.

@description
    Executes a structured folder rotation after a successful MT5 export.
    This ensures that Old_Data always contains the last merged dataset
    and New_Data always contains the latest MT5 exports before merging.

    Rotation order:
      1. BACKUP  : Copy Merged_Data  -> Backup_Data/YYYYMMDD_HHMMSS/
      2. CLEAR   : Delete all files in Merged_Data
      3. CLEAR   : Delete all files in Old_Data
      4. ROTATE  : Move New_Data     -> Old_Data
      5. FETCH   : Copy MT5 exports  -> New_Data

    After this step:
      Old_Data/  — contains the previous merged CSVs (full history)
      New_Data/  — contains the fresh MT5 exports (latest bars)
      Merged_Data/ — empty, ready for Step 5

@returns
    exit code 0 — rotation completed successfully
    exit code 1 — critical failure
"""

import sys
import os
import shutil
import glob
import json
from datetime import datetime

from config import (
    MT5_COMMON_FILES,
    NEW_DATA_DIR, OLD_DATA_DIR,
    MERGED_DATA_DIR, BACKUP_DATA_DIR,
    SYMBOL
)
from logger import log, sep

FOUND_FILES_JSON = os.path.join(os.path.dirname(__file__), "_found_files.json")


def ensure_folders() -> None:
    """
    @brief  Creates all required data folders if they do not exist.
    """
    for path in [NEW_DATA_DIR, OLD_DATA_DIR, MERGED_DATA_DIR, BACKUP_DATA_DIR]:
        os.makedirs(path, exist_ok=True)


def count_csv(folder: str) -> int:
    """
    @brief  Returns the number of CSV files in a folder.
    @param  folder  Path to the folder.
    @return Number of CSV files found.
    """
    return len(glob.glob(os.path.join(folder, "*.csv")))


def clear_folder(folder: str, label: str) -> int:
    """
    @brief  Deletes all CSV files in a folder.
    @param  folder  Path to the folder to clear.
    @param  label   Display label for logging.
    @return Number of files deleted.
    """
    files   = glob.glob(os.path.join(folder, "*.csv"))
    deleted = 0
    for f in files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            log(f"Cannot delete {os.path.basename(f)}: {e}", "WARN")
    log(f"{label}: {deleted} file(s) deleted", "OK" if deleted >= 0 else "WARN")
    return deleted


def copy_files(src: str, dst: str, label: str) -> int:
    """
    @brief  Copies all CSV files from src to dst folder.
    @param  src    Source folder path.
    @param  dst    Destination folder path.
    @param  label  Display label for logging.
    @return Number of files copied.
    """
    files  = glob.glob(os.path.join(src, "*.csv"))
    copied = 0
    for f in files:
        dst_path = os.path.join(dst, os.path.basename(f))
        try:
            shutil.copy2(f, dst_path)
            log(f"  Copied: {os.path.basename(f)}", "INFO")
            copied += 1
        except Exception as e:
            log(f"  Cannot copy {os.path.basename(f)}: {e}", "WARN")
    log(f"{label}: {copied} file(s) copied", "OK" if copied > 0 else "WARN")
    return copied


def move_files(src: str, dst: str, label: str) -> int:
    """
    @brief  Moves all CSV files from src to dst folder.
    @param  src    Source folder path.
    @param  dst    Destination folder path.
    @param  label  Display label for logging.
    @return Number of files moved.
    """
    files = glob.glob(os.path.join(src, "*.csv"))
    moved = 0
    for f in files:
        dst_path = os.path.join(dst, os.path.basename(f))
        try:
            shutil.move(f, dst_path)
            log(f"  Moved: {os.path.basename(f)}", "INFO")
            moved += 1
        except Exception as e:
            log(f"  Cannot move {os.path.basename(f)}: {e}", "WARN")
    log(f"{label}: {moved} file(s) moved", "OK" if moved > 0 else "WARN")
    return moved


def fetch_mt5_exports() -> int:
    """
    @brief  Copies fresh MT5 exports from Common/Files into New_Data.

    @description
        Reads _found_files.json to get exact paths of exported CSVs.
        Falls back to glob pattern if JSON is not available.
        Updates _found_files.json with new paths in New_Data.

    @return Number of files copied to New_Data.
    """
    if os.path.exists(FOUND_FILES_JSON):
        try:
            with open(FOUND_FILES_JSON, "r", encoding="utf-8") as f:
                found = json.load(f)

            copied  = 0
            updated = {}
            for tf, src_path in found.items():
                if not src_path or not os.path.exists(src_path):
                    log(f"  {tf}: source not found: {src_path}", "WARN")
                    continue
                dst_path = os.path.join(NEW_DATA_DIR, os.path.basename(src_path))
                try:
                    shutil.copy2(src_path, dst_path)
                    updated[tf] = dst_path
                    log(f"  {tf}: {os.path.basename(src_path)}", "INFO")
                    copied += 1
                except Exception as e:
                    log(f"  {tf}: copy failed: {e}", "WARN")

            # Update JSON with new paths in New_Data
            with open(FOUND_FILES_JSON, "w", encoding="utf-8") as f:
                json.dump(updated, f, indent=2)

            log(f"Fetch MT5 exports: {copied} file(s) copied to New_Data",
                "OK" if copied > 0 else "WARN")
            return copied

        except Exception as e:
            log(f"Error reading _found_files.json: {e}", "ERROR")

    # Fallback: glob pattern
    log("_found_files.json not found — using glob fallback", "WARN")
    files  = glob.glob(os.path.join(MT5_COMMON_FILES, f"{SYMBOL}_*.csv"))
    copied = 0
    for f in files:
        dst_path = os.path.join(NEW_DATA_DIR, os.path.basename(f))
        try:
            shutil.copy2(f, dst_path)
            log(f"  Fallback copied: {os.path.basename(f)}", "INFO")
            copied += 1
        except Exception as e:
            log(f"  Fallback copy failed: {e}", "WARN")
    return copied


def rotate() -> bool:
    """
    @brief  Executes the full folder rotation sequence.

    @description
        Runs all 5 rotation steps in order:
          1. Backup Merged_Data
          2. Clear Merged_Data
          3. Clear Old_Data
          4. Move New_Data to Old_Data
          5. Fetch MT5 exports into New_Data

    @return True if rotation completed without critical errors.
    """
    ensure_folders()

    # ── 1. BACKUP: Merged_Data → Backup_Data/TIMESTAMP/
    sep("1/5  BACKUP: Merged_Data -> Backup_Data")
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_sub = os.path.join(BACKUP_DATA_DIR, ts)
    os.makedirs(backup_sub, exist_ok=True)
    n_merged   = count_csv(MERGED_DATA_DIR)
    log(f"Backup folder: {backup_sub}", "INFO")
    log(f"Files in Merged_Data: {n_merged}", "INFO")
    if n_merged > 0:
        copy_files(MERGED_DATA_DIR, backup_sub, "Backup")
    else:
        log("Merged_Data is empty — nothing to backup", "WARN")

    # ── 2. CLEAR: Merged_Data
    sep("2/5  CLEAR: Merged_Data")
    clear_folder(MERGED_DATA_DIR, "Merged_Data")

    # ── 3. CLEAR: Old_Data
    sep("3/5  CLEAR: Old_Data")
    clear_folder(OLD_DATA_DIR, "Old_Data")

    # ── 4. ROTATE: New_Data → Old_Data
    sep("4/5  ROTATE: New_Data -> Old_Data")
    n_new = count_csv(NEW_DATA_DIR)
    log(f"Files in New_Data: {n_new}", "INFO")
    if n_new > 0:
        moved = move_files(NEW_DATA_DIR, OLD_DATA_DIR, "New_Data -> Old_Data")
    else:
        log("New_Data is empty — nothing to rotate", "WARN")
        moved = 0

    # ── 5. FETCH: MT5 exports → New_Data
    sep("5/5  FETCH: MT5 exports -> New_Data")
    fetched = fetch_mt5_exports()
    if fetched == 0:
        log("No MT5 exports found in Common/Files!", "ERROR")
        return False

    # ── Final state
    sep("ROTATION COMPLETE")
    log(f"Backup_Data : {count_csv(backup_sub)} file(s) in {ts}", "INFO")
    log(f"Old_Data    : {count_csv(OLD_DATA_DIR)} file(s)", "INFO")
    log(f"New_Data    : {count_csv(NEW_DATA_DIR)} file(s)", "INFO")
    log(f"Merged_Data : {count_csv(MERGED_DATA_DIR)} file(s) (empty, ready for merge)", "INFO")
    return True


if __name__ == "__main__":
    sep("STEP 4: Rotate Data Folders")
    success = rotate()
    sys.exit(0 if success else 1)