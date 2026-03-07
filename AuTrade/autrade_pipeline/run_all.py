#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file    run_all.py
@brief   Master orchestrator — runs the complete Autrade pipeline.

@description
    Executes all pipeline steps in sequence as subprocesses.
    Streams each step's output live to the console.
    Stops on critical step failures, continues on partial failures.

    Pipeline steps:
      Step 1 — Start MT5                   (critical)
      Step 2 — Trigger MT5 EA export       (critical)
      Step 3 — Verify exported CSV files   (partial)
      Step 4 — Rotate data folders         (critical)
      Step 5 — Merge Old_Data + New_Data   (partial)
      Step 6 — Verify merged CSV files     (partial)

    Exit code modes:
      critical — pipeline stops immediately if step fails
      partial  — pipeline logs warning but continues

@returns
    exit code 0 — all steps completed successfully
    exit code 1 — at least one critical step failed
"""

import sys
import os
import subprocess
import time
from datetime import datetime

from logger import log, sep

# ─────────────────────────────────────────────────────────────
# PIPELINE DEFINITION
# ─────────────────────────────────────────────────────────────

STEPS = [
    # (label,              script,                      mode)
    ("STEP 1", "step1_start_mt5.py",        "critical"),
    ("STEP 2", "step2_trigger_export.py",   "critical"),
    ("STEP 3", "step3_verify_exports.py",   "partial"),
    ("STEP 4", "step4_rotate_data.py",      "critical"),
    ("STEP 5", "step5_merge.py",            "partial"),
    ("STEP 6", "step6_verify_merged.py",    "partial"),
]

# Directory containing all step scripts
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# STEP RUNNER
# ─────────────────────────────────────────────────────────────

def run_step(label: str, script: str) -> int:
    """
    @brief  Runs a single pipeline step as a subprocess.

    @description
        Launches the script using the same Python interpreter as run_all.py.
        Streams stdout and stderr live to the console.
        Returns the exit code of the subprocess.

    @param  label   Display label e.g. 'STEP 1'.
    @param  script  Filename of the step script e.g. 'step1_start_mt5.py'.
    @return Exit code of the subprocess (0 = success).
    """
    script_path = os.path.join(SCRIPTS_DIR, script)

    if not os.path.exists(script_path):
        log(f"Script not found: {script_path}", "ERROR")
        return 1

    sep(f"{label}: {script}")
    start = time.time()

    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=SCRIPTS_DIR
    )

    # Stream output live
    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    elapsed = int(time.time() - start)

    if proc.returncode == 0:
        log(f"{label} completed in {elapsed}s", "OK")
    elif proc.returncode == 2:
        log(f"{label} partial success in {elapsed}s (exit 2)", "WARN")
    else:
        log(f"{label} FAILED in {elapsed}s (exit {proc.returncode})", "ERROR")

    return proc.returncode


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> int:
    """
    @brief  Executes all pipeline steps in sequence.

    @description
        Runs each step and evaluates the result based on its mode:
          critical — exit code != 0 and != 2 stops the pipeline
          partial  — any exit code is treated as a warning only

        Collects results and prints a final summary table.

    @return 0 if all critical steps passed, 1 if any critical step failed.
    """
    pipeline_start = datetime.now()

    sep("AUTRADE PIPELINE — START")
    log(f"Started  : {pipeline_start.strftime('%Y.%m.%d %H:%M:%S')}", "INFO")
    log(f"Steps    : {len(STEPS)}", "INFO")
    log(f"Scripts  : {SCRIPTS_DIR}", "INFO")

    results      = []
    any_critical = False

    for label, script, mode in STEPS:
        exit_code = run_step(label, script)

        success = exit_code in (0, 2)  # 0=OK, 2=partial OK
        results.append((label, script, mode, exit_code, success))

        if not success and mode == "critical":
            log(f"{label} is critical — stopping pipeline", "ERROR")
            any_critical = True
            break

        if not success and mode == "partial":
            log(f"{label} had issues but pipeline continues", "WARN")

    # ── Final summary
    elapsed_total = int((datetime.now() - pipeline_start).total_seconds())

    sep("PIPELINE SUMMARY")
    log(f"Duration : {elapsed_total}s", "INFO")
    log(f"Steps run: {len(results)}/{len(STEPS)}", "INFO")
    sep()

    for label, script, mode, exit_code, success in results:
        status = "OK  " if exit_code == 0 else ("PART" if exit_code == 2 else "FAIL")
        flag   = "critical" if mode == "critical" else "partial "
        print(f"  [{status}] {label}  exit={exit_code}  [{flag}]  {script}")

    # Steps that were not reached
    if len(results) < len(STEPS):
        for label, script, mode, *_ in STEPS[len(results):]:
            print(f"  [SKIP] {label}  [{mode:8s}]  {script}")

    sep()
    if any_critical:
        log("Pipeline FAILED — critical step did not pass", "ERROR")
        return 1
    else:
        all_ok = all(r[4] for r in results)
        if all_ok:
            log("Pipeline completed SUCCESSFULLY", "OK")
        else:
            log("Pipeline completed with WARNINGS", "WARN")
        return 0


if __name__ == "__main__":
    sys.exit(main())