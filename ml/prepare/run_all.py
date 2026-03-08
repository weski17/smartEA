#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py
==========
Master orchestrator for the ML prepare pipeline.

Runs ``prepare.py`` and ``check_labels.py`` in sequence as subprocesses,
streaming their output live.  Stops on a critical failure.

Pipeline steps
--------------
Step 1 — Feature engineering   (prepare.py)       critical
Step 2 — Label verification    (check_labels.py)  partial

Exit codes
----------
0 — all steps completed successfully.
1 — at least one critical step failed.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

# ==============================================================================
# PIPELINE DEFINITION
# ==============================================================================

STEPS = [
    # (label,    script,             mode)
    ("STEP 1", "prepare.py",      "critical"),
    ("STEP 2", "check_labels.py", "partial"),
]

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# HELPERS
# ==============================================================================

def _sep(title: str = "") -> None:
    """Print a section separator line.

    Args:
        title: Optional text to embed in the separator.
    """
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print("=" * 60)


def _log(msg: str, level: str = "INFO") -> None:
    """Print a timestamped log line.

    Args:
        msg:   Message text.
        level: Severity tag shown in brackets (INFO / OK / WARN / ERROR).
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{level:<5}] {ts}  {msg}")


# ==============================================================================
# STEP RUNNER
# ==============================================================================

def run_step(label: str, script: str) -> int:
    """Run a single pipeline step as a subprocess and stream its output.

    Args:
        label:  Display label e.g. ``'STEP 1'``.
        script: Filename of the script relative to SCRIPTS_DIR.

    Returns:
        Exit code of the subprocess (0 = success).
    """
    script_path = os.path.join(SCRIPTS_DIR, script)

    if not os.path.isfile(script_path):
        _log(f"Script not found: {script_path}", "ERROR")
        return 1

    _sep(f"{label}: {script}")
    start = time.time()

    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=SCRIPTS_DIR,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    elapsed = int(time.time() - start)

    if proc.returncode == 0:
        _log(f"{label} completed in {elapsed}s", "OK")
    elif proc.returncode == 2:
        _log(f"{label} partial success in {elapsed}s (exit 2)", "WARN")
    else:
        _log(f"{label} FAILED in {elapsed}s (exit {proc.returncode})", "ERROR")

    return proc.returncode


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    """Execute all pipeline steps in sequence and print a summary.

    Returns:
        0 if all critical steps passed, 1 if any critical step failed.
    """
    pipeline_start = datetime.now()

    _sep("ML PREPARE PIPELINE — START")
    _log(f"Started : {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    _log(f"Steps   : {len(STEPS)}", "INFO")
    _log(f"Dir     : {SCRIPTS_DIR}", "INFO")

    results      = []
    any_critical = False

    for label, script, mode in STEPS:
        exit_code = run_step(label, script)
        success   = exit_code in (0, 2)
        results.append((label, script, mode, exit_code, success))

        if not success and mode == "critical":
            _log(f"{label} is critical — stopping pipeline", "ERROR")
            any_critical = True
            break

        if not success and mode == "partial":
            _log(f"{label} had issues but pipeline continues", "WARN")

    elapsed_total = int((datetime.now() - pipeline_start).total_seconds())

    _sep("PIPELINE SUMMARY")
    _log(f"Duration : {elapsed_total}s", "INFO")
    _log(f"Steps run: {len(results)}/{len(STEPS)}", "INFO")
    print()

    for label, script, mode, exit_code, success in results:
        status = "OK  " if exit_code == 0 else ("PART" if exit_code == 2 else "FAIL")
        flag   = "critical" if mode == "critical" else "partial "
        print(f"  [{status}] {label}  exit={exit_code}  [{flag}]  {script}")

    if len(results) < len(STEPS):
        for label, script, mode, *_ in STEPS[len(results):]:
            print(f"  [SKIP] {label}  [{mode:8s}]  {script}")

    print()
    if any_critical:
        _log("Pipeline FAILED — critical step did not pass", "ERROR")
        return 1
    else:
        if all(r[4] for r in results):
            _log("Pipeline completed SUCCESSFULLY", "OK")
        else:
            _log("Pipeline completed with WARNINGS", "WARN")
        return 0


if __name__ == "__main__":
    sys.exit(main())
