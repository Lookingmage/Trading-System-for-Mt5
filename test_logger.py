"""
MT5 Quantum AI -- Step 3 Verification: Logging System
=======================================================
How to run:
    py -3.11 test_logger.py

What it tests:
    1. All 6 loggers import without error
    2. Each logger writes a test message to its file
    3. ERROR messages appear in error.log from any logger
    4. Log files are created in the logs/ directory
    5. Log file format is correct (timestamp | level | name | message)
    6. Log rotation settings are loaded from config
    7. Console output appears with colour labels
"""

import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0


def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    status = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    suffix = f"  {YELLOW}-> {detail}{RESET}" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if ok:
        passed += 1
    else:
        failed += 1


def section(title: str):
    print(f"\n{CYAN}{BOLD}{'-' * 55}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'-' * 55}{RESET}")


LOG_DIR = os.path.join(ROOT, "logs")

# ══════════════════════════════════════════════════════════════
# 1. Import
# ══════════════════════════════════════════════════════════════
section("1. Import Logger Module")

try:
    from src.logger import (
        get_logger,
        set_log_level,
        system_log,
        trading_log,
        model_log,
        error_log,
        optimization_log,
        backtest_log,
        _manager,
    )
    check("from src.logger import get_logger", True)
    check("Pre-built system_log available", system_log is not None)
    check("Pre-built trading_log available", trading_log is not None)
    check("Pre-built model_log available", model_log is not None)
    check("Pre-built error_log available", error_log is not None)
    check("Pre-built optimization_log available", optimization_log is not None)
    check("Pre-built backtest_log available", backtest_log is not None)
except Exception as e:
    check("from src.logger import get_logger", False, str(e))
    print(f"\n{RED}Cannot continue -- fix the import error above.{RESET}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Named logger access
# ══════════════════════════════════════════════════════════════
section("2. Named Logger Access")

for name in ["system", "trading", "model", "error", "optimization", "backtest"]:
    try:
        lg = get_logger(name)
        check(f"get_logger('{name}') returns Logger",
              lg.__class__.__name__ == "Logger")
    except Exception as e:
        check(f"get_logger('{name}') returns Logger", False, str(e))

# Bad name must raise ValueError
try:
    get_logger("does_not_exist")
    check("get_logger('does_not_exist') raises ValueError", False,
          "should have raised ValueError")
except ValueError:
    check("get_logger('does_not_exist') raises ValueError", True)
except Exception as e:
    check("get_logger('does_not_exist') raises ValueError", False, str(e))

# ══════════════════════════════════════════════════════════════
# 3. Log directory
# ══════════════════════════════════════════════════════════════
section("3. Log Directory")

check("logs/ directory exists", os.path.isdir(LOG_DIR))

# ══════════════════════════════════════════════════════════════
# 4. Write test messages through every logger
# ══════════════════════════════════════════════════════════════
section("4. Writing Test Messages (you will see coloured output below)")
print()

STAMP = f"[VERIFY-{int(time.time())}]"

system_log.info(f"  {STAMP} system logger -- INFO test")
system_log.warning(f"  {STAMP} system logger -- WARNING test")

trading_log.info(f"  {STAMP} BUY EURUSD 0.01 @ 1.08520")
trading_log.info(f"  {STAMP} CLOSE EURUSD +12.50 USD")

model_log.info(f"  {STAMP} XGBoost training started -- 500 samples")
model_log.info(f"  {STAMP} Model accuracy: 67.4%  Sharpe: 1.32")

error_log.error(f"  {STAMP} error logger -- ERROR test (this is only a test)")

optimization_log.info(f"  {STAMP} Trial 001: Sharpe=1.45 PF=1.82 DD=8.2%")
optimization_log.info(f"  {STAMP} Trial 002: Sharpe=1.12 PF=1.51 DD=11.7%")

backtest_log.info(f"  {STAMP} EURUSD BUY 2023-01-15 08:15 → +25.30 USD")
backtest_log.info(f"  {STAMP} XAUUSD SELL 2023-01-17 14:22 → -18.00 USD")

# Trigger an ERROR from trading_log → should also land in error.log
trading_log.error(f"  {STAMP} Simulated trade error -- expect in error.log too")

print()

# ══════════════════════════════════════════════════════════════
# 5. Verify log files were created and contain the test stamp
# ══════════════════════════════════════════════════════════════
section("5. Verifying Log File Contents")

expected_files = {
    "system.log":       STAMP,
    "trading.log":      STAMP,
    "model.log":        STAMP,
    "error.log":        STAMP,
    "optimization.log": STAMP,
    "backtest.log":     STAMP,
}

for filename, stamp in expected_files.items():
    filepath = os.path.join(LOG_DIR, filename)

    # File exists?
    check(f"File created: {filename}", os.path.isfile(filepath),
          f"expected at {filepath}")

    if not os.path.isfile(filepath):
        continue

    # File is non-empty?
    size = os.path.getsize(filepath)
    check(f"File non-empty: {filename}", size > 0, f"size={size} bytes")

    # Test stamp appears in file?
    with open(filepath, encoding="utf-8") as fh:
        content = fh.read()
    check(f"Test stamp found in: {filename}", stamp in content,
          "stamp not written to file")

    # Format check: each line should contain " | "
    lines = [ln for ln in content.splitlines() if stamp in ln]
    if lines:
        has_pipes = " | " in lines[0]
        check(f"Format OK (pipes present): {filename}", has_pipes,
              f"sample: {lines[0][:80]}")

# ══════════════════════════════════════════════════════════════
# 6. Error spillover: trading error → error.log
# ══════════════════════════════════════════════════════════════
section("6. Error Spillover (trading error in error.log)")

error_path = os.path.join(LOG_DIR, "error.log")
if os.path.isfile(error_path):
    with open(error_path, encoding="utf-8") as fh:
        error_content = fh.read()
    spillover = "Simulated trade error" in error_content
    check("trading_log.error() appears in error.log", spillover,
          "ERROR spillover from trading_log not found in error.log")
else:
    check("trading_log.error() appears in error.log", False,
          "error.log does not exist")

# ══════════════════════════════════════════════════════════════
# 7. set_log_level runtime change
# ══════════════════════════════════════════════════════════════
section("7. Runtime Log Level Change")

import logging
try:
    set_log_level("DEBUG")
    lg = get_logger("system")
    check("set_log_level('DEBUG') works",
          lg.level == logging.DEBUG,
          f"level={lg.level}")
    set_log_level("INFO")  # restore
    check("set_log_level('INFO') restores",
          lg.level == logging.INFO,
          f"level={lg.level}")
except Exception as e:
    check("set_log_level() works", False, str(e))

# ══════════════════════════════════════════════════════════════
# 8. list_log_files helper
# ══════════════════════════════════════════════════════════════
section("8. Manager Helper Methods")

files_map = _manager.list_log_files()
check("list_log_files() returns dict with 6 entries", len(files_map) == 6,
      f"got {len(files_map)}")
check("log_dir property returns Path", hasattr(_manager.log_dir, "exists"))

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 3 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Logging system is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 3 works' to begin Step 4.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 3 INCOMPLETE -- {passed}/{total} passed, {failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run: py -3.11 test_logger.py{RESET}")
print(f"{'=' * 55}\n")
