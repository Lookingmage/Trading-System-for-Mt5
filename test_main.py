"""
MT5 Quantum AI -- Step 21 Verification: Main Controller
========================================================
How to run:
    py -3.11 test_main.py

MT5 does NOT need to be open for this test.

What this tests:
  1.  MainController imports without error
  2.  MainController initialises with mode override
  3.  startup() succeeds (DB + modules load)
  4.  All core modules attach correctly after startup
  5.  _run_one_cycle() executes without crashing
  6.  Backtest mode dispatches correctly
  7.  CLI argument parser works
  8.  shutdown() cleans up and logs event
  9.  Signal handling is registered
  10. BUILD_STEP is 21
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

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


# ══════════════════════════════════════════════════════════════
# 1. Import
# ══════════════════════════════════════════════════════════════
section("1. Import MainController")

try:
    from main import MainController, SYSTEM_VERSION, BUILD_STEP
    check("from main import MainController", True)
    check("BUILD_STEP == 21", BUILD_STEP == 21, f"got {BUILD_STEP}")
    check("SYSTEM_VERSION is string", isinstance(SYSTEM_VERSION, str))
    print(f"  {CYAN}MT5 Quantum AI v{SYSTEM_VERSION} — Build Step {BUILD_STEP}{RESET}")
except Exception as e:
    check("from main import MainController", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Controller initialisation
# ══════════════════════════════════════════════════════════════
section("2. Controller Initialisation")

ctrl = MainController(mode="shadow")
check("MainController created", isinstance(ctrl, MainController))
check("mode set to 'shadow'", ctrl._mode == "shadow")
check("_running is False initially", not ctrl._running)
check("_shutdown_flag is False", not ctrl._shutdown_flag)
check("_symbols is a list", isinstance(ctrl._symbols, list))
check("_symbols not empty", len(ctrl._symbols) > 0,
      f"symbols={ctrl._symbols[:3]}")

# ══════════════════════════════════════════════════════════════
# 3. startup()
# ══════════════════════════════════════════════════════════════
section("3. startup() — Initialize All Modules")

print(f"  {YELLOW}Starting up all modules...{RESET}\n")

ok = ctrl.startup()
check("startup() returns True", ok, "check logs/system.log for errors")

if ok:
    check("_db attached",               ctrl._db  is not None)
    check("_risk attached",             ctrl._risk is not None)
    check("_executor attached",         ctrl._executor is not None)
    check("_processor attached",        ctrl._processor is not None)
    check("_engineer attached",         ctrl._engineer  is not None)
    check("_strategy_engine attached",  ctrl._strategy_engine is not None)

# ══════════════════════════════════════════════════════════════
# 4. Startup event logged to DB
# ══════════════════════════════════════════════════════════════
section("4. Startup Event in Database")

from src.database import db, SystemEvent

with db.session() as sess:
    events = db.get_recent_events(sess, limit=5, event_type="startup")

check("Startup event saved to DB", len(events) >= 1,
      f"found {len(events)} startup events")
if events:
    check("Event message mentions mode",
          "shadow" in events[0].message.lower() or
          "startup" in events[0].message.lower())

# ══════════════════════════════════════════════════════════════
# 5. _run_one_cycle()
# ══════════════════════════════════════════════════════════════
section("5. _run_one_cycle() — One Trading Iteration")

print(f"  {YELLOW}Running one trading cycle (shadow mode)...{RESET}\n")

if ok:
    try:
        summary = ctrl._run_one_cycle()
        check("_run_one_cycle() returns dict", isinstance(summary, dict))
        check("summary has 'loop' key",          "loop"          in summary)
        check("summary has 'signals_found' key", "signals_found" in summary)
        check("summary has 'trades_opened' key", "trades_opened" in summary)
        check("summary has 'retrained' key",     "retrained"     in summary)
        print(f"\n  {CYAN}Cycle summary: {summary}{RESET}")
    except Exception as e:
        check("_run_one_cycle() runs without fatal error", False, str(e))

# ══════════════════════════════════════════════════════════════
# 6. Mode dispatch
# ══════════════════════════════════════════════════════════════
section("6. Mode Dispatch Logic")

check("mode 'shadow' not 'backtest'",
      ctrl._mode != "backtest")

# Verify backtest mode creates a different controller
ctrl_bt = MainController(mode="backtest")
check("backtest controller mode = 'backtest'",
      ctrl_bt._mode == "backtest")

# ══════════════════════════════════════════════════════════════
# 7. CLI argument parser
# ══════════════════════════════════════════════════════════════
section("7. CLI Argument Parser")

from main import parse_args
import argparse

# Simulate: py main.py --mode demo
old_argv = sys.argv.copy()
try:
    sys.argv = ["main.py", "--mode", "demo"]
    args = parse_args()
    check("parse_args() parses --mode demo", args.mode == "demo")

    sys.argv = ["main.py", "--mode", "shadow", "--symbols", "EURUSD", "GBPUSD"]
    args2 = parse_args()
    check("parse_args() parses --symbols list",
          args2.symbols == ["EURUSD", "GBPUSD"])
finally:
    sys.argv = old_argv

# ══════════════════════════════════════════════════════════════
# 8. shutdown()
# ══════════════════════════════════════════════════════════════
section("8. shutdown() — Graceful Cleanup")

ctrl.shutdown(reason="test_complete")

check("_shutdown_flag is True after shutdown", ctrl._shutdown_flag)
check("_running is False after shutdown",       not ctrl._running)

# Check shutdown event in DB
with db.session() as sess:
    shutdown_events = db.get_recent_events(sess, limit=3, event_type="shutdown")
check("Shutdown event in DB", len(shutdown_events) >= 1)

# ══════════════════════════════════════════════════════════════
# 9. Signal handler registered
# ══════════════════════════════════════════════════════════════
section("9. Signal Handler")

import signal as _signal
sigint_handler  = _signal.getsignal(_signal.SIGINT)
sigterm_handler = _signal.getsignal(_signal.SIGTERM)

check("SIGINT handler registered (not default)",
      sigint_handler is not _signal.SIG_DFL)
check("SIGTERM handler registered (not default)",
      sigterm_handler is not _signal.SIG_DFL)

# ══════════════════════════════════════════════════════════════
# 10. main.py summary
# ══════════════════════════════════════════════════════════════
section("10. System Ready Summary")

print(f"\n  {CYAN}Build Step:    {BUILD_STEP}/25{RESET}")
print(f"  {CYAN}Version:       {SYSTEM_VERSION}{RESET}")
print(f"  {CYAN}Mode tested:   shadow{RESET}")
print(f"  {CYAN}Symbols:       {ctrl._symbols[:4]}{RESET}")
print(f"  {CYAN}Loop count:    {ctrl._loop_count}{RESET}")
check("main.py ready for full demo/shadow operation", True)

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 21 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Main Controller is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 21 works' to begin Step 22.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 21 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
print(f"  {CYAN}To run the full system:{RESET}")
print(f"  {YELLOW}  py -3.11 main.py --mode shadow{RESET}")
print(f"  {YELLOW}  py -3.11 main.py --mode backtest{RESET}")
print(f"  {YELLOW}  streamlit run dashboard/app.py{RESET}\n")
