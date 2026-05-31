"""
MT5 Quantum AI -- Step 12 Verification: Walk-Forward Optimizer
===============================================================
How to run:
    py -3.11 test_optimizer.py

MT5 does NOT need to be open.
Requires: EURUSD H1 data in database (from Step 6).

Uses only 10 Optuna trials and 2 windows for speed.
Production runs use 100 trials and 4 windows.
"""

import sys
import os

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


section("1. Import Optimizer")

try:
    from src.optimizer.optimizer import optimizer, WalkForwardOptimizer, WFOResult
    check("from src.optimizer.optimizer import optimizer", True)
    check("WalkForwardOptimizer singleton", optimizer is WalkForwardOptimizer())
except Exception as e:
    check("from src.optimizer.optimizer import optimizer", False, str(e))
    sys.exit(1)

section("2. WFOResult Structure")

r = WFOResult(symbol="EURUSD", timeframe="H1", strategy="ema_trend",
              n_trials=10, n_windows=2)
check("WFOResult has 'best_params'",   hasattr(r, "best_params"))
check("WFOResult has 'oos_sharpe'",    hasattr(r, "oos_sharpe"))
check("WFOResult has 'accepted'",      hasattr(r, "accepted"))
check("WFOResult has 'window_results'",hasattr(r, "window_results"))
check("WFOResult.summary() works",     isinstance(r.summary(), str))

section("3. Find Data")

from src.data.processor import processor

TEST_SYM = None
for sym in ["EURUSD", "GBPUSD"]:
    if processor.get_bar_count(sym, "H1") >= 200:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"\n{RED}  No data with >= 200 bars.{RESET}")
    sys.exit(1)

print(f"\n  {CYAN}Using {TEST_SYM} H1{RESET}\n")

section("4. Parameter Space Defined for All Strategies")

for strat in ["ema_trend", "breakout", "market_structure", "order_block",
              "fair_value_gap", "liquidity_sweep"]:
    space = optimizer._get_param_space(strat)
    check(f"  param space for '{strat}'", bool(space),
          f"params: {list(space.keys())}")

section("5. Run WFO (fast: 10 trials, 2 windows — ~20 seconds)")

print(f"  {YELLOW}Running Optuna optimization... please wait{RESET}\n")

result = optimizer.run(
    TEST_SYM, "H1", "ema_trend",
    n_trials=10, n_windows=2,
)

check("optimizer.run() returns WFOResult", isinstance(result, WFOResult))
check("result.strategy == 'ema_trend'", result.strategy == "ema_trend")
check("result.symbol correct", result.symbol == TEST_SYM)
check("result.n_windows >= 1", result.n_windows >= 1, f"got {result.n_windows}")
check("result.best_params is dict", isinstance(result.best_params, dict))
check("result.accepted is bool", isinstance(result.accepted, bool))
check("result.is_sharpe is float", isinstance(result.is_sharpe, float))
check("result.oos_sharpe is float", isinstance(result.oos_sharpe, float))
check("result.window_results is list", isinstance(result.window_results, list))
check("window_results has entries", len(result.window_results) >= 1)

if result.window_results:
    wr = result.window_results[0]
    check("window has 'is_sharpe'",  "is_sharpe"  in wr)
    check("window has 'oos_sharpe'", "oos_sharpe" in wr)
    check("window has 'params'",     "params"     in wr)

print(f"\n  {CYAN}{result.summary()}{RESET}")

section("6. Database Save Verified")

from src.database import db, OptimizationResult

with db.session() as sess:
    opts = (
        sess.query(OptimizationResult)
        .filter_by(strategy="ema_trend", symbol=TEST_SYM)
        .all()
    )

check("Optimization results saved to DB", len(opts) >= 1,
      f"found {len(opts)} records")
if opts:
    check("Record has strategy", opts[0].strategy == "ema_trend")
    check("Record has params JSON", bool(opts[0].parameters_json))

section("7. Run on a Second Strategy (breakout)")

result2 = optimizer.run(
    TEST_SYM, "H1", "breakout", n_trials=5, n_windows=2,
)
check("breakout optimization completes", isinstance(result2, WFOResult))
check("breakout has best_params", bool(result2.best_params))
print(f"  {CYAN}Breakout best: {result2.best_params}{RESET}")

section("8. Missing Symbol Returns Gracefully")

empty = optimizer.run("FAKESYM", "H1", "ema_trend", n_trials=5, n_windows=2)
check("Missing symbol returns WFOResult", isinstance(empty, WFOResult))
check("Empty result not accepted", not empty.accepted)

total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 12 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Walk-Forward Optimizer is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 12 works' to begin Step 13.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 12 INCOMPLETE -- {passed}/{total} passed, {failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
