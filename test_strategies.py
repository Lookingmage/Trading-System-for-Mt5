"""
MT5 Quantum AI -- Step 10 Verification: Strategies
====================================================
How to run:
    py -3.11 test_strategies.py

MT5 does NOT need to be open.
Requires: EURUSD H1 data in database (from Step 6).

What this tests:
  1.  All strategy imports succeed
  2.  TradeSignal dataclass validation works
  3.  TradeSignal properties (risk_reward, risk_pips, etc.)
  4.  Each strategy's generate() returns None or a valid TradeSignal
  5.  When a signal is generated: direction/SL/TP are logically correct
  6.  Strategy engine imports and has all 9 strategies
  7.  engine.run() returns a list (possibly empty)
  8.  engine.run_on_df() works on a pre-computed DataFrame
  9.  enabled_strategy_names reflects config
  10. Invalid TradeSignal raises AssertionError (validation works)
"""

import sys
import os
import numpy as np
import pandas as pd

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


# ══════════════════════════════════════════════════════════════
# 1. Imports
# ══════════════════════════════════════════════════════════════
section("1. Import All Strategy Modules")

try:
    from src.strategies.base import TradeSignal, BaseStrategy
    check("from src.strategies.base import TradeSignal", True)
except Exception as e:
    check("from src.strategies.base import TradeSignal", False, str(e))
    sys.exit(1)

strategy_classes = {}
for name, cls_path in [
    ("ema_trend",        ("src.strategies.ema_trend",          "EMATrendStrategy")),
    ("breakout",         ("src.strategies.breakout",           "BreakoutStrategy")),
    ("market_structure", ("src.strategies.market_structure",   "MarketStructureStrategy")),
    ("mtf_trend",        ("src.strategies.mtf_trend",          "MTFTrendStrategy")),
    ("order_block",      ("src.strategies.ict_smc.order_block","OrderBlockStrategy")),
    ("fair_value_gap",   ("src.strategies.ict_smc.fair_value_gap","FairValueGapStrategy")),
    ("liquidity_sweep",  ("src.strategies.ict_smc.liquidity_sweep","LiquiditySweepStrategy")),
    ("session_breakout", ("src.strategies.ict_smc.session_breakout","SessionBreakoutStrategy")),
    ("mss_choch",        ("src.strategies.ict_smc.mss_choch",  "MSSChoCHStrategy")),
]:
    try:
        mod = __import__(cls_path[0], fromlist=[cls_path[1]])
        cls = getattr(mod, cls_path[1])
        strategy_classes[name] = cls
        check(f"  Import: {cls_path[1]}", True)
    except Exception as e:
        check(f"  Import: {cls_path[1]}", False, str(e))

try:
    from src.strategies.engine import strategy_engine, StrategyEngine, ALL_STRATEGIES
    check("from src.strategies.engine import strategy_engine", True)
except Exception as e:
    check("from src.strategies.engine import strategy_engine", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. TradeSignal validation
# ══════════════════════════════════════════════════════════════
section("2. TradeSignal Dataclass Validation")

from datetime import datetime

# Valid BUY signal
try:
    sig = TradeSignal(
        strategy_name="test", symbol="EURUSD", timeframe="H1",
        direction="BUY", entry_price=1.1000, stop_loss=1.0950,
        take_profit=1.1100, signal_strength=0.75,
        bar_time=datetime.utcnow(), reason="test signal",
    )
    check("Valid BUY TradeSignal created", True)
    check("risk_pips correct", abs(sig.risk_pips - 0.0050) < 1e-6)
    check("reward_pips correct", abs(sig.reward_pips - 0.0100) < 1e-6)
    check("risk_reward = 2.0", abs(sig.risk_reward - 2.0) < 0.01)
except Exception as e:
    check("Valid BUY TradeSignal created", False, str(e))

# Valid SELL signal
try:
    sig_sell = TradeSignal(
        strategy_name="test", symbol="XAUUSD", timeframe="H4",
        direction="SELL", entry_price=2000.0, stop_loss=2010.0,
        take_profit=1980.0, signal_strength=0.60,
        bar_time=datetime.utcnow(), reason="test sell",
    )
    check("Valid SELL TradeSignal created", True)
    check("SELL risk_reward = 2.0", abs(sig_sell.risk_reward - 2.0) < 0.01)
except Exception as e:
    check("Valid SELL TradeSignal created", False, str(e))

# Invalid signal must raise AssertionError
invalid_raised = False
try:
    bad = TradeSignal(
        strategy_name="test", symbol="EURUSD", timeframe="H1",
        direction="BUY", entry_price=1.1000,
        stop_loss=1.1100,   # SL ABOVE entry for BUY — invalid
        take_profit=1.1200, signal_strength=0.5,
        bar_time=datetime.utcnow(), reason="bad",
    )
except AssertionError:
    invalid_raised = True
check("Invalid BUY (SL > entry) raises AssertionError", invalid_raised)

invalid_strength = False
try:
    bad2 = TradeSignal(
        strategy_name="test", symbol="EURUSD", timeframe="H1",
        direction="BUY", entry_price=1.1000, stop_loss=1.0950,
        take_profit=1.1100, signal_strength=1.5,  # > 1.0 — invalid
        bar_time=datetime.utcnow(), reason="bad",
    )
except AssertionError:
    invalid_strength = True
check("signal_strength > 1.0 raises AssertionError", invalid_strength)

# ══════════════════════════════════════════════════════════════
# 3. Load data for strategy testing
# ══════════════════════════════════════════════════════════════
section("3. Load EURUSD H1 Feature Data")

TEST_SYM = None
TEST_TF  = "H1"

from src.data.processor import processor
for sym in ["EURUSD", "GBPUSD", "XAUUSD"]:
    if processor.get_bar_count(sym, TEST_TF) >= 100:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"\n{RED}  No data with >= 100 bars.{RESET}")
    sys.exit(1)

from src.features.engineer import engineer
df_clean = processor.process(TEST_SYM, TEST_TF)
feat_df  = engineer.compute(df_clean, symbol=TEST_SYM, timeframe=TEST_TF)

check(f"Loaded {len(feat_df)} feature bars", not feat_df.empty)
print(f"  {CYAN}Using {TEST_SYM} {TEST_TF}: {len(feat_df)} bars{RESET}\n")

# ══════════════════════════════════════════════════════════════
# 4. Run each strategy individually
# ══════════════════════════════════════════════════════════════
section("4. Individual Strategy generate() Calls")

signals_found = []

for strat_name, StratCls in strategy_classes.items():
    try:
        strat = StratCls()
        result = strat.generate(feat_df, TEST_SYM, TEST_TF)

        # Must return None or a valid TradeSignal
        is_valid = result is None or isinstance(result, TradeSignal)
        check(f"  {strat_name}: returns None or TradeSignal", is_valid,
              type(result).__name__ if result is not None else "None")

        if isinstance(result, TradeSignal):
            signals_found.append(result)

            # Validate signal fields
            dir_ok = result.direction in ("BUY", "SELL")
            check(f"  {strat_name}: direction is BUY or SELL", dir_ok,
                  result.direction)

            if result.direction == "BUY":
                sl_ok = result.stop_loss < result.entry_price
                tp_ok = result.take_profit > result.entry_price
            else:
                sl_ok = result.stop_loss > result.entry_price
                tp_ok = result.take_profit < result.entry_price

            check(f"  {strat_name}: SL correct side", sl_ok,
                  f"entry={result.entry_price:.5f} sl={result.stop_loss:.5f}")
            check(f"  {strat_name}: TP correct side", tp_ok,
                  f"entry={result.entry_price:.5f} tp={result.take_profit:.5f}")
            check(f"  {strat_name}: strength in [0,1]",
                  0 <= result.signal_strength <= 1,
                  f"{result.signal_strength:.3f}")
            check(f"  {strat_name}: RR >= 1.0",
                  result.risk_reward >= 1.0,
                  f"RR={result.risk_reward:.2f}")
            print(f"    {CYAN}Signal: {result}{RESET}")

    except Exception as e:
        check(f"  {strat_name}: no exception raised", False, str(e))

# ══════════════════════════════════════════════════════════════
# 5. Strategy Engine
# ══════════════════════════════════════════════════════════════
section("5. Strategy Engine")

check("strategy_engine is StrategyEngine instance",
      isinstance(strategy_engine, StrategyEngine))
check("strategy_engine is singleton",
      strategy_engine is StrategyEngine())

check("ALL_STRATEGIES has 9 strategies", len(ALL_STRATEGIES) == 9,
      f"got {len(ALL_STRATEGIES)}")

names = strategy_engine.strategy_names
check("strategy_names has 9 entries", len(names) == 9, f"got {len(names)}")

expected_names = [
    "ema_trend", "breakout", "market_structure", "mtf_trend",
    "order_block", "fair_value_gap", "liquidity_sweep",
    "session_breakout", "mss_choch",
]
for n in expected_names:
    check(f"  '{n}' in strategy_names", n in names)

enabled = strategy_engine.enabled_strategy_names
check("enabled_strategy_names returns list", isinstance(enabled, list))
check("At least 5 strategies enabled", len(enabled) >= 5,
      f"enabled: {enabled}")

# ══════════════════════════════════════════════════════════════
# 6. engine.run_on_df()
# ══════════════════════════════════════════════════════════════
section("6. engine.run_on_df()")

engine_signals = strategy_engine.run_on_df(feat_df, TEST_SYM, TEST_TF)
check("run_on_df() returns list", isinstance(engine_signals, list))
check("Signals are sorted by strength (highest first)",
      all(
          engine_signals[i].signal_strength >= engine_signals[i+1].signal_strength
          for i in range(len(engine_signals) - 1)
      ) if len(engine_signals) > 1 else True)

print(f"\n  {CYAN}Engine found {len(engine_signals)} signal(s) on {TEST_SYM} {TEST_TF}{RESET}")
for s in engine_signals[:3]:
    print(f"    [{s.strategy_name}] {s.direction} str={s.signal_strength:.2f} | {s.reason[:50]}")

# ══════════════════════════════════════════════════════════════
# 7. engine.run() (loads data internally)
# ══════════════════════════════════════════════════════════════
section("7. engine.run() — Full Pipeline")

run_signals = strategy_engine.run(TEST_SYM, TEST_TF, limit=300)
check("engine.run() returns list", isinstance(run_signals, list))

# With min_strength filter
strong_signals = strategy_engine.run(
    TEST_SYM, TEST_TF, limit=300, min_strength=0.55
)
check("run(min_strength=0.55) returns list", isinstance(strong_signals, list))
check("All strong signals have strength >= 0.55",
      all(s.signal_strength >= 0.55 for s in strong_signals))

print(f"\n  {CYAN}Strong signals (>= 0.55): {len(strong_signals)}{RESET}")

# ══════════════════════════════════════════════════════════════
# 8. get_strategy() lookup
# ══════════════════════════════════════════════════════════════
section("8. Strategy Lookup")

for n in expected_names:
    s = strategy_engine.get_strategy(n)
    check(f"  get_strategy('{n}') returns BaseStrategy",
          isinstance(s, BaseStrategy))

none_strat = strategy_engine.get_strategy("nonexistent")
check("get_strategy('nonexistent') returns None", none_strat is None)

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 10 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Strategy Engine is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 10 works' to begin Step 11.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 10 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
