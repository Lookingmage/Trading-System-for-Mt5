"""
MT5 Quantum AI -- Step 11 Verification: Backtesting Engine
===========================================================
How to run:
    py -3.11 test_backtest.py

MT5 does NOT need to be open.
Requires: EURUSD H1 data in database (from Step 6).

What this tests:
  1.  BacktestEngine imports correctly
  2.  BacktestTrade dataclass is correct
  3.  BacktestMetrics has all required fields
  4.  BacktestResult has trades + equity_curve + metrics
  5.  engine.run() completes without error
  6.  metrics structure has all required keys
  7.  equity_curve is a pandas Series
  8.  If trades exist: all required fields present
  9.  If trades exist: BUY SL < entry, SELL SL > entry
  10. If trades exist: net_pnl = gross_pnl - commission
  11. Profit factor is correct: gross_profit / gross_loss
  12. Win rate is correct: winners / total
  13. result.summary() returns a string
  14. result.save_report() creates a file in reports/
  15. Empty result (no data) returns safely
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


# ══════════════════════════════════════════════════════════════
# 1. Import
# ══════════════════════════════════════════════════════════════
section("1. Import Backtest Engine")

try:
    from src.backtest.engine import (
        backtest_engine, BacktestEngine,
        BacktestResult, BacktestMetrics, BacktestTrade,
    )
    check("from src.backtest.engine import backtest_engine", True)
    check("BacktestEngine singleton",
          backtest_engine is BacktestEngine())
except Exception as e:
    check("from src.backtest.engine import backtest_engine", False, str(e))
    print(f"\n{RED}Cannot continue — fix import error above.{RESET}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Data classes
# ══════════════════════════════════════════════════════════════
section("2. Data Class Structure")

from datetime import datetime
import pandas as pd

# BacktestMetrics has required fields
m = BacktestMetrics()
required_metric_fields = [
    "initial_balance", "final_balance", "net_profit", "gross_profit",
    "gross_loss", "total_trades", "winning_trades", "losing_trades",
    "win_rate", "profit_factor", "avg_win", "avg_loss",
    "max_drawdown", "max_drawdown_pct", "recovery_factor",
    "sharpe_ratio", "sortino_ratio", "expectancy", "return_pct",
]
for f in required_metric_fields:
    check(f"BacktestMetrics has '{f}'", hasattr(m, f))

# BacktestTrade has required fields
dummy_trade = BacktestTrade(
    symbol="EURUSD", strategy="ema_trend", direction="BUY",
    entry_bar=0, exit_bar=5,
    entry_time=datetime(2024, 1, 1), exit_time=datetime(2024, 1, 2),
    entry_price=1.1000, exit_price=1.1050,
    stop_loss=1.0950, take_profit=1.1100,
    lot_size=0.01, gross_pnl=50.0, commission=0.07,
    net_pnl=49.93, pips=50.0, close_reason="tp",
)
check("BacktestTrade.is_winner True for positive pnl", dummy_trade.is_winner)
check("BacktestTrade.to_dict() returns dict",
      isinstance(dummy_trade.to_dict(), dict))
check("to_dict() has 'net_pnl' key", "net_pnl" in dummy_trade.to_dict())
check("to_dict() has 'close_reason' key",
      "close_reason" in dummy_trade.to_dict())

# ══════════════════════════════════════════════════════════════
# 3. Find data
# ══════════════════════════════════════════════════════════════
section("3. Find EURUSD H1 Data")

from src.data.processor import processor

TEST_SYM = None
for sym in ["EURUSD", "GBPUSD", "XAUUSD"]:
    if processor.get_bar_count(sym, "H1") >= 250:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"\n{RED}  No symbol with >= 250 bars found.{RESET}")
    print(f"{YELLOW}  Run: py -3.11 test_downloader.py first{RESET}")
    sys.exit(1)

n_bars = processor.get_bar_count(TEST_SYM, "H1")
print(f"\n  {CYAN}Using {TEST_SYM} H1: {n_bars} bars{RESET}\n")

# ══════════════════════════════════════════════════════════════
# 4. Run backtest
# ══════════════════════════════════════════════════════════════
section("4. Run Backtest (may take 15-30 seconds)")

print(f"  {YELLOW}Running bar-by-bar simulation...{RESET}\n")

result = backtest_engine.run(
    TEST_SYM, "H1",
    initial_balance=10_000.0,
    lot_size=0.01,
    commission_per_lot=7.0,
)

check("engine.run() returns BacktestResult",
      isinstance(result, BacktestResult))

# ══════════════════════════════════════════════════════════════
# 5. Result structure
# ══════════════════════════════════════════════════════════════
section("5. BacktestResult Structure")

check("result.metrics is BacktestMetrics",
      isinstance(result.metrics, BacktestMetrics))
check("result.trades is list",
      isinstance(result.trades, list))
check("result.equity_curve is pd.Series",
      isinstance(result.equity_curve, pd.Series))
check("equity_curve is not empty", len(result.equity_curve) > 0,
      f"len={len(result.equity_curve)}")
check("equity_curve starts at initial balance",
      abs(result.equity_curve.iloc[0] - 10_000.0) < 1.0,
      f"first={result.equity_curve.iloc[0]:.2f}")

# ══════════════════════════════════════════════════════════════
# 6. Metrics validation
# ══════════════════════════════════════════════════════════════
section("6. Metrics Validation")

m = result.metrics
check(f"total_bars > 0 (got {m.total_bars})", m.total_bars > 0)
check("symbol correct", m.symbol == TEST_SYM)
check("timeframe correct", m.timeframe == "H1")
check("initial_balance = 10000", m.initial_balance == 10_000.0)
check("period_from is set", bool(m.period_from))
check("period_to is set", bool(m.period_to))

# Metric consistency
check("final_balance = initial + net_profit",
      abs(m.final_balance - (m.initial_balance + m.net_profit)) < 0.01,
      f"final={m.final_balance} init+net={m.initial_balance + m.net_profit}")
check("net_profit = gross_profit - gross_loss",
      abs(m.net_profit - (m.gross_profit - m.gross_loss)) < 0.01)
check("return_pct = net_profit / initial * 100",
      abs(m.return_pct - m.net_profit / m.initial_balance * 100) < 0.01)

n_trades = m.total_trades
print(f"\n  {CYAN}Total trades: {n_trades}{RESET}")
print(f"  {CYAN}Net profit:   ${m.net_profit:+,.2f}  ({m.return_pct:+.2f}%){RESET}")

if n_trades > 0:
    check(f"winning + losing = total ({m.winning_trades}+{m.losing_trades}={n_trades})",
          m.winning_trades + m.losing_trades == n_trades)
    check("win_rate in [0, 100]",
          0 <= m.win_rate <= 100, f"{m.win_rate:.1f}%")
    check("profit_factor >= 0", m.profit_factor >= 0)
    check("max_drawdown >= 0", m.max_drawdown >= 0)
    check("max_drawdown_pct >= 0", m.max_drawdown_pct >= 0)
    print(f"  {CYAN}Win rate:     {m.win_rate:.1f}%{RESET}")
    print(f"  {CYAN}Profit factor:{m.profit_factor:.2f}{RESET}")
    print(f"  {CYAN}Max drawdown: ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2f}%){RESET}")
    print(f"  {CYAN}Sharpe ratio: {m.sharpe_ratio:.2f}{RESET}")
else:
    print(f"  {YELLOW}  0 trades taken — normal with 500 bars + many filters{RESET}")
    print(f"  {YELLOW}  Download more history to see more signals{RESET}")

# ══════════════════════════════════════════════════════════════
# 7. Individual trade validation
# ══════════════════════════════════════════════════════════════
if result.trades:
    section("7. Individual Trade Validation")

    for i, t in enumerate(result.trades):
        # SL direction check
        if t.direction == "BUY":
            sl_ok = t.stop_loss < t.entry_price
            tp_ok = t.take_profit > t.entry_price
        else:
            sl_ok = t.stop_loss > t.entry_price
            tp_ok = t.take_profit < t.entry_price

        check(f"  Trade {i+1}: {t.direction} SL correct side", sl_ok,
              f"entry={t.entry_price:.5f} sl={t.stop_loss:.5f}")
        check(f"  Trade {i+1}: {t.direction} TP correct side", tp_ok,
              f"entry={t.entry_price:.5f} tp={t.take_profit:.5f}")
        check(f"  Trade {i+1}: net_pnl = gross_pnl - commission",
              abs(t.net_pnl - (t.gross_pnl - t.commission)) < 0.01,
              f"net={t.net_pnl} gross={t.gross_pnl} comm={t.commission}")
        check(f"  Trade {i+1}: close_reason valid",
              t.close_reason in ("tp", "sl", "end_of_data"))
        check(f"  Trade {i+1}: exit_bar >= entry_bar",
              t.exit_bar >= t.entry_bar)

    # Profit factor verification
    if m.gross_loss > 0:
        expected_pf = round(m.gross_profit / m.gross_loss, 3)
        check("Profit factor = gross_profit / gross_loss",
              abs(m.profit_factor - expected_pf) < 0.01,
              f"expected={expected_pf} got={m.profit_factor}")

    # Win rate verification
    expected_wr = round(m.winning_trades / m.total_trades * 100, 2)
    check("Win rate = winners / total * 100",
          abs(m.win_rate - expected_wr) < 0.01,
          f"expected={expected_wr} got={m.win_rate}")

# ══════════════════════════════════════════════════════════════
# 8. Summary and report
# ══════════════════════════════════════════════════════════════
section("8. Summary and Report Generation")

summary = result.summary()
check("result.summary() returns string", isinstance(summary, str))
check("summary contains 'Net Profit'", "Net Profit" in summary)
check("summary contains 'Win Rate'", "Win Rate" in summary)
check("summary contains 'Sharpe'", "Sharpe" in summary)

print(f"\n{summary}\n")

report_path = result.save_report(tag="test")
check("save_report() returns Path", hasattr(report_path, "exists"))
check("Report file created", report_path.exists(),
      str(report_path))
if report_path.exists():
    print(f"  {CYAN}Report saved: {report_path.name}{RESET}")

# ══════════════════════════════════════════════════════════════
# 9. Empty result (missing symbol)
# ══════════════════════════════════════════════════════════════
section("9. Empty Result for Missing Symbol")

empty = backtest_engine.run("FAKESYMBOL", "H1")
check("Missing symbol returns BacktestResult", isinstance(empty, BacktestResult))
check("Empty result has 0 trades", len(empty.trades) == 0)
check("Empty result has 0 net_profit", empty.metrics.net_profit == 0.0)

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 11 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Backtesting Engine is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 11 works' to begin Step 12.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 11 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
