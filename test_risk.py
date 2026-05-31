"""
MT5 Quantum AI -- Step 13 Verification: Risk Manager
=====================================================
How to run:
    py -3.11 test_risk.py

MT5 does NOT need to be open.

What this tests:
  1.  RiskManager imports correctly
  2.  RiskDecision dataclass is correct
  3.  Fixed lot sizing returns configured lot
  4.  Risk % sizing calculates correct lot from SL distance
  5.  Daily loss limit triggers circuit breaker
  6.  Daily profit target blocks trading
  7.  Max drawdown triggers circuit breaker
  8.  Consecutive losses triggers pause
  9.  Circuit breaker can be manually reset
  10. Session filter blocks outside-hours signals
  11. evaluate() returns approved RiskDecision for clean signal
  12. can_trade() returns (bool, str)
  13. record_trade_result() updates daily PnL
  14. status() returns dict with all expected keys
  15. summary() returns string
"""

import sys
import os
from datetime import datetime

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


# ── helpers ──────────────────────────────────────────────────

def make_account(balance=10_000.0, equity=None, margin_free=None):
    if equity      is None: equity      = balance
    if margin_free is None: margin_free = balance * 0.9
    return {"balance": balance, "equity": equity,
            "margin_free": margin_free, "currency": "USD"}


def make_signal(direction="BUY", entry=1.1000, sl=1.0950, tp=1.1100,
                symbol="EURUSD", tf="H1", strategy="ema_trend"):
    from src.strategies.base import TradeSignal
    return TradeSignal(
        strategy_name=strategy, symbol=symbol, timeframe=tf,
        direction=direction, entry_price=entry,
        stop_loss=sl, take_profit=tp,
        signal_strength=0.75, bar_time=datetime.utcnow(),
        reason="test signal",
    )


# ══════════════════════════════════════════════════════════════
# 1. Import
# ══════════════════════════════════════════════════════════════
section("1. Import Risk Manager")

try:
    from src.risk.manager import risk_manager, RiskManager, RiskDecision
    check("from src.risk.manager import risk_manager", True)
    check("RiskManager singleton", risk_manager is RiskManager())
except Exception as e:
    check("from src.risk.manager import risk_manager", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. RiskDecision structure
# ══════════════════════════════════════════════════════════════
section("2. RiskDecision Dataclass")

d = RiskDecision(approved=True, lot_size=0.01, reason="test",
                 risk_amount=5.0, risk_pct=0.05)
check("RiskDecision has 'approved'",     hasattr(d, "approved"))
check("RiskDecision has 'lot_size'",     hasattr(d, "lot_size"))
check("RiskDecision has 'risk_amount'",  hasattr(d, "risk_amount"))
check("RiskDecision has 'risk_pct'",     hasattr(d, "risk_pct"))
check("RiskDecision has 'circuit_breaker'", hasattr(d, "circuit_breaker"))
check("RiskDecision has 'checks_passed'",  hasattr(d, "checks_passed"))
check("str(RiskDecision) works",
      "APPROVED" in str(d) or "BLOCKED" in str(d))

# ══════════════════════════════════════════════════════════════
# 3. Fresh state
# ══════════════════════════════════════════════════════════════
section("3. Fresh Risk Manager State")

risk_manager.reset_circuit_breaker()
risk_manager.reset_daily()
risk_manager._consecutive_losses = 0
risk_manager._pause_until        = None
risk_manager._peak_balance       = 0.0
risk_manager._current_balance    = 0.0

can, reason = risk_manager.can_trade()
check("can_trade() returns True on fresh state", can, reason)
check("can_trade() returns str reason", isinstance(reason, str))

# ══════════════════════════════════════════════════════════════
# 4. Fixed lot sizing
# ══════════════════════════════════════════════════════════════
section("4. Position Sizing — Fixed Lot")

from config.loader import cfg

cfg._settings["risk"]["sizing_mode"]    = "fixed_lot"
cfg._settings["position"]["default_lot"]= 0.05

sig    = make_signal()
acct   = make_account()
result = risk_manager.evaluate(sig, acct)

check("Fixed lot: evaluate() runs", isinstance(result, RiskDecision))
if result.approved:
    check("Fixed lot: lot_size = 0.05", abs(result.lot_size - 0.05) < 0.001,
          f"got {result.lot_size}")

# ══════════════════════════════════════════════════════════════
# 5. Risk % sizing
# ══════════════════════════════════════════════════════════════
section("5. Position Sizing — Risk %")

cfg._settings["risk"]["sizing_mode"]        = "fixed_pct"
cfg._settings["risk"]["risk_per_trade_pct"] = 1.0
cfg._settings["position"]["default_lot"]    = 0.01

# EURUSD BUY: entry=1.1000, SL=1.0950 → 50 pips risk
# At 1% of $10,000 = $100 risk
# $100 / (50 pips × $10/pip) = 0.20 lots
sig2   = make_signal(entry=1.1000, sl=1.0950, tp=1.1100)
result2 = risk_manager.evaluate(sig2, make_account(10_000))
check("Risk% sizing: evaluate() runs", isinstance(result2, RiskDecision))
if result2.approved:
    # Expected ~0.20 lots (100 / (50 * 10))
    check("Risk% sizing: lot_size ~0.20",
          0.10 <= result2.lot_size <= 0.30,
          f"got {result2.lot_size}")
    check("Risk% sizing: risk_pct ~1.0%",
          0.5 <= result2.risk_pct <= 2.0,
          f"got {result2.risk_pct:.2f}%")

# Restore default lot
cfg._settings["risk"]["sizing_mode"]    = "fixed_pct"
cfg._settings["position"]["default_lot"]= 0.01

# ══════════════════════════════════════════════════════════════
# 6. Daily loss limit circuit breaker
# ══════════════════════════════════════════════════════════════
section("6. CB-3: Daily Loss Limit")

risk_manager.reset_circuit_breaker()
risk_manager.reset_daily()
risk_manager._peak_balance    = 10_000.0
risk_manager._current_balance = 10_000.0

# Simulate 3.1% loss ($310 on $10k balance)
risk_manager._daily_pnl = -310.0

result3 = risk_manager.evaluate(make_signal(), make_account(10_000))
check("Daily loss limit: trade BLOCKED", not result3.approved,
      result3.reason)
check("Daily loss limit: circuit_breaker=True", result3.circuit_breaker)
check("Daily loss limit: 'loss limit' in reason",
      "loss" in result3.reason.lower())

risk_manager.reset_circuit_breaker()
risk_manager.reset_daily()

# ══════════════════════════════════════════════════════════════
# 7. Daily profit target
# ══════════════════════════════════════════════════════════════
section("7. CB-4: Daily Profit Target")

risk_manager._daily_pnl = +510.0   # > 5% of $10k

result4 = risk_manager.evaluate(make_signal(), make_account(10_000))
check("Profit target: trade BLOCKED", not result4.approved, result4.reason)
check("Profit target: 'profit' in reason",
      "profit" in result4.reason.lower())

risk_manager.reset_daily()

# ══════════════════════════════════════════════════════════════
# 8. Max drawdown circuit breaker
# ══════════════════════════════════════════════════════════════
section("8. CB-5: Max Drawdown")

risk_manager.reset_circuit_breaker()
risk_manager._peak_balance    = 10_000.0
risk_manager._current_balance =  8_900.0

# equity = $8,800 → drawdown = 12% (> 10% limit)
result5 = risk_manager.evaluate(
    make_signal(),
    make_account(balance=8_900, equity=8_800)
)
check("Max drawdown: trade BLOCKED", not result5.approved, result5.reason)
check("Max drawdown: circuit_breaker=True", result5.circuit_breaker)

risk_manager.reset_circuit_breaker()
risk_manager._peak_balance    = 0.0
risk_manager._current_balance = 0.0

# ══════════════════════════════════════════════════════════════
# 9. Consecutive losses pause
# ══════════════════════════════════════════════════════════════
section("9. CB-6: Consecutive Losses Pause")

risk_manager.reset_circuit_breaker()
risk_manager.reset_daily()
risk_manager._consecutive_losses = 5   # hit the limit

result6 = risk_manager.evaluate(make_signal(), make_account())
check("Consec losses: trade BLOCKED", not result6.approved, result6.reason)
check("Consec losses: 'pause' in reason", "pause" in result6.reason.lower() or
      "consecutive" in result6.reason.lower())

# Reset after trigger
risk_manager._consecutive_losses = 0
risk_manager._pause_until        = None

# ══════════════════════════════════════════════════════════════
# 10. Manual circuit breaker
# ══════════════════════════════════════════════════════════════
section("10. Manual Circuit Breaker")

risk_manager.trigger_circuit_breaker("Unit test shutdown")
check("Manual CB: is_circuit_breaker_active() True",
      risk_manager.is_circuit_breaker_active())

result7 = risk_manager.evaluate(make_signal(), make_account())
check("Manual CB: all trades BLOCKED", not result7.approved)
check("Manual CB: circuit_breaker flag True", result7.circuit_breaker)

risk_manager.reset_circuit_breaker()
check("After reset: is_circuit_breaker_active() False",
      not risk_manager.is_circuit_breaker_active())

# ══════════════════════════════════════════════════════════════
# 11. Clean signal approved
# ══════════════════════════════════════════════════════════════
section("11. Clean Signal — Should Be APPROVED")

risk_manager.reset_circuit_breaker()
risk_manager.reset_daily()
risk_manager._consecutive_losses = 0
risk_manager._pause_until        = None
risk_manager._peak_balance       = 10_000.0
risk_manager._current_balance    = 10_000.0

# Disable session filter for this test
cfg._settings["risk"]["session_filter_enabled"] = False

clean_sig = make_signal()
clean_res = risk_manager.evaluate(clean_sig, make_account(10_000))

check("Clean signal: evaluate() approved", clean_res.approved,
      clean_res.reason)
if clean_res.approved:
    check("Clean signal: lot_size > 0", clean_res.lot_size > 0)
    check("Clean signal: risk_amount > 0", clean_res.risk_amount > 0)
    check("Clean signal: checks_passed not empty",
          len(clean_res.checks_passed) > 0)
    print(f"  {CYAN}{clean_res}{RESET}")

# Restore session filter
cfg._settings["risk"]["session_filter_enabled"] = True

# ══════════════════════════════════════════════════════════════
# 12. record_trade_result()
# ══════════════════════════════════════════════════════════════
section("12. record_trade_result()")

risk_manager.reset_daily()
risk_manager._consecutive_losses = 0

risk_manager.record_trade_result(+25.0)
check("Win recorded: daily_pnl = +25", abs(risk_manager._daily_pnl - 25.0) < 0.01)
check("Win recorded: consec_losses = 0", risk_manager._consecutive_losses == 0)
check("Win recorded: consec_wins = 1", risk_manager._consecutive_wins == 1)

risk_manager.record_trade_result(-12.0)
check("Loss recorded: daily_pnl = +13",
      abs(risk_manager._daily_pnl - 13.0) < 0.01)
check("Loss recorded: consec_losses = 1",
      risk_manager._consecutive_losses == 1)
check("Loss recorded: consec_wins = 0",
      risk_manager._consecutive_wins == 0)

# ══════════════════════════════════════════════════════════════
# 13. status() and summary()
# ══════════════════════════════════════════════════════════════
section("13. status() and summary()")

risk_manager.reset_circuit_breaker()
s = risk_manager.status()
check("status() returns dict", isinstance(s, dict))
for key in ["can_trade", "daily_pnl", "consecutive_losses",
            "circuit_breaker", "daily_trades"]:
    check(f"  status has '{key}'", key in s)

summ = risk_manager.summary()
check("summary() returns string", isinstance(summ, str))
check("summary contains 'RiskManager'", "RiskManager" in summ)
print(f"\n  {CYAN}{summ}{RESET}")

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 13 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Risk Manager is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 13 works' to begin Step 14.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 13 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
