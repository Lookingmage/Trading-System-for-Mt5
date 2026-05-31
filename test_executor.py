"""
MT5 Quantum AI -- Step 14 Verification: Trade Executor
=======================================================
How to run:
    py -3.11 test_executor.py

MT5 does NOT need to be open for shadow mode tests.
Shadow mode: creates trades in DB, no real orders sent.

What this tests:
  1.  TradeExecutor imports correctly
  2.  Executor is in demo/shadow/live mode
  3.  Live mode blocked when live_trading.enabled=false
  4.  Shadow mode: place_order() creates DB record
  5.  Shadow mode: trade has correct direction/SL/TP
  6.  Shadow mode: close_position() updates DB status
  7.  Shadow mode: modify_sltp() updates SL/TP in DB
  8.  Shadow mode: move_to_breakeven() moves SL to entry
  9.  Shadow mode: apply_trailing_stop() trails SL
  10. Shadow mode: partial_close() reduces lot size
  11. manage_positions() runs without error
  12. get_open_trades() returns list
  13. summary() returns string
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


def make_signal(direction="BUY", entry=1.1000, sl=1.0950, tp=1.1100,
                symbol="EURUSD", tf="H1", strategy="ema_trend"):
    from src.strategies.base import TradeSignal
    return TradeSignal(
        strategy_name=strategy, symbol=symbol, timeframe=tf,
        direction=direction, entry_price=entry,
        stop_loss=sl, take_profit=tp,
        signal_strength=0.75, bar_time=datetime.utcnow(),
        reason="executor test",
    )


def make_decision(lot=0.01, sl=1.0950, tp=1.1100):
    from src.risk.manager import RiskDecision
    return RiskDecision(
        approved=True, lot_size=lot,
        adjusted_sl=sl, adjusted_tp=tp,
        risk_amount=5.0, risk_pct=0.05,
        reason="test approved",
    )


# ══════════════════════════════════════════════════════════════
# 1. Import
# ══════════════════════════════════════════════════════════════
section("1. Import Trade Executor")

try:
    from src.executor.executor import executor, TradeExecutor
    check("from src.executor.executor import executor", True)
    check("TradeExecutor singleton", executor is TradeExecutor())
except Exception as e:
    check("from src.executor.executor import executor", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Mode detection
# ══════════════════════════════════════════════════════════════
section("2. Mode and Structure")

check("executor.mode is string", isinstance(executor.mode, str))
check("mode is valid",
      executor.mode in ("demo", "shadow", "live", "backtest"),
      f"mode='{executor.mode}'")
print(f"  {CYAN}Current mode: {executor.mode}{RESET}")

# ── force shadow mode for testing ────────────────────────────
executor._mode = "shadow"
check("Forced to shadow mode", executor.mode == "shadow")

# ══════════════════════════════════════════════════════════════
# 3. Live mode safety gate
# ══════════════════════════════════════════════════════════════
section("3. Live Mode Safety Gate")

executor._mode = "live"
from config.loader import cfg
cfg._settings["live_trading"]["enabled"] = False

sig_live = make_signal()
dec_live = make_decision()
result_live = executor.place_order(sig_live, dec_live)

check("Live mode blocked when enabled=false",
      result_live is None,
      f"got trade_id={result_live}")

# Reset to shadow
executor._mode = "shadow"

# ══════════════════════════════════════════════════════════════
# 4. Shadow mode: place order
# ══════════════════════════════════════════════════════════════
section("4. Shadow Mode — Place Order")

sig  = make_signal(direction="BUY", entry=1.1000, sl=1.0950, tp=1.1100)
dec  = make_decision(lot=0.02, sl=1.0950, tp=1.1100)
tid  = executor.place_order(sig, dec)

check("place_order() returns int (trade_id)",
      isinstance(tid, int) and tid > 0,
      f"trade_id={tid}")

# Verify in DB
from src.database import db, Trade

with db.session() as sess:
    trade_db = sess.query(Trade).filter_by(id=tid).first()

check("Trade record created in DB", trade_db is not None)
if trade_db:
    check("DB mode = 'shadow'",     trade_db.mode      == "shadow")
    check("DB status = 'open'",     trade_db.status    == "open")
    check("DB direction = 'BUY'",   trade_db.direction == "BUY")
    check("DB symbol = 'EURUSD'",   trade_db.symbol    == "EURUSD")
    check("DB lot_size = 0.02",
          abs(trade_db.lot_size - 0.02) < 0.001)
    check("DB open_price = 1.1000",
          abs(trade_db.open_price - 1.1000) < 0.0001)
    check("DB stop_loss = 1.0950",
          abs(trade_db.stop_loss - 1.0950) < 0.0001)
    check("DB take_profit = 1.1100",
          abs(trade_db.take_profit - 1.1100) < 0.0001)
    check("DB strategy = 'ema_trend'",
          trade_db.strategy == "ema_trend")

# ══════════════════════════════════════════════════════════════
# 5. Shadow mode: modify SL/TP
# ══════════════════════════════════════════════════════════════
section("5. Shadow Mode — Modify SL/TP")

new_sl = 1.0960
new_tp = 1.1120
ok_mod = executor.modify_sltp(tid, new_sl=new_sl, new_tp=new_tp)
check("modify_sltp() returns True", ok_mod)

with db.session() as sess:
    t = sess.query(Trade).filter_by(id=tid).first()
check("DB stop_loss updated",
      t is not None and abs(t.stop_loss - new_sl) < 0.0001,
      f"got {t.stop_loss if t else 'N/A'}")
check("DB take_profit updated",
      t is not None and abs(t.take_profit - new_tp) < 0.0001)

# ══════════════════════════════════════════════════════════════
# 6. Shadow mode: break-even
# ══════════════════════════════════════════════════════════════
section("6. Shadow Mode — Move to Break-Even")

# entry=1.1000, SL=1.0960 → risk=40 pips
# BE triggers when profit >= 1×risk = 40 pips → price >= 1.1040
be_moved = executor.move_to_breakeven(tid, current_price=1.1050)
check("move_to_breakeven() returns True (profit > threshold)", be_moved)

with db.session() as sess:
    t = sess.query(Trade).filter_by(id=tid).first()
check("SL moved to entry price",
      t is not None and abs(t.stop_loss - 1.1000) < 0.0001,
      f"got sl={t.stop_loss if t else 'N/A'}")
check("'be_done' note recorded", "be_done" in (t.notes or ""))

# Calling BE again should not move SL (already done)
be_moved2 = executor.move_to_breakeven(tid, current_price=1.1060)
check("move_to_breakeven() False (already done)", not be_moved2)

# ══════════════════════════════════════════════════════════════
# 7. Shadow mode: trailing stop
# ══════════════════════════════════════════════════════════════
section("7. Shadow Mode — Trailing Stop")

# Current SL is at entry (1.1000), price moved to 1.1060
# ATR = 0.0020, trail_mult = 1.0 → trail_dist = 0.0020
# New SL = 1.1060 - 0.0020 = 1.1040 (above current SL of 1.1000 → move)
trailed = executor.apply_trailing_stop(tid, current_price=1.1060, atr=0.0020)
check("apply_trailing_stop() returns True", trailed)

with db.session() as sess:
    t = sess.query(Trade).filter_by(id=tid).first()
expected_sl = round(1.1060 - 0.0020, 4)
check(f"SL trailed to ~{expected_sl}",
      t is not None and abs(t.stop_loss - expected_sl) < 0.0002,
      f"got sl={t.stop_loss if t else 'N/A'}")

# Call again with same price — SL should not go lower
trailed2 = executor.apply_trailing_stop(tid, current_price=1.1060, atr=0.0020)
check("No SL move when price unchanged", not trailed2)

# ══════════════════════════════════════════════════════════════
# 8. Shadow mode: partial close
# ══════════════════════════════════════════════════════════════
section("8. Shadow Mode — Partial Close (50%)")

ok_partial = executor.partial_close(tid, close_pct=50.0)
check("partial_close() returns True", ok_partial)

with db.session() as sess:
    t = sess.query(Trade).filter_by(id=tid).first()
check("Lot size reduced to 50% (0.01)",
      t is not None and abs(t.lot_size - 0.01) < 0.001,
      f"got {t.lot_size if t else 'N/A'}")
check("Trade still open after partial close",
      t is not None and t.status == "open")

# ══════════════════════════════════════════════════════════════
# 9. Shadow mode: close position
# ══════════════════════════════════════════════════════════════
section("9. Shadow Mode — Close Position")

# Need a close price — inject it directly
with db.session() as sess:
    t = sess.query(Trade).filter_by(id=tid).first()
    if t:
        t.open_price = 1.1000  # reset for clean PnL calc

ok_close = executor._close_shadow(tid, reason="tp")
check("_close_shadow() returns True", ok_close)

with db.session() as sess:
    t = sess.query(Trade).filter_by(id=tid).first()
check("Trade status = 'closed' after close", t is not None and t.status == "closed")
check("close_reason = 'tp'", t is not None and t.close_reason == "tp")
check("close_time is set", t is not None and t.close_time is not None)

# ══════════════════════════════════════════════════════════════
# 10. Unapproved signal is blocked
# ══════════════════════════════════════════════════════════════
section("10. Unapproved RiskDecision Is Blocked")

from src.risk.manager import RiskDecision
bad_dec = RiskDecision(approved=False, lot_size=0.0, reason="test block")
blocked = executor.place_order(make_signal(), bad_dec)
check("Unapproved decision returns None", blocked is None)

# ══════════════════════════════════════════════════════════════
# 11. manage_positions() and get_open_trades()
# ══════════════════════════════════════════════════════════════
section("11. manage_positions() and get_open_trades()")

try:
    executor.manage_positions()
    check("manage_positions() runs without error", True)
except Exception as e:
    check("manage_positions() runs without error", False, str(e))

open_t = executor.get_open_trades()
check("get_open_trades() returns list", isinstance(open_t, list))

# ══════════════════════════════════════════════════════════════
# 12. summary()
# ══════════════════════════════════════════════════════════════
section("12. summary()")

summ = executor.summary()
check("summary() returns string", isinstance(summ, str))
check("summary contains 'TradeExecutor'", "TradeExecutor" in summ)
print(f"\n  {CYAN}{summ}{RESET}")

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 14 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Trade Executor is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 14 works' to begin Step 15.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 14 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
