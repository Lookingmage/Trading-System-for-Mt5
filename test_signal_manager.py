"""
MT5 Quantum AI — Signal Manager Test
======================================
Verifies that ONE SIGNAL = ONE TRADE is enforced correctly.

How to run:
    py -3.11 test_signal_manager.py

MT5 does NOT need to be open.
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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
    if ok: passed += 1
    else:  failed += 1
    return ok


def section(title: str):
    print(f"\n{CYAN}{BOLD}{'=' * 55}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{'=' * 55}{RESET}")


def make_signal(
    symbol="EURUSD", tf="H1", strategy="fair_value_gap",
    direction="BUY", bar_time=None
):
    """Create a minimal mock signal for testing."""
    from src.strategies.base import TradeSignal
    if bar_time is None:
        bar_time = datetime(2026, 5, 30, 10, 0, 0)  # fixed candle time
    return TradeSignal(
        strategy_name=strategy, symbol=symbol, timeframe=tf,
        direction=direction, entry_price=1.1000,
        stop_loss=1.0950 if direction=="BUY" else 1.1050,
        take_profit=1.1100 if direction=="BUY" else 1.0900,
        signal_strength=0.75, bar_time=bar_time,
        reason="test signal",
    )


# ══════════════════════════════════════════════════════════════
# 1. Import
# ══════════════════════════════════════════════════════════════
section("1. Import SignalManager")

# Use a temporary history file so tests don't pollute production
TEMP_HISTORY = Path(ROOT) / "database" / "_test_signal_history.json"

import src.signal_manager as sm_module
sm_module.HISTORY_FILE = TEMP_HISTORY   # redirect to temp file

# Clean up any previous test file
if TEMP_HISTORY.exists():
    TEMP_HISTORY.unlink()

try:
    from src.signal_manager import SignalManager, signal_manager
    check("from src.signal_manager import signal_manager", True)
    # Force re-init with temp file
    signal_manager._executed = {}
    signal_manager._load()
except Exception as e:
    check("from src.signal_manager import signal_manager", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. signal_id property on TradeSignal
# ══════════════════════════════════════════════════════════════
section("2. TradeSignal.signal_id Property")

bar_time = datetime(2026, 5, 30, 10, 0, 0)
sig = make_signal(bar_time=bar_time)
sid = sig.signal_id

check("signal_id is a string", isinstance(sid, str))
check("signal_id contains symbol",    "EURUSD"        in sid)
check("signal_id contains strategy",  "fair_value_gap" in sid)
check("signal_id contains direction", "BUY"           in sid)
check("signal_id contains bar_time",  "10:00:00"      in sid)
print(f"  {CYAN}signal_id = {sid}{RESET}")

# Same bar → same signal_id (deduplication key)
sig2 = make_signal(bar_time=bar_time)
check("Same bar → identical signal_id", sig.signal_id == sig2.signal_id)

# Different bar time → different signal_id
sig3 = make_signal(bar_time=datetime(2026, 5, 30, 11, 0, 0))
check("Different bar → different signal_id", sig.signal_id != sig3.signal_id)

# Different direction → different signal_id
sig4 = make_signal(direction="SELL", bar_time=bar_time)
check("Different direction → different signal_id", sig.signal_id != sig4.signal_id)

# ══════════════════════════════════════════════════════════════
# 3. First signal allowed through
# ══════════════════════════════════════════════════════════════
section("3. First Signal — Should Be ALLOWED")

signal_manager._executed = {}  # clear state

is_dup, reason = signal_manager.is_duplicate(sig)
check("First occurrence: is_duplicate = False", not is_dup,
      f"reason='{reason}'")

# ══════════════════════════════════════════════════════════════
# 4. Record execution → same signal now blocked
# ══════════════════════════════════════════════════════════════
section("4. After Execution — Same Signal BLOCKED")

signal_manager.record_execution(sig, trade_id=99)

# Simulate: same signal fires again (next loop cycle, same bar)
is_dup2, reason2 = signal_manager.is_duplicate(sig)
check("Second occurrence (same bar): is_duplicate = True", is_dup2,
      reason2)
check("Reason mentions trade #99", "99" in reason2)

# ══════════════════════════════════════════════════════════════
# 5. Different bar time → new signal allowed
# ══════════════════════════════════════════════════════════════
section("5. New Candle → New Signal ALLOWED")

sig_new_bar = make_signal(bar_time=datetime(2026, 5, 30, 11, 0, 0))
is_dup_new, reason_new = signal_manager.is_duplicate(sig_new_bar)
check("New bar (11:00) not duplicate", not is_dup_new,
      f"reason='{reason_new}'")
print(f"  {CYAN}Old signal_id: {sig.signal_id}{RESET}")
print(f"  {CYAN}New signal_id: {sig_new_bar.signal_id}{RESET}")

# ══════════════════════════════════════════════════════════════
# 6. Different symbol → not duplicate
# ══════════════════════════════════════════════════════════════
section("6. Different Symbol → Not Duplicate")

sig_gbp = make_signal(symbol="GBPUSD", bar_time=bar_time)
is_dup_gbp, _ = signal_manager.is_duplicate(sig_gbp)
check("GBPUSD BUY (same bar time) not duplicate of EURUSD", not is_dup_gbp)

# ══════════════════════════════════════════════════════════════
# 7. Opposite direction release
# ══════════════════════════════════════════════════════════════
section("7. Opposite Direction — Releases Previous Lock")

# Add a SELL lock for EURUSD
sig_sell = make_signal(direction="SELL", bar_time=bar_time)
signal_manager.record_execution(sig_sell, trade_id=100)

sell_locked = sig_sell.signal_id in signal_manager._executed
check("SELL signal locked", sell_locked)

# New BUY appears → should release the SELL lock
n_released = signal_manager.release_opposite("EURUSD", "BUY")
check("release_opposite() removes SELL lock when BUY appears",
      n_released >= 1, f"released {n_released}")

sell_still_locked = sig_sell.signal_id in signal_manager._executed
check("SELL lock removed after BUY signal", not sell_still_locked)

# ══════════════════════════════════════════════════════════════
# 8. Persistence — survives restart
# ══════════════════════════════════════════════════════════════
section("8. Persistence — Survives System Restart")

signal_manager._executed = {}
signal_manager.record_execution(sig, trade_id=42)

# Simulate restart: create fresh instance that loads from disk
fresh = SignalManager.__new__(SignalManager)
fresh._ready = False
fresh._executed = {}
fresh._log = signal_manager._log
fresh._load()

is_dup_restart, reason_restart = fresh.is_duplicate(sig)
check("After restart: previously executed signal still blocked",
      is_dup_restart, reason_restart)
check("Loaded from disk correctly",
      TEMP_HISTORY.exists())

# ══════════════════════════════════════════════════════════════
# 9. Expiry of old signals
# ══════════════════════════════════════════════════════════════
section("9. Signal Expiry (Stale Signals Auto-Removed)")

# Add a signal with an already-past expiry time
old_sid = "EURUSD_H1_old_strategy_BUY_2026-01-01T00:00:00"
signal_manager._executed[old_sid] = {
    "trade_id":    1,
    "executed_at": "2026-01-01T00:00:00",
    "expires_at":  "2026-01-01T08:00:00",   # already expired
    "symbol":      "EURUSD",
    "direction":   "BUY",
}

count_before = len(signal_manager._executed)
expired = signal_manager._expire_old(max_age_hours=8)
count_after = len(signal_manager._executed)

check("Expired signal removed", old_sid not in signal_manager._executed)
check("expire_old() removed 1 entry", expired >= 1,
      f"removed={expired}")
check("Other signals preserved", count_after == count_before - 1,
      f"before={count_before} after={count_after}")

# ══════════════════════════════════════════════════════════════
# 10. status() and summary()
# ══════════════════════════════════════════════════════════════
section("10. Status and Summary")

s = signal_manager.status()
check("status() returns dict", isinstance(s, dict))
check("status has 'active_locks'", "active_locks" in s)
check("status has 'locked_signals'", "locked_signals" in s)

summ = signal_manager.summary()
check("summary() returns string", isinstance(summ, str))
check("summary contains 'locked'", "locked" in summ.lower())
print(f"  {CYAN}{summ}{RESET}")

# ══════════════════════════════════════════════════════════════
# 11. Integration: simulate 5 loop cycles, same signal
# ══════════════════════════════════════════════════════════════
section("11. Integration — 5 Loops Same Signal = 1 Trade")

signal_manager._executed = {}

base_time = datetime(2026, 5, 30, 10, 0, 0)
trades_opened = 0
skips         = 0

for loop in range(1, 6):
    # Same bar_time every cycle — simulates 5 loop runs on same candle
    loop_sig = make_signal(bar_time=base_time)

    is_dup, reason = signal_manager.is_duplicate(loop_sig)
    if is_dup:
        skips += 1
        print(f"  Loop {loop}: {YELLOW}SKIPPED{RESET} — {reason[:50]}")
    else:
        trades_opened += 1
        signal_manager.record_execution(loop_sig, trade_id=200 + loop)
        print(f"  Loop {loop}: {GREEN}TRADE OPENED{RESET} (trade #{200+loop})")

check(f"5 loops → exactly 1 trade opened ({trades_opened})",
      trades_opened == 1, f"opened={trades_opened}")
check(f"5 loops → 4 loops skipped ({skips})",
      skips == 4, f"skips={skips}")

# ══════════════════════════════════════════════════════════════
# 12. Cleanup
# ══════════════════════════════════════════════════════════════
if TEMP_HISTORY.exists():
    TEMP_HISTORY.unlink()
    print(f"\n  {CYAN}Test signal history file cleaned up{RESET}")

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Signal Manager VERIFIED -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  ONE SIGNAL = ONE TRADE is now guaranteed.{RESET}")
    print(f"{GREEN}  Duplicate trade bug is fixed.{RESET}")
else:
    print(f"{RED}{BOLD}  INCOMPLETE -- {passed}/{total} passed, {failed} failed{RESET}")
print(f"{'=' * 55}\n")
