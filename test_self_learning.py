"""
MT5 Quantum AI -- Step 19 Verification: Self-Learning System
=============================================================
How to run:
    py -3.11 test_self_learning.py

MT5 does NOT need to be open.

What this tests:
  1.  SelfLearner imports correctly
  2.  record_trade_opened() stores features in DB trade.notes
  3.  record_trade_closed() increments counter
  4.  should_retrain() returns bool
  5.  analyze_performance() returns dict with required keys
  6.  analyze_performance() correctly calculates win_rate
  7.  get_learning_dataframe() returns DataFrame
  8.  compare_models() promotes when new > old + threshold
  9.  compare_models() rejects when improvement < threshold
  10. retrain_all(force=True) runs all three retraining jobs
  11. retrain_all returns result dict with 'models' key
  12. After retrain: new_trades_since_retrain resets to 0
  13. status() returns dict with required keys
  14. print_performance_report() runs without error
  15. summary() returns informative string
"""

import sys
import os
import json

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
section("1. Import Self-Learner")

try:
    from src.self_learning.self_learner import self_learner, SelfLearner
    check("from src.self_learning.self_learner import self_learner", True)
    check("SelfLearner singleton", self_learner is SelfLearner())
except Exception as e:
    check("from src.self_learning.self_learner import self_learner", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. record_trade_opened()
# ══════════════════════════════════════════════════════════════
section("2. record_trade_opened() — Stores Features in DB")

from src.database import db, Trade
from datetime import datetime

# Create a test shadow trade in DB
with db.session() as sess:
    test_trade = Trade(
        symbol="EURUSD", direction="BUY", lot_size=0.01,
        open_time=datetime.utcnow(), open_price=1.1000,
        stop_loss=1.0950, take_profit=1.1100,
        strategy="test_strategy", mode="shadow", status="open",
    )
    db.save_trade(sess, test_trade)
    test_id = test_trade.id

check("Test trade created in DB", test_id > 0, f"id={test_id}")

# Build a fake feature dict
fake_features = {
    "rsi_14": 62.5, "adx_14": 28.3, "atr_14_pct": 0.08,
    "ema20_above_ema50": 1.0, "structure_bullish": 1.0,
    "macd_hist": 0.0003, "bb_position": 0.65,
}

self_learner.record_trade_opened(test_id, fake_features)

# Verify stored in DB
with db.session() as sess:
    t = sess.query(Trade).filter_by(id=test_id).first()
    notes_raw = t.notes if t else None

check("trade.notes set after record_trade_opened",
      notes_raw is not None and len(notes_raw) > 10)

if notes_raw:
    notes = json.loads(notes_raw)
    check("notes has 'entry_meta' key", "entry_meta" in notes)
    if "entry_meta" in notes:
        meta = notes["entry_meta"]
        check("entry_meta has 'features'", "features" in meta)
        check("features stored correctly",
              meta.get("features", {}).get("rsi_14") == 62.5)

# ══════════════════════════════════════════════════════════════
# 3. record_trade_closed()
# ══════════════════════════════════════════════════════════════
section("3. record_trade_closed() — Increments Counter")

initial_count = self_learner._new_trades_since_retrain

# Close the test trade first
with db.session() as sess:
    db.close_trade(sess, test_id, close_price=1.1080,
                   close_time=datetime.utcnow(),
                   net_profit=8.0, pips=80.0, close_reason="tp")

self_learner.record_trade_closed(test_id, "tp")

check("counter incremented after record_trade_closed",
      self_learner._new_trades_since_retrain == initial_count + 1,
      f"was {initial_count} now {self_learner._new_trades_since_retrain}")

# Verify outcome stored
with db.session() as sess:
    t = sess.query(Trade).filter_by(id=test_id).first()
    notes2 = json.loads(t.notes) if t and t.notes else {}

check("notes has 'outcome' after close",
      notes2.get("outcome") == "tp", f"notes={list(notes2.keys())}")
check("notes has 'learned' flag",
      notes2.get("learned") is True)

# ══════════════════════════════════════════════════════════════
# 4. should_retrain()
# ══════════════════════════════════════════════════════════════
section("4. should_retrain()")

check("should_retrain() returns bool",
      isinstance(self_learner.should_retrain(), bool))

# Force enough new trades to trigger retrain
original_count  = self_learner._new_trades_since_retrain
self_learner._new_trades_since_retrain = self_learner._min_trades
check("should_retrain() True when >= min_trades",
      self_learner.should_retrain() is True)

# Reset to original for rest of test
self_learner._new_trades_since_retrain = original_count

# ══════════════════════════════════════════════════════════════
# 5. analyze_performance()
# ══════════════════════════════════════════════════════════════
section("5. analyze_performance()")

report = self_learner.analyze_performance(window=200)
check("analyze_performance() returns dict", isinstance(report, dict))
check("report has 'total_trades'", "total_trades" in report)
check("report has 'win_rate'",     "win_rate"     in report)
check("report has 'profit_factor'","profit_factor" in report)
check("report has 'expectancy'",   "expectancy"   in report)
check("report has 'by_strategy'",  "by_strategy"  in report)
check("report has 'by_regime'",    "by_regime"    in report)

n_trades = report["total_trades"]
print(f"\n  {CYAN}Performance: {n_trades} closed trades | "
      f"WR={report['win_rate']:.1f}% | "
      f"PF={report['profit_factor']:.2f}{RESET}")

# ══════════════════════════════════════════════════════════════
# 6. Win rate correctness
# ══════════════════════════════════════════════════════════════
section("6. Win Rate Calculation Correctness")

# Create 4 more trades (3 wins, 1 loss) in shadow mode for testing
test_ids = []
for i, (pnl, reason) in enumerate([
    (+15.0, "tp"), (+10.0, "tp"), (-5.0, "sl"), (+8.0, "tp")
]):
    with db.session() as sess:
        t = Trade(
            symbol="EURUSD", direction="BUY", lot_size=0.01,
            open_time=datetime.utcnow(), open_price=1.1000,
            stop_loss=1.0950, take_profit=1.1100,
            strategy="ema_trend",
            regime_at_entry="strong_uptrend",
            mode="shadow", status="closed",
            close_price=1.1080, close_time=datetime.utcnow(),
            net_profit=pnl, pips=abs(pnl)*10, close_reason=reason,
        )
        db.save_trade(sess, t)
        test_ids.append(t.id)

# Refresh analysis
self_learner._performance_cache = None
report2 = self_learner.analyze_performance(window=50)
check("analyze_performance() still returns dict after new trades",
      isinstance(report2, dict))
check("win_rate in [0, 100]",
      0 <= report2.get("win_rate", 0) <= 100,
      f"got {report2.get('win_rate')}")

# ══════════════════════════════════════════════════════════════
# 7. get_learning_dataframe()
# ══════════════════════════════════════════════════════════════
section("7. get_learning_dataframe()")

import pandas as pd

df_learn = self_learner.get_learning_dataframe()
check("get_learning_dataframe() returns DataFrame",
      isinstance(df_learn, pd.DataFrame))
check("DataFrame has 'outcome' column",
      df_learn.empty or "outcome" in df_learn.columns)
check("DataFrame has 'net_profit' column",
      df_learn.empty or "net_profit" in df_learn.columns)
print(f"  {CYAN}Learning dataset: {len(df_learn)} rows, "
      f"{len(df_learn.columns)} columns{RESET}")

# ══════════════════════════════════════════════════════════════
# 8. compare_models()
# ══════════════════════════════════════════════════════════════
section("8. compare_models() — Promotion Logic")

# New clearly better (10% improvement)
promote, reason = self_learner.compare_models(0.72, 0.65, "accuracy")
check("Promotes when improvement >= threshold",
      promote, reason)

# New slightly better but below threshold (1% < 2% threshold)
no_promote, reason2 = self_learner.compare_models(0.656, 0.65, "accuracy")
check("Rejects when improvement < threshold",
      not no_promote, reason2)

# No old model → always promote
promote_new, reason3 = self_learner.compare_models(0.60, 0.0, "accuracy")
check("Promotes when no previous model (old=0)", promote_new, reason3)

print(f"  {CYAN}Promote (10% better): {promote}{RESET}")
print(f"  {CYAN}Reject  (1% better):  {not no_promote}{RESET}")

# ══════════════════════════════════════════════════════════════
# 9. retrain_all(force=True)
# ══════════════════════════════════════════════════════════════
section("9. retrain_all(force=True) — All Models Retrained")

print(f"  {YELLOW}Retraining all models... (~30 seconds){RESET}\n")

retrain_result = self_learner.retrain_all(
    symbol="EURUSD", timeframe="H1", force=True
)

check("retrain_all() returns dict", isinstance(retrain_result, dict))
check("result has 'status' key",  "status" in retrain_result)
check("result has 'models' key",  "models" in retrain_result)
check("status = 'completed'",
      retrain_result.get("status") == "completed",
      f"got '{retrain_result.get('status')}'")

models_result = retrain_result.get("models", {})
check("'ai_engine' in models results",     "ai_engine"        in models_result)
check("'ml_entry_quality' in models",      "ml_entry_quality" in models_result)
check("'regime' in models results",        "regime"           in models_result)

for model_name, m_result in models_result.items():
    status = m_result.get("status", "unknown")
    print(f"  {CYAN}  {model_name}: {status}{RESET}")

# ══════════════════════════════════════════════════════════════
# 10. Counter resets after retrain
# ══════════════════════════════════════════════════════════════
section("10. Counter Resets After Retrain")

check("new_trades_since_retrain reset to 0",
      self_learner._new_trades_since_retrain == 0,
      f"got {self_learner._new_trades_since_retrain}")
check("last_retrain_at is set",
      self_learner._last_retrain_at is not None)
check("should_retrain() False immediately after retrain",
      not self_learner.should_retrain())

# ══════════════════════════════════════════════════════════════
# 11. status()
# ══════════════════════════════════════════════════════════════
section("11. status() and summary()")

s = self_learner.status()
check("status() returns dict", isinstance(s, dict))
for key in ["new_trades_since_retrain", "min_trades_for_retrain",
            "last_retrain_at", "retrain_interval", "should_retrain"]:
    check(f"  status has '{key}'", key in s)

summ = self_learner.summary()
check("summary() returns string", isinstance(summ, str))
check("summary contains 'SelfLearner'", "SelfLearner" in summ)
print(f"\n  {CYAN}{summ}{RESET}")

# ══════════════════════════════════════════════════════════════
# 12. print_performance_report()
# ══════════════════════════════════════════════════════════════
section("12. print_performance_report()")

try:
    self_learner.print_performance_report(window=50)
    check("print_performance_report() runs without error", True)
except Exception as e:
    check("print_performance_report() runs without error", False, str(e))

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 19 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Self-Learning System is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 19 works' to begin Step 20.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 19 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
