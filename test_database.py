"""
MT5 Quantum AI -- Step 4 Verification: Database Layer
=======================================================
How to run:
    py -3.11 test_database.py

Uses a TEMPORARY test database (database/_test.db).
All test data is cleaned up at the end. Your main
database is not touched.
"""

import sys
import os
import shutil
from datetime import datetime, date

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


# ── redirect db to test path before importing DatabaseManager ─
TEST_DB = os.path.join(ROOT, "database", "_test.db")

# Always start clean — delete any leftover test DB from a previous run
for _f in [TEST_DB, TEST_DB + "-wal", TEST_DB + "-shm"]:
    if os.path.exists(_f):
        try:
            os.remove(_f)
        except OSError:
            pass

# Patch config to use test DB
import config.loader as _cl_mod
_real_db_path = _cl_mod.cfg.db_path

def _test_db_path():
    from pathlib import Path
    return Path(TEST_DB)

_cl_mod.cfg.db_path = _test_db_path

# ══════════════════════════════════════════════════════════════
# 1. Import
# ══════════════════════════════════════════════════════════════
section("1. Import Database Module")

try:
    from src.database import (
        db, DatabaseManager,
        Candle, Tick, Trade, Feature,
        RegimePrediction, ModelVersion,
        OptimizationResult, SystemEvent, DailyStats,
    )
    check("from src.database import db", True)
    check("DatabaseManager singleton", db is DatabaseManager())
except Exception as e:
    check("from src.database import db", False, str(e))
    print(f"\n{RED}Cannot continue -- fix the import error above.{RESET}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Tables created
# ══════════════════════════════════════════════════════════════
section("2. All 9 Tables Created")

tables = db.get_table_names()
check(f"9 tables exist (found {len(tables)})", len(tables) == 9,
      f"tables: {tables}")

expected_tables = [
    "candles", "ticks", "trades", "features",
    "regime_predictions", "model_versions",
    "optimization_results", "system_events", "daily_stats",
]
for t in expected_tables:
    check(f"Table '{t}' exists", t in tables)

# ══════════════════════════════════════════════════════════════
# 3. Candle CRUD
# ══════════════════════════════════════════════════════════════
section("3. Candle Table — Insert, Query, Upsert")

now = datetime(2024, 1, 15, 10, 0, 0)

with db.session() as sess:
    c = Candle(
        symbol="EURUSD", timeframe="H1", open_time=now,
        open=1.08500, high=1.08750, low=1.08300, close=1.08620,
        tick_volume=1250, spread=2,
    )
    db.upsert_candle(sess, c)

with db.session() as sess:
    rows = db.get_candles(sess, "EURUSD", "H1", limit=10)
    check("Candle inserted", len(rows) >= 1)
    if rows:
        r = rows[0]
        check("Candle close price correct", abs(r.close - 1.08620) < 0.00001)
        check("Candle to_dict() works", isinstance(r.to_dict(), dict))

# upsert (update existing)
with db.session() as sess:
    c2 = Candle(
        symbol="EURUSD", timeframe="H1", open_time=now,
        open=1.08500, high=1.08800, low=1.08300, close=1.08700,
        tick_volume=1400, spread=2,
    )
    db.upsert_candle(sess, c2)

with db.session() as sess:
    rows = db.get_candles(sess, "EURUSD", "H1", limit=10)
    check("Candle upsert updates close price",
          abs(rows[0].close - 1.08700) < 0.00001,
          f"got close={rows[0].close}")

with db.session() as sess:
    n = db.count_candles(sess, "EURUSD", "H1")
    check("count_candles() returns int", isinstance(n, int))

# ══════════════════════════════════════════════════════════════
# 4. Tick CRUD
# ══════════════════════════════════════════════════════════════
section("4. Tick Table — Insert, Query")

with db.session() as sess:
    t = Tick(
        symbol="EURUSD", time=now,
        bid=1.08615, ask=1.08617, volume=1.0,
    )
    db.upsert_tick(sess, t)

with db.session() as sess:
    ticks = db.get_ticks(sess, "EURUSD", limit=10)
    check("Tick inserted", len(ticks) >= 1)
    if ticks:
        check("Tick bid correct", abs(ticks[0].bid - 1.08615) < 0.00001)
        spread = ticks[0].spread_pips(pip_size=0.0001)
        check("Tick spread_pips() works", isinstance(spread, float))

# duplicate tick must not raise
with db.session() as sess:
    t2 = Tick(symbol="EURUSD", time=now, bid=1.08620, ask=1.08622)
    try:
        db.upsert_tick(sess, t2)
        check("Duplicate tick silently ignored", True)
    except Exception as e:
        check("Duplicate tick silently ignored", False, str(e))

# ══════════════════════════════════════════════════════════════
# 5. Trade CRUD
# ══════════════════════════════════════════════════════════════
section("5. Trade Table — Open, Close, Stats")

trade_id = None
with db.session() as sess:
    tr = Trade(
        symbol="XAUUSD", direction="BUY", lot_size=0.01,
        open_time=now, open_price=2025.50,
        stop_loss=2015.00, take_profit=2050.00,
        strategy="ema_trend", regime_at_entry="strong_uptrend",
        ai_confidence=0.78, mode="demo",
    )
    db.save_trade(sess, tr)
    trade_id = tr.id

check("Trade saved with auto id", trade_id is not None,
      f"id={trade_id}")

with db.session() as sess:
    open_t = db.get_open_trades(sess, mode="demo")
    check("get_open_trades() finds 1", len(open_t) >= 1)

close_dt = datetime(2024, 1, 15, 14, 30, 0)
with db.session() as sess:
    ok = db.close_trade(
        sess, trade_id=trade_id,
        close_price=2048.75, close_time=close_dt,
        net_profit=23.25, pips=232.5,
        close_reason="tp",
    )
    check("close_trade() returns True", ok)

with db.session() as sess:
    closed = db.get_closed_trades(sess, symbol="XAUUSD", limit=10)
    check("Closed trade found", len(closed) >= 1)
    if closed:
        check("Trade close_reason = 'tp'",
              closed[0].close_reason == "tp")
        check("Trade net_profit correct",
              abs(closed[0].net_profit - 23.25) < 0.01)
        check("Trade to_dict() works",
              isinstance(closed[0].to_dict(), dict))

with db.session() as sess:
    stats = db.get_trade_stats(sess, mode="demo")
    check("get_trade_stats() returns dict", isinstance(stats, dict))
    check("Stats has 'win_rate' key", "win_rate" in stats)
    check("Stats has 'profit_factor' key", "profit_factor" in stats)

# ══════════════════════════════════════════════════════════════
# 6. Feature CRUD
# ══════════════════════════════════════════════════════════════
section("6. Feature Table — Store and Retrieve")

features_dict = {
    "rsi_14": 62.5, "atr_14": 0.0015, "adx_14": 28.3,
    "macd": 0.00045, "bb_width": 0.0032, "ema_20": 1.08540,
}

with db.session() as sess:
    f = Feature.from_dict("EURUSD", "H1", now, features_dict)
    db.upsert_feature(sess, f)

with db.session() as sess:
    feats = db.get_features(sess, "EURUSD", "H1", limit=10)
    check("Feature saved", len(feats) >= 1)
    if feats:
        retrieved = feats[0].get_features()
        check("Feature dict round-trips correctly",
              abs(retrieved["rsi_14"] - 62.5) < 0.001,
              f"got {retrieved.get('rsi_14')}")

# ══════════════════════════════════════════════════════════════
# 7. Regime Predictions
# ══════════════════════════════════════════════════════════════
section("7. Regime Predictions — Save and Query")

with db.session() as sess:
    rp = RegimePrediction(
        symbol="EURUSD", timeframe="H1", bar_time=now,
        regime="strong_uptrend", confidence=0.82,
        trade_allowed=True, model_version="regime_xgb_v1",
    )
    db.upsert_regime(sess, rp)

with db.session() as sess:
    latest = db.get_latest_regime(sess, "EURUSD", "H1")
    check("Regime prediction saved", latest is not None)
    if latest:
        check("Regime = 'strong_uptrend'",
              latest.regime == "strong_uptrend")
        check("Regime confidence = 0.82",
              abs(latest.confidence - 0.82) < 0.001)
        check("trade_allowed = True", latest.trade_allowed is True)

# ══════════════════════════════════════════════════════════════
# 8. Model Versions
# ══════════════════════════════════════════════════════════════
section("8. Model Versions — Save, Activate, History")

with db.session() as sess:
    v1 = db.get_next_version(sess, "regime_xgboost")
    check(f"Next version = 1 (found {v1})", v1 == 1)

with db.session() as sess:
    mv = ModelVersion(
        model_name="regime_xgboost", model_type="ml",
        version=1, file_path="models/ml/regime_xgb_v1.pkl",
        train_date=now, train_samples=5000,
        sharpe_ratio=1.45, profit_factor=1.82,
        win_rate=62.5, max_drawdown=8.2, accuracy=0.674,
    )
    mv.set_params({"n_estimators": 200, "max_depth": 6, "lr": 0.05})
    db.save_model_version(sess, mv)

with db.session() as sess:
    active = db.get_active_model(sess, "regime_xgboost")
    check("Active model found", active is not None)
    if active:
        check("Model name correct",
              active.model_name == "regime_xgboost")
        check("Model is_active = True", active.is_active is True)
        check("Model params round-trip",
              active.get_params().get("n_estimators") == 200)

# Save v2 — v1 should become inactive
with db.session() as sess:
    mv2 = ModelVersion(
        model_name="regime_xgboost", model_type="ml",
        version=2, file_path="models/ml/regime_xgb_v2.pkl",
        train_date=now, train_samples=7500,
        sharpe_ratio=1.62, profit_factor=1.95,
        win_rate=65.0, max_drawdown=7.1, accuracy=0.691,
    )
    db.save_model_version(sess, mv2)

with db.session() as sess:
    active = db.get_active_model(sess, "regime_xgboost")
    check("Active model is now v2", active.version == 2)
    history = db.get_model_history(sess, "regime_xgboost")
    check("Model history has 2 entries", len(history) == 2)

# ══════════════════════════════════════════════════════════════
# 9. Optimization Results
# ══════════════════════════════════════════════════════════════
section("9. Optimization Results — Save and Query Best")

with db.session() as sess:
    for i, (sharpe, pf) in enumerate([(1.2, 1.5), (1.6, 1.9), (0.9, 1.2)]):
        r = OptimizationResult(
            strategy="ema_trend", symbol="EURUSD",
            trial_number=i + 1,
            in_sample_sharpe=sharpe, out_sample_sharpe=sharpe - 0.1,
            in_sample_pf=pf, out_sample_pf=pf - 0.1,
            win_rate=55.0, max_drawdown=10.0,
            is_best=(i == 1),
        )
        r.set_params({"fast_ema": 20 + i * 5, "slow_ema": 50})
        db.save_opt_result(sess, r)

with db.session() as sess:
    best = db.get_best_params(sess, "ema_trend", "EURUSD")
    check("Best opt result found", best is not None)
    if best:
        check("Best result has is_best=True", best.is_best is True)
        p = best.get_params()
        check("Best result params readable", "fast_ema" in p)

# ══════════════════════════════════════════════════════════════
# 10. System Events
# ══════════════════════════════════════════════════════════════
section("10. System Events — Log and Query")

with db.session() as sess:
    db.log_event(sess, "startup", "System started in demo mode",
                 {"mode": "demo", "symbols": 9})
    db.log_event(sess, "circuit_breaker",
                 "Daily loss limit hit: -3.1%",
                 {"loss_pct": 3.1})

with db.session() as sess:
    events = db.get_recent_events(sess, limit=10)
    check("2 system events found", len(events) >= 2)
    if events:
        check("Event has event_type", bool(events[0].event_type))
        check("Event get_details() works",
              isinstance(events[0].get_details(), dict))

# ══════════════════════════════════════════════════════════════
# 11. Daily Stats
# ══════════════════════════════════════════════════════════════
section("11. Daily Stats — Upsert and Query")

today = date(2024, 1, 15)
with db.session() as sess:
    ds = DailyStats(
        date=today, mode="demo",
        opening_balance=10000.0, closing_balance=10023.25,
        gross_profit=23.25, gross_loss=0.0, net_profit=23.25,
        trades_opened=1, trades_closed=1,
        win_count=1, loss_count=0, win_rate=100.0, max_drawdown=0.5,
    )
    db.upsert_daily_stats(sess, ds)

with db.session() as sess:
    stats = db.get_daily_stats(sess, mode="demo", days=10)
    check("Daily stats saved", len(stats) >= 1)
    if stats:
        check("Daily net_profit correct",
              abs(stats[0].net_profit - 23.25) < 0.01)

# ══════════════════════════════════════════════════════════════
# 12. Utility Methods
# ══════════════════════════════════════════════════════════════
section("12. Utility Methods")

counts = db.table_row_counts()
check("table_row_counts() returns 9 tables", len(counts) == 9)
check("candles count >= 1", counts.get("candles", 0) >= 1)
check("trades count >= 1", counts.get("trades", 0) >= 1)

summary = db.summary()
check("db.summary() returns string", isinstance(summary, str))
print(f"\n  {CYAN}{summary}{RESET}")

# ══════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════
section("Cleanup — Removing Test Database")

try:
    # Dispose all SQLAlchemy connections before deleting the file
    db.engine().dispose()
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    for ext in ["-wal", "-shm"]:
        f = TEST_DB + ext
        if os.path.exists(f):
            os.remove(f)
    check("Test database deleted", not os.path.exists(TEST_DB))
except Exception as e:
    check("Test database deleted", False, str(e))

# Restore real db path
_cl_mod.cfg.db_path = _real_db_path

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 4 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Database layer is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 4 works' to begin Step 5.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 4 INCOMPLETE -- {passed}/{total} passed, {failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run: py -3.11 test_database.py{RESET}")
print(f"{'=' * 55}\n")
