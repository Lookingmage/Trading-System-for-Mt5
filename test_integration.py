"""
MT5 Quantum AI -- Step 22: Integration Testing
===============================================
Tests all 21 modules working together with real data.

How to run:
    py -3.11 test_integration.py

MT5 does NOT need to be open.
Requires: EURUSD H1 data in DB (from Step 6).

Integration flows tested:
  1. Data Pipeline     Processor → Features → DB
  2. Regime Flow       Features → Regime → DB save
  3. Signal Flow       Features → 9 Strategies → AI → Risk
  4. Trade Flow        Signal → Risk → Executor → DB → Self-Learner
  5. Backtest Flow     Strategies → Backtest Engine → Metrics
  6. Model Flow        ML train → save → load → predict
  7. Config Flow       All modules share the same config values
  8. Error Recovery    Graceful handling of missing data
"""

import sys
import os
import time
from datetime import datetime

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
flow_results: dict = {}


def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    status = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    suffix = f"  {YELLOW}-> {detail}{RESET}" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if ok: passed += 1
    else:  failed += 1
    return ok


def section(title: str):
    print(f"\n{CYAN}{BOLD}{'=' * 60}{RESET}")
    print(f"{CYAN}{BOLD}  INTEGRATION FLOW: {title}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 60}{RESET}")


def sub(title: str):
    print(f"\n{CYAN}  --- {title} ---{RESET}")


# ══════════════════════════════════════════════════════════════
# SETUP — find test data
# ══════════════════════════════════════════════════════════════

print(f"\n{BOLD}MT5 Quantum AI — Integration Test Suite{RESET}")
print(f"{'=' * 60}")
print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

from src.data.processor import processor

TEST_SYM = None
for sym in ["EURUSD", "GBPUSD", "XAUUSD"]:
    if processor.get_bar_count(sym, "H1") >= 250:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"{RED}No symbol with >= 250 bars. Run py -3.11 test_downloader.py first.{RESET}")
    sys.exit(1)

print(f"Test symbol: {CYAN}{TEST_SYM} H1{RESET} "
      f"({processor.get_bar_count(TEST_SYM, 'H1')} bars)\n")

# ══════════════════════════════════════════════════════════════
# FLOW 1: DATA PIPELINE
# ══════════════════════════════════════════════════════════════
section("1 — Data Pipeline: Processor → Features → DB")

t_start = time.monotonic()

sub("1a. Load and Process Candles")
df_clean = processor.process(TEST_SYM, "H1", limit=300, min_bars=100)
check("process() returns non-empty DataFrame", not df_clean.empty,
      f"rows={len(df_clean)}")
check("DatetimeIndex is set", hasattr(df_clean.index, "hour"),
      type(df_clean.index).__name__)
check("OHLCV columns present",
      all(c in df_clean.columns for c in ["open","high","low","close"]))

sub("1b. Feature Engineering")
from src.features.engineer import engineer, FEATURE_NAMES
feat_df = engineer.compute(df_clean, symbol=TEST_SYM, timeframe="H1")
check("compute() adds feature columns", not feat_df.empty)
check("All 67 features present",
      all(f in feat_df.columns for f in FEATURE_NAMES),
      f"missing: {[f for f in FEATURE_NAMES if f not in feat_df.columns][:3]}")

ml_df = engineer.get_ml_features(feat_df)
check("get_ml_features() returns NaN-free matrix",
      not ml_df.isna().any().any(),
      f"rows={len(ml_df)} cols={len(ml_df.columns)}")

sub("1c. Save Features to DB")
saved_df = engineer.compute_and_save(TEST_SYM, "H1", limit=300)
from src.database import db, Feature
with db.session() as sess:
    feat_count = sess.query(Feature).filter_by(
        symbol=TEST_SYM, timeframe="H1"
    ).count()
check("Feature vector saved to DB", feat_count >= 1,
      f"count={feat_count}")

flow_results["data_pipeline"] = {
    "rows": len(df_clean), "features": len(FEATURE_NAMES),
    "ml_rows": len(ml_df), "duration_s": round(time.monotonic()-t_start, 2)
}
print(f"  {CYAN}Flow 1 complete in {flow_results['data_pipeline']['duration_s']}s{RESET}")

# ══════════════════════════════════════════════════════════════
# FLOW 2: REGIME FLOW
# ══════════════════════════════════════════════════════════════
section("2 — Regime Flow: Features → Regime Detector → DB")

t_start = time.monotonic()

sub("2a. Detect Regime from DB Features")
from src.regime.detector import regime_detector, REGIME_CLASSES, RegimeResult

regime_result = regime_detector.detect(TEST_SYM, "H1", save_to_db=True)
check("detect() returns RegimeResult", isinstance(regime_result, RegimeResult))

if regime_result:
    check("regime in 10 valid classes",
          regime_result.regime in REGIME_CLASSES,
          f"got '{regime_result.regime}'")
    check("confidence in [0,1]",
          0.0 <= regime_result.confidence <= 1.0)
    check("trade_allowed is bool",
          isinstance(regime_result.trade_allowed, bool))
    check("probabilities sum to ~1.0",
          abs(sum(regime_result.probabilities.values()) - 1.0) < 0.01)

sub("2b. Regime Saved to DB")
from src.database import RegimePrediction
with db.session() as sess:
    saved_regime = db.get_latest_regime(sess, TEST_SYM, "H1")
check("Regime prediction in DB", saved_regime is not None)
if saved_regime and regime_result:
    check("DB regime matches detect() output",
          saved_regime.regime == regime_result.regime)

flow_results["regime_flow"] = {
    "regime":     regime_result.regime if regime_result else "N/A",
    "confidence": regime_result.confidence if regime_result else 0,
    "duration_s": round(time.monotonic()-t_start, 2),
}
print(f"  {CYAN}Regime: {flow_results['regime_flow']['regime']} "
      f"({flow_results['regime_flow']['confidence']:.1%}) | "
      f"{flow_results['regime_flow']['duration_s']}s{RESET}")

# ══════════════════════════════════════════════════════════════
# FLOW 3: SIGNAL FLOW
# ══════════════════════════════════════════════════════════════
section("3 — Signal Flow: Strategies → AI Engine → Risk Manager")

t_start = time.monotonic()

sub("3a. Run All Strategies")
from src.strategies.engine import strategy_engine
from src.strategies.base   import TradeSignal

signals = strategy_engine.run_on_df(feat_df, TEST_SYM, "H1")
check("strategy_engine.run_on_df() returns list",
      isinstance(signals, list))
print(f"  {CYAN}  {len(signals)} signal(s) found{RESET}")
for sig in signals[:3]:
    print(f"    [{sig.strategy_name}] {sig.direction} "
          f"str={sig.signal_strength:.2f}")

sub("3b. AI Decision Engine Evaluates Each Signal")
from src.ai_engine.decision_engine import ai_engine, AIDecision

feat_row = feat_df.dropna().iloc[-1].to_dict() if not feat_df.dropna().empty else {}

ai_results = []
for sig in signals:
    dec = ai_engine.evaluate(sig, feat_row, regime_result)
    check(f"  AI eval [{sig.strategy_name}]: returns AIDecision",
          isinstance(dec, AIDecision))
    ai_results.append((sig, dec))

sub("3c. Risk Manager Evaluates Approved Signals")
from src.risk.manager import risk_manager, RiskDecision

risk_manager.reset_circuit_breaker()
risk_manager.reset_daily()
# Disable session filter for test
from config.loader import cfg
cfg._settings["risk"]["session_filter_enabled"] = False

risk_results = []
for sig, dec in ai_results:
    fake_acc = {"balance": 10_000.0, "equity": 10_000.0, "margin_free": 9_000.0}
    rd = risk_manager.evaluate(sig, fake_acc)
    check(f"  Risk eval [{sig.strategy_name}]: returns RiskDecision",
          isinstance(rd, RiskDecision))
    if rd.approved:
        check(f"  Risk approved: lot_size > 0", rd.lot_size > 0)
    risk_results.append((sig, dec, rd))

cfg._settings["risk"]["session_filter_enabled"] = True

flow_results["signal_flow"] = {
    "signals_found": len(signals),
    "duration_s":    round(time.monotonic()-t_start, 2),
}
print(f"  {CYAN}Flow 3: {len(signals)} signals in {flow_results['signal_flow']['duration_s']}s{RESET}")

# ══════════════════════════════════════════════════════════════
# FLOW 4: TRADE FLOW
# ══════════════════════════════════════════════════════════════
section("4 — Trade Flow: Signal → Executor → DB → Self-Learner")

t_start = time.monotonic()

sub("4a. Place a Shadow Trade")
from src.executor.executor import executor
from src.database import Trade

executor._mode = "shadow"

# Use any signal (approved or not — executor checks internally)
trade_id = None
if signals:
    test_sig = signals[0]
    test_sig.regime = regime_result.regime if regime_result else ""

    # Force-approve via RiskDecision
    approved_rd = RiskDecision(
        approved=True, lot_size=0.01,
        adjusted_sl=test_sig.stop_loss,
        adjusted_tp=test_sig.take_profit,
        risk_amount=5.0, risk_pct=0.05,
        reason="integration test"
    )
    trade_id = executor.place_order(test_sig, approved_rd)
    check("place_order() returns int trade_id",
          isinstance(trade_id, int) and trade_id > 0,
          f"trade_id={trade_id}")

sub("4b. Trade in Database")
if trade_id:
    with db.session() as sess:
        t = sess.query(Trade).filter_by(id=trade_id).first()
    check("Trade in DB with status=open", t is not None and t.status == "open")
    check("Trade has correct symbol",     t is not None and t.symbol == TEST_SYM)
    check("Trade has correct direction",
          t is not None and t.direction in ("BUY", "SELL"))

sub("4c. Self-Learner Records Trade")
from src.self_learning.self_learner import self_learner

if trade_id:
    self_learner.record_trade_opened(trade_id, feat_row, test_sig)
    with db.session() as sess:
        t_check = sess.query(Trade).filter_by(id=trade_id).first()
        notes_set = bool(t_check and t_check.notes)
    check("Self-learner stored features in Trade.notes", notes_set)

    # Close the trade
    from src.executor.executor import TradeExecutor
    close_ok = executor._close_shadow(trade_id, "tp")
    check("_close_shadow() closes trade", close_ok)
    self_learner.record_trade_closed(trade_id, "tp")

    with db.session() as sess:
        t_closed = sess.query(Trade).filter_by(id=trade_id).first()
    check("Trade status = 'closed' after close",
          t_closed is not None and t_closed.status == "closed")

sub("4d. Performance Analysis")
perf = self_learner.analyze_performance(window=50)
check("analyze_performance() returns dict", isinstance(perf, dict))
check("total_trades > 0", perf.get("total_trades", 0) > 0,
      f"got {perf.get('total_trades')}")
check("win_rate in [0,100]", 0 <= perf.get("win_rate", 0) <= 100)

flow_results["trade_flow"] = {
    "trade_id":    trade_id,
    "win_rate":    perf.get("win_rate", 0),
    "duration_s":  round(time.monotonic()-t_start, 2),
}
print(f"  {CYAN}Trade #{trade_id} placed, closed, and recorded | "
      f"{flow_results['trade_flow']['duration_s']}s{RESET}")

# ══════════════════════════════════════════════════════════════
# FLOW 5: BACKTEST FLOW
# ══════════════════════════════════════════════════════════════
section("5 — Backtest Flow: Strategies → Engine → Metrics → Report")

t_start = time.monotonic()

sub("5a. Run Backtest")
from src.backtest.engine import backtest_engine, BacktestResult

bt_result = backtest_engine.run(
    TEST_SYM, "H1",
    initial_balance=10_000.0,
    lot_size=0.01,
    commission_per_lot=7.0,
)

check("backtest_engine.run() returns BacktestResult",
      isinstance(bt_result, BacktestResult))

sub("5b. Verify Metrics")
m = bt_result.metrics
check("total_bars > 0",       m.total_bars > 0)
check("final_balance > 0",    m.final_balance > 0)
check("equity_curve length",  len(bt_result.equity_curve) > 0)

if m.total_trades > 0:
    check("win_rate in [0,100]", 0 <= m.win_rate <= 100)
    check("profit_factor >= 0",  m.profit_factor >= 0)
    check("sharpe_ratio finite",
          abs(m.sharpe_ratio) < 1000)

sub("5c. Report Generation")
report_text = bt_result.summary()
check("summary() has key metrics", "Net Profit" in report_text)

report_path = bt_result.save_report(tag="integration_test")
check("Report file created", report_path.exists())

flow_results["backtest_flow"] = {
    "trades":     m.total_trades,
    "net_profit": m.net_profit,
    "sharpe":     m.sharpe_ratio,
    "duration_s": round(time.monotonic()-t_start, 2),
}
print(f"  {CYAN}Backtest: {m.total_trades} trades | "
      f"net=${m.net_profit:+,.2f} | "
      f"Sharpe={m.sharpe_ratio:.2f} | "
      f"{flow_results['backtest_flow']['duration_s']}s{RESET}")

# ══════════════════════════════════════════════════════════════
# FLOW 6: MODEL FLOW
# ══════════════════════════════════════════════════════════════
section("6 — Model Flow: ML Train → Save → Load → Predict")

t_start = time.monotonic()

sub("6a. Train ML Pipeline (trend task)")
from src.ml_models.ml_pipeline import MLPipeline

pipe = MLPipeline(task="trend")
result = pipe.train(TEST_SYM, "H1", min_bars=100, n_cv_folds=2)
check("ML train() completes", bool(result.best_model))
check("Best model selected",  bool(result.best_model),
      f"best={result.best_model}")
check("Accuracy > 0",        result.best_accuracy > 0)

sub("6b. Predict with Trained Model")
label, conf = pipe.predict(feat_row)
check("predict() returns (int, float)",
      isinstance(label, int) and 0.0 <= conf <= 1.0,
      f"label={label} conf={conf:.4f}")
name, _ = pipe.predict_label_name(feat_row)
check("Label name is valid", name in ("up","down","sideways","unknown"),
      f"got '{name}'")

sub("6c. Load from Disk and Predict Again")
pipe2 = MLPipeline(task="trend")
loaded = pipe2.load(TEST_SYM, "H1")
check("load() succeeds", loaded)
if loaded:
    label2, conf2 = pipe2.predict(feat_row)
    check("Loaded model predict() works",
          (label2 is None or isinstance(label2, int)) and 0.0 <= conf2 <= 1.0)

sub("6d. Feature Importance")
imp = pipe.feature_importance()
check("feature_importance() returns dict", isinstance(imp, dict))
check("Importance dict not empty", len(imp) > 0)

flow_results["model_flow"] = {
    "best_model":  result.best_model,
    "accuracy":    result.best_accuracy,
    "label":       name,
    "duration_s":  round(time.monotonic()-t_start, 2),
}
print(f"  {CYAN}Model: {result.best_model} acc={result.best_accuracy:.4f} | "
      f"prediction={name} | {flow_results['model_flow']['duration_s']}s{RESET}")

# ══════════════════════════════════════════════════════════════
# FLOW 7: CONFIG CONSISTENCY
# ══════════════════════════════════════════════════════════════
section("7 — Config Consistency Across Modules")

sub("7a. All modules use the same config singleton")
from config.loader import cfg

cfg2 = __import__("config.loader", fromlist=["cfg"]).cfg
check("cfg is singleton across imports", cfg is cfg2)

check("risk.max_drawdown_pct is float",
      isinstance(cfg.get("risk.max_drawdown_pct"), float))
check("regime.confidence_threshold is float",
      isinstance(cfg.get("regime.confidence_threshold"), float))
check("9 symbols in config", len(cfg.symbols()) == 9,
      f"got {len(cfg.symbols())}")

sub("7b. Regime tradeable list matches config")
tradeable_cfg = set(cfg.tradeable_regimes())
tradeable_rd  = regime_detector.tradeable_regimes
check("Regime tradeable set matches config",
      tradeable_cfg == tradeable_rd,
      f"cfg={tradeable_cfg} detector={tradeable_rd}")

# ══════════════════════════════════════════════════════════════
# FLOW 8: ERROR RECOVERY
# ══════════════════════════════════════════════════════════════
section("8 — Error Recovery: Graceful Handling of Missing Data")

sub("8a. Process returns empty for unknown symbol")
empty_df = processor.process("FAKESYMBOL", "H1", min_bars=1)
check("processor.process() returns empty for unknown symbol",
      empty_df.empty)

sub("8b. Regime detector handles no data")
regime_none = regime_detector.detect("FAKESYMBOL", "H1", save_to_db=False)
check("regime_detector.detect() returns None for unknown symbol",
      regime_none is None)

sub("8c. Strategy engine handles empty DataFrame")
import pandas as pd
empty_signals = strategy_engine.run_on_df(pd.DataFrame(), "FAKESYMBOL", "H1")
check("strategy_engine returns [] for empty DataFrame",
      empty_signals == [])

sub("8d. Executor rejects unapproved signal")
from src.risk.manager import RiskDecision as RD
bad_rd = RD(approved=False, lot_size=0.0, reason="test block")
bad_result = executor.place_order(signals[0] if signals else signals, bad_rd)
check("executor rejects unapproved RiskDecision", bad_result is None)

# ══════════════════════════════════════════════════════════════
# INTEGRATION SUMMARY
# ══════════════════════════════════════════════════════════════

total    = passed + failed
duration = sum(v.get("duration_s", 0) for v in flow_results.values())

print(f"\n{'=' * 60}")
print(f"{CYAN}{BOLD}  INTEGRATION TEST SUMMARY{RESET}")
print(f"{'=' * 60}")
for flow_name, data in flow_results.items():
    print(f"  {GREEN}✓{RESET} {flow_name:<20} {data.get('duration_s', 0):.1f}s")
print(f"\n  Total integration time: ~{duration:.1f}s")

print(f"\n{'=' * 60}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 22 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Integration tests passed.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 22 works' to begin Step 23.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 22 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 60}\n")
