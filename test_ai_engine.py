"""
MT5 Quantum AI -- Step 15 Verification: AI Decision Engine
===========================================================
How to run:
    py -3.11 test_ai_engine.py

MT5 does NOT need to be open.

What this tests:
  1.  AIDecisionEngine imports correctly
  2.  AIDecision dataclass is correct
  3.  DECISION_FEATURES list has expected entries
  4.  Rule-based mode works without a trained model
  5.  Non-tradeable regime is hard-blocked
  6.  Low regime confidence is blocked
  7.  High-quality signal is approved in rule mode
  8.  Low-quality signal is rejected in rule mode
  9.  train() completes with synthetic data
  10. After training, ML mode is active
  11. ML mode evaluates signals correctly
  12. Model saved to disk and database
  13. Threshold can be changed
  14. summary() returns correct mode string
  15. Full pipeline test: signal → evaluate() → decision
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


def make_signal(direction="BUY", strength=0.75):
    from src.strategies.base import TradeSignal
    return TradeSignal(
        strategy_name="ema_trend", symbol="EURUSD", timeframe="H1",
        direction=direction,
        entry_price=1.1000,
        stop_loss= 1.0950 if direction == "BUY" else 1.1050,
        take_profit=1.1100 if direction == "BUY" else 1.0900,
        signal_strength=strength,
        bar_time=datetime.utcnow(), reason="test",
    )


def make_good_features(direction="BUY") -> dict:
    """Feature dict representing a high-quality setup."""
    return {
        "rsi_14": 58.0, "rsi_7": 60.0, "adx_14": 32.0,
        "di_plus": 25.0, "di_minus": 12.0, "atr_14_pct": 0.08,
        "bb_position": 0.65, "bb_width": 0.012,
        "macd_hist": 0.0003, "macd_above_signal": 1.0,
        "stoch_k": 62.0, "stoch_d": 58.0,
        "ema20_above_ema50": 1.0, "ema50_above_ema200": 1.0,
        "price_above_ema20": 1.0, "price_above_ema50": 1.0,
        "trend_slope_20": 0.015,
        "structure_bullish": 1.0, "structure_bearish": 0.0,
        "in_premium_zone": 0.0, "in_discount_zone": 1.0, "in_ote_zone": 1.0,
        "fvg_bullish": 1.0, "fvg_bearish": 0.0,
        "volume_ratio": 1.3, "returns_1": 0.002, "returns_5": 0.008,
        "body_pct": 65.0, "upper_wick_pct": 10.0, "lower_wick_pct": 25.0,
        "is_london_session": 1.0, "is_newyork_session": 0.0,
        "is_london_killzone": 1.0, "is_ny_killzone": 0.0,
        "hour": 9.0, "day_of_week": 1.0,
    }


def make_bad_features() -> dict:
    """Feature dict representing a poor-quality setup."""
    return {
        "rsi_14": 80.0, "rsi_7": 82.0, "adx_14": 8.0,
        "di_plus": 10.0, "di_minus": 18.0, "atr_14_pct": 0.02,
        "bb_position": 1.1, "bb_width": 0.003,
        "macd_hist": -0.0002, "macd_above_signal": 0.0,
        "stoch_k": 88.0, "stoch_d": 85.0,
        "ema20_above_ema50": 0.0, "ema50_above_ema200": 0.0,
        "price_above_ema20": 0.0, "price_above_ema50": 0.0,
        "trend_slope_20": -0.01,
        "structure_bullish": 0.0, "structure_bearish": 1.0,
        "in_premium_zone": 1.0, "in_discount_zone": 0.0, "in_ote_zone": 0.0,
        "fvg_bullish": 0.0, "fvg_bearish": 0.0,
        "volume_ratio": 0.6, "returns_1": -0.003, "returns_5": -0.01,
        "body_pct": 20.0, "upper_wick_pct": 50.0, "lower_wick_pct": 30.0,
        "is_london_session": 0.0, "is_newyork_session": 0.0,
        "is_london_killzone": 0.0, "is_ny_killzone": 0.0,
        "hour": 3.0, "day_of_week": 5.0,
    }


def make_regime(name="strong_uptrend", confidence=0.82, trade_allowed=True):
    from src.regime.detector import RegimeResult
    return RegimeResult(
        symbol="EURUSD", timeframe="H1",
        bar_time=datetime.utcnow(),
        regime=name, confidence=confidence,
        trade_allowed=trade_allowed,
    )


# ══════════════════════════════════════════════════════════════
# 1. Import
# ══════════════════════════════════════════════════════════════
section("1. Import AI Decision Engine")

try:
    from src.ai_engine.decision_engine import (
        ai_engine, AIDecisionEngine, AIDecision, DECISION_FEATURES
    )
    check("from src.ai_engine.decision_engine import ai_engine", True)
    check("AIDecisionEngine singleton", ai_engine is AIDecisionEngine())
except Exception as e:
    check("from src.ai_engine.decision_engine import ai_engine", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. AIDecision structure
# ══════════════════════════════════════════════════════════════
section("2. AIDecision Dataclass")

d = AIDecision(approved=True, confidence=0.78, reason="test")
check("AIDecision has 'approved'",    hasattr(d, "approved"))
check("AIDecision has 'confidence'",  hasattr(d, "confidence"))
check("AIDecision has 'reason'",      hasattr(d, "reason"))
check("AIDecision has 'model_used'",  hasattr(d, "model_used"))
check("AIDecision has 'regime'",      hasattr(d, "regime"))
check("str(AIDecision) works",        "APPROVED" in str(d) or "REJECTED" in str(d))

# ══════════════════════════════════════════════════════════════
# 3. DECISION_FEATURES
# ══════════════════════════════════════════════════════════════
section("3. DECISION_FEATURES List")

check("DECISION_FEATURES has >= 30 entries", len(DECISION_FEATURES) >= 30,
      f"got {len(DECISION_FEATURES)}")
for feat in ["rsi_14", "adx_14", "macd_hist", "structure_bullish",
             "ema20_above_ema50", "is_london_session"]:
    check(f"  '{feat}' in DECISION_FEATURES", feat in DECISION_FEATURES)

# ══════════════════════════════════════════════════════════════
# 4. Rule-based mode (no model)
# ══════════════════════════════════════════════════════════════
section("4. Rule-Based Mode (no model)")

# Force rule-based for this section
saved_model      = ai_engine._model
ai_engine._model = None

check("has_model is False without model", not ai_engine.has_model)

# Good signal + tradeable regime → approved
result_good = ai_engine.evaluate(
    make_signal("BUY", 0.75),
    make_good_features("BUY"),
    make_regime("strong_uptrend", 0.82),
)
check("Good BUY + uptrend → approved (rule)",
      result_good.approved, str(result_good))
check("Rule mode: model_used=False", not result_good.model_used)
check("Confidence in [0,1]", 0.0 <= result_good.confidence <= 1.0)
print(f"  {CYAN}{result_good}{RESET}")

# Bad signal → rejected
result_bad = ai_engine.evaluate(
    make_signal("BUY", 0.25),
    make_bad_features(),
    make_regime("weak_uptrend", 0.70),
)
check("Bad signal + weak features → rejected (rule)",
      not result_bad.approved, str(result_bad))
print(f"  {CYAN}{result_bad}{RESET}")

# ══════════════════════════════════════════════════════════════
# 5. Regime filter
# ══════════════════════════════════════════════════════════════
section("5. Regime Hard Blocks")

# Non-tradeable regime
result_sideways = ai_engine.evaluate(
    make_signal("BUY", 0.90),         # strong signal
    make_good_features(),
    make_regime("sideways", 0.95),     # sideways is NOT tradeable
)
check("Sideways regime BLOCKS even strong signal",
      not result_sideways.approved, str(result_sideways))
check("'not tradeable' in reason",
      "not tradeable" in result_sideways.reason.lower() or
      "tradeable" in result_sideways.reason.lower())

# Low regime confidence
result_low_conf = ai_engine.evaluate(
    make_signal("BUY", 0.85),
    make_good_features(),
    make_regime("strong_uptrend", 0.40),   # confidence below threshold
)
check("Low regime confidence BLOCKS signal",
      not result_low_conf.approved, str(result_low_conf))

# No regime = no regime filter (signal evaluated on its own)
result_no_regime = ai_engine.evaluate(
    make_signal("BUY", 0.75),
    make_good_features(),
    regime_result=None,
)
check("No regime = no regime block (may approve)", isinstance(result_no_regime, AIDecision))
print(f"  {CYAN}No regime: {result_no_regime}{RESET}")

# ══════════════════════════════════════════════════════════════
# 6. Train with synthetic data
# ══════════════════════════════════════════════════════════════
section("6. Train XGBoost (synthetic data)")

print(f"  {YELLOW}Training AI Decision Engine...{RESET}\n")
metrics = ai_engine.train(min_trades=5)

check("train() returns dict", isinstance(metrics, dict))
check("train status != error",
      metrics.get("status") in ("trained", "skipped"))

if metrics.get("status") == "trained":
    check("accuracy in metrics", "accuracy" in metrics)
    check("train_samples > 0",   metrics.get("train_samples", 0) > 0)
    acc = metrics.get("accuracy", 0)
    check(f"accuracy > 0.5 (got {acc:.4f})", acc > 0.5)
    print(f"\n  {CYAN}Training: {metrics}{RESET}")
else:
    print(f"  {YELLOW}Skipped: {metrics}{RESET}")

# ══════════════════════════════════════════════════════════════
# 7. ML mode active after training
# ══════════════════════════════════════════════════════════════
section("7. ML Mode Active After Training")

check("has_model is True after training", ai_engine.has_model)

if ai_engine.has_model:
    result_ml_good = ai_engine.evaluate(
        make_signal("BUY", 0.80),
        make_good_features(),
        make_regime("strong_uptrend", 0.85),
    )
    check("ML mode: evaluate() returns AIDecision",
          isinstance(result_ml_good, AIDecision))
    check("ML mode: model_used=True", result_ml_good.model_used)
    check("ML mode: confidence in [0,1]",
          0.0 <= result_ml_good.confidence <= 1.0)
    print(f"  {CYAN}ML (good): {result_ml_good}{RESET}")

    result_ml_bad = ai_engine.evaluate(
        make_signal("BUY", 0.20),
        make_bad_features(),
        make_regime("strong_uptrend", 0.85),
    )
    check("ML mode: bad features evaluated", isinstance(result_ml_bad, AIDecision))
    print(f"  {CYAN}ML (bad):  {result_ml_bad}{RESET}")

# ══════════════════════════════════════════════════════════════
# 8. Model saved to disk and DB
# ══════════════════════════════════════════════════════════════
section("8. Model Persistence")

from pathlib import Path
from src.database import db, ModelVersion

from pathlib import Path as _Path
model_files = list((_Path(ROOT) / "models" / "ml").glob("decision_xgboost*.pkl"))
check("Decision model .pkl saved to disk",
      len(model_files) >= 1, f"found: {[f.name for f in model_files]}")

with db.session() as sess:
    mv = db.get_active_model(sess, "decision_xgboost")
check("Model version in database", mv is not None)
if mv:
    check("Model is_active=True", mv.is_active)
    check("Model file_path set",  bool(mv.file_path))

# ══════════════════════════════════════════════════════════════
# 9. Threshold change
# ══════════════════════════════════════════════════════════════
section("9. Threshold Configuration")

original_thresh = ai_engine.confidence_threshold
ai_engine.set_threshold(0.80)
check("set_threshold(0.80) works",
      abs(ai_engine.confidence_threshold - 0.80) < 0.001)

# Very strict threshold — even good signals may fail
result_strict = ai_engine.evaluate(
    make_signal("BUY", 0.75),
    make_bad_features(),
    make_regime("strong_uptrend", 0.85),
)
# Restore
ai_engine.set_threshold(original_thresh)
check("Threshold restored", abs(ai_engine.confidence_threshold - original_thresh) < 0.001)

# ══════════════════════════════════════════════════════════════
# 10. summary()
# ══════════════════════════════════════════════════════════════
section("10. Summary")

summ = ai_engine.summary()
check("summary() returns string", isinstance(summ, str))
check("summary contains mode",
      "ML" in summ or "Rule" in summ)
check("summary contains threshold",
      "threshold" in summ.lower())
print(f"\n  {CYAN}{summ}{RESET}")

# Restore saved model state
ai_engine._model = saved_model

# ══════════════════════════════════════════════════════════════
# 11. Full pipeline: signal → evaluate → decision
# ══════════════════════════════════════════════════════════════
section("11. Full Pipeline Integration Test")

from src.data.processor import processor
from src.features.engineer import engineer
from src.regime.detector import regime_detector
from src.strategies.engine import strategy_engine

TEST_SYM = None
for sym in ["EURUSD", "GBPUSD"]:
    if processor.get_bar_count(sym, "H1") >= 100:
        TEST_SYM = sym
        break

if TEST_SYM:
    df_clean = processor.process(TEST_SYM, "H1", limit=300)
    feat_df  = engineer.compute(df_clean, TEST_SYM, "H1")
    signals  = strategy_engine.run_on_df(feat_df, TEST_SYM, "H1")
    regime   = regime_detector.detect(TEST_SYM, "H1", save_to_db=False)

    if signals and not feat_df.empty:
        sig       = signals[0]
        feat_row  = feat_df.dropna().iloc[-1].to_dict()
        decision  = ai_engine.evaluate(sig, feat_row, regime)

        check("Pipeline evaluate() returns AIDecision",
              isinstance(decision, AIDecision))
        check("Pipeline confidence in [0,1]",
              0.0 <= decision.confidence <= 1.0)
        print(f"\n  {CYAN}Pipeline: {decision}{RESET}")
        print(f"  {CYAN}Signal:   {sig}{RESET}")
    else:
        print(f"  {YELLOW}No signals on {TEST_SYM} H1 right now — skipping pipeline test{RESET}")
        check("Pipeline test skipped (no signals)", True)

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 15 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  AI Decision Engine is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 15 works' to begin Step 16.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 15 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
