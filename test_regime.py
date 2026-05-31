"""
MT5 Quantum AI -- Step 9 Verification: Regime Detector
=======================================================
How to run:
    py -3.11 test_regime.py

MT5 does NOT need to be open.
Requires: EURUSD H1 data in database (from Step 6).

What this tests:
  1.  Import works
  2.  REGIME_CLASSES has 10 entries
  3.  Label distribution covers multiple regimes
  4.  Training completes and returns metrics
  5.  Ensemble accuracy > 0 (models learned something)
  6.  Model files saved to models/ml/
  7.  Model versions saved to database
  8.  detect() returns a RegimeResult
  9.  Regime name is one of the 10 valid classes
  10. Confidence is between 0 and 1
  11. trade_allowed is a boolean
  12. Probabilities sum to ~1.0
  13. Prediction saved to regime_predictions table
  14. predict_from_features() works with a feature dict
  15. Load model from disk works
  16. is_trade_allowed() returns a bool
"""

import sys
import os
import numpy as np

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
section("1. Import Regime Detector")

try:
    from src.regime.detector import (
        regime_detector, RegimeDetector, RegimeResult, REGIME_CLASSES
    )
    check("from src.regime.detector import regime_detector", True)
    check("RegimeDetector singleton",
          regime_detector is RegimeDetector())
except Exception as e:
    check("from src.regime.detector import regime_detector", False, str(e))
    print(f"\n{RED}Cannot continue — fix import error above.{RESET}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Regime class list
# ══════════════════════════════════════════════════════════════
section("2. Regime Class Definitions")

check("10 regime classes defined", len(REGIME_CLASSES) == 10,
      f"got {len(REGIME_CLASSES)}: {REGIME_CLASSES}")

expected = [
    "strong_uptrend", "strong_downtrend", "weak_uptrend",
    "weak_downtrend", "sideways", "choppy", "breakout",
    "volatile", "low_volatility", "news_shock",
]
for r in expected:
    check(f"  '{r}' in REGIME_CLASSES", r in REGIME_CLASSES)

check("regime_detector.regime_classes has 10 entries",
      len(regime_detector.regime_classes) == 10)
check("tradeable_regimes is a set",
      isinstance(regime_detector.tradeable_regimes, set))

# ══════════════════════════════════════════════════════════════
# 3. Find data and check label distribution
# ══════════════════════════════════════════════════════════════
section("3. Auto-Labeling Distribution")

# Find a symbol with enough data
TEST_SYM = None
TEST_TF  = "H1"
from src.data.processor import processor
for sym in ["EURUSD", "GBPUSD", "XAUUSD"]:
    if processor.get_bar_count(sym, TEST_TF) >= 150:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"\n{RED}  No data with >= 150 bars.{RESET}")
    print(f"{YELLOW}  Run: py -3.11 test_downloader.py{RESET}")
    sys.exit(1)

print(f"\n  {CYAN}Using {TEST_SYM} {TEST_TF} for training{RESET}\n")

dist = regime_detector.get_label_distribution(TEST_SYM, TEST_TF)
check("get_label_distribution() returns dict", isinstance(dist, dict))
check("At least 3 different regimes detected", len(dist) >= 3,
      f"got {len(dist)}: {dict(list(dist.items())[:5])}")

print(f"  {CYAN}Label distribution:{RESET}")
total_labels = sum(dist.values())
for regime, count in sorted(dist.items(), key=lambda x: -x[1]):
    pct = count / total_labels * 100 if total_labels > 0 else 0
    bar = "#" * int(pct / 2)
    print(f"    {regime:<22} {count:>4} ({pct:4.1f}%) {bar}")
print()

# ══════════════════════════════════════════════════════════════
# 4. Train models (fast: 50 estimators for testing)
# ══════════════════════════════════════════════════════════════
section("4. Model Training (50 estimators — fast test mode)")

print(f"  {YELLOW}Training XGBoost + LightGBM... this takes ~10 seconds{RESET}\n")

metrics = regime_detector.train(
    TEST_SYM, TEST_TF,
    n_estimators_xgb=50,
    n_estimators_lgb=50,
    min_samples=100,
)

check("train() returns non-empty metrics dict", bool(metrics))
check("metrics has 'xgb_accuracy'",  "xgb_accuracy"  in metrics)
check("metrics has 'lgb_accuracy'",  "lgb_accuracy"  in metrics)
check("metrics has 'ensemble_accuracy'",
      "ensemble_accuracy" in metrics)
check("metrics has 'train_samples'", "train_samples" in metrics)

if metrics:
    xgb_acc = metrics.get("xgb_accuracy", 0)
    lgb_acc = metrics.get("lgb_accuracy", 0)
    ens_acc = metrics.get("ensemble_accuracy", 0)
    check("XGBoost accuracy > 0", xgb_acc > 0, f"{xgb_acc:.4f}")
    check("LightGBM accuracy > 0", lgb_acc > 0, f"{lgb_acc:.4f}")
    check("Ensemble accuracy >= max(xgb, lgb) - 0.05",
          ens_acc >= max(xgb_acc, lgb_acc) - 0.05,
          f"ensemble={ens_acc:.4f}")
    print(f"\n  {CYAN}Results: XGB={xgb_acc:.2%} | "
          f"LGB={lgb_acc:.2%} | "
          f"Ensemble={ens_acc:.2%}{RESET}")

# ══════════════════════════════════════════════════════════════
# 5. Model files on disk
# ══════════════════════════════════════════════════════════════
section("5. Model Files Saved to Disk")

import os
from pathlib import Path
MODEL_DIR = Path(ROOT) / "models" / "ml"

model_files = list(MODEL_DIR.glob("regime_*.pkl"))
check("models/ml/ directory exists", MODEL_DIR.exists())
check("At least 2 model .pkl files created",
      len(model_files) >= 2,
      f"found: {[f.name for f in model_files]}")

for f in sorted(model_files):
    sz_kb = f.stat().st_size // 1024
    print(f"    {f.name}  ({sz_kb} KB)")

# ══════════════════════════════════════════════════════════════
# 6. Model versions in database
# ══════════════════════════════════════════════════════════════
section("6. Model Versions Saved to Database")

from src.database import db, ModelVersion

with db.session() as sess:
    xgb_mv = db.get_active_model(sess, "regime_xgboost")
    lgb_mv = db.get_active_model(sess, "regime_lightgbm")

check("XGBoost model version in DB", xgb_mv is not None)
check("LightGBM model version in DB", lgb_mv is not None)
if xgb_mv:
    check("XGBoost is_active=True", xgb_mv.is_active is True)
    check("XGBoost file_path set", bool(xgb_mv.file_path))
if lgb_mv:
    check("LightGBM is_active=True", lgb_mv.is_active is True)

# ══════════════════════════════════════════════════════════════
# 7. detect() — predict current regime
# ══════════════════════════════════════════════════════════════
section("7. detect() — Current Regime Prediction")

result = regime_detector.detect(TEST_SYM, TEST_TF, save_to_db=True)

check("detect() returns RegimeResult", isinstance(result, RegimeResult))

if result:
    check("regime is one of 10 valid classes",
          result.regime in REGIME_CLASSES,
          f"got '{result.regime}'")
    check("confidence in [0.0, 1.0]",
          0.0 <= result.confidence <= 1.0,
          f"got {result.confidence:.4f}")
    check("trade_allowed is bool",
          isinstance(result.trade_allowed, bool))
    check("probabilities is dict",
          isinstance(result.probabilities, dict))
    check("probabilities has 10 entries",
          len(result.probabilities) == 10,
          f"got {len(result.probabilities)}")

    prob_sum = sum(result.probabilities.values())
    check("probabilities sum to ~1.0",
          abs(prob_sum - 1.0) < 0.01,
          f"sum={prob_sum:.4f}")

    check("model_version string set", bool(result.model_version))
    check("bar_time is datetime",
          hasattr(result.bar_time, "year"))

    print(f"\n  {CYAN}{result}{RESET}")
    print(f"\n  {CYAN}Top 3 regime probabilities:{RESET}")
    top3 = sorted(result.probabilities.items(), key=lambda x: -x[1])[:3]
    for regime, prob in top3:
        bar = "#" * int(prob * 30)
        print(f"    {regime:<22} {prob:5.1%} {bar}")

# ══════════════════════════════════════════════════════════════
# 8. Prediction saved to database
# ══════════════════════════════════════════════════════════════
section("8. Prediction Saved to Database")

from src.database import RegimePrediction
with db.session() as sess:
    latest_pred = db.get_latest_regime(sess, TEST_SYM, TEST_TF)

check("Regime prediction found in DB", latest_pred is not None)
if latest_pred:
    check("DB regime matches result.regime",
          latest_pred.regime == result.regime,
          f"db='{latest_pred.regime}' result='{result.regime}'")
    check("DB confidence matches",
          abs(latest_pred.confidence - result.confidence) < 0.001)
    check("DB trade_allowed matches",
          latest_pred.trade_allowed == result.trade_allowed)

# ══════════════════════════════════════════════════════════════
# 9. predict_from_features()
# ══════════════════════════════════════════════════════════════
section("9. predict_from_features() — Single Feature Dict")

from src.features.engineer import FEATURE_NAMES

# Build a synthetic feature dict (simulate a bullish bar)
bullish_features = {f: 0.0 for f in FEATURE_NAMES}
bullish_features.update({
    "rsi_14": 65.0, "adx_14": 30.0, "atr_14_pct": 0.08,
    "ema20_above_ema50": 1.0, "price_above_ema200": 1.0,
    "returns_20": 0.008, "structure_bullish": 1.0,
    "macd_hist": 0.0003, "bb_position": 0.7,
})
feat_result = regime_detector.predict_from_features(bullish_features)
check("predict_from_features() returns RegimeResult",
      isinstance(feat_result, RegimeResult))
if feat_result:
    check("predict_from_features regime is valid",
          feat_result.regime in REGIME_CLASSES,
          f"got '{feat_result.regime}'")
    check("predict_from_features confidence in [0,1]",
          0.0 <= feat_result.confidence <= 1.0)
    print(f"  {CYAN}Bullish synthetic bar → {feat_result.regime} "
          f"({feat_result.confidence:.1%}){RESET}")

# ══════════════════════════════════════════════════════════════
# 10. Load model from disk
# ══════════════════════════════════════════════════════════════
section("10. Load Model from Disk")

loaded_xgb = regime_detector.load_model("regime_xgboost")
check("load_model('regime_xgboost') returns object",
      loaded_xgb is not None)
if loaded_xgb:
    # Payload is a dict {"model": ..., "classes": [...]}
    actual_model = loaded_xgb["model"] if isinstance(loaded_xgb, dict) else loaded_xgb
    check("Loaded model has predict_proba method",
          hasattr(actual_model, "predict_proba"))

# ══════════════════════════════════════════════════════════════
# 11. is_trade_allowed()
# ══════════════════════════════════════════════════════════════
section("11. is_trade_allowed() Quick Check")

allowed = regime_detector.is_trade_allowed(TEST_SYM, TEST_TF)
check("is_trade_allowed() returns bool", isinstance(allowed, bool))
print(f"  {CYAN}Trade allowed for {TEST_SYM} {TEST_TF}: {allowed}{RESET}")
print(f"  {CYAN}Tradeable regimes: "
      f"{sorted(regime_detector.tradeable_regimes)}{RESET}")

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 9 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Regime Detector is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 9 works' to begin Step 10.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 9 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
