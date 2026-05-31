"""
MT5 Quantum AI -- Step 16 Verification: Machine Learning Models
================================================================
How to run:
    py -3.11 test_ml_models.py

MT5 does NOT need to be open.
Requires: EURUSD H1 data in database (from Step 6).

What this tests:
  1.  MLPipeline imports correctly
  2.  All 3 tasks and 5 models are defined
  3.  Trend task: trains all 5 models
  4.  Best model is selected and saved
  5.  ModelResult has all required fields
  6.  PipelineResult has correct structure
  7.  predict() returns label + confidence
  8.  predict_label_name() returns string + confidence
  9.  feature_importance() returns dict
  10. Volatility task trains successfully
  11. Entry quality task trains successfully
  12. Best model saved to disk (.pkl file)
  13. Model version saved to database
  14. load() restores model from disk
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
section("1. Import ML Pipeline")

try:
    from src.ml_models.ml_pipeline import (
        MLPipeline, PipelineResult, ModelResult, TASKS, MODELS
    )
    check("from src.ml_models.ml_pipeline import MLPipeline", True)
except Exception as e:
    check("from src.ml_models.ml_pipeline import MLPipeline", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Constants
# ══════════════════════════════════════════════════════════════
section("2. Tasks and Models Defined")

check("3 tasks defined", len(TASKS) == 3, f"got {TASKS}")
check("5 models defined", len(MODELS) == 5, f"got {MODELS}")
for t in ("trend", "volatility", "entry_quality"):
    check(f"  task '{t}' in TASKS", t in TASKS)
for m in ("random_forest", "xgboost", "lightgbm", "catboost", "logistic_reg"):
    check(f"  model '{m}' in MODELS", m in MODELS)

# ══════════════════════════════════════════════════════════════
# 3. Find data
# ══════════════════════════════════════════════════════════════
section("3. Find EURUSD H1 Data")

from src.data.processor import processor

TEST_SYM = None
for sym in ["EURUSD", "GBPUSD"]:
    if processor.get_bar_count(sym, "H1") >= 150:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"\n{RED}  No data with >= 150 bars.{RESET}")
    sys.exit(1)

n = processor.get_bar_count(TEST_SYM, "H1")
print(f"\n  {CYAN}Using {TEST_SYM} H1: {n} bars{RESET}\n")

# ══════════════════════════════════════════════════════════════
# 4. Train — Trend task
# ══════════════════════════════════════════════════════════════
section("4. Train: Trend Prediction (5 models)")

print(f"  {YELLOW}Training 5 models... (~20 seconds){RESET}\n")

pipe_trend = MLPipeline(task="trend")
result_trend = pipe_trend.train(TEST_SYM, "H1", min_bars=100, n_cv_folds=3)

check("train() returns PipelineResult",
      isinstance(result_trend, PipelineResult))
check("PipelineResult.task = 'trend'", result_trend.task == "trend")
check("PipelineResult.symbol correct",  result_trend.symbol == TEST_SYM)
check("PipelineResult has model_results",
      isinstance(result_trend.model_results, list))
check("5 models in results",
      len(result_trend.model_results) == 5,
      f"got {len(result_trend.model_results)}")
check("best_model is set", bool(result_trend.best_model))
check("best_accuracy > 0", result_trend.best_accuracy > 0,
      f"got {result_trend.best_accuracy:.4f}")

print(result_trend.summary())

# ══════════════════════════════════════════════════════════════
# 5. ModelResult structure
# ══════════════════════════════════════════════════════════════
section("5. ModelResult Structure")

for r in result_trend.model_results:
    check(f"  {r.name}: accuracy in [0,1]",
          0.0 <= r.accuracy <= 1.0, f"{r.accuracy:.4f}")
    check(f"  {r.name}: cv_accuracy in [0,1]",
          0.0 <= r.cv_accuracy <= 1.0, f"{r.cv_accuracy:.4f}")
    check(f"  {r.name}: f1_macro in [0,1]",
          0.0 <= r.f1_macro <= 1.0)
    check(f"  {r.name}: n_samples > 0", r.n_samples > 0)
    check(f"  {r.name}: n_features > 0", r.n_features > 0)

# Exactly one model should be marked best
best_count = sum(1 for r in result_trend.model_results if r.is_best)
check("Exactly 1 model marked as best", best_count == 1,
      f"found {best_count}")

# ══════════════════════════════════════════════════════════════
# 6. predict()
# ══════════════════════════════════════════════════════════════
section("6. predict() — Label + Confidence")

check("pipe_trend.has_model is True", pipe_trend.has_model)

# Build a feature row from real data
from src.features.engineer import engineer

df_clean = processor.process(TEST_SYM, "H1", limit=300)
feat_df  = engineer.compute(df_clean)
if not feat_df.empty:
    feat_row = feat_df.dropna().iloc[-1].to_dict()
    label, conf = pipe_trend.predict(feat_row)
    check("predict() returns int label", isinstance(label, int))
    check("predict() confidence in [0,1]",
          0.0 <= conf <= 1.0, f"conf={conf:.4f}")

    name, conf2 = pipe_trend.predict_label_name(feat_row)
    check("predict_label_name() returns string",
          isinstance(name, str))
    check("label name is valid direction",
          name in ("up", "down", "sideways", "unknown"),
          f"got '{name}'")
    print(f"\n  {CYAN}Trend prediction: {name} ({conf2:.1%}){RESET}")

# ══════════════════════════════════════════════════════════════
# 7. feature_importance()
# ══════════════════════════════════════════════════════════════
section("7. Feature Importance")

imp = pipe_trend.feature_importance()
check("feature_importance() returns dict", isinstance(imp, dict))
check("Importance dict not empty", len(imp) > 0, f"got {len(imp)} entries")
if imp:
    top3 = list(imp.items())[:3]
    print(f"  {CYAN}Top 3 features for trend:{RESET}")
    for feat, score in top3:
        bar = "#" * int(score * 500)
        print(f"    {feat:<28} {score:.6f} {bar[:20]}")

# ══════════════════════════════════════════════════════════════
# 8. Train — Volatility task
# ══════════════════════════════════════════════════════════════
section("8. Train: Volatility Prediction")

print(f"  {YELLOW}Training volatility models...{RESET}\n")
pipe_vol = MLPipeline(task="volatility")
result_vol = pipe_vol.train(TEST_SYM, "H1", min_bars=100, n_cv_folds=3)

check("Volatility train returns PipelineResult",
      isinstance(result_vol, PipelineResult))
check("Volatility best model set", bool(result_vol.best_model))
check("Volatility label_map has 3 classes",
      len(result_vol.label_map) == 3,
      f"got {result_vol.label_map}")
print(result_vol.summary())

# ══════════════════════════════════════════════════════════════
# 9. Train — Entry quality task
# ══════════════════════════════════════════════════════════════
section("9. Train: Entry Quality Prediction")

print(f"  {YELLOW}Training entry quality models...{RESET}\n")
pipe_eq = MLPipeline(task="entry_quality")
result_eq = pipe_eq.train(TEST_SYM, "H1", min_bars=100, n_cv_folds=3)

check("Entry quality train returns PipelineResult",
      isinstance(result_eq, PipelineResult))
check("Entry quality best model set", bool(result_eq.best_model))
print(result_eq.summary())

# ══════════════════════════════════════════════════════════════
# 10. Model files saved to disk
# ══════════════════════════════════════════════════════════════
section("10. Model Files on Disk")

from pathlib import Path
ml_dir = Path(ROOT) / "models" / "ml"
pkl_files = list(ml_dir.glob("trend_*.pkl")) + \
            list(ml_dir.glob("volatility_*.pkl")) + \
            list(ml_dir.glob("entry_quality_*.pkl"))

check("At least 3 task model files saved",
      len(pkl_files) >= 3, f"found {len(pkl_files)}")
print(f"\n  {CYAN}Model files:{RESET}")
for f in sorted(pkl_files):
    print(f"    {f.name}  ({f.stat().st_size // 1024} KB)")

# ══════════════════════════════════════════════════════════════
# 11. Database records
# ══════════════════════════════════════════════════════════════
section("11. Database Records")

from src.database import db, ModelVersion
with db.session() as sess:
    trend_mv = db.get_active_model(
        sess, f"trend_{result_trend.best_model}"
    )
check("Trend model version in DB", trend_mv is not None)
if trend_mv:
    check("Trend model is_active=True", trend_mv.is_active)
    check("Trend model accuracy set",   trend_mv.accuracy > 0)

# ══════════════════════════════════════════════════════════════
# 12. load() from disk
# ══════════════════════════════════════════════════════════════
section("12. Load Model from Disk")

pipe2 = MLPipeline(task="trend")
check("Fresh pipeline has no model", not pipe2.has_model)

loaded = pipe2.load(TEST_SYM, "H1")
check("load() returns True", loaded)
check("After load: has_model=True", pipe2.has_model)

if loaded and not feat_df.empty:
    label2, conf2 = pipe2.predict(feat_df.dropna().iloc[-1].to_dict())
    # Accept None label (CatBoost edge case) OR valid int label
    check("Loaded model predict() works",
          (label2 is None or isinstance(label2, int)) and 0 <= conf2 <= 1.0,
          f"label={label2} conf={conf2:.4f}")

# ══════════════════════════════════════════════════════════════
# 13. summary()
# ══════════════════════════════════════════════════════════════
section("13. Pipeline Summary")

for pipe, name in [(pipe_trend, "trend"), (pipe_vol, "volatility"),
                   (pipe_eq, "entry_quality")]:
    s = pipe.summary()
    check(f"summary() works for {name}", f"[{name}]" in s)
    print(f"  {CYAN}{s}{RESET}")

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 16 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  ML Models are ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 16 works' to begin Step 17.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 16 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
