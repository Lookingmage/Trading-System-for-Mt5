"""
MT5 Quantum AI -- Step 17 Verification: Deep Learning Models
=============================================================
How to run:
    py -3.11 test_dl_models.py

MT5 does NOT need to be open. Runs on CPU.
Requires: EURUSD H1 data in database (from Step 6).

Uses max_epochs=5 for speed. Production uses 100 epochs.

What this tests:
  1.  DLPipeline imports correctly
  2.  All 5 model types defined
  3.  All 5 model architectures build without error
  4.  Model parameter counts are reasonable (> 1000)
  5.  LSTM trains successfully
  6.  GRU trains successfully
  7.  Training returns metrics dict with val_accuracy
  8.  val_accuracy > 0 after training
  9.  predict() returns (int, float)
  10. predict_latest() returns (str, float)
  11. Label name is valid (up/down/sideways/unknown)
  12. Model saved to models/dl/ directory
  13. Model version saved to database
  14. load() restores model from disk
  15. model_summary() returns informative string
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Suppress TF startup noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
section("1. Import DL Pipeline")

try:
    from src.dl_models.dl_pipeline import DLPipeline, DL_MODEL_TYPES, DL_FEATURES
    check("from src.dl_models.dl_pipeline import DLPipeline", True)
except Exception as e:
    check("from src.dl_models.dl_pipeline import DLPipeline", False, str(e))
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Constants
# ══════════════════════════════════════════════════════════════
section("2. Model Types and Features")

check("5 model types defined", len(DL_MODEL_TYPES) == 5,
      f"got {DL_MODEL_TYPES}")
for mt in ("lstm", "gru", "cnn_lstm", "transformer", "tcn"):
    check(f"  '{mt}' in DL_MODEL_TYPES", mt in DL_MODEL_TYPES)
check("DL_FEATURES list has >= 20 entries", len(DL_FEATURES) >= 20,
      f"got {len(DL_FEATURES)}")

# ══════════════════════════════════════════════════════════════
# 3. All 5 architectures build
# ══════════════════════════════════════════════════════════════
section("3. All 5 Architectures Build")

print(f"  {YELLOW}Building model architectures (TF loading)...{RESET}\n")

N_FEATS   = len(DL_FEATURES)
N_CLASSES = 3

for model_type in DL_MODEL_TYPES:
    try:
        pipe  = DLPipeline(model_type=model_type)
        model = pipe.build_only(N_FEATS, N_CLASSES)
        params = model.count_params()
        check(f"  {model_type}: builds OK ({params:,} params)",
              params > 1_000, f"params={params:,}")
    except Exception as e:
        check(f"  {model_type}: builds OK", False, str(e))

# ══════════════════════════════════════════════════════════════
# 4. Find data
# ══════════════════════════════════════════════════════════════
section("4. Find EURUSD H1 Data")

from src.data.processor import processor

TEST_SYM = None
for sym in ["EURUSD", "GBPUSD"]:
    if processor.get_bar_count(sym, "H1") >= 100:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"\n{RED}  No data with >= 100 bars.{RESET}")
    sys.exit(1)

n = processor.get_bar_count(TEST_SYM, "H1")
print(f"\n  {CYAN}Using {TEST_SYM} H1: {n} bars{RESET}\n")

# ══════════════════════════════════════════════════════════════
# 5. Train LSTM
# ══════════════════════════════════════════════════════════════
section("5. Train LSTM (5 epochs max)")

print(f"  {YELLOW}Training LSTM on {TEST_SYM} H1 — ~20 seconds...{RESET}\n")

pipe_lstm = DLPipeline(model_type="lstm")
metrics_lstm = pipe_lstm.train(TEST_SYM, "H1", min_bars=80, max_epochs=5)

check("LSTM train() returns dict", isinstance(metrics_lstm, dict))
check("LSTM status = 'trained'",
      metrics_lstm.get("status") == "trained",
      str(metrics_lstm.get("status")))
check("LSTM val_accuracy > 0",
      metrics_lstm.get("val_accuracy", 0) > 0,
      f"got {metrics_lstm.get('val_accuracy', 'N/A')}")
check("LSTM epochs_run > 0", metrics_lstm.get("epochs_run", 0) > 0)
check("LSTM n_samples > 0",  metrics_lstm.get("n_samples",  0) > 0)
print(f"\n  {CYAN}LSTM: val_acc={metrics_lstm.get('val_accuracy'):.4f} "
      f"val_loss={metrics_lstm.get('val_loss'):.6f} "
      f"epochs={metrics_lstm.get('epochs_run')}{RESET}")

# ══════════════════════════════════════════════════════════════
# 6. Train GRU
# ══════════════════════════════════════════════════════════════
section("6. Train GRU (5 epochs max)")

print(f"  {YELLOW}Training GRU — ~15 seconds...{RESET}\n")

pipe_gru = DLPipeline(model_type="gru")
metrics_gru = pipe_gru.train(TEST_SYM, "H1", min_bars=80, max_epochs=5)

check("GRU train() returns dict", isinstance(metrics_gru, dict))
check("GRU status = 'trained'",
      metrics_gru.get("status") == "trained")
check("GRU val_accuracy > 0",
      metrics_gru.get("val_accuracy", 0) > 0,
      f"got {metrics_gru.get('val_accuracy'):.4f}")
print(f"  {CYAN}GRU: val_acc={metrics_gru.get('val_accuracy'):.4f} "
      f"val_loss={metrics_gru.get('val_loss'):.6f}{RESET}")

# ══════════════════════════════════════════════════════════════
# 7. predict()
# ══════════════════════════════════════════════════════════════
section("7. predict() — Sequence Input")

import numpy as np

# Build a random sequence with correct shape
seq_len  = pipe_lstm._seq_len
n_feats  = len(pipe_lstm._feature_names or DL_FEATURES)
test_seq = np.random.randn(seq_len, n_feats).astype(np.float32)

# Scale to [0,1] to match training
test_seq = np.clip(test_seq, -3, 3)
test_seq = (test_seq + 3) / 6.0

label, conf = pipe_lstm.predict(test_seq)
check("predict() returns (int, float)", isinstance(label, int))
check("predict() confidence in [0,1]", 0.0 <= conf <= 1.0,
      f"conf={conf:.4f}")
check("predict() label in {0,1,2}", label in (0, 1, 2), f"label={label}")
print(f"  {CYAN}Random sequence → label={label} conf={conf:.4f}{RESET}")

# ══════════════════════════════════════════════════════════════
# 8. predict_latest()
# ══════════════════════════════════════════════════════════════
section("8. predict_latest() — Live Data")

name, conf2 = pipe_lstm.predict_latest(TEST_SYM, "H1")
check("predict_latest() returns (str, float)", isinstance(name, str))
check("label name is valid",
      name in ("up", "down", "sideways", "unknown"),
      f"got '{name}'")
check("confidence in [0,1]", 0.0 <= conf2 <= 1.0)
print(f"  {CYAN}LSTM prediction: {TEST_SYM} H1 → {name} ({conf2:.1%}){RESET}")

# ══════════════════════════════════════════════════════════════
# 9. Model files saved
# ══════════════════════════════════════════════════════════════
section("9. Model Files Saved to disk")

from pathlib import Path
dl_dir    = Path(ROOT) / "models" / "dl"
dl_keras  = list(dl_dir.glob("*.keras")) if dl_dir.exists() else []
dl_metas  = list(dl_dir.glob("*_meta.pkl")) if dl_dir.exists() else []

check("models/dl/ directory exists", dl_dir.exists())
check("At least 2 .keras model files (LSTM + GRU)",
      len(dl_keras) >= 2, f"found {len(dl_keras)}")
check("At least 2 meta .pkl files",
      len(dl_metas) >= 2, f"found {len(dl_metas)}")

print(f"  {CYAN}Saved model files:{RESET}")
for f in sorted(dl_keras):
    print(f"    {f.name}  ({f.stat().st_size // 1024} KB)")

# ══════════════════════════════════════════════════════════════
# 10. Database records
# ══════════════════════════════════════════════════════════════
section("10. Database Records")

from src.database import db, ModelVersion
with db.session() as sess:
    lstm_mv = db.get_active_model(sess, "dl_lstm")
    gru_mv  = db.get_active_model(sess, "dl_gru")

check("dl_lstm model version in DB", lstm_mv is not None)
check("dl_gru model version in DB",  gru_mv  is not None)
if lstm_mv:
    check("dl_lstm is_active=True", lstm_mv.is_active)
    check("dl_lstm accuracy > 0",   (lstm_mv.accuracy or 0) > 0)

# ══════════════════════════════════════════════════════════════
# 11. Load from disk
# ══════════════════════════════════════════════════════════════
section("11. Load LSTM from Disk")

pipe_loaded = DLPipeline(model_type="lstm")
check("Fresh pipe has no model", not pipe_loaded.has_model)

ok = pipe_loaded.load(TEST_SYM, "H1")
check("load() returns True", ok)
check("After load: has_model=True", pipe_loaded.has_model)

if ok:
    name_l, conf_l = pipe_loaded.predict_latest(TEST_SYM, "H1")
    check("Loaded model predict_latest() works",
          name_l in ("up", "down", "sideways", "unknown"),
          f"got '{name_l}' conf={conf_l:.4f}")

# ══════════════════════════════════════════════════════════════
# 12. model_summary()
# ══════════════════════════════════════════════════════════════
section("12. Model Summary")

for pipe, name in [(pipe_lstm, "lstm"), (pipe_gru, "gru")]:
    s = pipe.model_summary()
    check(f"model_summary() for {name} contains type",
          name in s.lower())
    print(f"  {CYAN}{s}{RESET}")

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 17 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Deep Learning Models are ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 17 works' to begin Step 18.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 17 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
