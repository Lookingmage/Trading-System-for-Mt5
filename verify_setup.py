"""
MT5 Quantum AI — Step 1 Verification Script
============================================
Run this script to confirm Step 1 is complete.

How to run:
    py -3.11 verify_setup.py

Expected output:
    Every line shows [PASS].
    Final line shows: "Step 1 COMPLETE — X/X checks passed"
"""

import sys
import os
import sqlite3
import importlib

# ── colour helpers (no external dep needed here) ────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

passed = 0
failed = 0
results = []


def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    status = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    suffix = f"  {YELLOW}→ {detail}{RESET}" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if ok:
        passed += 1
    else:
        failed += 1
    results.append((label, ok, detail))


def section(title: str):
    print(f"\n{CYAN}{BOLD}{'-' * 55}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'-' * 55}{RESET}")


# ══════════════════════════════════════════════════════════════
# 1. PYTHON VERSION
# ══════════════════════════════════════════════════════════════
section("1. Python Version")

ver = sys.version_info
check(
    f"Python >= 3.11  (found {ver.major}.{ver.minor}.{ver.micro})",
    ver >= (3, 11),
    "" if ver >= (3, 11) else "Use: py -3.11 verify_setup.py"
)

# ══════════════════════════════════════════════════════════════
# 2. DIRECTORY STRUCTURE
# ══════════════════════════════════════════════════════════════
section("2. Directory Structure")

required_dirs = [
    "config",
    "data/tick",
    "data/candles",
    "data/processed",
    "data/features",
    "models/ml",
    "models/dl",
    "models/rl",
    "models/archived",
    "logs",
    "reports",
    "database",
    "dashboard",
    "src",
    "src/connector",
    "src/data",
    "src/features",
    "src/regime",
    "src/strategies",
    "src/strategies/ict_smc",
    "src/backtest",
    "src/optimizer",
    "src/risk",
    "src/executor",
    "src/ai_engine",
    "src/ml_models",
    "src/dl_models",
    "src/rl_models",
    "src/self_learning",
]

for d in required_dirs:
    full = os.path.join(BASE_DIR, d.replace("/", os.sep))
    check(f"Directory: {d}", os.path.isdir(full))

# ══════════════════════════════════════════════════════════════
# 3. PACKAGE FILES
# ══════════════════════════════════════════════════════════════
section("3. Package __init__.py Files")

init_files = [
    "src/__init__.py",
    "src/connector/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/regime/__init__.py",
    "src/strategies/__init__.py",
    "src/strategies/ict_smc/__init__.py",
    "src/backtest/__init__.py",
    "src/optimizer/__init__.py",
    "src/risk/__init__.py",
    "src/executor/__init__.py",
    "src/ai_engine/__init__.py",
    "src/ml_models/__init__.py",
    "src/dl_models/__init__.py",
    "src/rl_models/__init__.py",
    "src/self_learning/__init__.py",
    "config/__init__.py",
    "dashboard/__init__.py",
]

for f in init_files:
    full = os.path.join(BASE_DIR, f.replace("/", os.sep))
    check(f"File: {f}", os.path.isfile(full))

# ══════════════════════════════════════════════════════════════
# 4. KEY FILES
# ══════════════════════════════════════════════════════════════
section("4. Key Project Files")

key_files = [
    "main.py",
    "requirements.txt",
    ".env",
]

for f in key_files:
    full = os.path.join(BASE_DIR, f)
    check(f"File: {f}", os.path.isfile(full))

# ══════════════════════════════════════════════════════════════
# 5. PYTHON PACKAGES — CORE
# ══════════════════════════════════════════════════════════════
section("5. Core Python Packages")

core_packages = [
    ("MetaTrader5",  "MetaTrader5"),
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("yaml",         "PyYAML"),
    ("ta",           "ta"),
    ("sklearn",      "scikit-learn"),
    ("xgboost",      "xgboost"),
    ("lightgbm",     "lightgbm"),
    ("catboost",     "catboost"),
    ("sqlalchemy",   "sqlalchemy"),
    ("streamlit",    "streamlit"),
    ("plotly",       "plotly"),
    ("tqdm",         "tqdm"),
    ("joblib",       "joblib"),
    ("optuna",       "optuna"),
    ("schedule",     "schedule"),
    ("dotenv",       "python-dotenv"),
    ("colorlog",     "colorlog"),
    ("requests",     "requests"),
    ("pytz",         "pytz"),
    ("tabulate",     "tabulate"),
    ("humanize",     "humanize"),
]

for import_name, package_name in core_packages:
    try:
        importlib.import_module(import_name)
        check(f"Package: {package_name}", True)
    except ImportError as e:
        check(
            f"Package: {package_name}",
            False,
            f"pip install {package_name}"
        )

# ══════════════════════════════════════════════════════════════
# 6. DEEP LEARNING PACKAGES
# ══════════════════════════════════════════════════════════════
section("6. Deep Learning Packages")

dl_packages = [
    ("tensorflow", "tensorflow"),
    ("torch",      "torch"),
]

for import_name, package_name in dl_packages:
    try:
        mod = importlib.import_module(import_name)
        ver_attr = getattr(mod, "__version__", "?")
        check(f"Package: {package_name} ({ver_attr})", True)
    except ImportError:
        check(f"Package: {package_name}", False, f"pip install {package_name}")

# ══════════════════════════════════════════════════════════════
# 7. FUNCTIONAL TESTS
# ══════════════════════════════════════════════════════════════
section("7. Functional Tests")

# 7a. SQLite
try:
    test_db = os.path.join(BASE_DIR, "database", "_test_verify.db")
    conn = sqlite3.connect(test_db)
    conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO test (val) VALUES ('ok')")
    conn.commit()
    row = conn.execute("SELECT val FROM test").fetchone()
    conn.close()
    os.remove(test_db)
    check("SQLite: create + write + read + delete", row[0] == "ok")
except Exception as e:
    check("SQLite: create + write + read + delete", False, str(e))

# 7b. pandas DataFrame
try:
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"price": np.random.randn(100) + 100})
    assert len(df) == 100
    check("pandas: create 100-row DataFrame", True)
except Exception as e:
    check("pandas: create 100-row DataFrame", False, str(e))

# 7c. ta library — RSI calculation
try:
    import pandas as pd
    import numpy as np
    from ta.momentum import RSIIndicator
    prices = pd.Series(np.random.randn(50).cumsum() + 100)
    rsi = RSIIndicator(close=prices, window=14).rsi()
    assert len(rsi) == 50
    check("ta: RSI calculation on 50 bars", True)
except Exception as e:
    check("ta: RSI calculation on 50 bars", False, str(e))

# 7d. XGBoost basic train
try:
    import xgboost as xgb
    import numpy as np
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=5, verbosity=0, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    check("XGBoost: train 5-estimator classifier", True)
except Exception as e:
    check("XGBoost: train 5-estimator classifier", False, str(e))

# 7e. LightGBM basic train
try:
    import lightgbm as lgb
    import numpy as np
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    ds = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "verbosity": -1, "num_leaves": 4}
    model = lgb.train(params, ds, num_boost_round=5)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    check("LightGBM: train 5-round binary classifier", True)
except Exception as e:
    check("LightGBM: train 5-round binary classifier", False, str(e))

# 7f. MT5 terminal path
mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
check("MT5 terminal64.exe found", os.path.isfile(mt5_path), mt5_path)

# 7g. MetaTrader5 Python package importable
try:
    import MetaTrader5 as mt5
    check("MetaTrader5 Python package importable", True)
except ImportError as e:
    check("MetaTrader5 Python package importable", False, str(e))

# 7h. TensorFlow CPU mode
try:
    import tensorflow as tf
    result = tf.constant([1.0, 2.0]) + tf.constant([3.0, 4.0])
    assert result.numpy().tolist() == [4.0, 6.0]
    check("TensorFlow: tensor addition (CPU)", True)
except Exception as e:
    check("TensorFlow: tensor addition (CPU)", False, str(e))

# 7i. PyTorch CPU mode
try:
    import torch
    a = torch.tensor([1.0, 2.0]) + torch.tensor([3.0, 4.0])
    assert a.tolist() == [4.0, 6.0]
    check("PyTorch: tensor addition (CPU)", True)
except Exception as e:
    check("PyTorch: tensor addition (CPU)", False, str(e))

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 1 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  The project skeleton is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 1 works' to begin Step 2.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 1 INCOMPLETE -- {passed}/{total} passed, {failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run this script.{RESET}")
    print()
    print(f"{YELLOW}  Quick fix for missing packages:{RESET}")
    print(f"    py -3.11 -m pip install python-dotenv colorlog schedule joblib optuna tabulate humanize")
print(f"{'=' * 55}\n")
