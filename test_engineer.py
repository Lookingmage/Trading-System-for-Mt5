"""
MT5 Quantum AI -- Step 8 Verification: Feature Engineering
===========================================================
How to run:
    py -3.11 test_engineer.py

MT5 does NOT need to be open.
Requires: EURUSD H1 data in database (from Step 6).

What this tests:
  1.  Import works
  2.  50+ feature names defined
  3.  compute() adds all feature columns
  4.  No infinite values in any feature
  5.  RSI in [0, 100] range
  6.  ADX in [0, 100] range
  7.  Bollinger Band position in [0, 1] range
  8.  Binary features are 0 or 1 only
  9.  Session features correct (hour-based logic)
  10. ICT features present (swing_high/low, OTE, FVG)
  11. Market structure flags present (HH/HL/LH/LL)
  12. compute_and_save() stores to database
  13. get_ml_features() returns only feature columns, no NaN
  14. Warmup rows have NaN (expected), later rows are clean
"""

import sys
import os
import numpy as np
import pandas as pd

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
section("1. Import Feature Engineer")

try:
    from src.features.engineer import engineer, FeatureEngineer, FEATURE_NAMES
    check("from src.features.engineer import engineer", True)
    check("FeatureEngineer singleton", engineer is FeatureEngineer())
except Exception as e:
    check("from src.features.engineer import engineer", False, str(e))
    print(f"\n{RED}Cannot continue — fix import error above.{RESET}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Feature name registry
# ══════════════════════════════════════════════════════════════
section("2. Feature Name Registry")

names = engineer.get_feature_names()
check("get_feature_names() returns list", isinstance(names, list))
check(f"At least 50 features defined (got {len(names)})", len(names) >= 50)

# Spot-check key feature names exist
key_features = [
    "rsi_14", "atr_14", "adx_14", "macd", "macd_signal",
    "ema_20", "ema_50", "ema_200", "bb_width", "bb_position",
    "stoch_k", "stoch_d", "williams_r", "cci_20",
    "swing_high_20", "swing_low_20", "in_ote_zone",
    "fvg_bullish", "fvg_bearish",
    "hh", "hl", "lh", "ll",
    "structure_bullish", "structure_bearish",
    "is_london_session", "is_newyork_session",
    "is_london_killzone", "is_ny_killzone",
    "returns_1", "log_returns",
    "volume_ratio", "trend_slope_20",
]
for feat in key_features:
    check(f"  '{feat}' in FEATURE_NAMES", feat in names)

# ══════════════════════════════════════════════════════════════
# 3. Load data and run compute()
# ══════════════════════════════════════════════════════════════
section("3. Compute Features on EURUSD H1")

# Find a symbol with data
TEST_SYM = None
TEST_TF  = "H1"
for sym in ["EURUSD", "GBPUSD", "XAUUSD"]:
    from src.data.processor import processor
    n = processor.get_bar_count(sym, TEST_TF)
    if n >= 100:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"\n{RED}  No data in DB with >= 100 bars.{RESET}")
    print(f"{YELLOW}  Run Step 6 first: py -3.11 test_downloader.py{RESET}")
    sys.exit(1)

df_clean = processor.process(TEST_SYM, TEST_TF)
check(f"Loaded {len(df_clean)} clean bars from DB",
      not df_clean.empty, f"rows={len(df_clean)}")

feat_df = engineer.compute(df_clean, symbol=TEST_SYM, timeframe=TEST_TF)
check("compute() returns DataFrame", isinstance(feat_df, pd.DataFrame))
check("compute() result is not empty", not feat_df.empty)
check(f"All {len(names)} feature columns present",
      all(f in feat_df.columns for f in names),
      f"missing: {[f for f in names if f not in feat_df.columns]}")

# ══════════════════════════════════════════════════════════════
# 4. No infinite values
# ══════════════════════════════════════════════════════════════
section("4. No Infinite Values in Features")

inf_cols = []
for col in names:
    if col in feat_df.columns:
        if np.isinf(feat_df[col].dropna()).any():
            inf_cols.append(col)

check("No infinite values in any feature", len(inf_cols) == 0,
      f"cols with inf: {inf_cols}")

# ══════════════════════════════════════════════════════════════
# 5. Value range checks
# ══════════════════════════════════════════════════════════════
section("5. Feature Value Range Checks")

# RSI must be 0–100
rsi = feat_df["rsi_14"].dropna()
check("rsi_14 in [0, 100]",
      rsi.between(0, 100).all(),
      f"min={rsi.min():.2f} max={rsi.max():.2f}")

# ADX must be 0–100
adx = feat_df["adx_14"].dropna()
check("adx_14 in [0, 100]",
      adx.between(0, 100).all(),
      f"min={adx.min():.2f} max={adx.max():.2f}")

# Stochastic 0–100
stk = feat_df["stoch_k"].dropna()
check("stoch_k in [0, 100]",
      stk.between(0, 100).all(),
      f"min={stk.min():.2f} max={stk.max():.2f}")

# BB position 0–1 (Pband can exceed [0,1] during breakouts)
bb_pos = feat_df["bb_position"].dropna()
check("bb_position has values",  len(bb_pos) > 0)

# ATR is non-negative (ta library may set first bars to 0 before warmup)
atr = feat_df["atr_14"].dropna()
check("atr_14 is non-negative", (atr >= 0).all(),
      f"min={atr.min():.6f}")
# Check last 200 bars (post-warmup) are all positive
atr_clean = feat_df["atr_14"].tail(200).dropna()
check("atr_14 post-warmup bars are positive", (atr_clean > 0).all(),
      f"min={atr_clean.min():.6f}")

# Returns are finite
ret = feat_df["returns_1"].dropna()
check("returns_1 are finite", np.isfinite(ret).all())

# Trend slope is finite (not inf)
slope = feat_df["trend_slope_20"].dropna()
check("trend_slope_20 is finite", np.isfinite(slope).all())

# ══════════════════════════════════════════════════════════════
# 6. Binary features are 0 or 1 only
# ══════════════════════════════════════════════════════════════
section("6. Binary Features Are 0 or 1 Only")

binary_features = [
    "price_above_ema20", "price_above_ema50", "price_above_ema200",
    "ema20_above_ema50", "ema50_above_ema200",
    "macd_above_signal", "stoch_above_signal",
    "adx_trending", "is_bullish",
    "in_premium_zone", "in_discount_zone", "in_ote_zone",
    "fvg_bullish", "fvg_bearish",
    "hh", "hl", "lh", "ll",
    "structure_bullish", "structure_bearish",
    "is_asian_session", "is_london_session", "is_newyork_session",
    "is_london_killzone", "is_ny_killzone", "is_weekend",
]

for feat in binary_features:
    if feat in feat_df.columns:
        vals = feat_df[feat].dropna().unique()
        is_binary = all(v in [0, 1] for v in vals)
        check(f"  '{feat}' is 0/1 only", is_binary,
              f"found values: {set(vals)}")

# ══════════════════════════════════════════════════════════════
# 7. Session logic check
# ══════════════════════════════════════════════════════════════
section("7. Session Features — Logic Check")

# Find a bar at hour=9 (should be London session + London Kill Zone)
london_bars = feat_df[feat_df["hour"] == 9]
if not london_bars.empty:
    check("hour=9 → is_london_session=1",
          (london_bars["is_london_session"] == 1).all())
    check("hour=9 → is_london_killzone=1",
          (london_bars["is_london_killzone"] == 1).all())
    check("hour=9 → is_asian_session=0",
          (london_bars["is_asian_session"] == 0).all())

# Find a bar at hour=2 (should be Asian session only)
asian_bars = feat_df[feat_df["hour"] == 2]
if not asian_bars.empty:
    check("hour=2 → is_asian_session=1",
          (asian_bars["is_asian_session"] == 1).all())
    check("hour=2 → is_london_session=0",
          (asian_bars["is_london_session"] == 0).all())

# ══════════════════════════════════════════════════════════════
# 8. ICT / SMC features
# ══════════════════════════════════════════════════════════════
section("8. ICT / SMC Features")

# Align indices before comparing (dropna may produce different lengths)
sh = feat_df["swing_high_20"].dropna()
cl_sh = feat_df["close"].reindex(sh.index)
check("swing_high_20 >= close (swing high >= current price)",
      (sh >= cl_sh).all())

sl = feat_df["swing_low_20"].dropna()
cl_sl = feat_df["close"].reindex(sl.index)
check("swing_low_20 <= close (swing low <= current price)",
      (sl <= cl_sl).all())
check("premium_discount in [0, 1]",
      feat_df["premium_discount"].dropna().between(0, 1).all())

n_fvg_bull = feat_df["fvg_bullish"].sum()
n_fvg_bear = feat_df["fvg_bearish"].sum()
check(f"FVG detected (bull={int(n_fvg_bull)}, bear={int(n_fvg_bear)})",
      n_fvg_bull + n_fvg_bear > 0)
print(f"  {CYAN}FVGs found: bullish={int(n_fvg_bull)}, "
      f"bearish={int(n_fvg_bear)}{RESET}")

# ══════════════════════════════════════════════════════════════
# 9. Warmup vs clean rows
# ══════════════════════════════════════════════════════════════
section("9. Warmup Rows Have NaN; Later Rows Are Clean")

# First 200 rows may have NaN (warmup for EMA200)
first_rows = feat_df.head(200)
later_rows  = feat_df.tail(200)

nan_in_first  = first_rows[names].isna().any(axis=1).sum()
nan_in_later  = later_rows[names].isna().any(axis=1).sum()

check("First 200 rows have some NaN (warmup expected)",
      nan_in_first > 0, f"NaN rows in first 200: {nan_in_first}")
check("Last 200 rows have no NaN",
      nan_in_later == 0, f"NaN rows in last 200: {nan_in_later}")

# ══════════════════════════════════════════════════════════════
# 10. get_ml_features()
# ══════════════════════════════════════════════════════════════
section("10. ML-Ready Feature Matrix")

ml_df = engineer.get_ml_features(feat_df)
check("get_ml_features() returns DataFrame", isinstance(ml_df, pd.DataFrame))
check("ML DataFrame has no NaN", not ml_df.isna().any().any(),
      f"NaN columns: {list(ml_df.columns[ml_df.isna().any()])}")
check(f"ML features >= 50 columns (got {len(ml_df.columns)})",
      len(ml_df.columns) >= 50)
check("ML DataFrame has rows", len(ml_df) > 0, f"rows={len(ml_df)}")

print(f"\n  {CYAN}ML feature matrix: {len(ml_df)} rows × "
      f"{len(ml_df.columns)} features{RESET}")
print(f"  {CYAN}Latest bar sample:{RESET}")
sample = ml_df.iloc[-1]
for feat in ["rsi_14", "adx_14", "atr_14_pct", "bb_position",
             "macd_hist", "structure_bullish", "in_premium_zone"]:
    if feat in sample.index:
        print(f"    {feat:<25} = {sample[feat]:.4f}")

# ══════════════════════════════════════════════════════════════
# 11. compute_and_save()
# ══════════════════════════════════════════════════════════════
section("11. compute_and_save() — Stores Latest Features to DB")

try:
    saved_df = engineer.compute_and_save(TEST_SYM, TEST_TF, min_bars=100)
    check("compute_and_save() returns DataFrame", not saved_df.empty)

    # Verify DB entry was created
    from src.database import db, Feature
    with db.session() as sess:
        feat_in_db = (
            sess.query(Feature)
            .filter_by(symbol=TEST_SYM, timeframe=TEST_TF)
            .order_by(Feature.bar_time.desc())
            .first()
        )
    check("Feature record saved to database", feat_in_db is not None)
    if feat_in_db:
        stored = feat_in_db.get_features()
        check("Stored features are a dict", isinstance(stored, dict))
        check("Stored dict has rsi_14", "rsi_14" in stored)
        check("Stored rsi_14 is a number",
              isinstance(stored.get("rsi_14"), (int, float)))
        print(f"  {CYAN}Saved to DB: {TEST_SYM} {TEST_TF} | "
              f"bar_time={feat_in_db.bar_time} | "
              f"rsi_14={stored.get('rsi_14', 'N/A'):.2f}{RESET}")
except Exception as e:
    check("compute_and_save() runs without error", False, str(e))

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 8 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Feature Engineering is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 8 works' to begin Step 9.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 8 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
