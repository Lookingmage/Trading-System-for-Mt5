"""
MT5 Quantum AI -- Step 7 Verification: Data Processor
=======================================================
How to run:
    py -3.11 test_processor.py

MT5 does NOT need to be open for this test.
Requires: EURUSD H1 data in database (from Step 6).

What this tests:
  1. Processor imports correctly
  2. load_candles() returns a DataFrame from the database
  3. clean() removes invalid bars
  4. clean() correctly identifies and removes bad OHLC bars
  5. validate() detects quality issues
  6. enrich() adds derived columns (returns, log_returns, range, etc.)
  7. process() full pipeline returns clean indexed DataFrame
  8. process() with limit parameter returns correct number of bars
  9. DatetimeIndex is set correctly
  10. quality_report() returns dict with expected keys
  11. print_quality_report() runs without error
  12. has_enough_data() works correctly
  13. get_latest_bar() returns a Series
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
section("1. Import Processor")

try:
    from src.data.processor import processor, DataProcessor
    check("from src.data.processor import processor", True)
    check("DataProcessor singleton", processor is DataProcessor())
except Exception as e:
    check("from src.data.processor import processor", False, str(e))
    print(f"\n{RED}Cannot continue — fix import error above.{RESET}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Load Candles
# ══════════════════════════════════════════════════════════════
section("2. Load Candles from Database")

# Find a symbol/timeframe that has data
TEST_SYM = None
TEST_TF  = "H1"
for sym in ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"]:
    n = processor.get_bar_count(sym, "H1")
    if n > 0:
        TEST_SYM = sym
        break

if TEST_SYM is None:
    print(f"\n{RED}  No candle data found in database.{RESET}")
    print(f"{YELLOW}  Run Step 6 first: py -3.11 test_downloader.py{RESET}")
    sys.exit(1)

print(f"\n  {CYAN}Using {TEST_SYM} H1 for all tests{RESET}\n")

raw = processor.load_candles(TEST_SYM, TEST_TF)
check("load_candles() returns DataFrame", isinstance(raw, pd.DataFrame))
check("DataFrame is not empty", not raw.empty, f"rows={len(raw)}")
check("Has 'open' column",  "open"  in raw.columns)
check("Has 'high' column",  "high"  in raw.columns)
check("Has 'low' column",   "low"   in raw.columns)
check("Has 'close' column", "close" in raw.columns)
check("Has 'time' column",  "time"  in raw.columns)
check(f"At least 100 rows (got {len(raw)})", len(raw) >= 100)

# ══════════════════════════════════════════════════════════════
# 3. Cleaning — valid data passes through unchanged
# ══════════════════════════════════════════════════════════════
section("3. Cleaning — Valid Data Passes Through")

cleaned, report = processor.clean(raw.copy())

check("clean() returns DataFrame", isinstance(cleaned, pd.DataFrame))
check("clean() returns report dict", isinstance(report, dict))
check("report has 'initial_bars'", "initial_bars" in report)
check("report has 'final_bars'",   "final_bars"   in report)
check("report has 'removed_total'","removed_total" in report)
check("Clean data has same or fewer rows",
      len(cleaned) <= len(raw), f"{len(cleaned)} <= {len(raw)}")
check("Cleaned data is not empty", not cleaned.empty)
check(f"Removed {report['removed_total']} invalid bars",
      report["removed_total"] >= 0,
      f"removed={report['removed_total']}")

# ══════════════════════════════════════════════════════════════
# 4. Cleaning — intentionally bad bars are removed
# ══════════════════════════════════════════════════════════════
section("4. Cleaning — Bad Bars Are Detected and Removed")

# Create a DataFrame with known bad bars
good_row = {
    "time":  pd.Timestamp("2024-01-01 10:00:00"),
    "open":  1.0850, "high": 1.0870,
    "low":   1.0830, "close": 1.0860,
    "tick_volume": 100, "spread": 2, "real_volume": 0,
}
bad_rows = [
    # zero price
    {**good_row, "time": pd.Timestamp("2024-01-01 11:00"), "open": 0.0},
    # high < low
    {**good_row, "time": pd.Timestamp("2024-01-01 12:00"),
     "high": 1.0820, "low": 1.0870},
    # open above high
    {**good_row, "time": pd.Timestamp("2024-01-01 13:00"),
     "open": 1.0900},
    # close below low
    {**good_row, "time": pd.Timestamp("2024-01-01 14:00"),
     "close": 1.0810},
    # duplicate timestamp (same as good_row)
    {**good_row},
]

test_df = pd.DataFrame([good_row] + bad_rows)
c_df, c_report = processor.clean(test_df)

check("4 bad bars + 1 duplicate removed from test data",
      c_report["removed_total"] >= 4,
      f"removed={c_report['removed_total']}")
check("Only 1 good bar remains",
      len(c_df) == 1, f"got {len(c_df)} rows")
check("Remaining bar has correct close",
      abs(c_df.iloc[0]["close"] - 1.0860) < 0.0001)

# ══════════════════════════════════════════════════════════════
# 5. Validate
# ══════════════════════════════════════════════════════════════
section("5. Validation")

is_valid, issues = processor.validate(cleaned, TEST_TF, min_bars=50)
check("validate() returns (bool, list)", isinstance(is_valid, bool) and
      isinstance(issues, list))
check(f"Real data is valid (min_bars=50)", is_valid,
      f"issues={issues}")
if issues:
    print(f"  {YELLOW}  Validation notes (non-fatal): {issues}{RESET}")

# Too few bars fails validation
tiny_df = cleaned.head(10)
is_valid_tiny, issues_tiny = processor.validate(tiny_df, TEST_TF, min_bars=50)
check("Tiny DataFrame (10 bars) fails validation (min=50)",
      not is_valid_tiny, f"is_valid={is_valid_tiny}")

# Empty DataFrame fails
is_valid_empty, _ = processor.validate(pd.DataFrame(), TEST_TF)
check("Empty DataFrame fails validation", not is_valid_empty)

# ══════════════════════════════════════════════════════════════
# 6. Enrich — derived columns added
# ══════════════════════════════════════════════════════════════
section("6. Enrichment — Derived Columns Added")

enriched = processor.enrich(cleaned.copy())

for col in ["returns", "log_returns", "range", "body", "body_pct", "is_bullish"]:
    check(f"Column '{col}' added by enrich()", col in enriched.columns)

# Validate derived column logic on a known bar
# range = high - low (must be >= 0)
check("range = high - low (all >= 0)",
      (enriched["range"] >= 0).all(),
      f"min range={enriched['range'].min():.6f}")

# is_bullish must be 0 or 1 only
check("is_bullish is 0 or 1 only",
      enriched["is_bullish"].isin([0, 1]).all())

# log_returns have finite values (excluding first row NaN)
lr = enriched["log_returns"].dropna()
check("log_returns are finite", np.isfinite(lr).all(),
      f"non-finite count={np.sum(~np.isfinite(lr))}")

# body_pct between 0 and 100
check("body_pct in [0, 100]",
      ((enriched["body_pct"] >= 0) & (enriched["body_pct"] <= 100.01)).all())

# ══════════════════════════════════════════════════════════════
# 7. Full pipeline — process()
# ══════════════════════════════════════════════════════════════
section("7. Full Pipeline — process()")

df = processor.process(TEST_SYM, TEST_TF)

check("process() returns DataFrame", isinstance(df, pd.DataFrame))
check("process() result is not empty", not df.empty)
check("process() result has DatetimeIndex",
      isinstance(df.index, pd.DatetimeIndex),
      f"index type={type(df.index).__name__}")
check("process() result has 'returns' column",
      "returns" in df.columns)
check("process() result has 'log_returns' column",
      "log_returns" in df.columns)
check("process() result is sorted chronologically",
      df.index.is_monotonic_increasing)

# ══════════════════════════════════════════════════════════════
# 8. process() with limit
# ══════════════════════════════════════════════════════════════
section("8. process() with limit=100")

df_100 = processor.process(TEST_SYM, TEST_TF, limit=100)
check("process(limit=100) returns DataFrame", not df_100.empty)
check("process(limit=100) returns <= 100 bars",
      len(df_100) <= 100, f"got {len(df_100)}")
check("Limited DataFrame still has DatetimeIndex",
      isinstance(df_100.index, pd.DatetimeIndex))

print(f"\n  {CYAN}{TEST_SYM} {TEST_TF} (100 bars): "
      f"range {str(df_100.index[0])[:16]} → {str(df_100.index[-1])[:16]}{RESET}")
print(f"  {CYAN}Latest close: {df_100['close'].iloc[-1]:.5f} | "
      f"Last return: {df_100['returns'].iloc[-1]:.4%}{RESET}")

# ══════════════════════════════════════════════════════════════
# 9. Empty symbol returns empty DataFrame
# ══════════════════════════════════════════════════════════════
section("9. Missing Data Returns Empty DataFrame")

df_missing = processor.process("FAKESYMBOL", "H1", min_bars=1)
check("process() on missing symbol returns empty DataFrame",
      df_missing.empty)

# ══════════════════════════════════════════════════════════════
# 10. Quality report
# ══════════════════════════════════════════════════════════════
section("10. Quality Report")

rpt = processor.quality_report(TEST_SYM, TEST_TF)
check("quality_report() returns dict", isinstance(rpt, dict))
for key in ["symbol","timeframe","status","raw_bars","clean_bars",
            "removed","date_from","date_to"]:
    check(f"quality_report has '{key}' key", key in rpt)
check("quality_report status is OK or WARN",
      rpt["status"] in ("OK","WARN","NO DATA"))

print(f"\n  {CYAN}Quality: {rpt['symbol']} {rpt['timeframe']} | "
      f"status={rpt['status']} | bars={rpt['clean_bars']} | "
      f"removed={rpt['removed']}{RESET}")

# ══════════════════════════════════════════════════════════════
# 11. print_quality_report()
# ══════════════════════════════════════════════════════════════
section("11. Quality Table (print_quality_report)")

try:
    processor.print_quality_report()
    check("print_quality_report() runs without error", True)
except Exception as e:
    check("print_quality_report() runs without error", False, str(e))

# ══════════════════════════════════════════════════════════════
# 12. Helpers
# ══════════════════════════════════════════════════════════════
section("12. Helper Methods")

count = processor.get_bar_count(TEST_SYM, TEST_TF)
check("get_bar_count() returns int", isinstance(count, int))
check("get_bar_count() > 0", count > 0, f"count={count}")

check("has_enough_data(min=50) True", processor.has_enough_data(TEST_SYM, TEST_TF, 50))
check("has_enough_data(min=999999) False",
      not processor.has_enough_data(TEST_SYM, TEST_TF, 999999))

latest = processor.get_latest_bar(TEST_SYM, TEST_TF)
check("get_latest_bar() returns Series", isinstance(latest, pd.Series))
if latest is not None:
    check("Latest bar has 'close'", "close" in latest.index)

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 7 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Data Processor is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 7 works' to begin Step 8.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 7 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
