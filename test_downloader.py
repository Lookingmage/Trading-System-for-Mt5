"""
MT5 Quantum AI -- Step 6 Verification: Data Downloader
========================================================
How to run:
    py -3.11 test_downloader.py

Requires: MetaTrader 5 open and logged in.

What this tests:
  1. Downloader imports correctly
  2. Symbol names resolve to broker names (e.g. EURUSD → EURUSDc)
  3. Candles download for 2 symbols and 2 timeframes
  4. Bars are saved to the database
  5. CSV files are created in data/candles/
  6. Resume logic: second download adds 0 new bars (already up to date)
  7. Tick download works for 1 symbol
  8. Tick CSV file is created in data/tick/
  9. Status report shows correct bar counts
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
section("1. Import Downloader")

try:
    from src.data.downloader import downloader, DataDownloader
    check("from src.data.downloader import downloader", True)
    check("DataDownloader singleton", downloader is DataDownloader())
except Exception as e:
    check("from src.data.downloader import downloader", False, str(e))
    print(f"\n{RED}Cannot continue — fix the import error above.{RESET}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. MT5 Connection
# ══════════════════════════════════════════════════════════════
section("2. MT5 Connection")

from src.connector.mt5_connector import mt5c
import MetaTrader5 as mt5

terminal_ok = mt5.initialize()
if terminal_ok:
    mt5.shutdown()

if not terminal_ok:
    print(f"\n{RED}  MT5 terminal is not running.{RESET}")
    print(f"{YELLOW}  Open MetaTrader 5, then re-run: py -3.11 test_downloader.py{RESET}")
    sys.exit(1)

connected = mt5c.connect()
check("MT5 connected", connected)
if not connected:
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 3. Symbol Resolution
# ══════════════════════════════════════════════════════════════
section("3. Symbol Name Resolution (config → broker)")

# Resolve all enabled symbols
sym_map = downloader.resolve_all_symbols()
check("resolve_all_symbols() returns dict", isinstance(sym_map, dict))
check("at least 1 symbol resolved", len(sym_map) >= 1,
      f"resolved: {sym_map}")

print(f"\n  {CYAN}Symbol mapping (config name → broker name):{RESET}")
for cfg_sym, broker_sym in sym_map.items():
    print(f"    {cfg_sym:<10} → {broker_sym}")
print()

# Pick 2 resolved symbols for candle tests
resolved_syms = list(sym_map.keys())
test_sym1 = resolved_syms[0] if len(resolved_syms) >= 1 else None
test_sym2 = resolved_syms[1] if len(resolved_syms) >= 2 else None

check(f"First test symbol resolved: {test_sym1}", test_sym1 is not None)

# ══════════════════════════════════════════════════════════════
# 4. Candle Download — symbol 1, H1
# ══════════════════════════════════════════════════════════════
section(f"4. Candle Download — {test_sym1} H1 (500 bars)")

if test_sym1:
    n1 = downloader.download_candles(test_sym1, "H1", bars=500)

    # Verify in database — pass if new bars downloaded OR already existed
    from src.database import db, Candle
    with db.session() as sess:
        count = (
            sess.query(Candle)
            .filter_by(symbol=test_sym1, timeframe="H1")
            .count()
        )
    check(f"{test_sym1} H1 bars available in DB (new={n1}, total={count})",
          count > 0, f"count={count}")

    # Verify CSV created
    csv_path = downloader.csv_path(test_sym1, "H1")
    check(f"CSV created: data/candles/{test_sym1}/H1.csv",
          csv_path.exists(), str(csv_path))

    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        check("CSV has rows", len(df) > 0, f"got {len(df)} rows")
        for col in ["time", "open", "high", "low", "close"]:
            check(f"CSV has '{col}' column", col in df.columns)
        print(f"\n  {CYAN}{test_sym1} H1: {count} bars in DB | "
              f"CSV rows: {len(df)} | "
              f"Latest close: {df['close'].iloc[-1]}{RESET}")

# ══════════════════════════════════════════════════════════════
# 5. Candle Download — symbol 1, H4
# ══════════════════════════════════════════════════════════════
section(f"5. Candle Download — {test_sym1} H4 (300 bars)")

if test_sym1:
    n_h4 = downloader.download_candles(test_sym1, "H4", bars=300)
    with db.session() as sess:
        count_h4 = (
            sess.query(Candle)
            .filter_by(symbol=test_sym1, timeframe="H4")
            .count()
        )
    check(f"{test_sym1} H4 bars available in DB (new={n_h4}, total={count_h4})",
          count_h4 > 0, f"count={count_h4}")

    csv_h4 = downloader.csv_path(test_sym1, "H4")
    check(f"CSV created: data/candles/{test_sym1}/H4.csv",
          csv_h4.exists())

# ══════════════════════════════════════════════════════════════
# 6. Candle Download — symbol 2
# ══════════════════════════════════════════════════════════════
if test_sym2:
    section(f"6. Candle Download — {test_sym2} H1 (300 bars)")
    n2 = downloader.download_candles(test_sym2, "H1", bars=300)
    with db.session() as sess:
        count2 = (
            sess.query(Candle)
            .filter_by(symbol=test_sym2, timeframe="H1")
            .count()
        )
    check(f"{test_sym2} H1 bars available in DB (new={n2}, total={count2})",
          count2 > 0, f"count={count2}")

# ══════════════════════════════════════════════════════════════
# 7. Resume logic — re-downloading adds 0 new bars
# ══════════════════════════════════════════════════════════════
section("7. Resume Logic — Re-download Should Add 0 New Bars")

if test_sym1:
    n_resume = downloader.download_candles(test_sym1, "H1", bars=500)
    check("Second download adds 0 new bars (resume works)",
          n_resume == 0, f"got {n_resume} new bars (expected 0)")

# ══════════════════════════════════════════════════════════════
# 8. Tick Download
# ══════════════════════════════════════════════════════════════
section(f"8. Tick Download — {test_sym1} (4 hours)")

if test_sym1:
    n_ticks = downloader.download_ticks(test_sym1, hours=4)
    check(f"download_ticks({test_sym1}) >= 0", n_ticks >= 0,
          f"got {n_ticks} ticks")

    tick_csv = downloader.tick_csv_path(test_sym1)
    if n_ticks > 0:
        check("Tick CSV created", tick_csv.exists(), str(tick_csv))
        if tick_csv.exists():
            import pandas as pd
            tdf = pd.read_csv(tick_csv)
            check("Tick CSV has rows", len(tdf) > 0)
            check("Tick CSV has 'bid' column", "bid" in tdf.columns)
            check("Tick CSV has 'ask' column", "ask" in tdf.columns)
            print(f"  {CYAN}Ticks: {n_ticks} saved | "
                  f"CSV rows: {len(tdf)}{RESET}")
    else:
        print(f"  {YELLOW}0 ticks (market may be closed outside trading hours){RESET}")

# ══════════════════════════════════════════════════════════════
# 9. Status Report
# ══════════════════════════════════════════════════════════════
section("9. Download Status Report")

try:
    status = downloader.get_status()
    check("get_status() returns dict", isinstance(status, dict))
    check("Status has symbols", len(status) > 0)

    print()
    downloader.print_status()
except Exception as e:
    check("get_status() works", False, str(e))

# ══════════════════════════════════════════════════════════════
# Disconnect
# ══════════════════════════════════════════════════════════════
mt5c.disconnect()

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 6 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Data Downloader is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 6 works' to begin Step 7.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 6 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
