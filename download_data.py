"""
MT5 Quantum AI — Data Download Utility
========================================
Downloads historical bars for all enabled symbols and timeframes.
Run this any time you need to refresh or expand your dataset.

How to run:
    py -3.11 download_data.py

MetaTrader 5 must be open and logged in.
"""

import sys
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

print(f"\n{CYAN}{BOLD}MT5 Quantum AI — Data Downloader{RESET}")
print(f"{'=' * 50}")
print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

from config.loader import cfg
from src.connector.mt5_connector import mt5c
from src.data.downloader import downloader
from src.database import db

# ── Connect ──────────────────────────────────────────────────
print(f"{YELLOW}Connecting to MT5...{RESET}")
if not mt5c.connect():
    print(f"{RED}MT5 connection failed. Open MetaTrader 5 and try again.{RESET}")
    sys.exit(1)

acc = mt5c.account_info()
if acc:
    print(f"{GREEN}Connected: #{acc['login']} | "
          f"Balance: {acc['balance']:.2f} {acc['currency']} | "
          f"Broker: {acc['company']}{RESET}\n")

# ── Resolve all symbols ───────────────────────────────────────
print(f"{YELLOW}Resolving broker symbol names...{RESET}")
sym_map = downloader.resolve_all_symbols()
print(f"{GREEN}Available symbols ({len(sym_map)}):{RESET}")
for cfg_name, broker_name in sym_map.items():
    print(f"  {cfg_name:<10} → {broker_name}")
print()

# ── Download configuration ────────────────────────────────────
BARS_PER_TF = 5000   # bars to download per timeframe
# This gives:
#   M5:  5000 × 5 min  = ~17 days of history
#   M15: 5000 × 15 min = ~52 days of history
#   H1:  5000 × 1 hr   = ~208 days of history
#   H4:  5000 × 4 hr   = ~833 days (~2.3 years) of history

total_new  = 0
total_syms = 0
errors     = []

print(f"{CYAN}Downloading {BARS_PER_TF} bars per timeframe...{RESET}\n")

for config_sym in cfg.enabled_symbols():
    if config_sym not in sym_map:
        print(f"  {YELLOW}SKIP {config_sym} — not available on this broker{RESET}")
        continue

    sym_cfg    = cfg.symbol(config_sym)
    timeframes = sym_cfg.get("timeframes", ["H1"])
    sym_total  = 0

    print(f"  {CYAN}{config_sym}{RESET} ({sym_map[config_sym]})")

    for tf in timeframes:
        try:
            n = downloader.download_candles(
                config_sym, tf,
                bars=BARS_PER_TF,
                save_csv=True,
            )
            # Count total bars in DB for this symbol/tf
            from src.database import Candle
            with db.session() as sess:
                total_in_db = (
                    sess.query(Candle)
                    .filter_by(symbol=config_sym, timeframe=tf)
                    .count()
                )

            status = f"{GREEN}+{n} new{RESET}" if n > 0 else f"{YELLOW}up to date{RESET}"
            print(f"    {tf:<6} {status} | total in DB: {total_in_db}")
            sym_total += n
        except Exception as e:
            msg = f"{RED}ERROR {config_sym} {tf}: {e}{RESET}"
            print(f"    {tf:<6} {msg}")
            errors.append(f"{config_sym} {tf}: {e}")

    total_new  += sym_total
    total_syms += 1
    print()

# ── Tick data (last 4 hours for live spread monitoring) ───────
print(f"{CYAN}Downloading recent tick data (4h)...{RESET}")
for config_sym in list(sym_map.keys())[:3]:   # first 3 symbols only
    try:
        n = downloader.download_ticks(config_sym, hours=4, save_csv=True)
        print(f"  {config_sym}: {n} ticks")
    except Exception as e:
        print(f"  {config_sym}: {RED}tick error — {e}{RESET}")

print()

# ── Summary ───────────────────────────────────────────────────
mt5c.disconnect()

print(f"{'=' * 50}")
print(f"{GREEN}{BOLD}Download complete{RESET}")
print(f"  Symbols processed:  {total_syms}")
print(f"  New bars saved:     {total_new:,}")

if errors:
    print(f"\n  {YELLOW}Errors ({len(errors)}):{RESET}")
    for e in errors:
        print(f"    {e}")

# ── Show database status ──────────────────────────────────────
print(f"\n{CYAN}Database bar counts:{RESET}")
print(f"{'Symbol':<10} {'TF':<6} {'Bars':>7}  {'Latest Bar'}")
print("-" * 50)

from src.database import Candle
for config_sym in cfg.enabled_symbols():
    sym_cfg    = cfg.symbol(config_sym)
    timeframes = sym_cfg.get("timeframes", ["H1"])
    for tf in timeframes:
        with db.session() as sess:
            count = sess.query(Candle).filter_by(
                symbol=config_sym, timeframe=tf
            ).count()
            latest = (
                sess.query(Candle)
                .filter_by(symbol=config_sym, timeframe=tf)
                .order_by(Candle.open_time.desc())
                .first()
            )
        latest_str = str(latest.open_time)[:16] if latest else "not downloaded"
        flag = "" if count >= 250 else f"  {YELLOW}<-- needs more{RESET}"
        print(f"{config_sym:<10} {tf:<6} {count:>7}  {latest_str}{flag}")

print(f"\n{GREEN}Done. Bars >= 250 are ready for feature engineering.{RESET}")
print(f"{YELLOW}Run the system: py -3.11 main.py --mode live{RESET}\n")
