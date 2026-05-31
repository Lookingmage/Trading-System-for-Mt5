"""
MT5 Quantum AI -- Step 5 Verification: MT5 Connector
======================================================
How to run:
    py -3.11 test_connector.py

BEFORE running this test you MUST:
  1. Open MetaTrader 5 on your computer
  2. Log in to your account inside MT5
  3. Fill in .env with:
       MT5_LOGIN=your_account_number
       MT5_PASSWORD=your_password
       MT5_SERVER=your_broker_server

The test has two phases:
  Phase A - does NOT need MT5 running (import, module structure)
  Phase B - NEEDS MT5 running (connection, account, data)

If Phase B is skipped, open MT5 and re-run.
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
skipped = 0


def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    status = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    suffix = f"  {YELLOW}-> {detail}{RESET}" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if ok:
        passed += 1
    else:
        failed += 1


def skip(label: str, reason: str = ""):
    global skipped
    print(f"  {YELLOW}[SKIP]{RESET}  {label}  {YELLOW}({reason}){RESET}")
    skipped += 1


def section(title: str):
    print(f"\n{CYAN}{BOLD}{'-' * 55}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'-' * 55}{RESET}")


# ══════════════════════════════════════════════════════════════
# PHASE A — no MT5 required
# ══════════════════════════════════════════════════════════════
section("Phase A-1: Import")

try:
    from src.connector.mt5_connector import mt5c, MT5Connector, tf, TIMEFRAME_MAP
    check("from src.connector.mt5_connector import mt5c", True)
    check("MT5Connector singleton", mt5c is MT5Connector())
except Exception as e:
    check("from src.connector.mt5_connector import mt5c", False, str(e))
    print(f"\n{RED}Cannot continue -- fix the import error above.{RESET}")
    sys.exit(1)

section("Phase A-2: Timeframe Converter")

tf_tests = {
    "M1": True, "M5": True, "M15": True, "M30": True,
    "H1": True, "H4": True, "D1": True, "W1": True, "MN1": True,
}
for name, expected in tf_tests.items():
    try:
        val = tf(name)
        check(f"tf('{name}') returns int", isinstance(val, int))
    except Exception as e:
        check(f"tf('{name}') returns int", False, str(e))

try:
    tf("INVALID")
    check("tf('INVALID') raises ValueError", False, "should have raised")
except ValueError:
    check("tf('INVALID') raises ValueError", True)

check("TIMEFRAME_MAP has 21 entries", len(TIMEFRAME_MAP) == 21,
      f"found {len(TIMEFRAME_MAP)}")

section("Phase A-3: Credentials Loaded")

# Check .env values without printing them
from config.loader import cfg
login = cfg.mt5_login()
pwd   = cfg.mt5_password()
server = cfg.mt5_server()

check("MT5_LOGIN is set (non-zero)", login != 0,
      "Edit .env: MT5_LOGIN=your_account_number")
check("MT5_PASSWORD is set", bool(pwd) and pwd != "your_password_here",
      "Edit .env: MT5_PASSWORD=your_real_password")
check("MT5_SERVER is set", bool(server) and server != "your_broker_server_here",
      "Edit .env: MT5_SERVER=YourBroker-Demo")

credentials_ok = (login != 0 and bool(pwd) and
                  pwd != "your_password_here" and
                  bool(server) and server != "your_broker_server_here")

if not credentials_ok:
    print(f"\n{YELLOW}  Credentials not set in .env.{RESET}")
    print(f"{YELLOW}  Edit: d:\\Mt5 Ai New\\MT5_Quantum_AI\\.env{RESET}")
    print(f"{YELLOW}  Then re-run: py -3.11 test_connector.py{RESET}")

# ══════════════════════════════════════════════════════════════
# PHASE B — MT5 must be running
# ══════════════════════════════════════════════════════════════
section("Phase B-1: Connect to MT5")

import MetaTrader5 as mt5

# Quick check: is MT5 terminal already running?
terminal_available = mt5.initialize()
if terminal_available:
    mt5.shutdown()

if not terminal_available:
    print(f"\n{YELLOW}  MetaTrader 5 terminal is not running.{RESET}")
    print(f"{YELLOW}  Please:{RESET}")
    print(f"{YELLOW}    1. Open MetaTrader 5{RESET}")
    print(f"{YELLOW}    2. Log in to your account{RESET}")
    print(f"{YELLOW}    3. Re-run: py -3.11 test_connector.py{RESET}")
    for lbl in [
        "MT5 terminal is running",
        "mt5c.connect() returns True",
        "is_connected() is True",
        "account_info() returns dict",
        "balance > 0",
        "equity > 0",
        "terminal_info() returns dict",
        "terminal connected flag is True",
        "symbol_info('EURUSD') returns dict",
        "EURUSD bid > 0",
        "EURUSD ask > 0",
        "check_symbol('EURUSD') returns True",
        "symbol_info_tick() returns dict",
        "tick bid > 0",
        "copy_rates_from_pos EURUSD H1 returns DataFrame",
        "DataFrame has >= 100 rows",
        "DataFrame has 'open' column",
        "DataFrame has 'close' column",
        "DataFrame has 'time' column",
        "copy_rates_from_pos XAUUSD H4 returns DataFrame",
        "mt5c.summary() contains 'CONNECTED'",
        "mt5c.disconnect() runs cleanly",
    ]:
        skip(lbl, "MT5 not running")
else:
    check("MT5 terminal is running", True)

    connected = mt5c.connect()
    check("mt5c.connect() returns True", connected,
          "Check credentials in .env and that MT5 is logged in")

    if not connected:
        print(f"\n{YELLOW}  Connection failed.{RESET}")
        print(f"{YELLOW}  Common fixes:{RESET}")
        print(f"{YELLOW}    - Make sure MT5 is open and logged in{RESET}")
        print(f"{YELLOW}    - Check MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env{RESET}")
        print(f"{YELLOW}    - To find your server: MT5 → File → Open Account → see server list{RESET}")
    else:
        check("is_connected() is True", mt5c.is_connected())

        section("Phase B-2: Account Information")

        acc = mt5c.account_info()
        check("account_info() returns dict", isinstance(acc, dict),
              str(mt5.last_error()) if acc is None else "")

        if acc:
            check("balance > 0", acc["balance"] > 0,
                  f"balance={acc['balance']}")
            check("equity > 0", acc["equity"] > 0,
                  f"equity={acc['equity']}")
            print(f"\n  {CYAN}Account: #{acc['login']} | "
                  f"Balance: {acc['balance']:.2f} {acc['currency']} | "
                  f"Broker: {acc['company']}{RESET}")
            print(f"  {CYAN}Server: {acc['server']} | "
                  f"Leverage: 1:{acc['leverage']}{RESET}\n")

        term = mt5c.get_terminal_info()
        check("terminal_info() returns dict", isinstance(term, dict))
        if term:
            check("terminal connected flag is True",
                  term.get("connected", False),
                  f"connected={term.get('connected')}")

        section("Phase B-3: Symbol Discovery")

        # Get available symbols and find test candidates automatically
        all_sym = mt5c.get_symbols()
        check("get_symbols() returns list", isinstance(all_sym, list))
        check("broker has symbols available", len(all_sym) > 0,
              f"found {len(all_sym)} symbols")

        # Find a symbol with live prices (bid > 0) — tries preferred names first
        FOREX_CANDIDATES = [
            "EURUSD","EURUSDc","EURUSDm","EURUSD.","EURUSD+","EURUSD_",
            "GBPUSD","GBPUSDc","GBPUSDm",
            "USDJPY","USDJPYc","USDJPYm",
        ]
        METAL_CANDIDATES = [
            "XAUUSD","XAUUSDc","XAUUSDm","GOLD","GOLDc","GOLD.","XAUUSD.",
        ]

        def has_price(sym: str) -> bool:
            mt5.symbol_select(sym, True)
            t = mt5.symbol_info_tick(sym)
            return t is not None and t.bid > 0

        # Try preferred names first, then scan all available symbols
        test_forex = next((s for s in FOREX_CANDIDATES if s in all_sym and has_price(s)), None)
        if test_forex is None:
            test_forex = next((s for s in all_sym if has_price(s)), None)

        test_metal = next((s for s in METAL_CANDIDATES if s in all_sym and has_price(s)), None)

        print(f"\n  {CYAN}Broker symbols: {len(all_sym)} available{RESET}")
        print(f"  {CYAN}Test forex symbol : {test_forex or 'not found'}{RESET}")
        print(f"  {CYAN}Test metal symbol : {test_metal or 'not found'}{RESET}\n")

        section("Phase B-4: Symbol Information")

        if test_forex:
            info = mt5c.symbol_info(test_forex)
            check(f"symbol_info('{test_forex}') returns dict",
                  isinstance(info, dict))
            if info:
                check(f"{test_forex} bid > 0", info["bid"] > 0,
                      f"bid={info['bid']}")
                check(f"{test_forex} ask > 0", info["ask"] > 0,
                      f"ask={info['ask']}")
                print(f"  {CYAN}{test_forex}: bid={info['bid']}  "
                      f"ask={info['ask']}  spread={info['spread']} pts{RESET}")

            check(f"check_symbol('{test_forex}') returns True",
                  mt5c.check_symbol(test_forex))

            tick = mt5c.symbol_info_tick(test_forex)
            check("symbol_info_tick() returns dict", isinstance(tick, dict))
            if tick:
                check("tick bid > 0", tick["bid"] > 0,
                      f"bid={tick['bid']}")
        else:
            skip("symbol_info()", "no forex symbol found on broker")
            skip("check_symbol()", "no forex symbol found on broker")
            skip("symbol_info_tick()", "no forex symbol found on broker")

        section("Phase B-5: Candle Data Download")

        if test_forex:
            df = mt5c.copy_rates_from_pos(test_forex, "H1", 0, 200)
            check(f"copy_rates_from_pos {test_forex} H1 returns DataFrame",
                  df is not None)
            if df is not None:
                check("DataFrame has >= 100 rows", len(df) >= 100,
                      f"got {len(df)} rows")
                for col in ["open", "high", "low", "close", "time"]:
                    check(f"DataFrame has '{col}' column", col in df.columns)
                print(f"  {CYAN}{test_forex} H1: {len(df)} bars | "
                      f"latest close={df['close'].iloc[-1]}{RESET}")
        else:
            skip("copy_rates_from_pos H1", "no symbol available")

        if test_metal:
            df_metal = mt5c.copy_rates_from_pos(test_metal, "H4", 0, 100)
            check(f"copy_rates_from_pos {test_metal} H4 returns DataFrame",
                  df_metal is not None)
            if df_metal is not None:
                print(f"  {CYAN}{test_metal} H4: {len(df_metal)} bars | "
                      f"latest close={df_metal['close'].iloc[-1]}{RESET}")
        else:
            skip("copy_rates_from_pos H4 metal", "no metal symbol found")

        section("Phase B-5: Summary and Disconnect")

        summary = mt5c.summary()
        check("mt5c.summary() contains 'CONNECTED'",
              "CONNECTED" in summary)
        print(f"\n  {CYAN}{summary}{RESET}\n")

        mt5c.disconnect()
        check("mt5c.disconnect() runs cleanly", True)

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if not terminal_available:
    print(f"{YELLOW}{BOLD}  Phase A: {passed}/{total} checks passed{RESET}")
    print(f"{YELLOW}  Phase B: {skipped} checks SKIPPED (MT5 not running){RESET}")
    print(f"{YELLOW}  Open MetaTrader 5, then re-run to complete Phase B.{RESET}")
elif failed == 0:
    print(f"{GREEN}{BOLD}  Step 5 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  MT5 Connector is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 5 works' to begin Step 6.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 5 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed, {skipped} skipped{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
