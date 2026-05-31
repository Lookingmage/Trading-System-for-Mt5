"""
MT5 Quantum AI -- Step 2 Verification: Config Files
=====================================================
How to run:
    py -3.11 test_config.py

Expected: all [PASS], final line shows Step 2 COMPLETE.
"""

import sys
import os

# ── make sure we can import from the project root ───────────
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
# 1. Import ConfigLoader
# ══════════════════════════════════════════════════════════════
section("1. Import Config Loader")

try:
    from config.loader import cfg, ConfigLoader
    check("from config.loader import cfg", True)
except Exception as e:
    check("from config.loader import cfg", False, str(e))
    print(f"\n{RED}Cannot continue — fix the import error above first.{RESET}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 2. Singleton behaviour
# ══════════════════════════════════════════════════════════════
section("2. Singleton Behaviour")

cfg2 = ConfigLoader()
check("ConfigLoader() returns same instance", cfg is cfg2)

# ══════════════════════════════════════════════════════════════
# 3. System settings
# ══════════════════════════════════════════════════════════════
section("3. System Settings")

check("system.name readable", cfg.get("system.name") is not None)
check("system.version readable", cfg.get("system.version") is not None)
check(
    f"system.mode is valid  (found: '{cfg.mode()}')",
    cfg.mode() in {"backtest", "demo", "shadow", "live"}
)
check("system.timezone readable", cfg.get("system.timezone") is not None)

# ══════════════════════════════════════════════════════════════
# 4. MT5 settings
# ══════════════════════════════════════════════════════════════
section("4. MT5 Settings")

check("mt5.terminal_path present", bool(cfg.get("mt5.terminal_path")))
check("mt5.magic_number present", cfg.get("mt5.magic_number") is not None)
check("mt5.timeout_ms present", cfg.get("mt5.timeout_ms") is not None)
check("mt5_login() returns int", isinstance(cfg.mt5_login(), int))
check("mt5_password() returns str", isinstance(cfg.mt5_password(), str))
check("mt5_server() returns str", isinstance(cfg.mt5_server(), str))

# ══════════════════════════════════════════════════════════════
# 5. Risk settings
# ══════════════════════════════════════════════════════════════
section("5. Risk Settings")

check("risk.sizing_mode present", bool(cfg.get("risk.sizing_mode")))
check("risk.risk_per_trade_pct is float",
      isinstance(cfg.get("risk.risk_per_trade_pct"), float))
check("risk.max_daily_loss_pct is float",
      isinstance(cfg.get("risk.max_daily_loss_pct"), float))
check("risk.max_drawdown_pct is float",
      isinstance(cfg.get("risk.max_drawdown_pct"), float))
check("risk.max_consecutive_losses is int",
      isinstance(cfg.get("risk.max_consecutive_losses"), int))
check("risk.circuit_breaker_enabled is bool",
      isinstance(cfg.get("risk.circuit_breaker_enabled"), bool))

# ══════════════════════════════════════════════════════════════
# 6. Position settings
# ══════════════════════════════════════════════════════════════
section("6. Position Settings")

check("position.sl_atr_multiplier present",
      cfg.get("position.sl_atr_multiplier") is not None)
check("position.tp_atr_multiplier present",
      cfg.get("position.tp_atr_multiplier") is not None)
check("position.max_open_trades is int",
      isinstance(cfg.get("position.max_open_trades"), int))

# ══════════════════════════════════════════════════════════════
# 7. Regime settings
# ══════════════════════════════════════════════════════════════
section("7. Regime Settings")

tradeable = cfg.tradeable_regimes()
check("regime.tradeable_regimes is list", isinstance(tradeable, list))
check("at least 3 tradeable regimes defined", len(tradeable) >= 3)
check("regime.confidence_threshold is float",
      isinstance(cfg.get("regime.confidence_threshold"), float))
check("regime.classes list present",
      isinstance(cfg.get("regime.classes"), list))
check("10 regime classes defined", len(cfg.get("regime.classes", [])) == 10)

# ══════════════════════════════════════════════════════════════
# 8. ICT / SMC settings
# ══════════════════════════════════════════════════════════════
section("8. ICT / SMC Settings")

check("ict_smc.order_block_lookback present",
      cfg.get("ict_smc.order_block_lookback") is not None)
check("ict_smc.ote_fib_low present",
      cfg.get("ict_smc.ote_fib_low") is not None)
check("ict_smc.ote_fib_high present",
      cfg.get("ict_smc.ote_fib_high") is not None)
check("ict_smc.premium_discount_level == 0.5",
      cfg.get("ict_smc.premium_discount_level") == 0.50)

# ══════════════════════════════════════════════════════════════
# 9. Strategy toggles
# ══════════════════════════════════════════════════════════════
section("9. Strategy Toggles")

enabled_strats = cfg.enabled_strategies()
check("enabled_strategies() returns list", isinstance(enabled_strats, list))
check("at least 5 strategies enabled", len(enabled_strats) >= 5,
      f"found: {enabled_strats}")

expected_strats = [
    "ema_trend", "breakout", "market_structure",
    "mtf_trend", "order_block", "fair_value_gap",
    "liquidity_sweep", "session_breakout", "mss_choch",
]
all_strats = cfg.get("strategies", {})
for strat in expected_strats:
    check(f"strategy '{strat}' defined in settings",
          strat in all_strats)

# ══════════════════════════════════════════════════════════════
# 10. Session settings
# ══════════════════════════════════════════════════════════════
section("10. Session Kill Zones")

for sess_name in ["asian", "london_killzone", "london",
                  "newyork_killzone", "newyork"]:
    try:
        s = cfg.session(sess_name)
        check(f"session '{sess_name}' has start/end",
              "start" in s and "end" in s)
    except KeyError as e:
        check(f"session '{sess_name}' has start/end", False, str(e))

# ══════════════════════════════════════════════════════════════
# 11. Symbols
# ══════════════════════════════════════════════════════════════
section("11. Symbols")

all_syms = cfg.symbols()
enabled_syms = cfg.enabled_symbols()

check("symbols() returns dict", isinstance(all_syms, dict))
check("enabled_symbols() returns list", isinstance(enabled_syms, list))
check("9 symbols defined", len(all_syms) == 9, f"found {len(all_syms)}")
check("all 9 symbols enabled", len(enabled_syms) == 9,
      f"enabled: {len(enabled_syms)}")

required_symbols = [
    "EURUSD", "GBPUSD", "USDJPY",
    "XAUUSD", "XAGUSD",
    "BTCUSD", "ETHUSD",
    "US30", "NAS100",
]
for sym in required_symbols:
    check(f"symbol '{sym}' present", sym in all_syms)

# Verify each symbol has required keys
required_keys = [
    "enabled", "category", "pip_size", "pip_value_usd",
    "min_lot", "max_lot", "spread_filter_pips",
    "timeframes", "sessions", "strategies_enabled",
]
for sym in required_symbols:
    if sym not in all_syms:
        continue
    conf = all_syms[sym]
    for key in required_keys:
        check(f"  {sym}.{key} present", key in conf)

# ══════════════════════════════════════════════════════════════
# 12. Helper methods
# ══════════════════════════════════════════════════════════════
section("12. Helper Methods")

check("is_demo() returns bool", isinstance(cfg.is_demo(), bool))
check("is_live() returns bool", isinstance(cfg.is_live(), bool))
check("is_backtest() returns bool", isinstance(cfg.is_backtest(), bool))
check("is_shadow() returns bool", isinstance(cfg.is_shadow(), bool))
check("db_path() returns Path", hasattr(cfg.db_path(), "exists"))

try:
    sym_conf = cfg.symbol("EURUSD")
    check("cfg.symbol('EURUSD') returns dict", isinstance(sym_conf, dict))
except Exception as e:
    check("cfg.symbol('EURUSD') returns dict", False, str(e))

# ══════════════════════════════════════════════════════════════
# 13. Summary method
# ══════════════════════════════════════════════════════════════
section("13. Summary")

try:
    summary = cfg.summary()
    check("cfg.summary() returns string", isinstance(summary, str))
    print(f"\n  Config summary: {CYAN}{summary}{RESET}")
except Exception as e:
    check("cfg.summary() returns string", False, str(e))

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 2 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Config system is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 2 works' to begin Step 3.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 2 INCOMPLETE -- {passed}/{total} passed, {failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run: py -3.11 test_config.py{RESET}")
print(f"{'=' * 55}\n")
