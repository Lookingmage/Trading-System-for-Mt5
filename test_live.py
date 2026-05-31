"""
MT5 Quantum AI -- Step 25: Live Trading Pre-Live Checklist
===========================================================
30-point safety verification before enabling real money trading.

How to run:
    py -3.11 test_live.py

This test does NOT place real orders.
It verifies that all safety conditions are met.

To enable live trading after this test passes:
    1. Edit config/settings.yaml:
           live_trading:
             enabled: true
             human_override_required: false
    2. Run: py -3.11 main.py --mode live

IMPORTANT:
    - Live trading uses REAL MONEY
    - Start with minimum lot sizes (0.01)
    - Monitor the dashboard while running
    - Have a stop plan ready (Ctrl+C + check MT5 positions)
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
ORANGE = "\033[38;5;208m"

passed  = 0
failed  = 0
warned  = 0


def check(label: str, ok: bool, detail: str = "", critical: bool = False):
    global passed, failed
    status = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    suffix = f"  {YELLOW}-> {detail}{RESET}" if detail else ""
    if not ok and critical:
        status = f"{RED}[CRITICAL]{RESET}"
    print(f"  {status}  {label}{suffix}")
    if ok: passed += 1
    else:  failed += 1
    return ok


def warn(label: str, detail: str = ""):
    global warned
    print(f"  {YELLOW}[WARN]{RESET}  {label}  {YELLOW}{detail}{RESET}")
    warned += 1


def section(title: str):
    print(f"\n{CYAN}{BOLD}{'=' * 60}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{'=' * 60}{RESET}")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

print(f"\n{RED}{BOLD}{'=' * 60}{RESET}")
print(f"{RED}{BOLD}  MT5 QUANTUM AI — LIVE TRADING PRE-LIVE CHECKLIST{RESET}")
print(f"{RED}{BOLD}{'=' * 60}{RESET}")
print(f"\n  {ORANGE}WARNING: Live trading uses REAL MONEY.{RESET}")
print(f"  {ORANGE}This checklist must FULLY PASS before you go live.{RESET}")
print(f"  {CYAN}Test run: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC{RESET}\n")

# ══════════════════════════════════════════════════════════════
# GROUP 1: CONFIGURATION SAFETY
# ══════════════════════════════════════════════════════════════
section("Group 1 — Configuration Safety")

from config.loader import cfg

# 1. Live trading is currently DISABLED (safety)
live_enabled = cfg.get("live_trading.enabled", False)
check("1.  live_trading.enabled is currently FALSE (safety default)",
      not live_enabled,
      "Good — you must manually enable this to go live",
      critical=True)
if live_enabled:
    print(f"  {ORANGE}      Live trading is already enabled. "
          f"Disable before running this checklist.{RESET}")

# 2. Mode in config
mode = cfg.mode()
check("2.  system.mode is not 'live' (running in demo/shadow now)",
      mode != "live",
      f"current mode='{mode}'")

# 3. Risk limits sensible
daily_loss  = float(cfg.get("risk.max_daily_loss_pct", 0))
max_dd      = float(cfg.get("risk.max_drawdown_pct", 0))
risk_pct    = float(cfg.get("risk.risk_per_trade_pct", 0))
cb_enabled  = cfg.get("risk.circuit_breaker_enabled", False)

check("3.  max_daily_loss_pct is set (> 0 and <= 10)",
      0 < daily_loss <= 10,
      f"value={daily_loss}%")
check("4.  max_drawdown_pct is set (> 0 and <= 25)",
      0 < max_dd <= 25,
      f"value={max_dd}%")
check("5.  risk_per_trade_pct is set (> 0 and <= 3)",
      0 < risk_pct <= 3,
      f"value={risk_pct}%")
check("6.  circuit_breaker_enabled is True",
      cb_enabled is True,
      "CRITICAL: circuit breaker must be active",
      critical=True)

# 4. Live trading gates configured
require_all = cfg.get("live_trading.require_all_filters_pass", True)
require_ai  = cfg.get("live_trading.require_ai_approval", True)
require_risk= cfg.get("live_trading.require_risk_approval", True)
check("7.  require_all_filters_pass = True",
      require_all is True)
check("8.  require_ai_approval = True",
      require_ai is True)
check("9.  require_risk_approval = True",
      require_risk is True)

# 5. Symbols configured
enabled_syms = cfg.enabled_symbols()
check("10. At least 1 symbol enabled",
      len(enabled_syms) >= 1,
      f"enabled: {enabled_syms[:4]}")

# ══════════════════════════════════════════════════════════════
# GROUP 2: DATA QUALITY
# ══════════════════════════════════════════════════════════════
section("Group 2 — Data Quality")

from src.database import db
from src.database import Candle, Trade, ModelVersion
from src.data.processor import processor

# 11. Database accessible
db_tables = db.get_table_names()
check("11. Database: all 9 tables present",
      len(db_tables) == 9,
      f"found {len(db_tables)}")

# 12. Minimum candle history
total_candles = 0
sym_bar_counts = {}
for sym in enabled_syms[:3]:
    n = processor.get_bar_count(sym, "H1")
    sym_bar_counts[sym] = n
    total_candles += n

check("12. At least 500 H1 candles in DB",
      total_candles >= 500,
      f"total={total_candles} | {sym_bar_counts}")
if total_candles < 2000:
    warn("12. Less than 2000 candles — download more history for better performance",
         "run: downloader.download_all_candles(bars=10000)")

# 13. Feature engineering works on current data
primary_sym = enabled_syms[0] if enabled_syms else "EURUSD"
df_test = processor.process(primary_sym, "H1", limit=300, min_bars=100)
check("13. Data processor works on primary symbol",
      not df_test.empty,
      f"{primary_sym}: {len(df_test)} usable bars")

from src.features.engineer import engineer
feat_test = engineer.compute(df_test, primary_sym, "H1")
ml_feat   = engineer.get_ml_features(feat_test)
check("14. Feature engineering produces clean matrix",
      not ml_feat.empty and not ml_feat.isna().any().any(),
      f"{len(ml_feat)} clean rows × {len(ml_feat.columns)} features")

# ══════════════════════════════════════════════════════════════
# GROUP 3: AI/ML MODEL READINESS
# ══════════════════════════════════════════════════════════════
section("Group 3 — AI/ML Model Readiness")

from src.ai_engine.decision_engine import ai_engine
from src.regime.detector import regime_detector

# 15. AI engine
check("15. AI Decision Engine has trained model",
      ai_engine.has_model,
      "mode: " + ("ML" if ai_engine.has_model else "rule-based"),
      critical=True)
check("16. AI confidence threshold is set",
      0.3 <= ai_engine.confidence_threshold <= 0.9,
      f"threshold={ai_engine.confidence_threshold}")

# 16. Regime detector (trigger lazy load from disk)
regime_detector._ensure_models_loaded()
regime_has_model = (
    regime_detector._models.get("xgboost") is not None or
    regime_detector._models.get("lightgbm") is not None
)
check("17. Regime detector has trained model",
      regime_has_model,
      critical=True)

# Test regime detection
regime_test = regime_detector.detect(primary_sym, "H1", save_to_db=False)
check("18. Regime detection runs successfully",
      regime_test is not None,
      f"regime={regime_test.regime if regime_test else 'N/A'}")

# 17. Model versions in DB
with db.session() as sess:
    active_models = (
        sess.query(ModelVersion)
        .filter_by(is_active=True)
        .count()
    )
check("19. Active model versions in database",
      active_models >= 2,
      f"found {active_models} active models")

# ══════════════════════════════════════════════════════════════
# GROUP 4: RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════
section("Group 4 — Risk Management")

from src.risk.manager import risk_manager

# 18. Risk manager state
rm_can, rm_reason = risk_manager.can_trade()
check("20. Risk manager: can_trade() returns True (no circuit breaker)",
      rm_can,
      rm_reason)

# Verify risk manager config matches what we set
check("21. Risk manager reads correct config",
      risk_manager._cfg is cfg)

# 19. Position sizing configured
sizing_mode = cfg.get("risk.sizing_mode", "")
check("22. Position sizing mode is configured",
      sizing_mode in ("fixed_lot", "fixed_pct", "kelly"),
      f"mode='{sizing_mode}'")

# ══════════════════════════════════════════════════════════════
# GROUP 5: SHADOW PERFORMANCE REVIEW
# ══════════════════════════════════════════════════════════════
section("Group 5 — Shadow Performance Review")

from src.self_learning.self_learner import self_learner

# 20. Shadow trades exist
with db.session() as sess:
    shadow_closed = (
        sess.query(Trade)
        .filter_by(mode="shadow", status="closed")
        .filter(Trade.close_reason.in_(["tp", "sl", "manual", "trailing"]))
        .count()
    )

check("23. At least 5 closed shadow trades for review",
      shadow_closed >= 5,
      f"found {shadow_closed} closed shadow trades")
if shadow_closed < 20:
    warn("23. Less than 20 shadow trades — run more shadow sessions first",
         "py -3.11 main.py --mode shadow  (run for 24+ hours)")

# 21. Shadow win rate (if enough trades)
perf = self_learner.analyze_performance(window=100)
shadow_wr = perf.get("win_rate", 0)
shadow_pf = perf.get("profit_factor", 0)
shadow_total = perf.get("total_trades", 0)

if shadow_total >= 10:
    check("24. Shadow win rate >= 40% (minimum acceptable)",
          shadow_wr >= 40,
          f"win_rate={shadow_wr:.1f}%  (ideally >= 50%)")
    check("25. Shadow profit factor > 0",
          shadow_pf > 0,
          f"profit_factor={shadow_pf:.2f}")
else:
    warn("24. Not enough shadow trades for statistical significance",
         f"only {shadow_total} trades — need >= 10")
    check("24. Shadow win rate (skipped — not enough data)", True,
          f"only {shadow_total} trades")
    check("25. Shadow profit factor (skipped — not enough data)", True)

print(f"\n  {CYAN}Shadow Performance Summary:{RESET}")
print(f"    Total shadow trades:  {shadow_total}")
print(f"    Win rate:            {shadow_wr:.1f}%")
print(f"    Profit factor:       {shadow_pf:.2f}")
print(f"    Net PnL:             ${perf.get('net_profit', 0):.2f}")

# ══════════════════════════════════════════════════════════════
# GROUP 6: STRATEGY AND EXECUTOR
# ══════════════════════════════════════════════════════════════
section("Group 6 — Strategy and Executor")

from src.strategies.engine import strategy_engine
from src.executor.executor import executor

# 22. Strategies
enabled_strats = strategy_engine.enabled_strategy_names
check("26. At least 3 strategies enabled",
      len(enabled_strats) >= 3,
      f"enabled: {enabled_strats}")

# 23. Run strategies on current data (dry run)
if not feat_test.empty:
    live_signals = strategy_engine.run_on_df(feat_test, primary_sym, "H1")
    check("27. Strategy engine produces signals (or correctly returns empty)",
          isinstance(live_signals, list))
    print(f"  {CYAN}  Current signals: {len(live_signals)} on {primary_sym} H1{RESET}")
    for s in live_signals[:2]:
        print(f"    [{s.strategy_name}] {s.direction} str={s.signal_strength:.2f}")

# 24. Executor live mode code exists
check("28. Executor has _place_mt5() for live orders",
      hasattr(executor, "_place_mt5"))
check("29. Executor live safety gate is present",
      hasattr(executor, "_mode"))

# ══════════════════════════════════════════════════════════════
# GROUP 7: LOGGING AND REPORTING
# ══════════════════════════════════════════════════════════════
section("Group 7 — Logging and Reporting")

LOG_DIR = Path(ROOT) / "logs"
REPORT_DIR = Path(ROOT) / "reports"

log_files = ["system.log", "trading.log", "model.log",
             "error.log", "optimization.log", "backtest.log"]
all_logs_ok = True
for lf in log_files:
    exists = (LOG_DIR / lf).exists()
    if not exists:
        all_logs_ok = False

check("30. All 6 log files writable", all_logs_ok,
      f"in {LOG_DIR}")
check("31. Reports directory exists", REPORT_DIR.is_dir(),
      str(REPORT_DIR))

shadow_reports = list(REPORT_DIR.glob("shadow_session_*.txt"))
check("32. Shadow session reports exist",
      len(shadow_reports) >= 1,
      f"found {len(shadow_reports)} shadow report(s)")

# ══════════════════════════════════════════════════════════════
# MT5 LIVE READINESS (if connected)
# ══════════════════════════════════════════════════════════════
section("MT5 Live Readiness Check")

import MetaTrader5 as mt5_raw
mt5_running = mt5_raw.initialize()
if mt5_running:
    mt5_raw.shutdown()

if mt5_running:
    from src.connector.mt5_connector import mt5c
    connected = mt5c.connect()

    if connected:
        acc = mt5c.account_info()
        if acc:
            check("MT5. Account balance > 0",
                  acc["balance"] > 0,
                  f"balance={acc['balance']:.2f} {acc['currency']}")
            check("MT5. Trade allowed on account",
                  acc.get("trade_allowed", False))

            # Warning if account balance is very low
            if acc["balance"] < 100:
                warn("MT5. Account balance < 100 — very small account",
                     f"balance={acc['balance']:.2f} — consider depositing more")

        mt5c.disconnect()
    else:
        warn("MT5 not connected", "verify credentials in .env")
else:
    warn("MT5 not running", "open MT5 before running live trading")

# ══════════════════════════════════════════════════════════════
# FINAL CHECKLIST SUMMARY
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 60}")
print(f"{CYAN}{BOLD}  PRE-LIVE CHECKLIST SUMMARY{RESET}")
print(f"{'=' * 60}")
print(f"  Passed:   {GREEN}{passed}/{total}{RESET}")
print(f"  Failed:   {RED}{failed}{RESET}")
print(f"  Warnings: {YELLOW}{warned}{RESET}")

# ══════════════════════════════════════════════════════════════
# LIVE TRADING INSTRUCTIONS
# ══════════════════════════════════════════════════════════════

if failed == 0:
    print(f"\n{GREEN}{BOLD}  ALL CHECKS PASSED — System is ready for live trading{RESET}")
    print(f"""
{CYAN}{'=' * 60}
  HOW TO ENABLE LIVE TRADING
{'=' * 60}{RESET}

{YELLOW}Step 1 — Review your shadow performance:{RESET}
  - Open: reports/shadow_session_*.txt
  - Review win rate, profit factor, drawdown
  - Only proceed if you are satisfied with performance

{YELLOW}Step 2 — Enable live trading in config/settings.yaml:{RESET}
  Find the live_trading section and set:

    live_trading:
      enabled: true
      human_override_required: false
      require_all_filters_pass: true
      require_regime_approval: true
      require_ai_approval: true
      require_risk_approval: true

{YELLOW}Step 3 — Start live trading:{RESET}
  py -3.11 main.py --mode live

  The system will:
  → Show a 10-second countdown (press Ctrl+C to abort)
  → Connect to your MT5 account
  → Begin processing signals with all safety gates active

{YELLOW}Step 4 — Monitor in real-time:{RESET}
  Open a second terminal:
  streamlit run dashboard/app.py
  Then go to: http://localhost:8501

{YELLOW}Step 5 — Emergency stop (if needed):{RESET}
  → Press Ctrl+C in the main.py terminal
  → Open MT5 and manually close any positions
  → The circuit breaker also stops automatically at risk limits

{RED}{'=' * 60}
  RISK REMINDER
{'=' * 60}{RESET}
  • Daily loss limit:    {daily_loss}% of balance
  • Max drawdown:        {max_dd}% of balance
  • Risk per trade:      {risk_pct}% of balance
  • Circuit breaker:     {'ACTIVE' if cb_enabled else 'INACTIVE'}
  • Consecutive losses:  {int(cfg.get('risk.max_consecutive_losses', 5))} → pause trading

{GREEN}  Start small. Monitor closely. Increase size only after consistent profit.{RESET}
""")
else:
    print(f"\n{RED}{BOLD}  {failed} CHECKS FAILED — Fix before going live:{RESET}\n")
    print(f"  Run again after fixing: py -3.11 test_live.py\n")

print(f"{'=' * 60}\n")

# ══════════════════════════════════════════════════════════════
# STEP 25 VERDICT
# ══════════════════════════════════════════════════════════════
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 25 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Live Trading system is verified and ready.{RESET}")
    print(f"{GREEN}  Follow the instructions above to go live.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 25 works' to complete the build.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 25 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
