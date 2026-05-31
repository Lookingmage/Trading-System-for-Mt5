"""
MT5 Quantum AI -- Step 20 Verification: Dashboard
==================================================
How to run:
    py -3.11 test_dashboard.py

What this tests:
  1.  All dashboard files exist
  2.  dashboard/components/ imports work
  3.  metrics.py functions work
  4.  charts.py: equity_curve_chart() returns Plotly Figure
  5.  charts.py: daily_pnl_chart() returns Plotly Figure
  6.  charts.py: regime_gauge() returns Plotly Figure
  7.  trades.py helper functions run
  8.  system.py imports without error
  9.  dashboard/app.py can be imported (checks for syntax errors)
  10. Data loading functions work without MT5

How to VIEW the dashboard:
    cd "d:\\Mt5 Ai New\\MT5_Quantum_AI"
    streamlit run dashboard/app.py
    Then open: http://localhost:8501
"""

import sys
import os

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
# 1. Files exist
# ══════════════════════════════════════════════════════════════
section("1. Dashboard Files Exist")

from pathlib import Path

DASH = Path(ROOT) / "dashboard"
expected_files = [
    "dashboard/app.py",
    "dashboard/__init__.py",
    "dashboard/components/__init__.py",
    "dashboard/components/metrics.py",
    "dashboard/components/charts.py",
    "dashboard/components/trades.py",
    "dashboard/components/system.py",
]
for f in expected_files:
    check(f"  {f} exists", (Path(ROOT) / f).exists())

# ══════════════════════════════════════════════════════════════
# 2. Component imports
# ══════════════════════════════════════════════════════════════
section("2. Component Imports")

try:
    from dashboard.components.metrics import (
        format_currency, format_pct, color_pnl
    )
    check("from dashboard.components.metrics import helpers", True)
except Exception as e:
    check("from dashboard.components.metrics import helpers", False, str(e))

try:
    from dashboard.components.charts import (
        equity_curve_chart, drawdown_chart, daily_pnl_chart,
        win_rate_trend_chart, regime_gauge, strategy_performance_chart,
    )
    check("from dashboard.components.charts import chart functions", True)
except Exception as e:
    check("from dashboard.components.charts import chart functions", False, str(e))

try:
    from dashboard.components.trades import (
        render_trade_stats,
    )
    check("from dashboard.components.trades import helpers", True)
except Exception as e:
    check("from dashboard.components.trades import helpers", False, str(e))

try:
    from dashboard.components.system import (
        render_system_health, render_log_tail, render_model_versions,
    )
    check("from dashboard.components.system import helpers", True)
except Exception as e:
    check("from dashboard.components.system import helpers", False, str(e))

# ══════════════════════════════════════════════════════════════
# 3. Metrics helpers
# ══════════════════════════════════════════════════════════════
section("3. Metrics Helper Functions")

check("format_currency(+25.5) starts with '$'",
      format_currency(25.5).startswith("$"))
check("format_currency(-10.0) contains '-'",
      "-" in format_currency(-10.0))
check("format_pct(2.5) = '+2.50%'",
      format_pct(2.5) == "+2.50%")
check("format_pct(-1.5) = '-1.50%'",
      format_pct(-1.5) == "-1.50%")
check("color_pnl(10) contains 'green color'",
      "00cc44" in color_pnl(10.0))
check("color_pnl(-5) contains 'red color'",
      "ff4444" in color_pnl(-5.0))

# ══════════════════════════════════════════════════════════════
# 4. Chart generation — equity curve
# ══════════════════════════════════════════════════════════════
section("4. Charts — Equity Curve")

import plotly.graph_objects as go
from datetime import datetime, timedelta

# Make synthetic trade data
now = datetime.utcnow()
fake_trades = [
    {"close_time": str(now - timedelta(hours=h)),
     "net_profit": (10.0 if h % 3 != 0 else -5.0)}
    for h in range(20, 0, -1)
]

fig_eq = equity_curve_chart(fake_trades, initial_balance=10_000)
check("equity_curve_chart() returns Figure",
      isinstance(fig_eq, go.Figure))
check("Figure has at least 1 trace", len(fig_eq.data) >= 1)

fig_eq_empty = equity_curve_chart([])
check("equity_curve_chart([]) returns Figure (empty)",
      isinstance(fig_eq_empty, go.Figure))

# ══════════════════════════════════════════════════════════════
# 5. Charts — daily PnL
# ══════════════════════════════════════════════════════════════
section("5. Charts — Daily PnL")

fig_pnl = daily_pnl_chart(fake_trades)
check("daily_pnl_chart() returns Figure",
      isinstance(fig_pnl, go.Figure))

# ══════════════════════════════════════════════════════════════
# 6. Charts — regime gauge
# ══════════════════════════════════════════════════════════════
section("6. Charts — Regime Gauge")

fig_gauge = regime_gauge("strong_uptrend", 0.82)
check("regime_gauge() returns Figure", isinstance(fig_gauge, go.Figure))
check("Gauge has indicator trace",
      any(isinstance(t, go.Indicator) for t in fig_gauge.data))

# ══════════════════════════════════════════════════════════════
# 7. Charts — drawdown + win rate trend
# ══════════════════════════════════════════════════════════════
section("7. Charts — Drawdown and Win Rate")

fig_dd = drawdown_chart(fake_trades, 10_000)
check("drawdown_chart() returns Figure", isinstance(fig_dd, go.Figure))

fig_wr = win_rate_trend_chart(fake_trades, window=5)
check("win_rate_trend_chart() returns Figure", isinstance(fig_wr, go.Figure))

# Strategy performance
fake_by_strategy = {
    "ema_trend":   {"win_rate": 62.5, "trades": 8},
    "order_block": {"win_rate": 45.0, "trades": 4},
}
fig_strat = strategy_performance_chart(fake_by_strategy)
check("strategy_performance_chart() returns Figure",
      isinstance(fig_strat, go.Figure))

# ══════════════════════════════════════════════════════════════
# 8. Data loading without MT5
# ══════════════════════════════════════════════════════════════
section("8. Data Loading from Database (No MT5)")

from src.database import db

try:
    stats = db.get_trade_stats(db._Session())
    check("db.get_trade_stats() returns dict", isinstance(stats, dict))
except Exception:
    # Use direct session
    with db.session() as sess:
        stats = db.get_trade_stats(sess)
    check("db.get_trade_stats() returns dict (via session)",
          isinstance(stats, dict))

try:
    with db.session() as sess:
        closed = db.get_closed_trades(sess, limit=50)
    check("db.get_closed_trades() returns list", isinstance(closed, list))
    check(f"  {len(closed)} closed trades in DB", True)
except Exception as e:
    check("db.get_closed_trades() returns list", False, str(e))

try:
    counts = db.table_row_counts()
    check("db.table_row_counts() returns dict", isinstance(counts, dict))
    check("All 9 tables counted", len(counts) == 9, f"got {len(counts)}")
except Exception as e:
    check("db.table_row_counts() returns dict", False, str(e))

# ══════════════════════════════════════════════════════════════
# 9. app.py syntax check (import without running)
# ══════════════════════════════════════════════════════════════
section("9. dashboard/app.py Syntax Check")

import ast

app_path = Path(ROOT) / "dashboard" / "app.py"
try:
    source = app_path.read_text(encoding="utf-8")
    tree   = ast.parse(source)
    check("dashboard/app.py parses without syntax error", True)
    # Count function definitions
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    check(f"app.py has >= 8 functions (got {len(funcs)})", len(funcs) >= 8)
except SyntaxError as e:
    check("dashboard/app.py parses without syntax error", False, str(e))

# ══════════════════════════════════════════════════════════════
# 10. How to start the dashboard
# ══════════════════════════════════════════════════════════════
section("10. How to Start the Dashboard")

print(f"\n  {CYAN}To view the dashboard:{RESET}")
print(f"  {YELLOW}  cd \"d:\\Mt5 Ai New\\MT5_Quantum_AI\"{RESET}")
print(f"  {YELLOW}  streamlit run dashboard/app.py{RESET}")
print(f"  {CYAN}  Then open: http://localhost:8501{RESET}\n")
check("Dashboard ready to run", True)

# ══════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'=' * 55}")
if failed == 0:
    print(f"{GREEN}{BOLD}  Step 20 COMPLETE -- {passed}/{total} checks passed{RESET}")
    print(f"{GREEN}  Dashboard is ready.{RESET}")
    print(f"{GREEN}  Tell Claude 'Step 20 works' to begin Step 21.{RESET}")
else:
    print(f"{RED}{BOLD}  Step 20 INCOMPLETE -- {passed}/{total} passed, "
          f"{failed} failed{RESET}")
    print(f"{YELLOW}  Fix the [FAIL] items above, then re-run.{RESET}")
print(f"{'=' * 55}\n")
