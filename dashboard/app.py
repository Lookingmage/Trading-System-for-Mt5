"""
MT5 Quantum AI — Streamlit Dashboard
======================================
Real-time trading dashboard with auto-refresh every 5 seconds.

How to run:
    cd d:\\Mt5 Ai New\\MT5_Quantum_AI
    streamlit run dashboard/app.py

Then open: http://localhost:8501

The dashboard reads from the local SQLite database.
MT5 does NOT need to be running to view the dashboard,
but balance/equity data requires MT5 connection.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── set up project root on path ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import streamlit as st

# ── page config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="MT5 Quantum AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── imports after path setup ─────────────────────────────────
from config.loader import cfg
from src.database  import db
from src.database  import Trade, ModelVersion, RegimePrediction, DailyStats
from dashboard.components.metrics import render_kpi_row, render_secondary_row
from dashboard.components.charts  import (
    equity_curve_chart, drawdown_chart, daily_pnl_chart,
    win_rate_trend_chart, regime_gauge, strategy_performance_chart,
)
from dashboard.components.trades  import (
    render_open_trades_table, render_closed_trades_table,
    render_trade_stats,
)
from dashboard.components.system  import (
    render_system_health, render_log_tail, render_model_versions,
)


# ════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def load_account_info() -> dict:
    """Try to get live MT5 account info, fallback to DB daily_stats."""
    try:
        from src.connector.mt5_connector import mt5c
        if mt5c.is_connected():
            return mt5c.account_info() or {}
    except Exception:
        pass

    # Fallback: use last known balance from daily_stats
    try:
        with db.session() as sess:
            stats = (
                sess.query(DailyStats)
                .order_by(DailyStats.date.desc())
                .first()
            )
            if stats:
                return {
                    "balance":  stats.closing_balance,
                    "equity":   stats.closing_balance,
                    "currency": "USD",
                    "company":  "Offline mode",
                }
    except Exception:
        pass

    return {"balance": 0.0, "equity": 0.0, "currency": "USD",
            "company": "No data"}


@st.cache_data(ttl=5)
def load_open_trades() -> list[dict]:
    mode = cfg.mode()
    try:
        with db.session() as sess:
            trades = db.get_open_trades(sess, mode=mode)
            return [t.to_dict() for t in trades]
    except Exception:
        return []


@st.cache_data(ttl=5)
def load_closed_trades(limit: int = 100) -> list[dict]:
    try:
        with db.session() as sess:
            trades = db.get_closed_trades(sess, limit=limit)
            return [t.to_dict() for t in trades]
    except Exception:
        return []


@st.cache_data(ttl=5)
def load_trade_stats() -> dict:
    try:
        with db.session() as sess:
            return db.get_trade_stats(sess)
    except Exception:
        return {}


@st.cache_data(ttl=5)
def load_latest_regime() -> dict:
    try:
        symbols = cfg.enabled_symbols()
        regimes = {}
        with db.session() as sess:
            for sym in symbols[:6]:   # show first 6
                r = db.get_latest_regime(sess, sym, "H1")
                if r:
                    regimes[sym] = {
                        "regime":    r.regime,
                        "confidence":r.confidence,
                        "trade_allowed": r.trade_allowed,
                        "bar_time":  str(r.bar_time)[:16],
                    }
        return regimes
    except Exception:
        return {}


@st.cache_data(ttl=30)
def load_model_versions() -> list:
    try:
        with db.session() as sess:
            mvs = (
                sess.query(ModelVersion)
                .filter_by(is_active=True)
                .order_by(ModelVersion.train_date.desc())
                .limit(20)
                .all()
            )
            return mvs
    except Exception:
        return []


@st.cache_data(ttl=10)
def load_daily_stats(days: int = 30) -> list[dict]:
    try:
        with db.session() as sess:
            stats = db.get_daily_stats(sess, mode=cfg.mode(), days=days)
            return [
                {
                    "date":        str(s.date),
                    "net_profit":  s.net_profit,
                    "win_rate":    s.win_rate,
                    "trades":      s.trades_closed,
                    "drawdown":    s.max_drawdown,
                }
                for s in stats
            ]
    except Exception:
        return []


def check_mt5_connected() -> bool:
    try:
        from src.connector.mt5_connector import mt5c
        return mt5c.is_connected()
    except Exception:
        return False


def check_db_ok() -> bool:
    try:
        counts = db.table_row_counts()
        return len(counts) == 9
    except Exception:
        return False


def check_circuit_breaker() -> tuple[bool, str]:
    try:
        from src.risk.manager import risk_manager
        return risk_manager.is_circuit_breaker_active(), risk_manager._cb_reason
    except Exception:
        return False, ""


def get_self_learner_status() -> dict:
    try:
        from src.self_learning.self_learner import self_learner
        return self_learner.status()
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
        st.title("MT5 Quantum AI")
        st.caption(f"v{cfg.get('system.version', '1.0.0')} | {cfg.mode().upper()}")
        st.divider()

        # Quick status
        mt5_ok = check_mt5_connected()
        db_ok  = check_db_ok()
        cb, _  = check_circuit_breaker()

        st.markdown("**System Status**")
        st.markdown(
            f"{'🟢' if mt5_ok else '🔴'} MT5  "
            f"{'🟢' if db_ok  else '🔴'} DB  "
            f"{'🔴 CB!' if cb else '🟢 OK'}"
        )
        st.divider()

        # Symbols
        enabled = cfg.enabled_symbols()
        st.markdown(f"**Symbols ({len(enabled)})**")
        st.code(", ".join(enabled[:6]))
        st.divider()

        # Self-learner status
        sl = get_self_learner_status()
        if sl:
            new_t = sl.get("new_trades_since_retrain", 0)
            min_t = sl.get("min_trades_for_retrain",  20)
            pct   = min(new_t / max(min_t, 1) * 100, 100)
            st.markdown("**Self-Learning**")
            st.progress(pct / 100, text=f"{new_t}/{min_t} trades to retrain")
        st.divider()

        # Auto-refresh
        refresh_secs = int(cfg.get("dashboard.auto_refresh_seconds", 5))
        st.markdown(f"*Auto-refreshes every {refresh_secs}s*")
        if st.button("Refresh Now"):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"Last refresh: {datetime.utcnow().strftime('%H:%M:%S')} UTC")


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════

def tab_overview(account: dict, open_trades: list, closed_trades: list,
                 stats: dict, regimes: dict) -> None:
    """Tab 1 — Overview."""
    # Current regime for primary metric row
    primary_sym    = cfg.enabled_symbols()[0] if cfg.enabled_symbols() else "EURUSD"
    regime_info    = regimes.get(primary_sym, {})
    regime_name    = regime_info.get("regime",     "unknown")
    regime_conf    = regime_info.get("confidence",  0.0)

    # Daily PnL from today's closed trades
    from datetime import date
    today_pnl = sum(
        float(t.get("net_profit", 0) or 0)
        for t in closed_trades
        if t.get("close_time", "")[:10] == str(date.today())
    )

    render_kpi_row(account, today_pnl, len(open_trades),
                   stats.get("win_rate", 0.0))
    st.divider()
    render_secondary_row(
        win_rate=stats.get("win_rate", 0.0),
        profit_factor=stats.get("profit_factor", 0.0),
        regime=regime_name,
        confidence=regime_conf,
        total_closed=stats.get("total", 0),
    )
    st.divider()

    # Charts
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.plotly_chart(
            equity_curve_chart(closed_trades,
                               account.get("balance", 10_000)),
            use_container_width=True,
        )
        st.plotly_chart(
            drawdown_chart(closed_trades, account.get("balance", 10_000)),
            use_container_width=True,
        )

    with col_right:
        st.plotly_chart(
            daily_pnl_chart(closed_trades),
            use_container_width=True,
        )
        st.plotly_chart(
            win_rate_trend_chart(closed_trades),
            use_container_width=True,
        )


def tab_trades(open_trades: list, closed_trades: list, stats: dict) -> None:
    """Tab 2 — Trades."""
    render_trade_stats(stats)
    st.divider()
    render_open_trades_table(open_trades)
    st.divider()
    render_closed_trades_table(closed_trades)


def tab_regime(regimes: dict) -> None:
    """Tab 3 — Market Regime."""
    st.subheader("Current Market Regime")

    if not regimes:
        st.info("No regime predictions yet. Run the regime detector first.")
        return

    # Show gauges in a grid
    cols = st.columns(min(len(regimes), 3))
    for i, (sym, info) in enumerate(regimes.items()):
        with cols[i % 3]:
            st.markdown(f"**{sym}**")
            st.plotly_chart(
                regime_gauge(info["regime"], info["confidence"]),
                use_container_width=True,
            )
            trade_ok = info.get("trade_allowed", False)
            st.markdown(
                f"Trade: {'✅ Allowed' if trade_ok else '🚫 Blocked'} | "
                f"Updated: {info.get('bar_time', '—')}"
            )
            st.divider()


def tab_models(model_versions: list) -> None:
    """Tab 4 — AI Models."""
    st.subheader("Active AI Models")
    render_model_versions(model_versions)

    # Feature importance from AI engine if available
    try:
        from src.ai_engine.decision_engine import ai_engine
        from src.ml_models.ml_pipeline import MLPipeline

        imp = {}
        if ai_engine.has_model:
            # Try to get feature importance from AI engine's model
            import plotly.express as px
            st.subheader("AI Engine Feature Importance")
            st.info("Feature importance available after AI engine training.")
    except Exception:
        pass


def tab_system(mt5_ok: bool, db_ok: bool, cb: bool, cb_reason: str) -> None:
    """Tab 5 — System Health."""
    col_left, col_right = st.columns(2)

    with col_left:
        render_system_health(mt5_ok, db_ok, cb, cb_reason, cfg.mode())
        st.divider()

        # Database row counts
        st.subheader("Database")
        try:
            counts = db.table_row_counts()
            import pandas as pd
            df = pd.DataFrame([
                {"Table": k, "Rows": v} for k, v in counts.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"DB error: {e}")

    with col_right:
        render_log_tail("system.log",  lines=15)
        render_log_tail("trading.log", lines=10)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main() -> None:
    render_sidebar()

    # Load all data
    account       = load_account_info()
    open_trades   = load_open_trades()
    closed_trades = load_closed_trades()
    stats         = load_trade_stats()
    regimes       = load_latest_regime()
    model_vers    = load_model_versions()
    mt5_ok        = check_mt5_connected()
    db_ok         = check_db_ok()
    cb, cb_reason = check_circuit_breaker()

    # Tabs
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Overview",
        "📋 Trades",
        "🌐 Regime",
        "🤖 Models",
        "⚙️ System",
    ])

    with t1: tab_overview(account, open_trades, closed_trades, stats, regimes)
    with t2: tab_trades(open_trades, closed_trades, stats)
    with t3: tab_regime(regimes)
    with t4: tab_models(model_vers)
    with t5: tab_system(mt5_ok, db_ok, cb, cb_reason)

    # Auto-refresh
    refresh_secs = int(cfg.get("dashboard.auto_refresh_seconds", 5))
    time.sleep(refresh_secs)
    st.rerun()


if __name__ == "__main__":
    main()
