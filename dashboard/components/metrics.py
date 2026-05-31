"""Dashboard metric cards and KPI helpers."""
from __future__ import annotations
import streamlit as st
from typing import Optional


def metric_card(label: str, value: str, delta: Optional[str] = None,
                delta_color: str = "normal") -> None:
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def format_currency(amount: float, symbol: str = "$") -> str:
    sign = "+" if amount > 0 else ""
    return f"{symbol}{sign}{amount:,.2f}"


def format_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def color_pnl(val: float) -> str:
    """Return HTML-colored PnL string."""
    color = "#00cc44" if val >= 0 else "#ff4444"
    sign  = "+" if val >= 0 else ""
    return f'<span style="color:{color}">{sign}{val:.2f}</span>'


def render_kpi_row(account: dict, daily_pnl: float, open_trades: int,
                   win_rate: float) -> None:
    """Render the top KPI row (4 columns)."""
    c1, c2, c3, c4 = st.columns(4)

    balance = account.get("balance", 0.0)
    equity  = account.get("equity",  0.0)
    dd_pct  = (balance - equity) / balance * 100 if balance > 0 else 0.0

    with c1:
        metric_card("Balance", f"${balance:,.2f}",
                    delta=format_currency(equity - balance),
                    delta_color="normal")
    with c2:
        metric_card("Equity", f"${equity:,.2f}",
                    delta=f"{format_pct(-dd_pct)} DD" if dd_pct > 0 else None,
                    delta_color="inverse" if dd_pct > 0 else "normal")
    with c3:
        metric_card("Daily PnL", format_currency(daily_pnl),
                    delta_color="normal" if daily_pnl >= 0 else "inverse")
    with c4:
        metric_card("Open Trades", str(open_trades))


def render_secondary_row(win_rate: float, profit_factor: float,
                          regime: str, confidence: float,
                          total_closed: int) -> None:
    """Render the secondary metric row."""
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Win Rate",     f"{win_rate:.1f}%")
    with c2:
        metric_card("Profit Factor", f"{profit_factor:.2f}")
    with c3:
        metric_card("Closed Trades", str(total_closed))
    with c4:
        metric_card("Regime", regime.replace("_", " ").title())
    with c5:
        metric_card("Regime Conf.", f"{confidence:.0%}")
