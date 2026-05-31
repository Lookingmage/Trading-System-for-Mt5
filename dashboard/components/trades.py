"""Trade table renderers for the dashboard."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from datetime import datetime


def _style_pnl(val):
    color = "#00cc44" if val > 0 else ("#ff4444" if val < 0 else "white")
    return f"color: {color}"


def render_open_trades_table(open_trades: list[dict]) -> None:
    """Display open trades in a styled table."""
    st.subheader(f"Open Trades ({len(open_trades)})")

    if not open_trades:
        st.info("No open trades right now.")
        return

    df = pd.DataFrame(open_trades)
    cols = ["symbol", "direction", "lot_size", "open_price",
            "stop_loss", "take_profit", "strategy", "mode"]
    show = [c for c in cols if c in df.columns]
    st.dataframe(df[show], use_container_width=True)


def render_closed_trades_table(closed_trades: list[dict]) -> None:
    """Display closed trades with PnL highlighting."""
    st.subheader(f"Closed Trades ({len(closed_trades)})")

    if not closed_trades:
        st.info("No closed trades yet.")
        return

    df = pd.DataFrame(closed_trades)
    df["net_profit"] = pd.to_numeric(df.get("net_profit", 0), errors="coerce").fillna(0)

    cols = ["symbol", "direction", "lot_size", "open_price",
            "close_price", "net_profit", "pips", "strategy",
            "close_reason", "close_time"]
    show = [c for c in cols if c in df.columns]
    df_show = df[show].copy()

    # Format numerics
    for c in ["open_price", "close_price"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].apply(lambda x: f"{x:.5f}" if x else "")
    if "net_profit" in df_show.columns:
        df_show["net_profit"] = df_show["net_profit"].apply(
            lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}"
        )

    st.dataframe(
        df_show.head(50),
        use_container_width=True,
        height=min(400, len(df_show) * 35 + 38),
    )


def render_trade_stats(stats: dict) -> None:
    """Show trade statistics in a compact grid."""
    if not stats or stats.get("total", 0) == 0:
        st.info("No closed trades to analyze.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades",    str(stats.get("total", 0)))
    c2.metric("Win Rate",        f"{stats.get('win_rate', 0):.1f}%")
    c3.metric("Profit Factor",   f"{stats.get('profit_factor', 0):.2f}")
    c4.metric("Net Profit",      f"${stats.get('net_profit', 0):+,.2f}")
