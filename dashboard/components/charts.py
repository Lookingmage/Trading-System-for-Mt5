"""Plotly chart generators for the dashboard."""
from __future__ import annotations
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ── Equity Curve ──────────────────────────────────────────────

def equity_curve_chart(
    trades: list[dict],
    initial_balance: float = 10_000.0,
) -> go.Figure:
    """Build an equity curve from a list of closed trade dicts."""
    if not trades:
        fig = go.Figure()
        fig.update_layout(title="Equity Curve (no trades yet)",
                          height=300, template="plotly_dark")
        return fig

    df = pd.DataFrame(trades)
    df["net_profit"] = pd.to_numeric(df.get("net_profit", 0), errors="coerce").fillna(0)
    df = df.sort_values("close_time")
    df["equity"] = initial_balance + df["net_profit"].cumsum()
    df["drawdown"] = df["equity"].cummax() - df["equity"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["close_time"], y=df["equity"],
        mode="lines", name="Equity",
        line=dict(color="#00cc44", width=2),
        fill="tozeroy", fillcolor="rgba(0,204,68,0.1)",
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        height=320,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── Drawdown Chart ────────────────────────────────────────────

def drawdown_chart(trades: list[dict], initial_balance: float = 10_000.0) -> go.Figure:
    """Build a drawdown chart from trade list."""
    fig = go.Figure()

    if trades:
        df = pd.DataFrame(trades)
        df["net_profit"] = pd.to_numeric(df.get("net_profit", 0), errors="coerce").fillna(0)
        df = df.sort_values("close_time")
        equity = initial_balance + df["net_profit"].cumsum()
        peak   = equity.cummax()
        dd_pct = (peak - equity) / peak * 100

        fig.add_trace(go.Scatter(
            x=df["close_time"], y=-dd_pct,
            mode="lines", name="Drawdown %",
            line=dict(color="#ff4444", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,68,68,0.15)",
        ))

    fig.update_layout(
        title="Drawdown (%)",
        height=220, template="plotly_dark",
        yaxis_title="Drawdown %",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── Daily PnL Bar Chart ───────────────────────────────────────

def daily_pnl_chart(trades: list[dict]) -> go.Figure:
    """Aggregate trades by day and show daily PnL bar chart."""
    fig = go.Figure()

    if trades:
        df = pd.DataFrame(trades)
        df["net_profit"] = pd.to_numeric(df.get("net_profit", 0), errors="coerce").fillna(0)
        df["date"] = pd.to_datetime(df["close_time"]).dt.date
        daily = df.groupby("date")["net_profit"].sum().reset_index()

        colors = ["#00cc44" if v >= 0 else "#ff4444"
                  for v in daily["net_profit"]]

        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["net_profit"],
            marker_color=colors, name="Daily PnL",
        ))

    fig.update_layout(
        title="Daily PnL ($)",
        height=250, template="plotly_dark",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── Win Rate Trend ────────────────────────────────────────────

def win_rate_trend_chart(trades: list[dict], window: int = 10) -> go.Figure:
    """Rolling win rate over the last `window` trades."""
    fig = go.Figure()

    if len(trades) >= 5:
        df = pd.DataFrame(trades)
        df["net_profit"] = pd.to_numeric(df.get("net_profit", 0), errors="coerce").fillna(0)
        df = df.sort_values("close_time")
        df["win"] = (df["net_profit"] > 0).astype(int)
        df["rolling_wr"] = df["win"].rolling(window, min_periods=1).mean() * 100

        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["rolling_wr"],
            mode="lines", name="Win Rate %",
            line=dict(color="#4488ff", width=2),
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray",
                       annotation_text="50%")

    fig.update_layout(
        title=f"Rolling Win Rate (last {window} trades)",
        yaxis_title="Win Rate %", height=220,
        template="plotly_dark", showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── Regime Gauge ──────────────────────────────────────────────

def regime_gauge(regime: str, confidence: float) -> go.Figure:
    """Display a gauge for regime confidence."""
    color = "#00cc44" if confidence >= 0.65 else "#ffaa00"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": regime.replace("_", " ").title(), "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#333"},
                {"range": [40, 65], "color": "#444"},
                {"range": [65, 100],"color": "#222"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 65,
            },
        },
        number={"suffix": "%", "font": {"size": 24}},
    ))
    fig.update_layout(
        height=200, template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# ── Strategy Performance Bar ──────────────────────────────────

def strategy_performance_chart(by_strategy: dict) -> go.Figure:
    """Bar chart of win rate by strategy."""
    fig = go.Figure()

    if by_strategy:
        names  = list(by_strategy.keys())
        wr     = [by_strategy[n].get("win_rate", 0) for n in names]
        colors = ["#00cc44" if w >= 50 else "#ff4444" for w in wr]

        fig.add_trace(go.Bar(
            x=names, y=wr,
            marker_color=colors, name="Win Rate %",
            text=[f"{w:.1f}%" for w in wr],
            textposition="auto",
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Win Rate by Strategy",
        yaxis_title="Win Rate %", height=250,
        template="plotly_dark", showlegend=False,
        margin=dict(l=40, r=20, t=40, b=60),
    )
    return fig
