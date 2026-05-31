"""System health display for the dashboard."""
from __future__ import annotations
from pathlib import Path
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent.parent


def render_system_health(
    mt5_connected: bool,
    db_ok: bool,
    cb_active: bool,
    cb_reason: str,
    mode: str,
) -> None:
    """Display system health status indicators."""
    st.subheader("System Status")

    def _badge(ok: bool, label: str, detail: str = "") -> str:
        color = "#00cc44" if ok else "#ff4444"
        icon  = "✓" if ok else "✗"
        d     = f" — {detail}" if detail else ""
        return f'<span style="color:{color}; font-weight:bold">{icon} {label}{d}</span>'

    st.markdown(_badge(mt5_connected, "MT5 Terminal"), unsafe_allow_html=True)
    st.markdown(_badge(db_ok,         "Database"),     unsafe_allow_html=True)
    st.markdown(_badge(not cb_active, "Circuit Breaker",
                       cb_reason if cb_active else "OK"),
                unsafe_allow_html=True)

    st.markdown(f"**Mode:** `{mode.upper()}`")


def render_log_tail(log_file: str, lines: int = 20) -> None:
    """Show the last N lines of a log file."""
    log_path = ROOT / "logs" / log_file
    st.subheader(f"Log: {log_file}")

    if not log_path.exists():
        st.info(f"Log file not found: {log_path}")
        return

    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
        tail = "".join(all_lines[-lines:])
        st.code(tail, language=None)
    except Exception as e:
        st.error(f"Could not read log: {e}")


def render_model_versions(model_versions: list) -> None:
    """Display active model versions in a table."""
    st.subheader("Active Models")

    if not model_versions:
        st.info("No trained models found.")
        return

    import pandas as pd
    rows = []
    for mv in model_versions:
        rows.append({
            "Name":      mv.model_name,
            "Type":      mv.model_type.upper(),
            "Version":   f"v{mv.version}",
            "Accuracy":  f"{(mv.accuracy or 0):.4f}",
            "Trained":   str(mv.train_date)[:16] if mv.train_date else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
