"""
Config Loader
=============
Loads settings.yaml and symbols.yaml, merges .env overrides,
validates required fields, and exposes a single ConfigLoader
singleton that every other module imports.

Usage in any module:
    from config.loader import cfg
    mode  = cfg.get("system.mode")
    syms  = cfg.symbols()
    risk  = cfg.get("risk.risk_per_trade_pct")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

# ── locate project root ──────────────────────────────────────
# This file lives at <root>/config/loader.py → root is one level up.
ROOT = Path(__file__).resolve().parent.parent


# ── load .env manually (no external dep needed here) ─────────
def _load_env() -> None:
    """Read .env and inject values into os.environ (simple parser)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # don't overwrite values already set by the OS environment
            if key and key not in os.environ:
                os.environ[key] = value


_load_env()


# ────────────────────────────────────────────────────────────
class ConfigLoader:
    """
    Singleton configuration manager.

    - Loads config/settings.yaml and config/symbols.yaml once.
    - Provides dot-notation access: cfg.get("risk.max_daily_loss_pct")
    - Exposes helper methods for common lookups.
    """

    _instance: ConfigLoader | None = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    # ── init ────────────────────────────────────────────────
    def __init__(self) -> None:
        if self._loaded:
            return
        self._settings: dict = {}
        self._symbols: dict = {}
        self._load_settings()
        self._load_symbols()
        self._validate()
        self._loaded = True

    # ── internal loaders ────────────────────────────────────
    def _load_settings(self) -> None:
        path = ROOT / "config" / "settings.yaml"
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found: {path}")
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError("settings.yaml must be a YAML mapping (dict).")
        self._settings = data

    def _load_symbols(self) -> None:
        path = ROOT / "config" / "symbols.yaml"
        if not path.exists():
            raise FileNotFoundError(f"symbols.yaml not found: {path}")
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict) or "symbols" not in data:
            raise ValueError("symbols.yaml must have a top-level 'symbols' key.")
        self._symbols = data["symbols"]

    # ── validation ──────────────────────────────────────────
    def _validate(self) -> None:
        required_sections = [
            "system", "mt5", "database", "logging", "risk",
            "position", "backtest", "regime", "ai_engine",
            "strategies", "sessions", "live_trading",
        ]
        missing = [s for s in required_sections if s not in self._settings]
        if missing:
            raise KeyError(
                f"settings.yaml is missing required sections: {missing}"
            )

        # Validate mode value
        valid_modes = {"backtest", "demo", "shadow", "live"}
        mode = self._settings.get("system", {}).get("mode", "")
        if mode not in valid_modes:
            raise ValueError(
                f"system.mode must be one of {valid_modes}, got '{mode}'"
            )

        # Validate at least one symbol is enabled
        enabled = [k for k, v in self._symbols.items() if v.get("enabled", False)]
        if not enabled:
            raise ValueError("symbols.yaml: no symbols have enabled: true")

    # ── public API ──────────────────────────────────────────
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Dot-notation access into settings.

        Examples:
            cfg.get("system.mode")               → "demo"
            cfg.get("risk.risk_per_trade_pct")   → 1.0
            cfg.get("strategies.ema_trend")      → {"enabled": True, ...}
        """
        keys = key_path.split(".")
        node: Any = self._settings
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def symbols(self) -> dict:
        """Return the full symbols dict."""
        return self._symbols

    def enabled_symbols(self) -> list[str]:
        """Return list of symbol names where enabled: true."""
        return [name for name, conf in self._symbols.items()
                if conf.get("enabled", False)]

    def symbol(self, name: str) -> dict:
        """Return config for one symbol. Raises KeyError if not found."""
        if name not in self._symbols:
            raise KeyError(f"Symbol '{name}' not in symbols.yaml")
        return self._symbols[name]

    def mode(self) -> str:
        """Return current trading mode: backtest | demo | shadow | live."""
        return self.get("system.mode", "demo")

    def is_live(self) -> bool:
        return self.mode() == "live"

    def is_demo(self) -> bool:
        return self.mode() == "demo"

    def is_backtest(self) -> bool:
        return self.mode() == "backtest"

    def is_shadow(self) -> bool:
        return self.mode() == "shadow"

    def mt5_login(self) -> int:
        """MT5 account login from environment variable."""
        val = os.environ.get("MT5_LOGIN", "0")
        try:
            return int(val)
        except ValueError:
            return 0

    def mt5_password(self) -> str:
        """MT5 account password from environment variable."""
        return os.environ.get("MT5_PASSWORD", "")

    def mt5_server(self) -> str:
        """MT5 broker server from environment variable."""
        return os.environ.get("MT5_SERVER", "")

    def db_path(self) -> Path:
        """Absolute path to the SQLite database file."""
        rel = self.get("database.path", "database/mt5_quantum_ai.db")
        return ROOT / rel

    def log_level(self) -> str:
        return self.get("logging.level", "INFO")

    def tradeable_regimes(self) -> list[str]:
        return self.get("regime.tradeable_regimes", [])

    def enabled_strategies(self) -> list[str]:
        strats = self.get("strategies", {})
        return [name for name, conf in strats.items()
                if isinstance(conf, dict) and conf.get("enabled", False)]

    def session(self, name: str) -> dict:
        """Return start/end times for a named session."""
        sess = self.get(f"sessions.{name}", {})
        if not sess:
            raise KeyError(f"Session '{name}' not in settings.yaml")
        return sess

    def raw_settings(self) -> dict:
        """Return the full settings dict (use sparingly)."""
        return self._settings

    def summary(self) -> str:
        """One-line summary for startup logging."""
        n_sym = len(self.enabled_symbols())
        n_str = len(self.enabled_strategies())
        return (
            f"ConfigLoader | mode={self.mode()} | "
            f"symbols={n_sym} | strategies={n_str} | "
            f"regime_threshold={self.get('regime.confidence_threshold')}"
        )


# ── module-level singleton ────────────────────────────────────
cfg = ConfigLoader()
