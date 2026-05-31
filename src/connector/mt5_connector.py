"""
MT5 Quantum AI — MT5 Connector
================================
Single point of access to the MetaTrader 5 terminal.
Every other module imports this; nothing calls mt5 directly.

Usage:
    from src.connector.mt5_connector import mt5c

    if mt5c.connect():
        info = mt5c.account_info()
        tick = mt5c.symbol_info_tick("EURUSD")
        df   = mt5c.copy_rates_from_pos("EURUSD", "H1", 0, 200)
        mt5c.disconnect()

Auto-reconnect:
    mt5c.ensure_connected()   # call before any data request
    # reconnects automatically if MT5 dropped
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

# ── timeframe string → MT5 constant ─────────────────────────
TIMEFRAME_MAP: dict[str, int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M2":  mt5.TIMEFRAME_M2,
    "M3":  mt5.TIMEFRAME_M3,
    "M4":  mt5.TIMEFRAME_M4,
    "M5":  mt5.TIMEFRAME_M5,
    "M6":  mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H2":  mt5.TIMEFRAME_H2,
    "H3":  mt5.TIMEFRAME_H3,
    "H4":  mt5.TIMEFRAME_H4,
    "H6":  mt5.TIMEFRAME_H6,
    "H8":  mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def tf(name: str) -> int:
    """Convert timeframe string to MT5 constant. E.g. 'H4' → mt5.TIMEFRAME_H4"""
    key = name.upper()
    if key not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unknown timeframe '{name}'. "
            f"Valid: {list(TIMEFRAME_MAP.keys())}"
        )
    return TIMEFRAME_MAP[key]


# ════════════════════════════════════════════════════════════
class MT5Connector:
    """
    Singleton wrapper around the MetaTrader5 Python API.

    Responsibilities:
    - Initialize and authenticate to the MT5 terminal
    - Auto-reconnect when the terminal disconnects
    - Wrap every MT5 call with error logging
    - Convert raw MT5 data to pandas DataFrames
    """

    _instance: Optional["MT5Connector"] = None

    def __new__(cls) -> "MT5Connector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return

        from config.loader import cfg
        from src.logger import get_logger

        self._cfg = cfg
        self._log = get_logger("system")
        self._err = get_logger("error")

        # credentials from .env (never from settings.yaml)
        self._login:    int  = cfg.mt5_login()
        self._password: str  = cfg.mt5_password()
        self._server:   str  = cfg.mt5_server()

        # settings from settings.yaml
        self._terminal_path: str = cfg.get(
            "mt5.terminal_path",
            "C:/Program Files/MetaTrader 5/terminal64.exe"
        )
        self._timeout:      int = int(cfg.get("mt5.timeout_ms",           60000))
        self._max_retries:  int = int(cfg.get("mt5.max_retries",          5))
        self._retry_delay:  int = int(cfg.get("mt5.retry_delay_seconds",  3))

        self._connected: bool = False
        self._ready = True

    # ── connection ───────────────────────────────────────────

    def connect(self) -> bool:
        """
        Initialize MT5 and login.
        Returns True on success, False on failure.
        Call this once at system startup.
        """
        self._log.info("MT5 Connector: connecting ...")

        # Step 1 — initialize the terminal
        init_ok = mt5.initialize(
            path=self._terminal_path,
            timeout=self._timeout,
        )
        if not init_ok:
            err = mt5.last_error()
            self._err.error(
                f"MT5 initialize() failed: {err}. "
                f"Is MetaTrader 5 open?"
            )
            self._connected = False
            return False

        # Step 2 — login (only if credentials are provided)
        if self._login and self._password and self._server:
            login_ok = mt5.login(
                login=self._login,
                password=self._password,
                server=self._server,
                timeout=self._timeout,
            )
            if not login_ok:
                err = mt5.last_error()
                self._err.error(
                    f"MT5 login() failed for account {self._login} "
                    f"on {self._server}: {err}"
                )
                mt5.shutdown()
                self._connected = False
                return False
            self._log.info(
                f"MT5 logged in: account={self._login} "
                f"server={self._server}"
            )
        else:
            self._log.warning(
                "MT5: no credentials in .env — using terminal's active login"
            )

        # Step 3 — confirm connection
        info = mt5.terminal_info()
        if info is None:
            self._err.error("MT5 terminal_info() returned None after login")
            mt5.shutdown()
            self._connected = False
            return False

        acc = mt5.account_info()
        if acc:
            self._log.info(
                f"MT5 connected: "
                f"account={acc.login}  balance={acc.balance:.2f} {acc.currency}  "
                f"server={acc.server}  broker={acc.company}"
            )

        self._connected = True
        return True

    def connect_with_retry(self) -> bool:
        """
        Try to connect, retrying up to max_retries times.
        Useful at startup when the terminal may still be loading.
        """
        for attempt in range(1, self._max_retries + 1):
            self._log.info(
                f"MT5 connect attempt {attempt}/{self._max_retries}"
            )
            if self.connect():
                return True
            if attempt < self._max_retries:
                self._log.warning(
                    f"MT5 connect failed — waiting {self._retry_delay}s ..."
                )
                time.sleep(self._retry_delay)
        self._err.error(
            f"MT5: all {self._max_retries} connection attempts failed"
        )
        return False

    def disconnect(self) -> None:
        """Cleanly shut down the MT5 connection."""
        mt5.shutdown()
        self._connected = False
        self._log.info("MT5 disconnected (shutdown called)")

    def is_connected(self) -> bool:
        """
        Fast health check — returns True if terminal is connected and alive.
        Does NOT attempt reconnection.
        """
        if not self._connected:
            return False
        try:
            info = mt5.terminal_info()
            return info is not None and info.connected
        except Exception:
            return False

    def ensure_connected(self) -> bool:
        """
        Call before any data request.
        If already connected → returns True immediately (no overhead).
        If disconnected → reconnects and returns True/False.
        """
        if self.is_connected():
            return True
        self._log.warning("MT5: connection lost — attempting reconnect ...")
        return self.connect_with_retry()

    # ── account information ──────────────────────────────────

    def account_info(self) -> Optional[dict]:
        """
        Return current account state as a plain dict.
        Returns None if not connected.
        """
        if not self.ensure_connected():
            return None
        acc = mt5.account_info()
        if acc is None:
            self._err.error(f"MT5 account_info() failed: {mt5.last_error()}")
            return None
        return {
            "login":        acc.login,
            "balance":      acc.balance,
            "equity":       acc.equity,
            "margin":       acc.margin,
            "margin_free":  acc.margin_free,
            "margin_level": acc.margin_level,
            "profit":       acc.profit,
            "currency":     acc.currency,
            "server":       acc.server,
            "company":      acc.company,
            "leverage":     acc.leverage,
            "trade_allowed": acc.trade_allowed,
            "trade_mode":   acc.trade_mode,
        }

    def get_terminal_info(self) -> Optional[dict]:
        """Return MT5 terminal info as a dict."""
        info = mt5.terminal_info()
        if info is None:
            return None
        return {
            "name":         info.name,
            "path":         info.path,
            "connected":    info.connected,
            "dlls_allowed": info.dlls_allowed,
            "trade_allowed": info.trade_allowed,
            "ping_last":    info.ping_last,
            "community_balance": info.community_balance,
        }

    # ── symbol information ───────────────────────────────────

    def symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Return static symbol properties (pip size, lot limits, etc.).
        Calls symbol_select() first to ensure symbol is in Market Watch.
        Returns None if symbol not found on this broker.
        """
        if not self.ensure_connected():
            return None
        mt5.symbol_select(symbol, True)   # add to Market Watch if needed
        info = mt5.symbol_info(symbol)
        if info is None:
            self._log.warning(
                f"Symbol '{symbol}' not found on broker. "
                f"Check Market Watch in MT5."
            )
            return None
        return {
            "name":          info.name,
            "bid":           info.bid,
            "ask":           info.ask,
            "spread":        info.spread,
            "digits":        info.digits,
            "point":         info.point,
            "trade_mode":    info.trade_mode,
            "volume_min":    info.volume_min,
            "volume_max":    info.volume_max,
            "volume_step":   info.volume_step,
            "trade_stops_level": info.trade_stops_level,
            "trade_freeze_level": info.trade_freeze_level,
            "currency_base": info.currency_base,
            "currency_profit": info.currency_profit,
            "path":          info.path,
            "description":   info.description,
        }

    def symbol_info_tick(self, symbol: str) -> Optional[dict]:
        """
        Return the latest tick (bid/ask/last/time) for a symbol.
        This is the fastest way to get the current price.
        """
        if not self.ensure_connected():
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self._err.error(
                f"symbol_info_tick('{symbol}') failed: {mt5.last_error()}"
            )
            return None
        return {
            "symbol": symbol,
            "time":   datetime.utcfromtimestamp(tick.time),
            "bid":    tick.bid,
            "ask":    tick.ask,
            "last":   tick.last,
            "volume": tick.volume,
            "spread": round((tick.ask - tick.bid) / (mt5.symbol_info(symbol).point or 0.0001), 1)
                      if mt5.symbol_info(symbol) else 0.0,
        }

    def check_symbol(self, symbol: str) -> bool:
        """
        Ensure symbol is visible in Market Watch (required before data fetch).
        Attempts symbol_select() first — MT5 requires this before symbol_info()
        returns data for symbols not yet in Market Watch.
        """
        if not self.ensure_connected():
            return False
        # Always try to select (add/confirm) the symbol first
        if not mt5.symbol_select(symbol, True):
            self._log.warning(
                f"symbol_select('{symbol}') failed — "
                f"symbol not offered by this broker"
            )
            return False
        # Verify info is now available
        info = mt5.symbol_info(symbol)
        if info is None:
            self._log.warning(
                f"symbol_info('{symbol}') still None after symbol_select — "
                f"symbol not available on broker"
            )
            return False
        return True

    def get_symbols(self, group: str = "*") -> list[str]:
        """Return list of all available symbol names matching a group filter."""
        if not self.ensure_connected():
            return []
        symbols = mt5.symbols_get(group=group)
        if symbols is None:
            return []
        return [s.name for s in symbols]

    # ── candle data ──────────────────────────────────────────

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: str,
        start_pos: int,
        count: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch `count` candles starting at bar `start_pos` from the current bar.
        start_pos=0 means the most recent (current) bar.

        Returns a DataFrame with columns:
            time, open, high, low, close, tick_volume, spread, real_volume
        Returns None on failure.
        """
        if not self.ensure_connected():
            return None
        if not self.check_symbol(symbol):
            return None
        rates = mt5.copy_rates_from_pos(symbol, tf(timeframe), start_pos, count)
        if rates is None or len(rates) == 0:
            self._err.error(
                f"copy_rates_from_pos({symbol},{timeframe}) failed: "
                f"{mt5.last_error()}"
            )
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"]    = symbol
        df["timeframe"] = timeframe
        return df

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch candles for a date range (inclusive).
        Useful for historical downloads.
        """
        if not self.ensure_connected():
            return None
        if not self.check_symbol(symbol):
            return None
        rates = mt5.copy_rates_range(symbol, tf(timeframe), date_from, date_to)
        if rates is None or len(rates) == 0:
            self._err.error(
                f"copy_rates_range({symbol},{timeframe},"
                f"{date_from}→{date_to}) failed: {mt5.last_error()}"
            )
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"]    = symbol
        df["timeframe"] = timeframe
        return df

    def copy_rates_from(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        count: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch `count` candles starting at a specific datetime."""
        if not self.ensure_connected():
            return None
        if not self.check_symbol(symbol):
            return None
        rates = mt5.copy_rates_from(symbol, tf(timeframe), date_from, count)
        if rates is None or len(rates) == 0:
            self._err.error(
                f"copy_rates_from({symbol},{timeframe},{date_from}) "
                f"failed: {mt5.last_error()}"
            )
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"]    = symbol
        df["timeframe"] = timeframe
        return df

    # ── tick data ────────────────────────────────────────────

    def copy_ticks_from(
        self,
        symbol: str,
        date_from: datetime,
        count: int,
        flags: int = mt5.COPY_TICKS_ALL,
    ) -> Optional[pd.DataFrame]:
        """Fetch `count` ticks starting at date_from."""
        if not self.ensure_connected():
            return None
        ticks = mt5.copy_ticks_from(symbol, date_from, count, flags)
        if ticks is None or len(ticks) == 0:
            self._err.error(
                f"copy_ticks_from({symbol}) failed: {mt5.last_error()}"
            )
            return None
        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"] = symbol
        return df

    def copy_ticks_range(
        self,
        symbol: str,
        date_from: datetime,
        date_to: datetime,
        flags: int = mt5.COPY_TICKS_ALL,
    ) -> Optional[pd.DataFrame]:
        """Fetch all ticks in a date range."""
        if not self.ensure_connected():
            return None
        ticks = mt5.copy_ticks_range(symbol, date_from, date_to, flags)
        if ticks is None or len(ticks) == 0:
            self._err.error(
                f"copy_ticks_range({symbol}) failed: {mt5.last_error()}"
            )
            return None
        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"] = symbol
        return df

    # ── positions & orders ───────────────────────────────────

    def positions_get(
        self, symbol: Optional[str] = None
    ) -> list:
        """Return all open positions (optionally filtered by symbol)."""
        if not self.ensure_connected():
            return []
        if symbol:
            result = mt5.positions_get(symbol=symbol)
        else:
            result = mt5.positions_get()
        return list(result) if result else []

    def orders_get(
        self, symbol: Optional[str] = None
    ) -> list:
        """Return all pending orders (optionally filtered by symbol)."""
        if not self.ensure_connected():
            return []
        if symbol:
            result = mt5.orders_get(symbol=symbol)
        else:
            result = mt5.orders_get()
        return list(result) if result else []

    def order_send(self, request: dict):
        """
        Send a trade request.
        Returns the MT5 OrderSendResult or None on failure.
        """
        if not self.ensure_connected():
            return None
        result = mt5.order_send(request)
        if result is None:
            self._err.error(
                f"order_send failed: {mt5.last_error()} | "
                f"request={request}"
            )
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self._log.warning(
                f"order_send retcode={result.retcode} "
                f"comment='{result.comment}' | request={request}"
            )
        return result

    def history_deals_get(
        self,
        date_from: datetime,
        date_to: datetime,
        group: str = "*",
    ) -> list:
        """Return historical deals in a date range."""
        if not self.ensure_connected():
            return []
        deals = mt5.history_deals_get(date_from, date_to, group=group)
        return list(deals) if deals else []

    def history_orders_get(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> list:
        """Return historical orders in a date range."""
        if not self.ensure_connected():
            return []
        orders = mt5.history_orders_get(date_from, date_to)
        return list(orders) if orders else []

    # ── utilities ────────────────────────────────────────────

    def last_error(self) -> tuple:
        """Return the last MT5 error code and description."""
        return mt5.last_error()

    def summary(self) -> str:
        """One-line status for logging."""
        if not self._connected:
            return "MT5Connector: DISCONNECTED"
        acc = mt5.account_info()
        if acc:
            return (
                f"MT5Connector: CONNECTED | "
                f"account={acc.login} | "
                f"balance={acc.balance:.2f} {acc.currency} | "
                f"equity={acc.equity:.2f}"
            )
        return "MT5Connector: CONNECTED (no account info)"


# ── module-level singleton ────────────────────────────────────
mt5c = MT5Connector()
