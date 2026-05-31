"""
MT5 Quantum AI — Data Downloader
==================================
Downloads historical candle and tick data from MT5.
Stores to database AND saves CSV files for inspection.

Smart resume: checks the latest bar already in the database
and only fetches newer data — safe to run repeatedly.

Usage:
    from src.data.downloader import downloader

    # Download everything defined in symbols.yaml
    report = downloader.download_all_candles()
    report = downloader.download_all_ticks(hours=48)

    # Download one symbol/timeframe
    n = downloader.download_candles("EURUSD", "H1")

    # Check what's in the database
    status = downloader.get_status()
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# Broker-name variants to try when broker_symbol is not set in config
_BROKER_VARIANTS: list[str] = ["", "c", "m", ".", "+", "_", "z", "pro"]


class DataDownloader:
    """
    Singleton that downloads and persists OHLCV and tick data.

    Symbol name resolution order:
      1. Use broker_symbol from symbols.yaml if set
      2. Auto-detect by trying common broker suffix variants
      3. Skip if symbol not available on the broker
    """

    _instance: Optional["DataDownloader"] = None

    def __new__(cls) -> "DataDownloader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return

        from config.loader import cfg
        from src.logger import get_logger
        from src.connector.mt5_connector import mt5c
        from src.database import db

        self._cfg   = cfg
        self._log   = get_logger("system")
        self._mt5c  = mt5c
        self._db    = db

        # Resolved broker symbol cache: config_name → broker_name
        self._broker_map: dict[str, str] = {}

        self._candle_dir = ROOT / "data" / "candles"
        self._tick_dir   = ROOT / "data" / "tick"
        self._candle_dir.mkdir(parents=True, exist_ok=True)
        self._tick_dir.mkdir(parents=True, exist_ok=True)

        self._ready = True

    # ── symbol name resolution ───────────────────────────────

    def resolve_broker_symbol(self, config_symbol: str) -> Optional[str]:
        """
        Return the exact broker symbol name for a config symbol.
        Result is cached so the lookup only happens once per session.

        Priority:
          1. Cached result from this session
          2. broker_symbol field in symbols.yaml
          3. Auto-detect by trying common suffix variants
        """
        if config_symbol in self._broker_map:
            return self._broker_map[config_symbol]

        sym_cfg = self._cfg.symbols().get(config_symbol, {})

        # Check explicit broker_symbol in config
        explicit = sym_cfg.get("broker_symbol")
        if explicit and explicit != "null":
            if self._mt5c.check_symbol(explicit):
                self._broker_map[config_symbol] = explicit
                self._log.info(
                    f"Symbol {config_symbol} → {explicit} (from config)"
                )
                return explicit
            else:
                self._log.warning(
                    f"broker_symbol '{explicit}' for {config_symbol} "
                    f"not found on broker — trying auto-detect"
                )

        # Auto-detect: try config name + common suffixes
        available = set(self._mt5c.get_symbols())
        for suffix in _BROKER_VARIANTS:
            candidate = config_symbol + suffix
            if candidate in available:
                if self._mt5c.check_symbol(candidate):
                    self._broker_map[config_symbol] = candidate
                    self._log.info(
                        f"Symbol {config_symbol} → {candidate} (auto-detected)"
                    )
                    return candidate

        self._log.warning(
            f"Symbol {config_symbol} not available on this broker — skipping"
        )
        return None

    def resolve_all_symbols(self) -> dict[str, str]:
        """
        Resolve broker symbol names for all enabled symbols.
        Returns dict: config_name → broker_name
        """
        if not self._mt5c.ensure_connected():
            self._log.error("Cannot resolve symbols: MT5 not connected")
            return {}
        result = {}
        for sym in self._cfg.enabled_symbols():
            broker_sym = self.resolve_broker_symbol(sym)
            if broker_sym:
                result[sym] = broker_sym
        return result

    # ── candle download ──────────────────────────────────────

    def _latest_bar_time(self, config_symbol: str, timeframe: str) -> Optional[datetime]:
        """Return the most recent bar time in the database for this symbol/timeframe."""
        from src.database import Candle
        with self._db.session() as sess:
            row = (
                sess.query(Candle)
                .filter_by(symbol=config_symbol, timeframe=timeframe)
                .order_by(Candle.open_time.desc())
                .first()
            )
            if row:
                dt = row.open_time
                # Ensure timezone-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
        return None

    def download_candles(
        self,
        config_symbol: str,
        timeframe: str,
        bars: int = 10_000,
        save_csv: bool = True,
        min_bars: int = 250,
    ) -> int:
        """
        Download candle history for one symbol/timeframe.

        Args:
            config_symbol: config name (e.g. "EURUSD")
            timeframe:     e.g. "H1", "M15"
            bars:          max bars to download
            save_csv:      also write data/candles/{symbol}/{timeframe}.csv
            min_bars:      if DB has fewer than this, force a full fresh download
                           regardless of what already exists (default 250).

        Returns:
            Number of new bars saved to the database.
        """
        if not self._mt5c.ensure_connected():
            return 0

        broker_sym = self.resolve_broker_symbol(config_symbol)
        if not broker_sym:
            return 0

        # Determine download range
        existing_count = self._db.count_candles_by_symbol(config_symbol, timeframe)
        latest = self._latest_bar_time(config_symbol, timeframe)

        # Force fresh when too few bars exist (extends history backwards)
        force_fresh = existing_count < min_bars

        if latest and not force_fresh:
            # Resume: fetch from 2 bars before latest (overlap for safety)
            from src.connector.mt5_connector import tf as tf_fn
            now_utc = datetime.now(timezone.utc)
            date_from = latest - timedelta(hours=2)
            df = self._mt5c.copy_rates_range(
                broker_sym, timeframe, date_from, now_utc
            )
            mode = "resume"
        else:
            # Fresh download (either first time or force_fresh for small datasets)
            df = self._mt5c.copy_rates_from_pos(
                broker_sym, timeframe, 0, bars
            )
            mode = "fresh" if not force_fresh else f"fresh (extend from {existing_count})"

        if df is None or df.empty:
            self._log.warning(
                f"No data returned for {config_symbol} {timeframe}"
            )
            return 0

        # Save to database
        from src.database import Candle
        new_bars = 0
        with self._db.session() as sess:
            for _, row in df.iterrows():
                bar_time = row["time"]
                # Convert timezone-aware to naive UTC for DB storage
                if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is not None:
                    bar_time = bar_time.replace(tzinfo=None)

                # Skip if we already have this exact bar (in resume mode)
                if mode == "resume" and latest:
                    bar_dt = bar_time if isinstance(bar_time, datetime) else bar_time.to_pydatetime()
                    latest_naive = latest.replace(tzinfo=None)
                    if bar_dt <= latest_naive:
                        continue

                candle = Candle(
                    symbol=config_symbol,
                    timeframe=timeframe,
                    open_time=bar_time,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    tick_volume=int(row.get("tick_volume", 0)),
                    real_volume=int(row.get("real_volume", 0)),
                    spread=int(row.get("spread", 0)),
                )
                self._db.upsert_candle(sess, candle)
                new_bars += 1

        # Save CSV
        if save_csv and not df.empty:
            self._save_candle_csv(config_symbol, timeframe, df)

        total = self._db.count_candles_by_symbol(config_symbol, timeframe)
        self._log.info(
            f"Candles {config_symbol} {timeframe}: "
            f"+{new_bars} new ({mode}) | total in DB: {total}"
        )
        return new_bars

    def download_all_candles(
        self, bars: int = 10_000, save_csv: bool = True
    ) -> dict[str, dict]:
        """
        Download candles for ALL enabled symbols and their configured timeframes.

        Returns:
            dict: { "EURUSD": {"H1": 5000, "H4": 2000, ...}, ... }
        """
        if not self._mt5c.ensure_connected():
            self._log.error("Cannot download: MT5 not connected")
            return {}

        report: dict[str, dict] = {}
        symbols = self._cfg.enabled_symbols()
        self._log.info(
            f"Starting full candle download: "
            f"{len(symbols)} symbols"
        )

        for config_sym in symbols:
            sym_cfg    = self._cfg.symbol(config_sym)
            timeframes = sym_cfg.get("timeframes", ["H1"])
            report[config_sym] = {}

            for tf in timeframes:
                n = self.download_candles(
                    config_sym, tf, bars=bars, save_csv=save_csv
                )
                report[config_sym][tf] = n

        total_bars = sum(
            v for sym in report.values() for v in sym.values()
        )
        self._log.info(
            f"Candle download complete: "
            f"{total_bars} total new bars across {len(report)} symbols"
        )
        return report

    # ── tick download ────────────────────────────────────────

    def download_ticks(
        self,
        config_symbol: str,
        hours: int = 24,
        save_csv: bool = True,
    ) -> int:
        """
        Download recent tick data for one symbol.

        Args:
            config_symbol: config name (e.g. "EURUSD")
            hours:         how many hours of history to download
            save_csv:      also write data/tick/{symbol}/ticks.csv

        Returns:
            Number of ticks saved to the database.
        """
        if not self._mt5c.ensure_connected():
            return 0

        broker_sym = self.resolve_broker_symbol(config_symbol)
        if not broker_sym:
            return 0

        date_from = datetime.now(timezone.utc) - timedelta(hours=hours)
        date_to   = datetime.now(timezone.utc)

        df = self._mt5c.copy_ticks_range(
            broker_sym, date_from, date_to,
            flags=mt5.COPY_TICKS_ALL,
        )

        if df is None or df.empty:
            self._log.warning(f"No tick data for {config_symbol}")
            return 0

        from src.database import Tick
        new_ticks = 0
        with self._db.session() as sess:
            for _, row in df.iterrows():
                tick_time = row["time"]
                if hasattr(tick_time, "tzinfo") and tick_time.tzinfo is not None:
                    tick_time = tick_time.replace(tzinfo=None)

                tick = Tick(
                    symbol=config_symbol,
                    time=tick_time,
                    bid=float(row.get("bid", 0)),
                    ask=float(row.get("ask", 0)),
                    last=float(row.get("last", 0)),
                    volume=float(row.get("volume", 0)),
                    flags=int(row.get("flags", 0)),
                )
                self._db.upsert_tick(sess, tick)
                new_ticks += 1

        if save_csv and not df.empty:
            self._save_tick_csv(config_symbol, df)

        self._log.info(
            f"Ticks {config_symbol}: +{new_ticks} new ({hours}h window)"
        )
        return new_ticks

    def download_all_ticks(
        self, hours: int = 24, save_csv: bool = True
    ) -> dict[str, int]:
        """
        Download tick data for all enabled symbols.

        Returns:
            dict: { "EURUSD": 45000, "GBPUSD": 38000, ... }
        """
        if not self._mt5c.ensure_connected():
            return {}

        report: dict[str, int] = {}
        for sym in self._cfg.enabled_symbols():
            n = self.download_ticks(sym, hours=hours, save_csv=save_csv)
            report[sym] = n
        return report

    # ── CSV helpers ──────────────────────────────────────────

    def _save_candle_csv(
        self, config_symbol: str, timeframe: str, df: pd.DataFrame
    ) -> None:
        """Append (or create) candle CSV for this symbol/timeframe."""
        folder = self._candle_dir / config_symbol
        folder.mkdir(exist_ok=True)
        csv_path = folder / f"{timeframe}.csv"

        cols = ["time", "open", "high", "low", "close",
                "tick_volume", "spread", "real_volume"]
        save_df = df[[c for c in cols if c in df.columns]].copy()

        if csv_path.exists():
            # Append only newer rows
            existing = pd.read_csv(csv_path, parse_dates=["time"])
            if not existing.empty:
                last_time = pd.to_datetime(existing["time"].max(), utc=True)
                save_df["time"] = pd.to_datetime(save_df["time"], utc=True)
                save_df = save_df[save_df["time"] > last_time]
                if save_df.empty:
                    return
                save_df.to_csv(csv_path, mode="a", header=False, index=False)
                return

        save_df.to_csv(csv_path, index=False)

    def _save_tick_csv(
        self, config_symbol: str, df: pd.DataFrame
    ) -> None:
        """Overwrite tick CSV (ticks are not accumulated long-term)."""
        folder = self._tick_dir / config_symbol
        folder.mkdir(exist_ok=True)
        csv_path = folder / "ticks.csv"

        cols = ["time", "bid", "ask", "last", "volume", "flags"]
        save_df = df[[c for c in cols if c in df.columns]].copy()
        save_df.to_csv(csv_path, index=False)

    # ── status / reporting ───────────────────────────────────

    def get_status(self) -> dict:
        """
        Return a summary of what's in the database for every
        enabled symbol and timeframe.

        Returns:
            {
              "EURUSD": {
                "H1":  {"count": 5000, "latest": "2024-01-15 14:00"},
                "H4":  {"count": 2000, "latest": "2024-01-15 12:00"},
              },
              ...
            }
        """
        from src.database import Candle
        status: dict = {}

        with self._db.session() as sess:
            for sym in self._cfg.enabled_symbols():
                sym_cfg = self._cfg.symbol(sym)
                timeframes = sym_cfg.get("timeframes", ["H1"])
                status[sym] = {}

                for tf in timeframes:
                    count = (
                        sess.query(Candle)
                        .filter_by(symbol=sym, timeframe=tf)
                        .count()
                    )
                    latest_row = (
                        sess.query(Candle)
                        .filter_by(symbol=sym, timeframe=tf)
                        .order_by(Candle.open_time.desc())
                        .first()
                    )
                    status[sym][tf] = {
                        "count":  count,
                        "latest": str(latest_row.open_time)
                                  if latest_row else "none",
                    }
        return status

    def print_status(self) -> None:
        """Print a formatted table of download status to console."""
        status = self.get_status()
        print(f"\n{'Symbol':<10} {'TF':<6} {'Bars':>8}  {'Latest Bar'}")
        print("-" * 52)
        for sym, tfs in status.items():
            for tf, info in tfs.items():
                count = info["count"]
                latest = info["latest"][:16] if info["latest"] != "none" else "not downloaded"
                flag = "" if count > 0 else "  <- MISSING"
                print(f"{sym:<10} {tf:<6} {count:>8}  {latest}{flag}")
        print()

    def csv_path(self, config_symbol: str, timeframe: str) -> Path:
        """Return the CSV file path for a symbol/timeframe."""
        return self._candle_dir / config_symbol / f"{timeframe}.csv"

    def tick_csv_path(self, config_symbol: str) -> Path:
        """Return the tick CSV file path for a symbol."""
        return self._tick_dir / config_symbol / "ticks.csv"


# ── patch DatabaseManager to add count helper used above ─────
from src.database import DatabaseManager as _DBM, Candle as _Candle

if not hasattr(_DBM, "count_candles_by_symbol"):
    def _count_candles_by_symbol(
        self, symbol: str, timeframe: str
    ) -> int:
        with self.session() as s:
            return (
                s.query(_Candle)
                .filter_by(symbol=symbol, timeframe=timeframe)
                .count()
            )
    _DBM.count_candles_by_symbol = _count_candles_by_symbol


# ── module-level singleton ────────────────────────────────────
downloader = DataDownloader()
