"""
MT5 Quantum AI — Data Processor
==================================
Loads raw candle data from the database, cleans it, validates it,
and returns ready-to-use pandas DataFrames for feature engineering
and AI model training.

No MT5 connection required — works entirely from the local database.

Usage:
    from src.data.processor import processor

    # Full pipeline: load → clean → validate
    df = processor.process("EURUSD", "H1")

    # Process all enabled symbols and timeframes
    data_map = processor.process_all()

    # Quality report
    processor.print_quality_report()
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent.parent

# Timeframe → expected seconds between bars
_TF_SECONDS: dict[str, int] = {
    "M1":  60,       "M2":  120,    "M3":  180,
    "M4":  240,      "M5":  300,    "M6":  360,
    "M10": 600,      "M12": 720,    "M15": 900,
    "M20": 1200,     "M30": 1800,
    "H1":  3600,     "H2":  7200,   "H3":  10800,
    "H4":  14400,    "H6":  21600,  "H8":  28800,
    "H12": 43200,    "D1":  86400,  "W1":  604800,
}


class DataProcessor:
    """
    Singleton that cleans, validates, and enriches OHLCV DataFrames.

    Pipeline:
      load_candles() → clean() → validate() → enrich()

    Or use the convenience method:
      process(symbol, timeframe) — runs the full pipeline.
    """

    _instance: Optional["DataProcessor"] = None

    def __new__(cls) -> "DataProcessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return
        from config.loader import cfg
        from src.logger import get_logger
        from src.database import db
        self._cfg = cfg
        self._log = get_logger("system")
        self._db  = db
        self._ready = True

    # ── load ─────────────────────────────────────────────────

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load raw candles from the database into a DataFrame.

        Returns an unsorted, unvalidated DataFrame with columns:
            time, open, high, low, close, tick_volume, spread, real_volume

        Returns empty DataFrame if no data exists.
        """
        from src.database import Candle

        with self._db.session() as sess:
            q = sess.query(Candle).filter_by(
                symbol=symbol, timeframe=timeframe
            )
            if from_date:
                q = q.filter(Candle.open_time >= from_date)
            if to_date:
                q = q.filter(Candle.open_time <= to_date)
            q = q.order_by(Candle.open_time.asc())
            if limit:
                # Get last N bars: query descending, then reverse
                q = (
                    sess.query(Candle)
                    .filter_by(symbol=symbol, timeframe=timeframe)
                    .order_by(Candle.open_time.desc())
                    .limit(limit)
                )
                rows = q.all()[::-1]   # reverse to chronological
            else:
                rows = q.all()

        if not rows:
            return pd.DataFrame()

        data = [
            {
                "time":        r.open_time,
                "open":        r.open,
                "high":        r.high,
                "low":         r.low,
                "close":       r.close,
                "tick_volume": r.tick_volume,
                "spread":      r.spread,
                "real_volume": r.real_volume,
            }
            for r in rows
        ]
        return pd.DataFrame(data)

    # ── clean ────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Remove invalid candles from a raw DataFrame.

        Returns:
            (cleaned_df, report) where report = {
                "initial_bars":    int,
                "removed_zero":    int,
                "removed_ohlc":    int,
                "removed_dup":     int,
                "final_bars":      int,
                "removed_total":   int,
            }
        """
        if df.empty:
            return df, {"initial_bars": 0, "final_bars": 0,
                        "removed_total": 0}

        report = {"initial_bars": len(df)}

        # Ensure correct dtypes
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 1. Remove bars with NaN prices
        before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])

        # 2. Remove zero or negative prices
        mask_zero = (
            (df["open"]  <= 0) |
            (df["high"]  <= 0) |
            (df["low"]   <= 0) |
            (df["close"] <= 0)
        )
        df = df[~mask_zero]
        report["removed_zero"] = before - len(df)
        before = len(df)

        # 3. Remove invalid OHLC: high must be >= low
        mask_invalid = df["high"] < df["low"]
        df = df[~mask_invalid]

        # 4. Remove bars where open or close is outside [low, high]
        mask_bad_oc = (
            (df["open"]  < df["low"]) | (df["open"]  > df["high"]) |
            (df["close"] < df["low"]) | (df["close"] > df["high"])
        )
        df = df[~mask_bad_oc]
        report["removed_ohlc"] = before - len(df)
        before = len(df)

        # 5. Remove duplicate timestamps (keep first occurrence)
        df = df.drop_duplicates(subset=["time"], keep="first")
        report["removed_dup"] = before - len(df)

        # 6. Sort chronologically
        df = df.sort_values("time").reset_index(drop=True)

        report["final_bars"]    = len(df)
        report["removed_total"] = report["initial_bars"] - report["final_bars"]
        return df, report

    # ── validate ─────────────────────────────────────────────

    def validate(
        self, df: pd.DataFrame, timeframe: str = "H1",
        min_bars: int = 50,
    ) -> tuple[bool, list[str]]:
        """
        Check that a cleaned DataFrame meets quality requirements.

        Returns:
            (is_valid: bool, warnings: list[str])
            is_valid = False if minimum bar count not met.
            warnings lists non-fatal issues (gaps, spikes, etc.).
        """
        issues: list[str] = []

        if df.empty:
            return False, ["DataFrame is empty"]

        if len(df) < min_bars:
            return False, [
                f"Only {len(df)} bars — need at least {min_bars}"
            ]

        # Check for large gaps
        tf_sec = _TF_SECONDS.get(timeframe.upper(), 3600)
        if "time" in df.columns:
            times = pd.to_datetime(df["time"])
            diffs = times.diff().dt.total_seconds().dropna()
            # A gap is any interval > 3× the expected bar period
            # (allows for weekends and bank holidays)
            gap_threshold = tf_sec * 3
            gaps = diffs[diffs > gap_threshold]
            if len(gaps) > 0:
                max_gap_h = gaps.max() / 3600
                issues.append(
                    f"{len(gaps)} time gaps detected "
                    f"(max gap: {max_gap_h:.1f}h) — "
                    f"normal for weekends/holidays"
                )

        # Check for price spikes (bar range > 5× rolling average range)
        bar_range = df["high"] - df["low"]
        rolling_avg = bar_range.rolling(20, min_periods=5).mean()
        spikes = bar_range[bar_range > rolling_avg * 5]
        if len(spikes) > 0:
            issues.append(
                f"{len(spikes)} potential price spike(s) detected — "
                f"review manually"
            )

        # Check for flatline (constant close for 5+ consecutive bars)
        close_diff = df["close"].diff().abs()
        flat_runs = (close_diff == 0).astype(int)
        if flat_runs.sum() > 10:
            issues.append(
                f"{flat_runs.sum()} bars with unchanged close — "
                f"possible flatline / broker issue"
            )

        return True, issues

    # ── enrich ───────────────────────────────────────────────

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns to a cleaned DataFrame.

        Adds:
            returns      — percentage change in close price
            log_returns  — log(close / prev_close)
            range        — high - low (bar size in price)
            body         — abs(close - open)
            body_pct     — body as % of range
            is_bullish   — 1 if close >= open, else 0
        """
        if df.empty:
            return df

        df = df.copy()
        df["returns"]     = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["range"]       = df["high"] - df["low"]
        df["body"]        = (df["close"] - df["open"]).abs()
        df["body_pct"]    = np.where(
            df["range"] > 0,
            df["body"] / df["range"] * 100,
            0.0
        )
        df["is_bullish"]  = (df["close"] >= df["open"]).astype(int)

        return df

    # ── full pipeline ────────────────────────────────────────

    def process(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None,
        min_bars: int = 50,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        set_index: bool = True,
    ) -> pd.DataFrame:
        """
        Full data pipeline: load → clean → validate → enrich.

        Args:
            symbol:    config symbol name (e.g. "EURUSD")
            timeframe: e.g. "H1", "M15"
            limit:     load last N bars only
            min_bars:  minimum bars required; returns empty df if not met
            set_index: if True, sets 'time' as the DatetimeIndex

        Returns:
            Enriched, clean DataFrame. Empty if validation fails.
        """
        # 1. Load
        raw = self.load_candles(
            symbol, timeframe, limit=limit,
            from_date=from_date, to_date=to_date,
        )

        if raw.empty:
            self._log.warning(
                f"No data in DB for {symbol} {timeframe}. "
                f"Run the downloader first."
            )
            return pd.DataFrame()

        # 2. Clean
        cleaned, clean_report = self.clean(raw)

        if clean_report["removed_total"] > 0:
            self._log.info(
                f"{symbol} {timeframe} | cleaned "
                f"{clean_report['removed_total']} invalid bars "
                f"({clean_report['initial_bars']} → "
                f"{clean_report['final_bars']})"
            )

        if cleaned.empty:
            self._log.warning(
                f"{symbol} {timeframe}: all bars removed during cleaning"
            )
            return pd.DataFrame()

        # 3. Validate
        is_valid, warnings = self.validate(cleaned, timeframe, min_bars)

        for w in warnings:
            self._log.warning(f"{symbol} {timeframe}: {w}")

        if not is_valid:
            self._log.warning(
                f"{symbol} {timeframe}: validation failed — {warnings}"
            )
            return pd.DataFrame()

        # 4. Enrich
        enriched = self.enrich(cleaned)

        # 5. Optionally set DatetimeIndex
        if set_index and "time" in enriched.columns:
            enriched["time"] = pd.to_datetime(enriched["time"])
            enriched = enriched.set_index("time")
            enriched.index.name = "time"

        self._log.info(
            f"{symbol} {timeframe}: processed {len(enriched)} bars | "
            f"range {str(enriched.index[0])[:16]} → "
            f"{str(enriched.index[-1])[:16]}"
        )
        return enriched

    def process_all(
        self, min_bars: int = 50, limit: Optional[int] = None
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Process all enabled symbols and their configured timeframes.

        Returns:
            {
              "EURUSD": {"H1": df, "H4": df, ...},
              "GBPUSD": {"H1": df, ...},
              ...
            }
        """
        result: dict[str, dict[str, pd.DataFrame]] = {}
        symbols = self._cfg.enabled_symbols()

        self._log.info(
            f"Processing all data: {len(symbols)} symbols"
        )

        for sym in symbols:
            sym_cfg    = self._cfg.symbol(sym)
            timeframes = sym_cfg.get("timeframes", ["H1"])
            result[sym] = {}

            for tf in timeframes:
                df = self.process(
                    sym, tf, min_bars=min_bars, limit=limit
                )
                result[sym][tf] = df

        loaded = sum(
            1 for sym in result.values()
            for df in sym.values()
            if not df.empty
        )
        self._log.info(
            f"Data processing complete: {loaded} symbol/timeframe "
            f"combinations ready"
        )
        return result

    # ── quality report ───────────────────────────────────────

    def quality_report(
        self, symbol: str, timeframe: str
    ) -> dict:
        """
        Return a quality dict for one symbol/timeframe.
        """
        raw = self.load_candles(symbol, timeframe)
        if raw.empty:
            return {
                "symbol": symbol, "timeframe": timeframe,
                "status": "NO DATA",
                "raw_bars": 0, "clean_bars": 0,
                "removed": 0, "date_from": "—", "date_to": "—",
                "gaps": 0, "spikes": 0,
            }

        cleaned, report = self.clean(raw)
        is_valid, issues = self.validate(cleaned, timeframe)

        gaps   = sum(1 for i in issues if "gap" in i.lower())
        spikes = sum(1 for i in issues if "spike" in i.lower())

        date_from = str(cleaned["time"].min())[:16] if not cleaned.empty else "—"
        date_to   = str(cleaned["time"].max())[:16] if not cleaned.empty else "—"

        return {
            "symbol":    symbol,
            "timeframe": timeframe,
            "status":    "OK" if is_valid else "WARN",
            "raw_bars":  report["initial_bars"],
            "clean_bars": report["final_bars"],
            "removed":   report["removed_total"],
            "date_from": date_from,
            "date_to":   date_to,
            "gaps":      gaps,
            "spikes":    spikes,
            "issues":    issues,
        }

    def print_quality_report(self) -> None:
        """Print a formatted quality table for all enabled symbols."""
        header = (
            f"\n{'Symbol':<10} {'TF':<6} {'Status':<8} "
            f"{'Bars':>6} {'Removed':>8}  "
            f"{'From':<17} {'To':<17} {'Issues'}"
        )
        print(header)
        print("-" * 90)

        for sym in self._cfg.enabled_symbols():
            sym_cfg    = self._cfg.symbol(sym)
            timeframes = sym_cfg.get("timeframes", ["H1"])
            for tf in timeframes:
                r = self.quality_report(sym, tf)
                issue_str = "; ".join(r["issues"]) if r.get("issues") else ""
                print(
                    f"{r['symbol']:<10} {r['timeframe']:<6} "
                    f"{r['status']:<8} {r['clean_bars']:>6} "
                    f"{r['removed']:>8}  "
                    f"{r['date_from']:<17} {r['date_to']:<17} "
                    f"{issue_str[:40]}"
                )
        print()

    # ── helpers ──────────────────────────────────────────────

    def get_latest_bar(
        self, symbol: str, timeframe: str
    ) -> Optional[pd.Series]:
        """Return the most recent bar as a Series, or None."""
        df = self.load_candles(symbol, timeframe, limit=1)
        if df.empty:
            return None
        return df.iloc[-1]

    def get_bar_count(self, symbol: str, timeframe: str) -> int:
        """Return number of clean bars available."""
        from src.database import Candle
        with self._db.session() as sess:
            return (
                sess.query(Candle)
                .filter_by(symbol=symbol, timeframe=timeframe)
                .count()
            )

    def has_enough_data(
        self, symbol: str, timeframe: str, min_bars: int = 200
    ) -> bool:
        """Return True if enough bars are available for this symbol/tf."""
        return self.get_bar_count(symbol, timeframe) >= min_bars


# ── module-level singleton ────────────────────────────────────
processor = DataProcessor()
