"""
MT5 Quantum AI — Backtesting Engine
=====================================
Bar-by-bar simulation of all strategies on historical OHLCV data.

Execution model:
  - Features are pre-computed once for the full period (fast).
  - Simulation steps from bar 200 (after EMA-200 warmup) to end.
  - Signal detected on bar[i] close → order filled at bar[i+1] open.
  - SL/TP checked against each bar's low/high.
  - Max 1 open trade per symbol at a time.

Usage:
    from src.backtest.engine import backtest_engine

    result = backtest_engine.run("EURUSD", "H1")
    print(result.summary())
    result.save_report()

    # Run all enabled symbols
    results = backtest_engine.run_all()
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── trade record ─────────────────────────────────────────────
@dataclass
class BacktestTrade:
    """One completed simulated trade."""
    symbol:       str
    strategy:     str
    direction:    str           # BUY | SELL
    entry_bar:    int
    exit_bar:     int
    entry_time:   datetime
    exit_time:    datetime
    entry_price:  float
    exit_price:   float
    stop_loss:    float
    take_profit:  float
    lot_size:     float
    gross_pnl:    float         # before commission
    commission:   float
    net_pnl:      float         # after commission
    pips:         float
    close_reason: str           # tp | sl | end_of_data

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    def to_dict(self) -> dict:
        return {
            "symbol":       self.symbol,
            "strategy":     self.strategy,
            "direction":    self.direction,
            "entry_time":   str(self.entry_time)[:19],
            "exit_time":    str(self.exit_time)[:19],
            "entry_price":  round(self.entry_price,  5),
            "exit_price":   round(self.exit_price,   5),
            "stop_loss":    round(self.stop_loss,     5),
            "take_profit":  round(self.take_profit,   5),
            "lot_size":     self.lot_size,
            "net_pnl":      round(self.net_pnl,      2),
            "pips":         round(self.pips,          1),
            "commission":   round(self.commission,    2),
            "close_reason": self.close_reason,
        }


# ── metrics ──────────────────────────────────────────────────
@dataclass
class BacktestMetrics:
    initial_balance:  float = 10_000.0
    final_balance:    float = 10_000.0
    net_profit:       float = 0.0
    gross_profit:     float = 0.0
    gross_loss:       float = 0.0
    total_trades:     int   = 0
    winning_trades:   int   = 0
    losing_trades:    int   = 0
    win_rate:         float = 0.0
    profit_factor:    float = 0.0
    avg_win:          float = 0.0
    avg_loss:         float = 0.0
    max_drawdown:     float = 0.0
    max_drawdown_pct: float = 0.0
    recovery_factor:  float = 0.0
    sharpe_ratio:     float = 0.0
    sortino_ratio:    float = 0.0
    expectancy:       float = 0.0
    return_pct:       float = 0.0
    total_bars:       int   = 0
    symbol:           str   = ""
    timeframe:        str   = ""
    period_from:      str   = ""
    period_to:        str   = ""


# ── backtest result ───────────────────────────────────────────
@dataclass
class BacktestResult:
    metrics:      BacktestMetrics
    trades:       list[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.Series          = field(default_factory=pd.Series)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"{'=' * 52}",
            f"  BACKTEST RESULT — {m.symbol} {m.timeframe}",
            f"  Period: {m.period_from} → {m.period_to}",
            f"{'=' * 52}",
            f"  Initial Balance :  ${m.initial_balance:>10,.2f}",
            f"  Final Balance   :  ${m.final_balance:>10,.2f}",
            f"  Net Profit      :  ${m.net_profit:>10,.2f}  ({m.return_pct:+.2f}%)",
            f"  Gross Profit    :  ${m.gross_profit:>10,.2f}",
            f"  Gross Loss      :  ${m.gross_loss:>10,.2f}",
            f"{'-' * 52}",
            f"  Total Trades    :  {m.total_trades:>10}",
            f"  Win Rate        :  {m.win_rate:>10.1f}%",
            f"  Profit Factor   :  {m.profit_factor:>10.2f}",
            f"  Avg Win         :  ${m.avg_win:>10,.2f}",
            f"  Avg Loss        :  ${m.avg_loss:>10,.2f}",
            f"  Expectancy      :  ${m.expectancy:>10,.2f}",
            f"{'-' * 52}",
            f"  Max Drawdown    :  ${m.max_drawdown:>10,.2f}  ({m.max_drawdown_pct:.2f}%)",
            f"  Recovery Factor :  {m.recovery_factor:>10.2f}",
            f"  Sharpe Ratio    :  {m.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio   :  {m.sortino_ratio:>10.2f}",
            f"{'=' * 52}",
        ]
        return "\n".join(lines)

    def save_report(self, tag: str = "") -> Path:
        """Save summary text + trades CSV to reports/."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"backtest_{self.metrics.symbol}_{self.metrics.timeframe}"
        if tag:
            base += f"_{tag}"
        base += f"_{ts}"

        # Summary text
        txt_path = REPORT_DIR / f"{base}.txt"
        txt_path.write_text(self.summary(), encoding="utf-8")

        # Trades CSV
        if self.trades:
            csv_path = REPORT_DIR / f"{base}_trades.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=list(self.trades[0].to_dict().keys())
                )
                writer.writeheader()
                writer.writerows(t.to_dict() for t in self.trades)

        return txt_path


# ════════════════════════════════════════════════════════════
class BacktestEngine:
    """
    Singleton bar-by-bar backtesting engine.

    The engine never modifies the database during a backtest run.
    Reports are saved to the reports/ directory.
    """

    _instance: Optional["BacktestEngine"] = None

    def __new__(cls) -> "BacktestEngine":
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
        self._log = get_logger("backtest")
        self._ready = True

    # ── public API ───────────────────────────────────────────

    def run(
        self,
        symbol: str,
        timeframe: str,
        initial_balance: Optional[float] = None,
        lot_size: Optional[float] = None,
        commission_per_lot: Optional[float] = None,
        warmup_bars: int = 210,
    ) -> BacktestResult:
        """
        Run a full backtest for one symbol/timeframe.

        Args:
            symbol:            config symbol name (e.g. "EURUSD")
            timeframe:         e.g. "H1"
            initial_balance:   starting balance (default from config)
            lot_size:          fixed lot size for all trades
            commission_per_lot: commission per lot (round-trip, USD)
            warmup_bars:       bars skipped for indicator warmup

        Returns:
            BacktestResult with metrics, trades, and equity curve.
        """
        from src.data.processor import processor
        from src.features.engineer import engineer
        from src.strategies.engine import strategy_engine

        # Config defaults
        if initial_balance is None:
            initial_balance = float(
                self._cfg.get("backtest.initial_balance", 10_000.0)
            )
        if lot_size is None:
            lot_size = float(self._cfg.get("position.default_lot", 0.01))
        if commission_per_lot is None:
            commission_per_lot = float(
                self._cfg.get("backtest.commission_per_lot", 7.0)
            )

        try:
            sym_cfg  = self._cfg.symbol(symbol)
        except KeyError:
            self._log.warning(f"Symbol '{symbol}' not in symbols.yaml — skipping")
            return self._empty_result(symbol, timeframe, initial_balance)

        pip_size = float(sym_cfg.get("pip_size", 0.0001))
        pip_val  = float(sym_cfg.get("pip_value_usd", 10.0))

        self._log.info(
            f"Backtest START: {symbol} {timeframe} | "
            f"balance=${initial_balance:,.0f} lot={lot_size} "
            f"commission=${commission_per_lot}/lot"
        )

        # 1. Load and process all candles
        df = processor.process(symbol, timeframe, min_bars=warmup_bars + 20)
        if df.empty:
            self._log.warning(f"No data for {symbol} {timeframe}")
            return self._empty_result(symbol, timeframe, initial_balance)

        # 2. Pre-compute all features once (fast)
        feat_df = engineer.compute(df, symbol=symbol, timeframe=timeframe)
        if feat_df.empty:
            return self._empty_result(symbol, timeframe, initial_balance)

        n_bars    = len(feat_df)
        opens     = feat_df["open"].values
        highs     = feat_df["high"].values
        lows      = feat_df["low"].values
        closes    = feat_df["close"].values
        times     = feat_df.index.to_list()

        # 3. Simulation state
        balance      = initial_balance
        equity_vals  = [balance]
        completed    = []        # list[BacktestTrade]
        open_trade   = None      # only 1 at a time

        start_bar = warmup_bars
        self._log.info(
            f"Simulating {n_bars - start_bar} bars "
            f"({str(times[start_bar])[:16]} → {str(times[-1])[:16]})"
        )

        for i in range(start_bar, n_bars):

            bar_open  = float(opens[i])
            bar_high  = float(highs[i])
            bar_low   = float(lows[i])
            bar_close = float(closes[i])
            bar_time  = times[i]

            # ── A. Update open trade ─────────────────────────
            if open_trade is not None:
                ot = open_trade
                hit_sl = hit_tp = False

                if ot["direction"] == "BUY":
                    if bar_low  <= ot["sl"]:
                        hit_sl = True
                        exit_p = ot["sl"]
                    elif bar_high >= ot["tp"]:
                        hit_tp = True
                        exit_p = ot["tp"]
                else:
                    if bar_high >= ot["sl"]:
                        hit_sl = True
                        exit_p = ot["sl"]
                    elif bar_low  <= ot["tp"]:
                        hit_tp = True
                        exit_p = ot["tp"]

                if hit_sl or hit_tp:
                    reason = "tp" if hit_tp else "sl"
                    closed = self._close_trade(
                        ot, exit_p, bar_time, i,
                        reason, pip_size, pip_val,
                        lot_size, commission_per_lot,
                    )
                    balance += closed.net_pnl
                    completed.append(closed)
                    self._log.info(
                        f"  [{reason.upper()}] {ot['direction']} {symbol} | "
                        f"entry={ot['entry']:.5f} exit={exit_p:.5f} "
                        f"pnl=${closed.net_pnl:+.2f}"
                    )
                    open_trade = None

            # ── B. Generate signal (no open trade) ───────────
            if open_trade is None and i + 1 < n_bars:
                view = feat_df.iloc[:i + 1]
                signals = strategy_engine.run_on_df(
                    view, symbol, timeframe, min_strength=0.30
                )

                if signals:
                    sig = signals[0]      # take highest-strength signal
                    # Fill at next bar's open (realistic)
                    fill_price = float(opens[i + 1])
                    # Validate SL/TP still make sense at fill price
                    if sig.direction == "BUY":
                        sl = min(sig.stop_loss, fill_price - (sig.entry_price - sig.stop_loss))
                        tp = fill_price + (sig.take_profit - sig.entry_price)
                        if sl >= fill_price or tp <= fill_price:
                            continue
                    else:
                        sl = max(sig.stop_loss, fill_price + (sig.stop_loss - sig.entry_price))
                        tp = fill_price - (sig.entry_price - sig.take_profit)
                        if sl <= fill_price or tp >= fill_price:
                            continue

                    open_trade = {
                        "direction": sig.direction,
                        "entry":     fill_price,
                        "sl":        sl,
                        "tp":        tp,
                        "strategy":  sig.strategy_name,
                        "entry_bar": i + 1,
                        "entry_time": times[i + 1] if i + 1 < n_bars else bar_time,
                    }
                    self._log.info(
                        f"  [{sig.strategy_name}] OPEN {sig.direction} {symbol} "
                        f"@ {fill_price:.5f} SL={sl:.5f} TP={tp:.5f}"
                    )

            # Track equity (open trade unrealized PnL)
            if open_trade is not None:
                ot = open_trade
                if ot["direction"] == "BUY":
                    unreal = (bar_close - ot["entry"]) / pip_size * pip_val * lot_size
                else:
                    unreal = (ot["entry"] - bar_close) / pip_size * pip_val * lot_size
                equity_vals.append(balance + unreal)
            else:
                equity_vals.append(balance)

        # Close any open trade at end of data
        if open_trade is not None:
            last_close = float(closes[-1])
            last_time  = times[-1]
            closed = self._close_trade(
                open_trade, last_close, last_time, n_bars - 1,
                "end_of_data", pip_size, pip_val, lot_size, commission_per_lot,
            )
            balance += closed.net_pnl
            completed.append(closed)
            self._log.info(
                f"  [EOD] {open_trade['direction']} {symbol} closed at data end | "
                f"pnl=${closed.net_pnl:+.2f}"
            )

        # 4. Compute metrics
        eq_series = pd.Series(equity_vals, name="equity")
        metrics   = self._compute_metrics(
            completed, balance, initial_balance,
            eq_series, symbol, timeframe,
            str(times[start_bar])[:16], str(times[-1])[:16],
            n_bars - start_bar,
        )

        result = BacktestResult(
            metrics=metrics,
            trades=completed,
            equity_curve=eq_series,
        )

        self._log.info(
            f"Backtest END: {symbol} {timeframe} | "
            f"trades={metrics.total_trades} "
            f"net_pnl=${metrics.net_profit:+,.2f} "
            f"win_rate={metrics.win_rate:.1f}%"
        )
        return result

    def run_all(
        self, timeframe: str = "H1"
    ) -> dict[str, BacktestResult]:
        """Run backtest for all enabled symbols."""
        results = {}
        for sym in self._cfg.enabled_symbols():
            self._log.info(f"Backtesting {sym} {timeframe} ...")
            try:
                r = self.run(sym, timeframe)
                results[sym] = r
            except Exception as e:
                self._log.error(
                    f"Backtest failed for {sym} {timeframe}: {e}",
                    exc_info=True,
                )
        return results

    # ── private helpers ──────────────────────────────────────

    @staticmethod
    def _close_trade(
        ot: dict, exit_price: float, exit_time, exit_bar: int,
        reason: str, pip_size: float, pip_val: float,
        lot_size: float, commission: float,
    ) -> BacktestTrade:
        """Convert an open trade dict into a completed BacktestTrade."""
        if ot["direction"] == "BUY":
            pips      = (exit_price - ot["entry"]) / pip_size
            gross_pnl = pips * pip_val * lot_size
        else:
            pips      = (ot["entry"] - exit_price) / pip_size
            gross_pnl = pips * pip_val * lot_size

        net_pnl = gross_pnl - commission * lot_size

        return BacktestTrade(
            symbol=      "",           # filled by caller if needed
            strategy=    ot["strategy"],
            direction=   ot["direction"],
            entry_bar=   ot["entry_bar"],
            exit_bar=    exit_bar,
            entry_time=  ot["entry_time"],
            exit_time=   exit_time if isinstance(exit_time, datetime)
                         else datetime.utcnow(),
            entry_price= ot["entry"],
            exit_price=  exit_price,
            stop_loss=   ot["sl"],
            take_profit= ot["tp"],
            lot_size=    lot_size,
            gross_pnl=   round(gross_pnl, 2),
            commission=  round(commission * lot_size, 2),
            net_pnl=     round(net_pnl, 2),
            pips=        round(pips, 1),
            close_reason=reason,
        )

    @staticmethod
    def _compute_metrics(
        trades: list[BacktestTrade],
        final_balance: float,
        initial_balance: float,
        equity: pd.Series,
        symbol: str,
        timeframe: str,
        period_from: str,
        period_to: str,
        total_bars: int,
    ) -> BacktestMetrics:
        """Compute all performance metrics from completed trades."""
        m = BacktestMetrics(
            initial_balance=initial_balance,
            final_balance=round(final_balance, 2),
            symbol=symbol,
            timeframe=timeframe,
            period_from=period_from,
            period_to=period_to,
            total_bars=total_bars,
        )

        if not trades:
            return m

        m.total_trades   = len(trades)
        wins  = [t for t in trades if t.net_pnl > 0]
        losses= [t for t in trades if t.net_pnl <= 0]

        m.winning_trades = len(wins)
        m.losing_trades  = len(losses)
        m.win_rate       = round(len(wins) / len(trades) * 100, 2)

        m.gross_profit   = round(sum(t.net_pnl for t in wins),   2)
        m.gross_loss     = round(abs(sum(t.net_pnl for t in losses)), 2)
        m.net_profit     = round(m.gross_profit - m.gross_loss, 2)
        m.return_pct     = round(m.net_profit / initial_balance * 100, 2)
        m.final_balance  = round(initial_balance + m.net_profit, 2)

        m.avg_win  = round(m.gross_profit / len(wins),   2) if wins   else 0.0
        m.avg_loss = round(m.gross_loss   / len(losses), 2) if losses else 0.0

        m.profit_factor = round(
            m.gross_profit / m.gross_loss, 3
        ) if m.gross_loss > 0 else 0.0

        # Expectancy: expected PnL per trade
        lr = 1 - m.win_rate / 100
        m.expectancy = round(
            (m.win_rate / 100 * m.avg_win) - (lr * m.avg_loss), 2
        )

        # Max Drawdown from equity curve
        eq = equity.values
        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
        m.max_drawdown     = round(max_dd, 2)
        m.max_drawdown_pct = round(max_dd / initial_balance * 100, 2)

        # Recovery Factor
        m.recovery_factor = round(
            m.net_profit / m.max_drawdown, 2
        ) if m.max_drawdown > 0 else 0.0

        # Bar-by-bar returns for Sharpe/Sortino
        eq_series  = pd.Series(eq)
        bar_returns= eq_series.pct_change().dropna()

        if len(bar_returns) > 1:
            mean_r = bar_returns.mean()
            std_r  = bar_returns.std()
            # Annualise for H1 (≈6240 trading bars/year)
            tf_bars = {"M5": 63_504, "M15": 21_168, "H1": 6_240,
                       "H4": 1_560, "D1": 252}.get(timeframe, 6_240)
            ann_factor = math.sqrt(tf_bars)

            m.sharpe_ratio = round(
                (mean_r / std_r * ann_factor) if std_r > 0 else 0.0, 2
            )

            down_returns = bar_returns[bar_returns < 0]
            down_std = down_returns.std() if len(down_returns) > 1 else 0.0
            m.sortino_ratio = round(
                (mean_r / down_std * ann_factor) if down_std > 0 else 0.0, 2
            )

        return m

    @staticmethod
    def _empty_result(
        symbol: str, timeframe: str, initial_balance: float
    ) -> BacktestResult:
        m = BacktestMetrics(
            initial_balance=initial_balance,
            final_balance=initial_balance,
            symbol=symbol,
            timeframe=timeframe,
        )
        return BacktestResult(metrics=m)


# ── module-level singleton ────────────────────────────────────
backtest_engine = BacktestEngine()
