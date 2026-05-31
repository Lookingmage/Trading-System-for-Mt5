"""
MT5 Quantum AI — Main Controller
==================================
Master entry point. Ties all 20 modules into one trading loop.

How to run:
    py -3.11 main.py                  # uses mode from settings.yaml
    py -3.11 main.py --mode demo      # override mode
    py -3.11 main.py --mode backtest  # run backtest and exit
    py -3.11 main.py --mode shadow    # shadow trading
    py -3.11 main.py --symbols EURUSD GBPUSD  # specific symbols only

How to run the dashboard (separate terminal):
    streamlit run dashboard/app.py

Press Ctrl+C to gracefully shutdown.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── ensure project root is on path ───────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

SYSTEM_VERSION = "1.0.0"
BUILD_STEP     = 21


# ════════════════════════════════════════════════════════════
class MainController:
    """
    Orchestrates all system modules for one complete trading session.

    Lifecycle:
        ctrl = MainController(mode="demo")
        if ctrl.startup():
            ctrl.run()          # blocks until shutdown
        ctrl.shutdown()
    """

    def __init__(
        self,
        mode:    Optional[str]       = None,
        symbols: Optional[list[str]] = None,
    ) -> None:
        # Lazy imports (modules may print warnings on import)
        from config.loader import cfg
        from src.logger    import get_logger, system_log, trading_log, error_log

        self._cfg        = cfg
        self._log        = system_log
        self._trade_log  = trading_log
        self._err_log    = error_log

        # Override mode if provided on command line
        if mode:
            self._cfg._settings["system"]["mode"] = mode
        self._mode = self._cfg.mode()

        # Symbol filter (None = all enabled)
        self._symbols = symbols or self._cfg.enabled_symbols()

        # Runtime state
        self._running          = False
        self._shutdown_flag    = False
        self._last_bar_times:  dict[str, dict[str, datetime]] = {}
        self._loop_count:      int = 0
        self._trades_this_session: int = 0
        self._signal_mgr                    = None   # set in startup()

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ── lifecycle ─────────────────────────────────────────────

    def startup(self) -> bool:
        """
        Initialize all modules.  Returns True if ready to run.
        On failure, logs the error and returns False.
        """
        self._log.info(
            f"{'=' * 60}\n"
            f"  MT5 Quantum AI v{SYSTEM_VERSION}  —  Step {BUILD_STEP}\n"
            f"  Mode: {self._mode.upper()}\n"
            f"  Symbols: {', '.join(self._symbols)}\n"
            f"  {'=' * 60}"
        )

        # 1. Database
        try:
            from src.database import db
            self._db = db
            self._log.info(f"Database: {db.db_path}")
        except Exception as e:
            self._err_log.error(f"Database startup failed: {e}", exc_info=True)
            return False

        # 2. Live mode safety gate
        if self._mode == "live":
            if not self._cfg.get("live_trading.enabled", False):
                self._err_log.error(
                    "LIVE TRADING BLOCKED: live_trading.enabled is false.\n"
                    "To enable live trading, edit config/settings.yaml:\n"
                    "  live_trading:\n"
                    "    enabled: true\n"
                    "Then re-run: py -3.11 main.py --mode live"
                )
                return False
            self._log.warning("LIVE TRADING ACTIVE — REAL MONEY AT RISK")
            self._log.warning("Press Ctrl+C within 10 seconds to abort")
            for i in range(10, 0, -1):
                self._log.info(f"  Live trading starts in {i}s ...")
                time.sleep(1)

        # 4. MT5 connection
        self._mt5c = None
        if self._mode != "backtest":
            try:
                from src.connector.mt5_connector import mt5c
                self._mt5c = mt5c
                connected  = mt5c.connect_with_retry()
                if not connected:
                    self._err_log.error("MT5 connection failed at startup")
                    if self._mode == "live":
                        return False
                    self._log.warning(
                        "MT5 not connected — running in data-only mode"
                    )
                else:
                    acc = mt5c.account_info()
                    if acc:
                        self._log.info(
                            f"MT5: account={acc['login']} "
                            f"balance={acc['balance']:.2f} {acc['currency']} "
                            f"broker={acc['company']}"
                        )
            except Exception as e:
                self._err_log.error(f"MT5 startup error: {e}", exc_info=True)
                if self._mode == "live":
                    return False

        # 5. Downloader (resolve broker symbol names)
        try:
            from src.data.downloader import downloader
            self._downloader = downloader
            if self._mt5c and self._mt5c.is_connected():
                sym_map = downloader.resolve_all_symbols()
                self._log.info(f"Symbols resolved: {sym_map}")
        except Exception as e:
            self._log.warning(f"Downloader init warning: {e}")
            self._downloader = None

        # 6. Risk Manager
        try:
            from src.risk.manager import risk_manager
            self._risk = risk_manager
            self._log.info("Risk Manager: initialized")
        except Exception as e:
            self._err_log.error(f"Risk Manager init failed: {e}")
            return False

        # 5. Trade Executor
        try:
            from src.executor.executor import executor
            executor._mode = self._mode
            self._executor = executor
            self._log.info(f"Executor: mode={self._mode}")
        except Exception as e:
            self._err_log.error(f"Executor init failed: {e}")
            return False

        # 6. Load ML Models (non-blocking — system works without them)
        self._load_models()

        # 7. Signal Manager — duplicate trade prevention
        try:
            from src.signal_manager import signal_manager
            self._signal_mgr = signal_manager
            self._log.info(
                f"Signal Manager: {signal_manager.summary()}"
            )
        except Exception as e:
            self._log.warning(f"Signal Manager init warning: {e}")
            self._signal_mgr = None

        # 8. Self-Learner
        try:
            from src.self_learning.self_learner import self_learner
            self._learner = self_learner
        except Exception as e:
            self._log.warning(f"Self-learner init warning: {e}")
            self._learner = None

        # 9. Log startup event
        with self._db.session() as sess:
            self._db.log_event(sess, "startup", f"System started in {self._mode} mode",
                               {"mode": self._mode, "symbols": self._symbols,
                                "version": SYSTEM_VERSION})

        self._log.info("Startup complete — all modules initialized")
        return True

    def _load_models(self) -> None:
        """Load active ML/DL/AI models from disk. Non-fatal if missing."""
        # Regime detector
        try:
            from src.regime.detector import regime_detector
            self._regime = regime_detector
            self._log.info("Regime detector: loaded")
        except Exception as e:
            self._log.warning(f"Regime detector not loaded: {e}")
            self._regime = None

        # AI Decision Engine
        try:
            from src.ai_engine.decision_engine import ai_engine
            self._ai = ai_engine
            self._log.info(
                f"AI Engine: {'ML mode' if ai_engine.has_model else 'rule-based mode'}"
            )
        except Exception as e:
            self._log.warning(f"AI Engine not loaded: {e}")
            self._ai = None

        # Strategy Engine
        try:
            from src.strategies.engine import strategy_engine
            self._strategy_engine = strategy_engine
            n_strats = len(strategy_engine.enabled_strategy_names)
            self._log.info(f"Strategies: {n_strats} enabled")
        except Exception as e:
            self._err_log.error(f"Strategy Engine failed: {e}")
            self._strategy_engine = None

        # Processor + Feature Engineer
        try:
            from src.data.processor   import processor
            from src.features.engineer import engineer
            self._processor = processor
            self._engineer  = engineer
        except Exception as e:
            self._err_log.error(f"Processor/Engineer init failed: {e}")
            self._processor = None
            self._engineer  = None

    def run(self) -> None:
        """
        Main loop. Blocks until shutdown signal received.
        Dispatches to the correct loop based on mode.
        """
        self._running = True

        if self._mode == "backtest":
            self._run_backtest_mode()
        else:
            self._run_live_mode()

    def shutdown(self, reason: str = "user_request") -> None:
        """Graceful shutdown — close connections, save state."""
        self._shutdown_flag = True
        self._running       = False

        self._log.info(f"Shutting down: {reason}")

        # Log shutdown event
        try:
            with self._db.session() as sess:
                self._db.log_event(
                    sess, "shutdown",
                    f"System shutdown: {reason}",
                    {"trades_this_session": self._trades_this_session,
                     "loops_completed":    self._loop_count},
                )
        except Exception:
            pass

        # Disconnect MT5
        if self._mt5c:
            try:
                self._mt5c.disconnect()
            except Exception:
                pass

        self._log.info(
            f"Shutdown complete | "
            f"trades={self._trades_this_session} | "
            f"loops={self._loop_count}"
        )

    # ── backtest mode ─────────────────────────────────────────

    def _run_backtest_mode(self) -> None:
        """Run backtest for all symbols and exit."""
        self._log.info("Starting backtest ...")
        from src.backtest.engine import backtest_engine

        for sym in self._symbols:
            sym_cfg    = self._cfg.symbol(sym)
            timeframes = sym_cfg.get("timeframes", ["H1"])

            for tf in timeframes:
                self._log.info(f"Backtesting {sym} {tf} ...")
                result = backtest_engine.run(sym, tf)
                print(result.summary())
                result.save_report(tag=f"{sym}_{tf}")
                self._log.info(
                    f"Backtest saved: {sym} {tf} | "
                    f"trades={result.metrics.total_trades} "
                    f"net=${result.metrics.net_profit:+,.2f}"
                )

        self._log.info("Backtest complete — exiting")

    # ── live / demo / shadow loop ────────────────────────────

    def _run_live_mode(self) -> None:
        """
        Main trading loop for demo, shadow, and live modes.
        Runs every `loop_interval` seconds and checks for new bars.
        """
        loop_interval = 60   # seconds between iterations

        self._log.info(
            f"Entering {self._mode} trading loop "
            f"(interval={loop_interval}s)"
        )

        while not self._shutdown_flag:
            try:
                self._loop_count += 1
                self._log.info(
                    f"--- Loop #{self._loop_count} "
                    f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ---"
                )

                self._run_one_cycle()

            except Exception as e:
                self._err_log.error(
                    f"Loop error (will retry): {e}", exc_info=True
                )

            # Wait for next iteration (check shutdown every second)
            for _ in range(loop_interval):
                if self._shutdown_flag:
                    break
                time.sleep(1)

    def _sync_open_trades(self) -> int:
        """
        Reconcile the local database against actual MT5 positions.

        For live/demo modes: any trade in DB with status='open' that
        no longer exists in MT5 is marked as 'closed' (closed externally —
        e.g. manually closed in MT5, SL/TP hit while system was offline,
        or order expired at broker).

        Returns number of stale trades cleaned up.
        """
        if self._mode not in ("live", "demo"):
            return 0
        if not self._mt5c or not self._mt5c.is_connected():
            return 0

        try:
            from src.database import Trade
            from src.data.downloader import downloader
            import datetime as _dt

            magic = int(self._cfg.get("mt5.magic_number", 20250001))

            # Get all actual open MT5 positions with our magic number
            all_positions = self._mt5c.positions_get()
            live_tickets  = {
                p.ticket for p in all_positions
                if p.magic == magic
            }

            # Query DB for open trades in this mode
            with self._db.session() as sess:
                db_open = (
                    sess.query(Trade)
                    .filter_by(status="open", mode=self._mode)
                    .all()
                )
                stale_ids = [
                    t.id for t in db_open
                    if t.mt5_ticket not in live_tickets
                ]

                if stale_ids:
                    # Mark stale trades as closed
                    for t in db_open:
                        if t.id in stale_ids:
                            t.status       = "closed"
                            t.close_time   = _dt.datetime.utcnow()
                            t.close_reason = "closed_externally"
                            t.updated_at   = _dt.datetime.utcnow()
                            self._log.warning(
                                f"Stale trade #{t.id} ({t.direction} {t.symbol} "
                                f"ticket={t.mt5_ticket}) not found in MT5 — "
                                f"marked as closed_externally"
                            )
            return len(stale_ids)
        except Exception as e:
            self._log.warning(f"sync_open_trades() error: {e}")
            return 0

    def _run_one_cycle(self) -> dict:
        """
        Execute one full trading cycle across all symbols.
        Returns a summary dict of what happened.
        """
        summary = {
            "loop": self._loop_count,
            "signals_found":   0,
            "trades_opened":   0,
            "trades_managed":  0,
            "retrained":       False,
        }

        if not self._processor or not self._engineer:
            self._log.warning("Processor/Engineer not loaded — skipping cycle")
            return summary

        # Ensure MT5 is connected (if needed)
        if self._mt5c and not self._mt5c.is_connected():
            self._log.warning("MT5 disconnected — attempting reconnect")
            if not self._mt5c.connect_with_retry():
                self._log.error("Reconnect failed — skipping cycle")
                return summary

        # Sync DB open trades against actual MT5 positions.
        # Cleans up stale "open" DB records for trades that were closed
        # in MT5 externally (manual close, SL/TP, system restart).
        n_synced = self._sync_open_trades()
        if n_synced > 0:
            self._log.info(
                f"DB sync: {n_synced} stale trade(s) marked as closed_externally"
            )

        # Update risk manager balance
        if self._mt5c and self._mt5c.is_connected():
            acc = self._mt5c.account_info()
            if acc:
                self._risk.update_balance(acc["balance"])

        # Check if trading is allowed (circuit breakers)
        can_trade, reason = self._risk.can_trade()
        if not can_trade:
            self._log.warning(f"Trading halted: {reason}")

        # Process each symbol
        for sym in self._symbols:
            try:
                self._process_symbol(sym, can_trade, summary)
            except Exception as e:
                self._err_log.error(
                    f"Error processing {sym}: {e}", exc_info=True
                )

        # Self-learning: retrain if triggered
        if self._learner and self._learner.should_retrain():
            self._log.info("Self-learning: retraining models ...")
            result = self._learner.retrain_all(
                symbol=self._symbols[0], timeframe="H1"
            )
            summary["retrained"] = result.get("status") == "completed"

        return summary

    def _process_symbol(
        self, sym: str, can_trade: bool, summary: dict
    ) -> None:
        """Run the full pipeline for one symbol."""
        sym_cfg    = self._cfg.symbol(sym)
        timeframes = sym_cfg.get("timeframes", ["H1"])
        tf         = timeframes[0]   # use first timeframe for signals

        # 1. Download latest bars
        if self._mt5c and self._mt5c.is_connected() and self._downloader:
            new_bars = self._downloader.download_candles(sym, tf, bars=50)
            if new_bars > 0:
                self._log.info(f"  {sym} {tf}: +{new_bars} new bars")

        # 2. Process + feature engineer
        df   = self._processor.process(sym, tf, limit=300, min_bars=50)
        if df.empty:
            return

        feat_df = self._engineer.compute(df, symbol=sym, timeframe=tf)
        if feat_df.empty:
            return

        feat_row = feat_df.dropna().iloc[-1].to_dict() if not feat_df.dropna().empty else {}

        # 3. Regime detection
        regime_result = None
        if self._regime:
            regime_result = self._regime.detect(sym, tf, save_to_db=True)
            if regime_result:
                self._log.info(
                    f"  {sym}: regime={regime_result.regime} "
                    f"conf={regime_result.confidence:.1%} "
                    f"trade={'OK' if regime_result.trade_allowed else 'BLOCKED'}"
                )

        # 4. Strategy signals
        if not can_trade:
            return

        signals = self._strategy_engine.run_on_df(feat_df, sym, tf, min_strength=0.35)
        if not signals:
            return

        summary["signals_found"] += len(signals)

        # 5. Process each signal
        for signal in signals:
            self._process_signal(signal, feat_row, regime_result, summary)
            break   # only first (strongest) signal per symbol per cycle

        # 6. Manage open positions
        self._executor.manage_positions(sym)
        summary["trades_managed"] += 1

    def _process_signal(
        self, signal, feat_row: dict, regime_result, summary: dict
    ) -> None:
        """
        Run a single signal through:
          SignalManager → AI Engine → Risk Manager → Executor

        The SignalManager check is FIRST — it is the deduplication gate
        that guarantees ONE SIGNAL = ONE TRADE with zero sleep/delay logic.
        """
        # ── GATE 0: SignalManager — duplicate check ───────────
        # This is the primary fix for the duplicate trade bug.
        # A signal_id encodes: symbol+timeframe+strategy+direction+bar_time
        # The bar_time is the CANDLE timestamp — constant until the next
        # candle forms. So the same bar generates the same signal_id every
        # loop cycle, and only the FIRST cycle is allowed through.
        if self._signal_mgr:
            is_dup, dup_reason = self._signal_mgr.is_duplicate(signal)
            if is_dup:
                # Already executed or open position exists — skip silently
                return

            # Release opposite-direction locks for this symbol
            # (e.g. if we had a BUY lock and now see a SELL signal)
            self._signal_mgr.release_opposite(signal.symbol, signal.direction)

        # ── GATE 1: AI Decision Engine ────────────────────────
        ai_decision = None
        if self._ai:
            ai_decision = self._ai.evaluate(signal, feat_row, regime_result)
            if not ai_decision.approved:
                self._log.info(
                    f"  AI blocked: {signal.strategy_name} {signal.direction} "
                    f"{signal.symbol} — {ai_decision.reason}"
                )
                return

        # Attach regime to signal
        if regime_result:
            signal.regime = regime_result.regime

        # ── GATE 2: Risk Manager ──────────────────────────────
        acc = {"balance": 10_000.0, "equity": 10_000.0, "margin_free": 9_000.0}
        if self._mt5c and self._mt5c.is_connected():
            live_acc = self._mt5c.account_info()
            if live_acc:
                acc = live_acc

        risk_decision = self._risk.evaluate(signal, acc)
        if not risk_decision.approved:
            self._log.info(
                f"  Risk blocked: {signal.strategy_name} — {risk_decision.reason}"
            )
            return

        # ── GATE 3: Executor ──────────────────────────────────
        trade_id = self._executor.place_order(signal, risk_decision)
        if trade_id:
            self._trades_this_session += 1
            summary["trades_opened"]  += 1

            # Lock this signal_id — no more trades from this candle's signal
            if self._signal_mgr:
                self._signal_mgr.record_execution(signal, trade_id)

            # Record in self-learner for retraining
            if self._learner:
                self._learner.record_trade_opened(trade_id, feat_row, signal)

            ai_conf_str = f"{ai_decision.confidence:.2f}" if ai_decision else "N/A"
            self._trade_log.info(
                f"TRADE OPENED: #{trade_id} "
                f"{signal.direction} {signal.symbol} {signal.timeframe} "
                f"lot={risk_decision.lot_size} "
                f"@ ~{signal.entry_price:.5f} "
                f"SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f} "
                f"strategy={signal.strategy_name} "
                f"signal_id={signal.signal_id} "
                f"ai_conf={ai_conf_str} "
                f"regime={signal.regime}"
            )

    # ── signal handling ───────────────────────────────────────

    def _handle_signal(self, signum, frame) -> None:
        """Handle Ctrl+C / SIGTERM gracefully."""
        print()
        self._log.info("Shutdown signal received")
        self.shutdown(reason="signal_interrupt")
        sys.exit(0)


# ════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MT5 Quantum AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  backtest  Run backtest on all symbols and exit
  demo      Live loop on MT5 demo account
  shadow    Paper trading (no real orders)
  live      Real money trading (requires live_trading.enabled=true)

Examples:
  py -3.11 main.py
  py -3.11 main.py --mode backtest
  py -3.11 main.py --mode shadow --symbols EURUSD GBPUSD
        """,
    )
    parser.add_argument(
        "--mode", choices=["backtest", "demo", "shadow", "live"],
        help="Override trading mode from settings.yaml"
    )
    parser.add_argument(
        "--symbols", nargs="+",
        help="Comma/space-separated symbols to trade (default: all enabled)"
    )
    parser.add_argument(
        "--version", action="version",
        version=f"MT5 Quantum AI v{SYSTEM_VERSION} (Build Step {BUILD_STEP})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ctrl = MainController(
        mode=args.mode,
        symbols=args.symbols,
    )

    if not ctrl.startup():
        print("Startup failed — see logs/system.log for details")
        sys.exit(1)

    try:
        ctrl.run()
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.shutdown()


if __name__ == "__main__":
    main()
