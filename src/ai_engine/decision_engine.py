"""
MT5 Quantum AI — AI Decision Engine
=====================================
XGBoost signal quality classifier that approves or rejects
every trade signal before it reaches the executor.

Operating modes:
  ML mode:        XGBoost trained on historical TP/SL outcomes
  Fallback mode:  Weighted rule-based scoring (when no model exists)

Both modes always enforce:
  - Regime filter (non-tradeable regimes are hard-blocked)
  - Confidence threshold from config

Usage:
    from src.ai_engine.decision_engine import ai_engine

    decision = ai_engine.evaluate(signal, feature_row, regime_result)
    if decision.approved:
        executor.place_order(signal, risk_decision)

    # Train / retrain from closed trades
    metrics = ai_engine.train()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent.parent
MODEL_DIR  = ROOT / "models" / "ml"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "decision_xgboost"

# Features used by the AI Decision Engine
# Subset of the 67 features most relevant to signal quality
DECISION_FEATURES = [
    "rsi_14", "rsi_7", "adx_14", "di_plus", "di_minus",
    "atr_14_pct", "bb_position", "bb_width",
    "macd_hist", "macd_above_signal",
    "stoch_k", "stoch_d",
    "ema20_above_ema50", "ema50_above_ema200",
    "price_above_ema20", "price_above_ema50",
    "trend_slope_20",
    "structure_bullish", "structure_bearish",
    "in_premium_zone", "in_discount_zone", "in_ote_zone",
    "fvg_bullish", "fvg_bearish",
    "volume_ratio",
    "returns_1", "returns_5",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "is_london_session", "is_newyork_session",
    "is_london_killzone", "is_ny_killzone",
    "hour", "day_of_week",
]


@dataclass
class AIDecision:
    """Result of the AI Decision Engine's evaluation of a trade signal."""
    approved:        bool
    confidence:      float       = 0.0     # 0.0–1.0
    reason:          str         = ""
    model_used:      bool        = False   # True = ML; False = rule-based
    signal_strength: float       = 0.0
    regime:          str         = ""
    feature_scores:  dict        = field(default_factory=dict)

    def __str__(self) -> str:
        mode   = "ML" if self.model_used else "Rules"
        status = "APPROVED" if self.approved else "REJECTED"
        return (
            f"[{status}|{mode}] conf={self.confidence:.2f} "
            f"regime={self.regime} | {self.reason}"
        )


class AIDecisionEngine:
    """
    Singleton AI decision engine.

    On first use with no trained model, falls back to rule-based scoring.
    Call train() after accumulating ≥10 closed trades to enable ML mode.
    """

    _instance: Optional["AIDecisionEngine"] = None

    def __new__(cls) -> "AIDecisionEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return

        from config.loader import cfg
        from src.logger    import get_logger
        from src.database  import db

        self._cfg   = cfg
        self._log   = get_logger("model")
        self._db    = db

        self._conf_threshold = float(cfg.get("ai_engine.confidence_threshold", 0.60))
        self._tradeable      = set(cfg.tradeable_regimes())

        # ML model (None = not trained yet)
        self._model         = None
        self._model_classes: list[int] = [0, 1]

        # Try to load from disk on startup
        self._try_load_model()

        self._ready = True

    # ── main gate ─────────────────────────────────────────────

    def evaluate(
        self,
        signal,                           # TradeSignal
        feature_row: dict,                # latest feature dict
        regime_result = None,             # RegimeResult or None
    ) -> AIDecision:
        """
        Evaluate a trade signal and return an AIDecision.

        Args:
            signal:        TradeSignal from a strategy
            feature_row:   dict of feature values for the current bar
                           (use engineer.compute() then iloc[-1].to_dict())
            regime_result: RegimeResult from regime_detector.detect()
                           (can be None — regime filter skipped)

        Returns:
            AIDecision with .approved and .confidence
        """
        regime_name = ""
        if regime_result is not None:
            regime_name = getattr(regime_result, "regime", "")

            # Hard block: non-tradeable regime
            if regime_name and regime_name not in self._tradeable:
                return AIDecision(
                    approved=False,
                    confidence=getattr(regime_result, "confidence", 0.0),
                    reason=f"Regime '{regime_name}' is not tradeable",
                    regime=regime_name,
                    signal_strength=signal.signal_strength,
                )

            # Hard block: regime confidence below threshold
            reg_conf = getattr(regime_result, "confidence", 1.0)
            if reg_conf < self._conf_threshold:
                return AIDecision(
                    approved=False,
                    confidence=reg_conf,
                    reason=f"Regime confidence too low ({reg_conf:.2f} < {self._conf_threshold})",
                    regime=regime_name,
                    signal_strength=signal.signal_strength,
                )

        # ML path or rule-based path
        if self._model is not None:
            return self._ml_evaluate(signal, feature_row, regime_name)
        else:
            return self._rule_evaluate(signal, feature_row, regime_name)

    # ── ML evaluation ─────────────────────────────────────────

    def _ml_evaluate(
        self, signal, feature_row: dict, regime: str
    ) -> AIDecision:
        """Use the trained XGBoost model to score the signal."""
        X = self._build_feature_vector(signal, feature_row)
        raw_proba = self._model.predict_proba(X)[0]

        # Map to binary [p_bad, p_good]
        if len(raw_proba) == 2:
            p_good = float(raw_proba[1])
        else:
            p_good = float(raw_proba[0])   # fallback

        approved = p_good >= self._conf_threshold

        return AIDecision(
            approved=approved,
            confidence=round(p_good, 4),
            reason=(
                f"ML score={p_good:.2f} "
                f"({'≥' if approved else '<'} threshold {self._conf_threshold})"
            ),
            model_used=True,
            signal_strength=signal.signal_strength,
            regime=regime,
        )

    # ── rule-based evaluation ─────────────────────────────────

    def _rule_evaluate(
        self, signal, feature_row: dict, regime: str
    ) -> AIDecision:
        """
        Score the signal using weighted technical rules.
        Used when no trained model exists.

        Score components (max 1.0):
          0.25  signal_strength base
          0.20  ADX trend strength
          0.15  EMA alignment with direction
          0.15  RSI not overbought/oversold
          0.15  Market structure aligned
          0.10  MACD confirms direction
        """
        def get(key: str, default: float = 0.0) -> float:
            v = feature_row.get(key, default)
            return float(v) if v is not None and str(v) != "nan" else default

        score = signal.signal_strength * 0.25

        # ADX strength
        adx = get("adx_14")
        if adx > 30:   score += 0.20
        elif adx > 20: score += 0.12
        elif adx > 15: score += 0.06

        # EMA alignment
        e20_50  = get("ema20_above_ema50")
        e50_200 = get("ema50_above_ema200")
        if signal.direction == "BUY"  and e20_50 == 1 and e50_200 == 1: score += 0.15
        if signal.direction == "SELL" and e20_50 == 0 and e50_200 == 0: score += 0.15

        # RSI
        rsi = get("rsi_14", 50)
        if signal.direction == "BUY"  and 45 < rsi < 70: score += 0.15
        if signal.direction == "SELL" and 30 < rsi < 55: score += 0.15

        # Structure
        if signal.direction == "BUY"  and get("structure_bullish") == 1: score += 0.15
        if signal.direction == "SELL" and get("structure_bearish") == 1: score += 0.15

        # MACD
        macd_h = get("macd_hist")
        if signal.direction == "BUY"  and macd_h > 0: score += 0.10
        if signal.direction == "SELL" and macd_h < 0: score += 0.10

        score = round(min(score, 1.0), 4)
        approved = score >= self._conf_threshold

        return AIDecision(
            approved=approved,
            confidence=score,
            reason=(
                f"Rule score={score:.2f} "
                f"({'≥' if approved else '<'} threshold {self._conf_threshold}) "
                f"[no ML model — run ai_engine.train()]"
            ),
            model_used=False,
            signal_strength=signal.signal_strength,
            regime=regime,
        )

    # ── training ──────────────────────────────────────────────

    def train(self, min_trades: int = 10) -> dict:
        """
        Train the XGBoost decision classifier.

        Data sources (in priority order):
          1. Real closed trades from the database (TP=1, SL=0)
          2. Synthetic data generated from feature statistics

        Returns:
            metrics dict with accuracy and trade count.
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        self._log.info("AI Decision Engine: training started")

        X, y = self._build_training_data(min_trades)

        if X is None or len(X) < 10:
            self._log.warning(
                "Insufficient training data for AI Engine — "
                "using rule-based fallback"
            )
            return {"status": "skipped", "reason": "insufficient data"}

        self._log.info(f"Training on {len(X)} samples | features={X.shape[1]}")

        # Split
        test_size = 0.2 if len(X) >= 50 else 0.0
        if test_size > 0:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
        else:
            X_tr, X_te, y_tr, y_te = X, X, y, y

        # Count classes present
        classes_present = np.unique(y_tr)
        n_classes       = len(classes_present)

        params = dict(
            n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0,
            random_state=42, n_jobs=-1,
        )
        if n_classes > 2:
            params["objective"]  = "multi:softprob"
            params["num_class"]  = n_classes
        else:
            params["objective"] = "binary:logistic"

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        preds    = model.predict(X_te)
        accuracy = round(accuracy_score(y_te, preds), 4)

        self._log.info(
            f"AI Engine XGBoost accuracy: {accuracy:.4f} "
            f"on {len(X_te)} test samples"
        )

        self._model = model
        self._save_model(model, accuracy, len(X))

        return {
            "status":        "trained",
            "accuracy":      accuracy,
            "train_samples": len(X_tr),
            "test_samples":  len(X_te),
            "n_classes":     n_classes,
        }

    def _build_training_data(
        self, min_trades: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build X, y arrays for training. Falls back to synthetic data."""
        from src.database import Trade, Feature

        # Try real trades first
        with self._db.session() as sess:
            closed = (
                sess.query(Trade)
                .filter(
                    Trade.status == "closed",
                    Trade.close_reason.in_(["tp", "sl"])
                )
                .order_by(Trade.close_time.desc())
                .limit(500)
                .all()
            )

        rows_X, rows_y = [], []

        for trade in closed:
            label = 1 if trade.close_reason == "tp" else 0
            vec   = self._trade_to_feature_vector(trade)
            rows_X.append(vec)
            rows_y.append(label)

        # We only use real trades if they have matching feature dimensions.
        # Real trades store only metadata (4 values) which cannot be stacked
        # with the 38-feature synthetic data.  Once Step 19 (Self-Learning)
        # is active, full feature vectors will be stored per trade and used here.
        # Until then, use synthetic data exclusively.
        if len(rows_X) >= min_trades:
            self._log.info(
                f"AI Engine: {len(rows_X)} real trades available — "
                f"using synthetic training (full features arrive in Step 19)"
            )

        self._log.info(
            f"AI Engine: generating 200 synthetic training samples"
        )
        X_syn, y_syn = self._generate_synthetic_data(200)
        return X_syn, y_syn

    @staticmethod
    def _trade_to_feature_vector(trade) -> list[float]:
        """Convert a Trade DB record to a feature vector."""
        direction = 1.0 if (trade.direction or "BUY") == "BUY" else 0.0
        return [
            float(trade.ai_confidence or 0.5),
            direction,
            float(trade.lot_size or 0.01),
            # Approximate features from trade metadata
            float(trade.lot_size or 0.01) * 100,  # proxy for signal strength
        ]

    @staticmethod
    def _generate_synthetic_data(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training samples.

        Good signals (label=1): high ADX, aligned EMAs, confirming RSI,
                                 positive structure, confirming MACD
        Bad signals  (label=0): low ADX, misaligned EMAs, extreme RSI,
                                 no structure, counter MACD
        """
        rng = np.random.default_rng(42)
        half = n // 2

        def make_samples(good: bool, count: int) -> np.ndarray:
            rows = []
            for _ in range(count):
                adx   = rng.uniform(25, 50) if good else rng.uniform(5,  20)
                rsi   = rng.uniform(50, 68) if good else rng.uniform(72, 90)
                ema   = 1.0                 if good else 0.0
                struct= 1.0                 if good else 0.0
                macd  = rng.uniform(0, 0.001) if good else rng.uniform(-0.001, 0)
                atr   = rng.uniform(0.03, 0.12)
                bb    = rng.uniform(0.3, 0.7) if good else rng.uniform(0.8, 1.2)
                strength = rng.uniform(0.6, 0.95) if good else rng.uniform(0.2, 0.55)
                rows.append([
                    rsi, rsi + rng.uniform(-5, 5),  # rsi_14, rsi_7
                    adx,                              # adx_14
                    adx * 0.6, adx * 0.4,            # di_plus, di_minus
                    atr, bb, atr * 2,                 # atr_pct, bb_pos, bb_width
                    macd, 1.0 if macd > 0 else 0.0,  # macd_hist, above_signal
                    rsi - 10, rsi,                    # stoch_k, stoch_d
                    ema, ema, ema, ema,               # ema alignments
                    rng.uniform(-0.05, 0.05),         # trend_slope
                    struct, 1 - struct,               # structure bull/bear
                    0.0, 0.0, 0.0,                    # premium/discount/ote
                    1.0 if good else 0.0,             # fvg_bull
                    0.0,                              # fvg_bear
                    rng.uniform(0.8, 1.5),            # volume_ratio
                    rng.uniform(-0.002, 0.005) if good else rng.uniform(-0.005, 0),
                    rng.uniform(0, 0.01),             # returns_5
                    rng.uniform(40, 80),              # body_pct
                    rng.uniform(5, 20),               # upper_wick
                    rng.uniform(5, 20),               # lower_wick
                    1.0, 1.0, 0.0, 0.0,              # session flags
                    float(rng.integers(8, 16)),       # hour
                    float(rng.integers(0, 5)),        # day_of_week
                    strength,                         # signal_strength proxy
                    1.0,                              # direction numeric
                ])
            return np.array(rows, dtype=float)

        X_good = make_samples(True,  half)
        X_bad  = make_samples(False, n - half)
        X = np.vstack([X_good, X_bad])
        y = np.array([1] * half + [0] * (n - half), dtype=int)

        # Shuffle
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    def _build_feature_vector(self, signal, feature_row: dict) -> np.ndarray:
        """Build the feature vector for ML prediction."""
        vec = []
        for feat in DECISION_FEATURES:
            v = feature_row.get(feat, 0.0)
            vec.append(float(v) if v is not None and str(v) != "nan" else 0.0)

        # Add signal-level features
        vec.append(signal.signal_strength)
        vec.append(1.0 if signal.direction == "BUY" else 0.0)

        arr = np.array(vec, dtype=float).reshape(1, -1)

        # If model has different n_features, pad/trim
        if self._model is not None:
            expected = self._model.n_features_in_
            if arr.shape[1] < expected:
                arr = np.pad(arr, ((0, 0), (0, expected - arr.shape[1])))
            elif arr.shape[1] > expected:
                arr = arr[:, :expected]

        return arr

    # ── model persistence ─────────────────────────────────────

    def _save_model(
        self, model, accuracy: float, n_samples: int
    ) -> None:
        from src.database import ModelVersion
        version_num = 1
        with self._db.session() as sess:
            version_num = self._db.get_next_version(sess, MODEL_NAME)
            file_path   = MODEL_DIR / f"{MODEL_NAME}_v{version_num}.pkl"
            joblib.dump({"model": model, "features": DECISION_FEATURES}, file_path)
            mv = ModelVersion(
                model_name=    MODEL_NAME,
                model_type=    "ml",
                version=       version_num,
                file_path=     str(file_path),
                train_date=    datetime.utcnow(),
                train_samples= n_samples,
                accuracy=      accuracy,
                notes=f"AI Decision Engine v{version_num}",
            )
            self._db.save_model_version(sess, mv)
        self._log.info(
            f"AI Engine model saved: {file_path.name} (acc={accuracy:.4f})"
        )

    def _try_load_model(self) -> bool:
        """Try to load the most recent active model from disk."""
        try:
            with self._db.session() as sess:
                mv = self._db.get_active_model(sess, MODEL_NAME)
            if mv and mv.file_path and Path(mv.file_path).exists():
                payload      = joblib.load(mv.file_path)
                self._model  = payload["model"] if isinstance(payload, dict) else payload
                self._log.info(f"AI Engine: loaded model {Path(mv.file_path).name}")
                return True
        except Exception as e:
            self._log.warning(f"AI Engine: could not load model — {e}")
        return False

    def load_model(self) -> bool:
        return self._try_load_model()

    # ── utilities ─────────────────────────────────────────────

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def confidence_threshold(self) -> float:
        return self._conf_threshold

    def set_threshold(self, threshold: float) -> None:
        self._conf_threshold = max(0.0, min(1.0, threshold))

    def summary(self) -> str:
        mode = "ML" if self._model else "Rule-based"
        return (
            f"AIDecisionEngine | mode={mode} | "
            f"threshold={self._conf_threshold} | "
            f"tradeable_regimes={len(self._tradeable)}"
        )


# ── module-level singleton ────────────────────────────────────
ai_engine = AIDecisionEngine()
