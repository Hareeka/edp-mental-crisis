"""Crisis-risk model (training + inference) and the runtime classifier with safety overrides."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.nlp.crisis import detected_indicators, explicit_matches
from app.nlp.features import BehavioralContext, dense_columns, dense_features

log = logging.getLogger(__name__)

RISK_LABELS = ["low", "moderate", "high"]
DEFAULT_THRESHOLDS = {"high": 0.5, "moderate": 0.5}


class RiskModel:
    """TF-IDF + scaled dense features (+ optional sentence embeddings) feeding any sklearn-style classifier."""

    def __init__(
        self,
        estimator,
        groups: tuple[str, ...],
        use_tfidf: bool = True,
        use_embeddings: bool = False,
        tfidf_params: dict | None = None,
    ) -> None:
        self.estimator = estimator
        self.groups = tuple(groups)
        self.use_tfidf = use_tfidf and "linguistic" in self.groups
        self.use_embeddings = use_embeddings
        self.tfidf_params = tfidf_params or {
            "ngram_range": (1, 2), "max_features": 20000, "min_df": 2, "max_df": 0.9, "sublinear_tf": True,
        }
        self.columns = dense_columns(self.groups)
        self.tfidf: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None
        self.thresholds = dict(DEFAULT_THRESHOLDS)

    def _matrix(self, clean_texts, dense: pd.DataFrame, emb: np.ndarray | None, fit: bool):
        blocks = []
        if self.columns:
            x = dense[self.columns].to_numpy(dtype=float)
            if fit:
                self.scaler = StandardScaler().fit(x)
            blocks.append(sparse.csr_matrix(self.scaler.transform(x)))
        if self.use_tfidf:
            if fit:
                self.tfidf = TfidfVectorizer(**self.tfidf_params).fit(clean_texts)
            blocks.append(self.tfidf.transform(clean_texts))
        if self.use_embeddings:
            if emb is None:
                raise ValueError("embeddings required for this model")
            blocks.append(sparse.csr_matrix(emb))
        return sparse.hstack(blocks).tocsr()

    def fit(self, clean_texts, dense: pd.DataFrame, y, emb: np.ndarray | None = None, sample_weight=None):
        x = self._matrix(clean_texts, dense, emb, fit=True)
        if sample_weight is not None:
            self.estimator.fit(x, y, sample_weight=sample_weight)
        else:
            self.estimator.fit(x, y)
        return self

    def predict_proba(self, clean_texts, dense: pd.DataFrame, emb: np.ndarray | None = None) -> np.ndarray:
        return self.estimator.predict_proba(self._matrix(clean_texts, dense, emb, fit=False))

    def decide(self, proba: np.ndarray) -> np.ndarray:
        return apply_thresholds(proba, self.thresholds)


def apply_thresholds(proba: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Recall-oriented ordinal decision: High if P(high) >= t_high, else Moderate if P(mod)+P(high) >= t_mod."""
    proba = np.atleast_2d(proba)
    out = np.zeros(len(proba), dtype=int)
    out[proba[:, 1] + proba[:, 2] >= thresholds["moderate"]] = 1
    out[proba[:, 2] >= thresholds["high"]] = 2
    return out


@dataclass
class RiskResult:
    level: str
    probabilities: dict[str, float]
    indicators: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    model_version: str = "rules"


class RiskClassifier:
    def __init__(self, model_path: Path | None, backend: str) -> None:
        self.model: RiskModel | None = None
        self.meta: dict = {}
        self._embedder = None
        if model_path and Path(model_path).exists():
            import joblib

            bundle = joblib.load(model_path)
            if bundle["meta"].get("nlp_backend") == backend:
                self.model, self.meta = bundle["model"], bundle["meta"]
            else:
                log.warning(
                    "Risk model trained with %s features but runtime backend is %s; using rule-based scorer",
                    bundle["meta"].get("nlp_backend"), backend,
                )
        elif model_path:
            log.warning("Risk model not found at %s; using rule-based scorer", model_path)

    @property
    def version(self) -> str:
        return self.meta.get("version", "rules")

    def _embed(self, text: str) -> np.ndarray:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.meta["embedding_model"])
        return self._embedder.encode([text], normalize_embeddings=True)

    def _rule_proba(self, feats: dict[str, float]) -> np.ndarray:
        score = (
            1.2 * feats["sent_negative"] + 0.8 * (feats["emo_sadness"] + feats["emo_loneliness"] + feats["emo_anxiety"])
            + 1.5 * (feats["crisis_hopelessness"] + feats["crisis_burden"] + feats["crisis_entrapment"])
            + 1.0 * feats["crisis_death_reference"] + 1.5 * feats["crisis_method_or_means"]
        )
        high = 1 / (1 + np.exp(-(score - 3.0) * 2))
        mod = (1 - high) / (1 + np.exp(-(score - 1.2) * 3))
        return np.array([[1 - high - mod, mod, high]])

    def classify(
        self,
        text: str,
        clean_text: str,
        sentiment: dict[str, float],
        emotion: dict[str, float],
        behavior: BehavioralContext | None = None,
    ) -> RiskResult:
        feats = dense_features(text, sentiment, emotion, behavior)
        if self.model is not None:
            emb = self._embed(text) if self.model.use_embeddings else None
            proba = self.model.predict_proba([clean_text], pd.DataFrame([feats]), emb)
            thresholds = self.model.thresholds
        else:
            proba = self._rule_proba(feats)
            thresholds = DEFAULT_THRESHOLDS
        level_idx = int(apply_thresholds(proba, thresholds)[0])
        reasons = [f"model:{RISK_LABELS[level_idx]}"]

        if explicit_matches(text):
            level_idx = 2
            reasons.append("override:explicit_crisis_language")
        elif level_idx == 0 and behavior is not None:
            b = behavior.as_features()
            if b["prior_high_risk_7d"] >= 1 or (b["mood_avg_7d"] <= -0.5 and b["mood_slope_7d"] < 0):
                level_idx = 1
                reasons.append("escalation:behavioral_history")

        return RiskResult(
            level=RISK_LABELS[level_idx],
            probabilities={k: round(float(v), 4) for k, v in zip(RISK_LABELS, proba[0])},
            indicators=detected_indicators(text),
            reasons=reasons,
            model_version=self.version,
        )


@lru_cache(maxsize=1)
def get_risk_classifier() -> RiskClassifier:
    s = get_settings()
    from app.nlp.analyzers import get_analyzer

    return RiskClassifier(s.risk_model_path, get_analyzer().backend_name)
