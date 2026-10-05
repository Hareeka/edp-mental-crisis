"""Feature extraction for crisis-risk classification.

Feature groups (used by the ablation experiments):
* sentiment  — sentiment probabilities and polarity
* emotion    — SRS emotion probabilities and intensity
* linguistic — TF-IDF n-grams, stylometric statistics and lexical crisis indicators
* behavioral — mood history and interaction patterns
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.nlp.analyzers import EMOTION_LABELS, SENTIMENT_LABELS
from app.nlp.crisis import INDICATOR_NAMES, indicator_counts
from app.nlp.preprocess import preprocess

FEATURE_GROUPS = ("sentiment", "emotion", "linguistic", "behavioral")

SENTIMENT_FEATURES = [f"sent_{k}" for k in SENTIMENT_LABELS] + ["sent_polarity"]
EMOTION_FEATURES = [f"emo_{k}" for k in EMOTION_LABELS] + ["emo_intensity"]
STYLE_FEATURES = [
    "len_chars_log", "len_tokens_log", "first_person_ratio", "negation_ratio",
    "question_marks", "exclamations", "uppercase_ratio",
]
CRISIS_FEATURES = [f"crisis_{k}" for k in INDICATOR_NAMES]
BEHAVIORAL_FEATURES = [
    "mood_avg_7d", "mood_slope_7d", "mood_volatility_7d", "msgs_24h_log",
    "prior_high_risk_7d", "neg_ratio_recent",
]

_FIRST_PERSON = {"i", "me", "my", "myself", "mine"}
_NEGATIONS = {"not", "no", "never", "nothing", "nobody", "none", "nor", "cannot"}


@dataclass
class BehavioralContext:
    """Summary of a user's recent history. Moods are on a 1 (very low) .. 5 (very good) scale."""

    recent_moods: list[float] = field(default_factory=list)
    msgs_24h: int = 0
    prior_high_risk_7d: int = 0
    recent_negative_ratio: float = 0.0

    def as_features(self) -> dict[str, float]:
        moods = np.asarray(self.recent_moods, dtype=float)
        if moods.size:
            avg = float(moods.mean())
            slope = float(np.polyfit(np.arange(moods.size), moods, 1)[0]) if moods.size >= 2 else 0.0
            vol = float(moods.std())
        else:
            avg, slope, vol = 3.0, 0.0, 0.0
        return {
            "mood_avg_7d": (avg - 3.0) / 2.0,
            "mood_slope_7d": slope,
            "mood_volatility_7d": vol,
            "msgs_24h_log": math.log1p(self.msgs_24h),
            "prior_high_risk_7d": float(min(self.prior_high_risk_7d, 5)),
            "neg_ratio_recent": float(self.recent_negative_ratio),
        }


def style_features(text: str) -> dict[str, float]:
    p = preprocess(text, remove_stopwords=False)
    n = max(len(p.tokens), 1)
    letters = [c for c in text if c.isalpha()]
    return {
        "len_chars_log": math.log1p(len(text)),
        "len_tokens_log": math.log1p(len(p.tokens)),
        "first_person_ratio": sum(t in _FIRST_PERSON for t in p.tokens) / n,
        "negation_ratio": sum(t in _NEGATIONS for t in p.tokens) / n,
        "question_marks": float(min(text.count("?"), 5)),
        "exclamations": float(min(text.count("!"), 5)),
        "uppercase_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
    }


def crisis_features(text: str) -> dict[str, float]:
    return {f"crisis_{k}": math.log1p(v) for k, v in indicator_counts(text).items()}


def sentiment_features(probs: dict[str, float]) -> dict[str, float]:
    f = {f"sent_{k}": probs.get(k, 0.0) for k in SENTIMENT_LABELS}
    f["sent_polarity"] = probs.get("positive", 0.0) - probs.get("negative", 0.0)
    return f


def emotion_features(probs: dict[str, float]) -> dict[str, float]:
    f = {f"emo_{k}": probs.get(k, 0.0) for k in EMOTION_LABELS}
    f["emo_intensity"] = 1.0 - probs.get("neutral", 0.0)
    return f


def dense_features(
    text: str,
    sentiment: dict[str, float],
    emotion: dict[str, float],
    behavior: BehavioralContext | None = None,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(sentiment_features(sentiment))
    feats.update(emotion_features(emotion))
    feats.update(style_features(text))
    feats.update(crisis_features(text))
    feats.update((behavior or BehavioralContext()).as_features())
    return feats


def dense_columns(groups: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    if "sentiment" in groups:
        cols += SENTIMENT_FEATURES
    if "emotion" in groups:
        cols += EMOTION_FEATURES
    if "linguistic" in groups:
        cols += STYLE_FEATURES + CRISIS_FEATURES
    if "behavioral" in groups:
        cols += BEHAVIORAL_FEATURES
    return cols
