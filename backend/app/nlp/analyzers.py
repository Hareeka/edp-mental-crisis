"""Sentiment and emotion analysers.

Two interchangeable backends:
* ``transformer`` — cardiffnlp/twitter-roberta-base-sentiment-latest and
  j-hartmann/emotion-english-distilroberta-base (Hugging Face).
* ``lexicon`` — VADER sentiment plus keyword emotion scoring; no model download, used in CI.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from functools import lru_cache

from app.config import get_settings

log = logging.getLogger(__name__)

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
EMOTION_LABELS = ["sadness", "fear", "anger", "anxiety", "loneliness", "joy", "neutral"]

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

_ANXIETY_WORDS = re.compile(
    r"\b(anxious|anxiety|panic|panicking|nervous|worried|worry|worrying|overthink\w*|"
    r"restless|on edge|dread\w*|stress\w*|tense|uneasy|racing thoughts|can'?t breathe)\b",
    re.IGNORECASE,
)
_LONELY_WORDS = re.compile(
    r"\b(lonely|loneliness|alone|isolated|no friends|no one|nobody|left out|abandoned|"
    r"unwanted|disconnected|by myself|miss (them|him|her|my))\b",
    re.IGNORECASE,
)

_LEXICON = {
    "sadness": r"\b(sad|unhappy|depress\w*|down|cry\w*|tears|hopeless|empty|miserable|grief|heartbroken|hurt|numb|tired|exhausted|worthless)\b",
    "fear": r"\b(afraid|scared|fear\w*|terrified|frightened|unsafe|threat\w*)\b",
    "anger": r"\b(angry|mad|furious|hate|annoy\w*|irritat\w*|frustrat\w*|rage|pissed|resent\w*)\b",
    "joy": r"\b(happy|glad|great|good|joy\w*|excited|love|grateful|thankful|calm|relaxed|proud|better|awesome|wonderful)\b",
}
_LEXICON_RE = {k: re.compile(v, re.IGNORECASE) for k, v in _LEXICON.items()}


def _cue_strength(pattern: re.Pattern, text: str) -> float:
    hits = len(pattern.findall(text or ""))
    return 1.0 - math.exp(-0.9 * hits)


def _normalize(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values()) or 1.0
    return {k: v / total for k, v in d.items()}


def map_emotions(base: dict[str, float], text: str) -> dict[str, float]:
    """Map a 7-way Ekman-style distribution to the SRS categories.

    Anxiety and loneliness are not output by the base model; mass is shifted from fear/sadness/neutral
    towards them in proportion to lexical cue strength.
    """
    p = {
        "sadness": base.get("sadness", 0.0),
        "fear": base.get("fear", 0.0),
        "anger": base.get("anger", 0.0) + base.get("disgust", 0.0),
        "joy": base.get("joy", 0.0) + 0.5 * base.get("surprise", 0.0),
        "neutral": base.get("neutral", 0.0) + 0.5 * base.get("surprise", 0.0),
        "anxiety": 0.0,
        "loneliness": 0.0,
    }
    anx = _cue_strength(_ANXIETY_WORDS, text)
    lon = _cue_strength(_LONELY_WORDS, text)
    move = 0.25 * p["fear"] + anx * (0.6 * p["fear"] + 0.3 * p["neutral"] + 0.2 * p["sadness"])
    p["fear"] -= 0.25 * p["fear"] + anx * 0.6 * p["fear"]
    p["neutral"] -= anx * 0.3 * p["neutral"]
    p["sadness"] -= anx * 0.2 * p["sadness"]
    p["anxiety"] += move
    move = lon * (0.6 * p["sadness"] + 0.3 * p["neutral"])
    p["sadness"] -= lon * 0.6 * p["sadness"]
    p["neutral"] -= lon * 0.3 * p["neutral"]
    p["loneliness"] += move
    p = _normalize({k: max(v, 0.0) for k, v in p.items()})
    return {k: p[k] for k in EMOTION_LABELS}


class _TransformerBackend:
    def __init__(self) -> None:
        from transformers import pipeline

        self._sent = pipeline("text-classification", model=SENTIMENT_MODEL, top_k=None, truncation=True, max_length=256)
        self._emo = pipeline("text-classification", model=EMOTION_MODEL, top_k=None, truncation=True, max_length=256)
        self._lock = threading.Lock()

    def analyze_batch(self, texts: list[str]) -> list[tuple[dict, dict]]:
        safe = [t if t.strip() else "." for t in texts]
        with self._lock:
            sent = self._sent(safe, batch_size=16)
            emo = self._emo(safe, batch_size=16)
        out = []
        for t, s, e in zip(texts, sent, emo):
            sp = {d["label"].lower(): float(d["score"]) for d in s}
            ep = {d["label"].lower(): float(d["score"]) for d in e}
            out.append(({k: sp.get(k, 0.0) for k in SENTIMENT_LABELS}, map_emotions(ep, t)))
        return out


class _LexiconBackend:
    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._vader = SentimentIntensityAnalyzer()

    def analyze_batch(self, texts: list[str]) -> list[tuple[dict, dict]]:
        out = []
        for t in texts:
            v = self._vader.polarity_scores(t or "")
            sent = _normalize({"negative": v["neg"], "neutral": v["neu"], "positive": v["pos"]})
            base = {k: 0.15 + 2.0 * _cue_strength(r, t) for k, r in _LEXICON_RE.items()}
            base["neutral"] = 0.6
            out.append((sent, map_emotions(_normalize(base), t)))
        return out


class TextAnalyzer:
    def __init__(self, backend: str) -> None:
        self.backend_name = backend
        if backend == "transformer":
            try:
                self._impl = _TransformerBackend()
            except Exception:  # pragma: no cover - network/model failure
                log.exception("Transformer models unavailable; falling back to lexicon backend")
                self.backend_name = "lexicon"
                self._impl = _LexiconBackend()
        else:
            self._impl = _LexiconBackend()

    def analyze(self, text: str) -> tuple[dict[str, float], dict[str, float]]:
        return self._impl.analyze_batch([text])[0]

    def analyze_batch(self, texts: list[str]) -> list[tuple[dict, dict]]:
        return self._impl.analyze_batch(texts)


@lru_cache(maxsize=2)
def get_analyzer(backend: str | None = None) -> TextAnalyzer:
    return TextAnalyzer(backend or get_settings().nlp_backend)


def sentiment_label(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def dominant_emotion(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)
