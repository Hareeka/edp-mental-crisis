"""End-to-end analysis: preprocess -> sentiment/emotion -> features -> risk -> reply."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.nlp.analyzers import dominant_emotion, get_analyzer, sentiment_label
from app.nlp.features import BehavioralContext
from app.nlp.preprocess import preprocess
from app.nlp.responder import CompanionReply, generate_reply
from app.nlp.risk import RiskResult, get_risk_classifier


@dataclass
class Analysis:
    sentiment: str
    sentiment_scores: dict[str, float]
    emotion: str
    emotion_scores: dict[str, float]
    risk: RiskResult
    reply: CompanionReply
    latency_ms: float


def analyze_message(text: str, behavior: BehavioralContext | None = None, history: list[dict] | None = None) -> Analysis:
    t0 = time.perf_counter()
    processed = preprocess(text)
    sent, emo = get_analyzer().analyze(text)
    risk = get_risk_classifier().classify(text, processed.clean, sent, emo, behavior)
    emotion = dominant_emotion(emo)
    reply = generate_reply(text, emotion, risk.level, history)
    return Analysis(
        sentiment=sentiment_label(sent),
        sentiment_scores={k: round(v, 4) for k, v in sent.items()},
        emotion=emotion,
        emotion_scores={k: round(v, 4) for k, v in emo.items()},
        risk=risk,
        reply=reply,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
