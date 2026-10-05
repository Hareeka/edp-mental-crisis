from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crypto import decrypt, encrypt
from app.db import get_db
from app.models import Interaction, MoodRecord, User
from app.nlp.pipeline import analyze_message
from app.nlp.responder import CRISIS_RESOURCES, DISCLAIMER
from app.schemas import MOODS, ChatIn, ChatOut, InteractionOut, RiskOut
from app.security import get_current_user
from app.services import behavioral_context, chat_history

router = APIRouter(prefix="/api", tags=["chat"])
log = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ChatOut:
    if body.mood:
        db.add(MoodRecord(user_id=user.id, mood=body.mood, score=MOODS[body.mood], intensity=5))
        db.flush()
    try:
        a = analyze_message(body.message, behavioral_context(db, user), chat_history(db, user))
    except Exception:
        log.exception("analysis failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Analysis temporarily unavailable") from None

    store = get_settings().store_message_text
    row = Interaction(
        user_id=user.id,
        message=encrypt(body.message) if store else None,
        response=encrypt(a.reply.text) if store else None,
        sentiment=a.sentiment, sentiment_scores=a.sentiment_scores,
        emotion=a.emotion, emotion_scores=a.emotion_scores,
        risk_level=a.risk.level, risk_scores=a.risk.probabilities, indicators=a.risk.indicators,
        model_version=a.risk.model_version, latency_ms=a.latency_ms,
    )
    db.add(row)
    db.commit()
    # Log metadata only — never message content.
    log.info("chat user=%s interaction=%s risk=%s emotion=%s mode=%s latency_ms=%.0f",
             user.id, row.id, a.risk.level, a.emotion, a.reply.mode, a.latency_ms)
    return ChatOut(
        interaction_id=row.id, reply=a.reply.text, reply_mode=a.reply.mode,
        sentiment=a.sentiment, sentiment_scores=a.sentiment_scores,
        emotion=a.emotion, emotion_scores=a.emotion_scores,
        risk=RiskOut(**a.risk.__dict__),
        recommendations=a.reply.recommendations, resources=a.reply.resources,
        disclaimer=DISCLAIMER, latency_ms=a.latency_ms, created_at=row.created_at,
    )


@router.get("/interactions", response_model=list[InteractionOut])
def interactions(
    limit: int = Query(50, ge=1, le=200), user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[InteractionOut]:
    rows = db.scalars(
        select(Interaction).where(Interaction.user_id == user.id).order_by(Interaction.created_at.desc()).limit(limit)
    ).all()
    return [
        InteractionOut(id=r.id, message=decrypt(r.message), response=decrypt(r.response), sentiment=r.sentiment,
                       emotion=r.emotion, risk_level=r.risk_level, created_at=r.created_at)
        for r in reversed(rows)
    ]


@router.delete("/interactions", status_code=status.HTTP_204_NO_CONTENT)
def clear_interactions(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    for r in db.scalars(select(Interaction).where(Interaction.user_id == user.id)):
        db.delete(r)
    db.commit()


@router.get("/resources")
def resources() -> dict:
    return {"crisis": CRISIS_RESOURCES, "disclaimer": DISCLAIMER}
