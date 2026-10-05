from __future__ import annotations

import datetime as dt
from collections import Counter

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.crypto import decrypt, encrypt
from app.db import get_db
from app.models import Feedback, Interaction, MoodRecord, User
from app.schemas import MOODS, DashboardOut, FeedbackIn, FeedbackOut, InteractionOut, MoodIn, MoodOut, TrendPoint
from app.security import get_current_user
from app.services import trend

router = APIRouter(prefix="/api", tags=["moods"])


def _mood_out(m: MoodRecord) -> MoodOut:
    return MoodOut(id=m.id, mood=m.mood, score=m.score, intensity=m.intensity, note=decrypt(m.note),
                   created_at=m.created_at)


@router.post("/moods", response_model=MoodOut, status_code=status.HTTP_201_CREATED)
def add_mood(body: MoodIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> MoodOut:
    m = MoodRecord(user_id=user.id, mood=body.mood, score=MOODS[body.mood], intensity=body.intensity,
                   note=encrypt(body.note) if body.note else None)
    db.add(m)
    db.commit()
    return _mood_out(m)


@router.get("/moods", response_model=list[MoodOut])
def list_moods(
    days: int = Query(30, ge=1, le=365), user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[MoodOut]:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    rows = db.scalars(
        select(MoodRecord).where(MoodRecord.user_id == user.id, MoodRecord.created_at >= since)
        .order_by(MoodRecord.created_at)
    )
    return [_mood_out(m) for m in rows]


@router.get("/trends", response_model=list[TrendPoint])
def trends(
    days: int = Query(14, ge=1, le=90), user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> list[TrendPoint]:
    return [TrendPoint(**p) for p in trend(db, user, days)]


@router.get("/dashboard", response_model=DashboardOut)
def dashboard(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> DashboardOut:
    week = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
    moods = db.scalars(select(MoodRecord).where(MoodRecord.user_id == user.id, MoodRecord.created_at >= week)).all()
    latest = db.scalar(select(MoodRecord).where(MoodRecord.user_id == user.id).order_by(MoodRecord.created_at.desc()))
    inter = db.scalars(
        select(Interaction).where(Interaction.user_id == user.id, Interaction.created_at >= week)
        .order_by(Interaction.created_at.desc())
    ).all()
    emo = Counter(i.emotion for i in inter)
    return DashboardOut(
        mood_average_7d=round(sum(m.score for m in moods) / len(moods), 2) if moods else None,
        latest_mood=_mood_out(latest) if latest else None,
        interactions_7d=len(inter),
        dominant_emotion_7d=emo.most_common(1)[0][0] if emo else None,
        recent_interactions=[
            InteractionOut(id=r.id, message=decrypt(r.message), response=None, sentiment=r.sentiment,
                           emotion=r.emotion, risk_level=r.risk_level, created_at=r.created_at)
            for r in inter[:5]
        ],
        trend=[TrendPoint(**p) for p in trend(db, user, 14)],
    )


@router.post("/feedback", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
def feedback(body: FeedbackIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> FeedbackOut:
    if body.interaction_id is not None:
        owned = db.scalar(select(Interaction.id).where(Interaction.id == body.interaction_id,
                                                       Interaction.user_id == user.id))
        if owned is None:
            body.interaction_id = None
    f = Feedback(user_id=user.id, interaction_id=body.interaction_id, usefulness=body.usefulness,
                 relevance=body.relevance, appropriateness=body.appropriateness, experience=body.experience,
                 comment=encrypt(body.comment) if body.comment else None)
    db.add(f)
    db.commit()
    return FeedbackOut(id=f.id, created_at=f.created_at)
