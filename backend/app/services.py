"""Queries shared by routers: behavioural context, chat history and trends."""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.crypto import decrypt
from app.models import Interaction, MoodRecord, User
from app.nlp.features import BehavioralContext


def _utc(d: dt.datetime) -> dt.datetime:
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def behavioral_context(db: Session, user: User) -> BehavioralContext:
    now = dt.datetime.now(dt.timezone.utc)
    week = now - dt.timedelta(days=7)
    moods = db.scalars(
        select(MoodRecord.score).where(MoodRecord.user_id == user.id, MoodRecord.created_at >= week)
        .order_by(MoodRecord.created_at)
    ).all()
    msgs_24h = db.scalar(
        select(func.count()).select_from(Interaction)
        .where(Interaction.user_id == user.id, Interaction.created_at >= now - dt.timedelta(hours=24))
    )
    prior_high = db.scalar(
        select(func.count()).select_from(Interaction)
        .where(Interaction.user_id == user.id, Interaction.created_at >= week, Interaction.risk_level == "high")
    )
    recent = db.scalars(
        select(Interaction.sentiment).where(Interaction.user_id == user.id)
        .order_by(Interaction.created_at.desc()).limit(10)
    ).all()
    neg_ratio = (sum(s == "negative" for s in recent) / len(recent)) if recent else 0.0
    return BehavioralContext(list(map(float, moods)), int(msgs_24h or 0), int(prior_high or 0), neg_ratio)


def chat_history(db: Session, user: User, limit: int = 6) -> list[dict]:
    rows = db.scalars(
        select(Interaction).where(Interaction.user_id == user.id).order_by(Interaction.created_at.desc()).limit(limit)
    ).all()
    hist: list[dict] = []
    for r in reversed(rows):
        if (m := decrypt(r.message)) and (resp := decrypt(r.response)):
            hist += [{"role": "user", "content": m}, {"role": "assistant", "content": resp}]
    return hist


def trend(db: Session, user: User, days: int) -> list[dict]:
    today = dt.datetime.now(dt.timezone.utc).date()
    start = today - dt.timedelta(days=days - 1)
    since = dt.datetime.combine(start, dt.time.min, tzinfo=dt.timezone.utc)
    buckets: dict[dt.date, dict] = defaultdict(lambda: {"moods": [], "sent": [], "emo": Counter(), "risk": Counter()})
    for m in db.scalars(select(MoodRecord).where(MoodRecord.user_id == user.id, MoodRecord.created_at >= since)):
        buckets[_utc(m.created_at).date()]["moods"].append(m.score)
    for i in db.scalars(select(Interaction).where(Interaction.user_id == user.id, Interaction.created_at >= since)):
        b = buckets[_utc(i.created_at).date()]
        b["sent"].append(i.sentiment)
        b["emo"][i.emotion] += 1
        b["risk"][i.risk_level] += 1
    out = []
    for k in range(days):
        d = start + dt.timedelta(days=k)
        b = buckets.get(d, {"moods": [], "sent": [], "emo": Counter(), "risk": Counter()})
        out.append({
            "date": d,
            "avg_mood": round(sum(b["moods"]) / len(b["moods"]), 2) if b["moods"] else None,
            "mood_count": len(b["moods"]),
            "interactions": len(b["sent"]),
            "negative_ratio": round(sum(s == "negative" for s in b["sent"]) / len(b["sent"]), 2) if b["sent"] else None,
            "emotions": dict(b["emo"]),
            "risk": dict(b["risk"]),
        })
    return out
