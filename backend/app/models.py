from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alias: Mapped[str] = mapped_column(String(80))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_to_research: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    moods: Mapped[list[MoodRecord]] = relationship(back_populates="user", cascade="all, delete-orphan")
    interactions: Mapped[list[Interaction]] = relationship(back_populates="user", cascade="all, delete-orphan")
    feedback: Mapped[list[Feedback]] = relationship(back_populates="user", cascade="all, delete-orphan")


class MoodRecord(Base):
    __tablename__ = "mood_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    mood: Mapped[str] = mapped_column(String(32))
    score: Mapped[int] = mapped_column(Integer)  # 1 (very low) .. 5 (very good)
    intensity: Mapped[int] = mapped_column(Integer)  # 1..10
    note: Mapped[str | None] = mapped_column(Text, nullable=True)  # encrypted
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    user: Mapped[User] = relationship(back_populates="moods")


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)  # encrypted
    response: Mapped[str | None] = mapped_column(Text, nullable=True)  # encrypted
    sentiment: Mapped[str] = mapped_column(String(16))
    sentiment_scores: Mapped[dict] = mapped_column(JSON)
    emotion: Mapped[str] = mapped_column(String(16))
    emotion_scores: Mapped[dict] = mapped_column(JSON)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    risk_scores: Mapped[dict] = mapped_column(JSON)
    indicators: Mapped[list] = mapped_column(JSON, default=list)
    model_version: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    user: Mapped[User] = relationship(back_populates="interactions")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    interaction_id: Mapped[int | None] = mapped_column(ForeignKey("interactions.id", ondelete="SET NULL"), nullable=True)
    usefulness: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1..5
    relevance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    appropriateness: Mapped[int | None] = mapped_column(Integer, nullable=True)
    experience: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)  # encrypted
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="feedback")
