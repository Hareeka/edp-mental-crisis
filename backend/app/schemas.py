from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, EmailStr, Field, field_validator

MOODS = {"very_low": 1, "low": 2, "okay": 3, "good": 4, "very_good": 5}


class RegisterIn(BaseModel):
    alias: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    alias: str
    email: str
    is_admin: bool
    consent_to_research: bool
    created_at: dt.datetime


class ConsentIn(BaseModel):
    consent_to_research: bool


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    mood: str | None = None

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v

    @field_validator("mood")
    @classmethod
    def valid_mood(cls, v: str | None) -> str | None:
        if v is not None and v not in MOODS:
            raise ValueError(f"mood must be one of {sorted(MOODS)}")
        return v


class RiskOut(BaseModel):
    level: str
    probabilities: dict[str, float]
    indicators: list[str]
    reasons: list[str]
    model_version: str


class ChatOut(BaseModel):
    interaction_id: int
    reply: str
    reply_mode: str
    sentiment: str
    sentiment_scores: dict[str, float]
    emotion: str
    emotion_scores: dict[str, float]
    risk: RiskOut
    recommendations: list[str]
    resources: list[dict]
    disclaimer: str
    latency_ms: float
    created_at: dt.datetime


class InteractionOut(BaseModel):
    id: int
    message: str | None
    response: str | None
    sentiment: str
    emotion: str
    risk_level: str
    created_at: dt.datetime


class MoodIn(BaseModel):
    mood: str
    intensity: int = Field(ge=1, le=10)
    note: str | None = Field(default=None, max_length=1000)

    @field_validator("mood")
    @classmethod
    def valid_mood(cls, v: str) -> str:
        if v not in MOODS:
            raise ValueError(f"mood must be one of {sorted(MOODS)}")
        return v


class MoodOut(BaseModel):
    id: int
    mood: str
    score: int
    intensity: int
    note: str | None
    created_at: dt.datetime


class TrendPoint(BaseModel):
    date: dt.date
    avg_mood: float | None
    mood_count: int
    interactions: int
    negative_ratio: float | None
    emotions: dict[str, int]
    risk: dict[str, int]


class DashboardOut(BaseModel):
    mood_average_7d: float | None
    latest_mood: MoodOut | None
    interactions_7d: int
    dominant_emotion_7d: str | None
    recent_interactions: list[InteractionOut]
    trend: list[TrendPoint]


class FeedbackIn(BaseModel):
    interaction_id: int | None = None
    usefulness: int | None = Field(default=None, ge=1, le=5)
    relevance: int | None = Field(default=None, ge=1, le=5)
    appropriateness: int | None = Field(default=None, ge=1, le=5)
    experience: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackOut(BaseModel):
    id: int
    created_at: dt.datetime
