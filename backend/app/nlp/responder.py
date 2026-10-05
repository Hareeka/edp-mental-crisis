"""Risk-aware response generation.

High risk always returns a fixed safety-focused message (never free-form generation).
Low/Moderate use templates keyed by emotion, optionally rephrased by an OpenAI-compatible LLM.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

DISCLAIMER = (
    "MindCompanion is an AI support tool, not a therapist. Risk levels are automated indications, "
    "not a diagnosis. If you are in danger, contact local emergency services."
)

CRISIS_RESOURCES = [
    {"name": "Emergency services", "detail": "Call your local emergency number (e.g. 911 US/Canada, 112 EU, 999 UK)."},
    {"name": "988 Suicide & Crisis Lifeline (US)", "detail": "Call or text 988, or chat at 988lifeline.org."},
    {"name": "Samaritans (UK & ROI)", "detail": "Call 116 123, free, any time."},
    {"name": "Tele-MANAS (India)", "detail": "Call 14416 or 1-800-891-4416."},
    {"name": "Find a helpline (worldwide)", "detail": "findahelpline.com lists free, confidential services by country."},
]

WELLNESS = {
    "sadness": ["Take a short walk outside, even five minutes", "Write down one small thing that went okay today",
                "Reach out to someone you trust for a quick chat"],
    "fear": ["Try 4-7-8 breathing: in for 4, hold for 7, out for 8", "Name five things you can see around you",
             "Write the worry down, then one small step you could take"],
    "anxiety": ["Try box breathing: 4 seconds in, hold, out, hold", "Limit caffeine and screens for the next hour",
                "Break the next task into one tiny, doable step"],
    "anger": ["Step away for a few minutes before responding", "Do something physical: stretch, walk, shake it out",
              "Put the feeling into words in a private note"],
    "loneliness": ["Send a message to one person you haven't talked to in a while",
                   "Look for a local or online group around something you enjoy",
                   "Spend time somewhere with gentle background company, like a cafe or library"],
    "joy": ["Savour it: note what made today good", "Share the good news with someone",
            "Keep a gratitude list going"],
    "neutral": ["Check in with your body: are you rested, fed and hydrated?", "Try a 3-minute mindfulness pause",
                "Plan one enjoyable thing for later today"],
}

COPING = [
    "Grounding: slowly notice 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste",
    "Name the feeling and rate it from 1 to 10; naming it can make it a little more manageable",
    "Keep basics steady: sleep, food, water and a little movement",
    "Consider talking to a counsellor, doctor or mental health professional",
]

_LOW_OPENERS = {
    "sadness": "I'm sorry today feels heavy. Thank you for sharing it with me.",
    "fear": "That sounds unsettling. It makes sense to feel on edge about it.",
    "anxiety": "It sounds like your mind is working overtime right now. That's exhausting.",
    "anger": "That sounds really frustrating. Your feelings are valid.",
    "loneliness": "Feeling disconnected is hard. I'm glad you reached out.",
    "joy": "That's lovely to hear. I'm glad things are going well.",
    "neutral": "Thanks for checking in. I'm here to listen.",
}

_MOD_OPENERS = {
    "sadness": "It sounds like you've been carrying a lot, and it's been hard for a while.",
    "fear": "It sounds like you're feeling really frightened, and that's a lot to hold on your own.",
    "anxiety": "It sounds like the worry has been building and it's wearing you down.",
    "anger": "It sounds like a lot of hurt and frustration has been building up.",
    "loneliness": "It sounds like you've been feeling very alone, and that really hurts.",
    "joy": "I hear some good moments, and also that things have been difficult.",
    "neutral": "It sounds like things have been difficult lately.",
}

HIGH_RISK_MESSAGE = (
    "I'm really glad you told me, and I'm concerned about your safety. You deserve support from a real person "
    "right now. If you might act on these thoughts or are in immediate danger, please call your local emergency "
    "number now. You can also contact a crisis line (below) — they're free, confidential and available any time. "
    "If you can, reach out to someone you trust and let them know how you're feeling, and move away from anything "
    "you could use to hurt yourself. I'm an AI and can't help in an emergency, but I'm here to keep talking while "
    "you reach out."
)

_SYSTEM_PROMPT = (
    "You are a supportive, non-judgemental wellbeing companion. You are not a therapist and must not diagnose. "
    "Reply in 2-4 short sentences, warm and specific to what the user said, end with one gentle open question. "
    "Do not give medical advice. Detected emotion: {emotion}. Risk level: {risk}."
)


@dataclass
class CompanionReply:
    text: str
    recommendations: list[str] = field(default_factory=list)
    resources: list[dict] = field(default_factory=list)
    mode: str = "template"  # template | llm | safety


def _pick(options: list[str], seed: str, k: int) -> list[str]:
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    start = h % len(options)
    return [options[(start + i) % len(options)] for i in range(min(k, len(options)))]


def _llm_reply(message: str, history: list[dict], emotion: str, risk: str) -> str | None:
    s = get_settings()
    if not (s.llm_api_base and s.llm_api_key):
        return None
    msgs = [{"role": "system", "content": _SYSTEM_PROMPT.format(emotion=emotion, risk=risk)}]
    msgs += history[-6:] + [{"role": "user", "content": message}]
    try:
        r = httpx.post(
            f"{s.llm_api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
            json={"model": s.llm_model, "messages": msgs, "temperature": 0.6, "max_tokens": 200},
            timeout=s.llm_timeout_s,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        log.warning("LLM request failed; using template reply", exc_info=True)
        return None


def generate_reply(message: str, emotion: str, risk: str, history: list[dict] | None = None) -> CompanionReply:
    if risk == "high":
        return CompanionReply(text=HIGH_RISK_MESSAGE, recommendations=[], resources=CRISIS_RESOURCES, mode="safety")

    emotion = emotion if emotion in WELLNESS else "neutral"
    if risk == "moderate":
        recs = _pick(COPING, message, 2) + _pick(WELLNESS[emotion], message, 1)
        text = (
            f"{_MOD_OPENERS[emotion]} You don't have to go through this alone — talking with someone you trust or a "
            "mental health professional could really help. Would you like to tell me more about what's been going on?"
        )
        resources = CRISIS_RESOURCES[-1:]
    else:
        recs = _pick(WELLNESS[emotion], message, 2)
        text = f"{_LOW_OPENERS[emotion]} Would you like to talk more about it, or try something small together?"
        resources = []

    llm = _llm_reply(message, history or [], emotion, risk)
    return CompanionReply(text=llm or text, recommendations=recs, resources=resources, mode="llm" if llm else "template")
