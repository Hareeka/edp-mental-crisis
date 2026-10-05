"""Lexical crisis indicators.

These patterns are deliberately recall-oriented: they feed the classifier as features and
drive a deterministic safety override so explicit statements are never treated as ordinary chat.
"""

from __future__ import annotations

import re

_FLAGS = re.IGNORECASE

# Explicit statements of suicidal intent, plans, or self-harm. Any match forces High risk.
EXPLICIT_PATTERNS = [
    r"\b(kill|killing|hang|hanging|shoot|shooting)\s+my\s?self\b",
    r"\b(end|ending|take|taking)\s+my\s+(own\s+)?life\b",
    r"\bcommit(ting)?\s+suicide\b",
    r"\bsuicid(e|al)\b",
    r"\bwant(s)?\s+to\s+die\b",
    r"\bgoing\s+to\s+die\s+(tonight|today|soon)\b",
    r"\b(plan|planning|planned)\s+(to|on)\s+(die|dying|end\s+it|kill)\b",
    r"\b(cut|cutting|hurt|hurting|harm|harming|burn|burning)\s+my\s?self\b",
    r"\bself[\s-]?harm\b",
    r"\boverdos(e|ed|ing)\b",
    r"\b(wrote|writing|write)\s+(a|my)\s+(suicide\s+)?note\b",
    r"\bbetter\s+off\s+(dead|without\s+me)\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on|keep\s+going)\b",
    r"\bdon'?t\s+want\s+to\s+(live|be\s+alive|exist|wake\s+up)\b",
    r"\b(say|saying)\s+goodbye\s+(to\s+everyone|forever)\b",
]

INDICATOR_GROUPS: dict[str, list[str]] = {
    "hopelessness": [
        r"\bhopeless\b", r"\bno\s+hope\b", r"\bpointless\b", r"\bno\s+point\b",
        r"\bnothing\s+(matters|will\s+change|helps)\b", r"\bnever\s+get\s+better\b",
        r"\bgive\s+up\b", r"\bgiving\s+up\b", r"\bno\s+future\b", r"\bcan'?t\s+go\s+on\b",
    ],
    "burden": [
        r"\b(a\s+)?burden\b", r"\beveryone\s+would\s+be\s+better\b", r"\bnobody\s+would\s+(care|notice|miss)\b",
        r"\bno\s+one\s+would\s+(care|notice|miss)\b", r"\bworthless\b", r"\bwaste\s+of\s+space\b",
    ],
    "entrapment": [
        r"\bcan'?t\s+(do\s+this|take\s+(it|this))\s+anymore\b", r"\btrapped\b", r"\bno\s+way\s+out\b",
        r"\bcan'?t\s+escape\b", r"\bstuck\s+forever\b", r"\btoo\s+much\s+to\s+(bear|handle)\b",
    ],
    "death_reference": [
        r"\bdie\b", r"\bdying\b", r"\bdead\b", r"\bdeath\b", r"\bdisappear\b", r"\bnot\s+be\s+here\b",
        r"\bsleep\s+forever\b", r"\bend\s+it(\s+all)?\b", r"\bnot\s+wake\s+up\b",
    ],
    "method_or_means": [
        r"\bpills\b", r"\bbottle\s+of\b", r"\bpainkillers\b", r"\bparacetamol\b", r"\btylenol\b",
        r"\brope\b", r"\bnoose\b", r"\bbridge\b", r"\bgun\b", r"\bblade\b", r"\brazor\b",
        r"\bjump\s+(off|in\s+front)\b",
    ],
    "isolation": [
        r"\balone\b", r"\blonely\b", r"\bno\s+friends\b", r"\bno\s+one\s+(cares|understands)\b",
        r"\bnobody\s+(cares|understands)\b", r"\bisolated\b",
    ],
}

_EXPLICIT = [re.compile(p, _FLAGS) for p in EXPLICIT_PATTERNS]
_GROUPS = {k: [re.compile(p, _FLAGS) for p in v] for k, v in INDICATOR_GROUPS.items()}

INDICATOR_NAMES = ["explicit", *INDICATOR_GROUPS.keys()]


def _prep(text: str) -> str:
    return (text or "").replace("\u2019", "'").lower()


def explicit_matches(text: str) -> list[str]:
    t = _prep(text)
    return [p.pattern for p in _EXPLICIT if p.search(t)]


def has_explicit_crisis_language(text: str) -> bool:
    t = _prep(text)
    return any(p.search(t) for p in _EXPLICIT)


def indicator_counts(text: str) -> dict[str, int]:
    t = _prep(text)
    counts = {"explicit": sum(1 for p in _EXPLICIT if p.search(t))}
    for name, pats in _GROUPS.items():
        counts[name] = sum(len(p.findall(t)) for p in pats)
    return counts


def detected_indicators(text: str) -> list[str]:
    return [k for k, v in indicator_counts(text).items() if v > 0]
