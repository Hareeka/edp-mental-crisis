"""Text preprocessing: normalisation, tokenisation, stop-word handling, lemmatisation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

MAX_INPUT_CHARS = 5000

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"[@#]\w+")
_NON_ALPHA_RE = re.compile(r"[^a-z' ]+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

_CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'m": " am",
    "'ll": " will",
    "'ve": " have",
    "'d": " would",
}

# Negations and first-person pronouns carry signal for distress detection, so they are kept.
_KEEP = {
    "no", "not", "nor", "never", "nothing", "nobody", "none", "i", "me", "my", "myself",
    "alone", "against", "very", "too", "more", "most", "again", "all", "any", "only",
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with", "about",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "then", "once", "here", "there",
    "when", "where", "why", "how", "both", "each", "few", "other", "some", "such", "own",
    "same", "so", "than", "s", "t", "just", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "it", "its", "itself",
    "this", "that", "these", "those", "am", "which", "who", "whom", "what", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them", "their", "would",
    "should", "could", "will", "shall", "can", "may", "might", "must", "im", "ive",
}
_STOPWORDS -= _KEEP


@dataclass
class ProcessedText:
    raw: str
    normalized: str
    tokens: list[str] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)

    @property
    def clean(self) -> str:
        return " ".join(self.lemmas)


@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer

        lem = WordNetLemmatizer()
        lem.lemmatize("tests")
        return lem.lemmatize
    except Exception:  # WordNet data not installed: fall back to a light suffix stripper.
        return _simple_lemma


def _simple_lemma(token: str) -> str:
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("s", "")):
        if token.endswith(suffix) and len(token) > len(suffix) + 2 and not token.endswith("ss"):
            return token[: -len(suffix)] + repl
    return token


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text[:MAX_INPUT_CHARS].lower().replace("\u2019", "'")
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def preprocess(text: str, remove_stopwords: bool = True) -> ProcessedText:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    norm = normalize(text)
    tokens = tokenize(norm)
    kept = [t for t in tokens if not (remove_stopwords and t in _STOPWORDS)]
    lemma = _lemmatizer()
    return ProcessedText(raw=text, normalized=norm, tokens=tokens, lemmas=[lemma(t) for t in kept])
