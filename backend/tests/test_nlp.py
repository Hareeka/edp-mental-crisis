import numpy as np
import pytest

from app.nlp.analyzers import EMOTION_LABELS, get_analyzer, map_emotions
from app.nlp.crisis import detected_indicators, has_explicit_crisis_language
from app.nlp.features import BehavioralContext, dense_columns, dense_features
from app.nlp.preprocess import normalize, preprocess
from app.nlp.responder import CRISIS_RESOURCES, HIGH_RISK_MESSAGE, generate_reply
from app.nlp.risk import RiskClassifier, apply_thresholds


def test_preprocess_normalizes_and_keeps_negations():
    p = preprocess("I CAN'T sleep!!!! Visit https://x.com @bob  sooooo tired")
    assert "http" not in p.normalized and "@" not in p.normalized
    assert "not" in p.lemmas and "i" in p.lemmas
    assert "soo" in p.normalized


@pytest.mark.parametrize("bad", ["", "   ", "🙂🙂", "\x00\x01"])
def test_preprocess_handles_odd_input(bad):
    assert isinstance(preprocess(bad).clean, str)


def test_preprocess_rejects_non_string():
    with pytest.raises(TypeError):
        preprocess(None)  # type: ignore[arg-type]


def test_normalize_truncates_long_input():
    assert len(normalize("a " * 10000)) <= 5000


def test_emotion_mapping_is_distribution_over_srs_labels():
    p = map_emotions({"fear": 0.6, "sadness": 0.3, "neutral": 0.1}, "I'm so anxious and alone")
    assert set(p) == set(EMOTION_LABELS)
    assert abs(sum(p.values()) - 1) < 1e-6
    assert p["anxiety"] > 0.2 and p["loneliness"] > 0.05


def test_lexicon_analyzer_outputs():
    sent, emo = get_analyzer("lexicon").analyze("I am so happy and grateful today")
    assert max(sent, key=sent.get) == "positive"
    assert max(emo, key=emo.get) == "joy"


@pytest.mark.parametrize("text", [
    "I want to kill myself", "i'm going to end my life tonight", "I've been cutting myself again",
    "everyone would be better off without me", "I don't want to be alive anymore",
])
def test_explicit_crisis_detected(text):
    assert has_explicit_crisis_language(text)


@pytest.mark.parametrize("text", ["this exam is killing me", "I'm dying to see that movie", "had a rough day at work"])
def test_common_idioms_not_explicit(text):
    assert not has_explicit_crisis_language(text)


def test_indicators():
    assert "hopelessness" in detected_indicators("It all feels hopeless, nothing will change")


def test_dense_features_columns_complete():
    sent, emo = get_analyzer("lexicon").analyze("hello")
    feats = dense_features("hello", sent, emo, BehavioralContext([2, 2, 1], 3, 1, 0.5))
    for c in dense_columns(("sentiment", "emotion", "linguistic", "behavioral")):
        assert c in feats and np.isfinite(feats[c])


def test_threshold_rule_is_recall_oriented():
    p = np.array([[0.5, 0.2, 0.3], [0.55, 0.37, 0.08], [0.9, 0.05, 0.05]])
    assert apply_thresholds(p, {"high": 0.3, "moderate": 0.4}).tolist() == [2, 1, 0]


def _classify(text, behavior=None):
    clf = RiskClassifier(None, "lexicon")
    sent, emo = get_analyzer("lexicon").analyze(text)
    return clf.classify(text, preprocess(text).clean, sent, emo, behavior)


def test_explicit_language_overrides_to_high():
    r = _classify("I am going to kill myself")
    assert r.level == "high" and "override:explicit_crisis_language" in r.reasons


def test_behavioral_escalation():
    r = _classify("had an okay lunch", BehavioralContext([2, 2, 1, 1], 1, 1, 0.6))
    assert r.level in {"moderate", "high"}


def test_high_risk_reply_is_fixed_safety_template():
    r = generate_reply("anything", "sadness", "high")
    assert r.text == HIGH_RISK_MESSAGE and r.mode == "safety" and r.resources == CRISIS_RESOURCES


def test_low_and_moderate_replies_include_support():
    low = generate_reply("tired today", "sadness", "low")
    mod = generate_reply("tired today", "sadness", "moderate")
    assert low.recommendations and mod.recommendations
    assert "professional" in mod.text
