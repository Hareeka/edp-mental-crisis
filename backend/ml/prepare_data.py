"""Download and assemble the 3-class crisis-risk dataset.

Sources
* av9ash/CSSR-S_labelled_suicidewatch_posts_reddit (CC BY 4.0): r/SuicideWatch posts annotated with
  Columbia Suicide Severity Rating Scale levels 0-6. Mapping: 0 -> low, 1-2 -> moderate
  (passive / non-specific ideation), 3-6 -> high (method, intent, plan, behaviour).
* dair-ai/emotion: everyday emotional statements, sampled as additional *low* examples so the
  classifier sees ordinary difficulties typical of companion chat. Texts matching explicit crisis
  patterns are excluded.

Output: data/risk_dataset.csv with columns text, label, source, split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.nlp.crisis import has_explicit_crisis_language  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSSRS_URL = (
    "https://huggingface.co/datasets/av9ash/CSSR-S_labelled_suicidewatch_posts_reddit/"
    "resolve/main/labeled_rSuicidewatch_posts.csv"
)
EMOTION_URL = "https://huggingface.co/api/datasets/dair-ai/emotion/parquet/split/train/0.parquet"
EMOTION_NAMES = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
EMOTION_SAMPLE = {"sadness": 220, "fear": 160, "anger": 160, "joy": 120, "love": 40, "surprise": 40}
SEED = 42


def severity_to_risk(sev: int) -> str:
    if sev <= 0:
        return "low"
    if sev <= 2:
        return "moderate"
    return "high"


def load_cssrs() -> pd.DataFrame:
    df = pd.read_csv(CSSRS_URL)
    df = df.dropna(subset=["content", "severity"])
    df["text"] = df["content"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["label"] = df["severity"].astype(int).map(severity_to_risk)
    df["source"] = "cssrs"
    return df[["text", "label", "source"]]


def load_emotion() -> pd.DataFrame:
    df = pd.read_parquet(EMOTION_URL)
    df["emotion"] = df["label"].map(EMOTION_NAMES)
    df = df[~df["text"].map(has_explicit_crisis_language)]
    parts = [df[df.emotion == e].sample(n, random_state=SEED) for e, n in EMOTION_SAMPLE.items()]
    out = pd.concat(parts)
    return pd.DataFrame({"text": out["text"].str.strip(), "label": "low", "source": "emotion"})


def build(test_size: float = 0.15, val_size: float = 0.15) -> pd.DataFrame:
    df = pd.concat([load_cssrs(), load_emotion()], ignore_index=True)
    df = df[df["text"].str.len() >= 3].drop_duplicates(subset="text").reset_index(drop=True)
    strat = df["label"] + "_" + df["source"]
    train_idx, test_idx = train_test_split(df.index, test_size=test_size, stratify=strat, random_state=SEED)
    rel_val = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(train_idx, test_size=rel_val, stratify=strat[train_idx], random_state=SEED)
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_DIR / "risk_dataset.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(args.out, index=False)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
