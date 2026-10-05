"""Run the NLP analysers over the dataset once and cache the outputs (slow on CPU)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.nlp.analyzers import get_analyzer  # noqa: E402
from app.nlp.features import dense_features  # noqa: E402
from app.nlp.preprocess import preprocess  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="transformer", choices=["transformer", "lexicon"])
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(DATA_DIR / "risk_dataset.csv")
    analyzer = get_analyzer(args.backend)
    t0 = time.time()
    texts = df["text"].tolist()
    results = []
    for i in range(0, len(texts), 32):
        results += analyzer.analyze_batch(texts[i : i + 32])
        print(f"analysed {min(i + 32, len(texts))}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for text, (sent, emo) in zip(texts, results):
        f = dense_features(text, sent, emo)
        f["clean_text"] = preprocess(text).clean
        rows.append(f)
    feats = pd.DataFrame(rows)
    feats.to_parquet(DATA_DIR / f"features_{args.backend}.parquet")

    if not args.no_embeddings:
        from sentence_transformers import SentenceTransformer

        emb = SentenceTransformer(EMBEDDING_MODEL).encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        np.save(DATA_DIR / "embeddings.npy", emb)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
