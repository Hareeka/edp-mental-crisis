# MindCompanion — AI-Powered Mental Health Companion and Crisis Detection System

A working prototype of the SRS: a supportive chat companion with sentiment and emotion analysis, three-level crisis-risk classification (Low / Moderate / High), risk-appropriate responses, mood tracking and trends, feedback collection, and an experimental evaluation (model comparison + feature ablation).

> **Not a medical device.** Risk levels are automated indications, not diagnoses. The system does not replace professional care and cannot respond to emergencies.

## Architecture

```
React (Vite) UI ──/api──▶ FastAPI
                           ├─ auth (JWT, bcrypt) · moods · trends · feedback · admin stats
                           └─ NLP pipeline (backend/app/nlp)
                                preprocess.py   normalise, tokenise, stop-words, lemmatise
                                analyzers.py    sentiment (RoBERTa) + emotion (DistilRoBERTa → 7 SRS emotions)
                                features.py     sentiment · emotion · linguistic (TF-IDF, style, crisis lexicon) · behavioural
                                risk.py         trained classifier + recall-oriented thresholds + safety overrides
                                responder.py    templates / optional LLM for Low–Moderate; fixed safety message for High
SQLite (SQLAlchemy): users, mood_records, interactions, feedback — message text Fernet-encrypted at rest
```

| SRS module | Implementation |
|---|---|
| 3.2 Preprocessing | `app/nlp/preprocess.py` |
| 3.3 Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` (VADER in `lexicon` mode) |
| 3.4 Emotion | `j-hartmann/emotion-english-distilroberta-base`, mapped to sadness/fear/anger/anxiety/loneliness/joy/neutral |
| 3.5 Features | `app/nlp/features.py`, `app/nlp/crisis.py` |
| 3.6 Risk classification | `app/nlp/risk.py`, trained by `ml/train.py` (LR, RF, XGBoost, MiniLM-embedding classifier) |
| 3.7 / 3.8 Companion & crisis support | `app/nlp/responder.py` |
| 3.9 Mood tracking, 10.1 Dashboard | `app/routers/moods.py`, `frontend/src/pages` |
| 3.10 Feedback | `POST /api/feedback`, per-message thumbs and Settings page |
| 13 Evaluation | [`backend/artifacts/EVALUATION.md`](backend/artifacts/EVALUATION.md) |

### Safety design
- **Explicit crisis language always forces High**, regardless of model output.
- Thresholds are tuned for **High recall ≥ 0.90**; model selection never uses accuracy alone (SRS 13.3).
- High-risk replies are a **fixed, reviewed safety message** plus crisis resources — never free-form LLM text.
- Low predictions are raised to Moderate when recent history shows a High-risk message or a sustained declining mood.
- A disclaimer is returned with every chat response and shown in the UI.

### Privacy
Messages, mood notes and feedback comments are encrypted (Fernet) at rest; logs record only metadata (risk level, emotion, latency). Users can delete their history or account; research consent is opt-in and off by default. Admin endpoints return aggregates only.

## Running locally

Requires Python 3.10+ and Node 20+.

```bash
# Backend
cd backend
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip if you have a GPU build
pip install -r requirements-ml.txt -r requirements-dev.txt
cp .env.example .env    # set MC_JWT_SECRET and MC_ENCRYPTION_KEY
uvicorn app.main:app --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev             # http://localhost:5173, proxies /api to :8000
```

The first account you register becomes the admin (`/api/admin/stats`, `/api/admin/model`). API docs are at http://localhost:8000/docs.

**Lightweight mode:** `MC_NLP_BACKEND=lexicon` uses VADER and keyword emotions, so no model download or torch is needed (`pip install -r requirements.txt`). It loads `artifacts/risk_model_lexicon.joblib` if you point `MC_RISK_MODEL_PATH` at it, otherwise a rule-based scorer. Tests and CI run in this mode.

**Optional LLM replies:** set `MC_LLM_API_BASE`, `MC_LLM_API_KEY` and `MC_LLM_MODEL` (any OpenAI-compatible endpoint) to have Low/Moderate replies generated. Without these, curated templates are used.

## Reproducing the experiments

```bash
cd backend
python ml/prepare_data.py                 # downloads datasets → data/risk_dataset.csv
python ml/featurize.py                    # sentiment/emotion/embeddings cache (~10 min on 2 CPUs)
python ml/train.py                        # comparison, ablation, artifacts/{risk_model.joblib,metrics.json,EVALUATION.md}
python ml/featurize.py --backend lexicon --no-embeddings && python ml/train.py --backend lexicon --skip-cv
```

**Data:** [`av9ash/CSSR-S_labelled_suicidewatch_posts_reddit`](https://huggingface.co/datasets/av9ash/CSSR-S_labelled_suicidewatch_posts_reddit) (CC BY 4.0; Columbia severity 0 → Low, 1–2 → Moderate, 3–6 → High) plus everyday emotional statements from [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) as extra Low examples. Stratified 70/15/15 split.

## Tests

```bash
cd backend && pytest -q && ruff check .
cd frontend && npx oxlint src && npm run build
```

## Limitations
See SRS §16 and the limitations section of `EVALUATION.md`. In particular: small, single-source crisis dataset; English only; behavioural features in ablation Experiment 4 are simulated because no public dataset pairs crisis text with mood logs.
