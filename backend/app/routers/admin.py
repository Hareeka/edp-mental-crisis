from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import BACKEND_DIR
from app.db import get_db
from app.models import Feedback, Interaction, MoodRecord, User
from app.nlp.analyzers import get_analyzer
from app.nlp.risk import get_risk_classifier
from app.security import require_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
def stats(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> dict:
    """Aggregate, non-identifying statistics. No message content is exposed."""
    risk = dict(db.execute(select(Interaction.risk_level, func.count()).group_by(Interaction.risk_level)).all())
    fb = db.execute(select(func.avg(Feedback.usefulness), func.avg(Feedback.relevance),
                           func.avg(Feedback.appropriateness), func.avg(Feedback.experience))).one()
    return {
        "users": db.scalar(select(func.count()).select_from(User)),
        "interactions": db.scalar(select(func.count()).select_from(Interaction)),
        "mood_records": db.scalar(select(func.count()).select_from(MoodRecord)),
        "risk_distribution": risk,
        "avg_latency_ms": db.scalar(select(func.avg(Interaction.latency_ms))),
        "feedback_avg": dict(zip(["usefulness", "relevance", "appropriateness", "experience"], fb)),
    }


@router.get("/model")
def model_info(_: User = Depends(require_admin)) -> dict:
    clf = get_risk_classifier()
    metrics_path = BACKEND_DIR / "artifacts" / "metrics.json"
    summary = None
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        sel = m["selected_model"]
        t = m["models"][sel]["test_thresholded"]
        summary = {"selected_model": sel, "macro_f1": t["macro_f1"], "high_recall": t["per_class"]["high"]["recall"],
                   "roc_auc": t["roc_auc_ovr_macro"]}
    return {"nlp_backend": get_analyzer().backend_name, "risk_model_version": clf.version,
            "meta": clf.meta, "test_summary": summary}
