from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import init_db
from app.routers import admin, auth, chat, moods

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mindcompanion")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    from app.nlp.analyzers import get_analyzer
    from app.nlp.risk import get_risk_classifier

    analyzer = get_analyzer()
    clf = get_risk_classifier()
    log.info("NLP backend=%s risk model=%s", analyzer.backend_name, clf.version)
    yield


settings = get_settings()
app = FastAPI(title=f"{settings.app_name} API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


for r in (auth.router, chat.router, moods.router, admin.router):
    app.include_router(r)
