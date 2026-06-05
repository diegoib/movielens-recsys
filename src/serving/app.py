"""FastAPI serving app for the two-tower recommender."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis as redis_lib
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.data.schemas import Event
from src.serving.candidates import generate_candidates
from src.serving.scorer import OnnxScorer

_MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/models")
_REDIS_HOST = os.environ.get("REDIS_HOST")
_REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

_scorer: OnnxScorer | None = None
_redis: redis_lib.Redis | None = None  # type: ignore[type-arg]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _scorer, _redis
    try:
        _scorer = OnnxScorer(_MODEL_DIR)
    except Exception as exc:
        # Start without model — /recommendations returns 503 until artifacts are uploaded
        print(f"WARNING: model loading failed ({exc}). Run make train-gcp to generate artifacts.")
    if _REDIS_HOST:
        _redis = redis_lib.Redis(host=_REDIS_HOST, port=_REDIS_PORT, decode_responses=True)
    yield
    if _redis:
        _redis.close()


app = FastAPI(title="recsys-serving", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


# ── response schemas ──────────────────────────────────────────────────────────


class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendedMovie]


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_dir": _MODEL_DIR}


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(user_id: int, n: int = 5) -> RecommendationResponse:
    scorer = _scorer
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Fetch user features from Redis (empty dict = cold start → zero vector)
    user_data: dict = {}
    if _redis is not None:
        try:
            raw = _redis.hgetall(f"user:{user_id}")
            if raw:
                user_data = raw
        except redis_lib.RedisError:
            pass  # serve with cold-start features rather than failing

    user_behavior = scorer.build_user_behavior(user_data)
    uid_idx = scorer.user_idx(user_id)

    candidate_ids = generate_candidates(user_data, scorer.movie_cache)
    # Filter to movies that are in the cache (safety check)
    candidate_ids = [mid for mid in candidate_ids if mid in scorer.movie_cache]
    if not candidate_ids:
        return RecommendationResponse(user_id=user_id, recommendations=[])

    scores = scorer.score(uid_idx, user_behavior, candidate_ids)

    top_indices = scores.argsort()[::-1][:n]
    recs = [
        RecommendedMovie(
            movie_id=candidate_ids[i],
            title=scorer.movie_cache[candidate_ids[i]].title,
            score=float(scores[i]),
        )
        for i in top_indices
    ]
    return RecommendationResponse(user_id=user_id, recommendations=recs)


@app.post("/events", status_code=200)
def ingest_event(event: Event) -> dict:
    # Phase 7 will wire this to RedPanda; for now just validate and acknowledge
    return {"status": "accepted", "event_id": str(event.event_id)}
