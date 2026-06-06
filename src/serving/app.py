"""FastAPI serving app for the two-tower recommender."""

from __future__ import annotations

import json
import logging
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

log = logging.getLogger(__name__)

_MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/models")
_REDIS_HOST = os.environ.get("REDIS_HOST")
_REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
_REDPANDA_BROKERS = os.environ.get("REDPANDA_BROKERS")

_scorer: OnnxScorer | None = None
_redis: redis_lib.Redis | None = None  # type: ignore[type-arg]
_producer = None  # kafka.KafkaProducer | None, imported lazily to avoid hard dep


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _scorer, _redis, _producer
    try:
        _scorer = OnnxScorer(_MODEL_DIR)
    except Exception as exc:
        # Start without model — /recommendations returns 503 until artifacts are uploaded
        log.warning("Model loading failed (%s). Run make train-gcp to generate artifacts.", exc)
    if _REDIS_HOST:
        _redis = redis_lib.Redis(host=_REDIS_HOST, port=_REDIS_PORT, decode_responses=True)
    if _REDPANDA_BROKERS:
        try:
            from kafka import KafkaProducer

            _producer = KafkaProducer(
                bootstrap_servers=_REDPANDA_BROKERS,
                value_serializer=lambda v: json.dumps(v).encode(),
            )
            log.info("Kafka producer connected to %s", _REDPANDA_BROKERS)
        except Exception as exc:
            log.warning("Kafka producer init failed (%s). Events will not be streamed.", exc)
    yield
    if _redis:
        _redis.close()
    if _producer:
        _producer.close()


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
    if _producer is not None:
        payload = event.model_dump()
        payload["event_id"] = str(payload["event_id"])
        payload["session_id"] = str(payload["session_id"])
        # Enrich with genres_vector so the streaming processor can compute genre affinity
        # without loading its own copy of movie_features.parquet.
        if _scorer is not None and event.movie_id in _scorer.movie_cache:
            payload["genres_vector"] = _scorer.movie_cache[event.movie_id].genres_vector
        _producer.send("events", payload)
    return {"status": "accepted", "event_id": str(event.event_id)}
