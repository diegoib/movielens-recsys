"""FastAPI serving app for the two-tower recommender."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis as redis_lib
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram
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

# Sliding window of (timestamp, movie_id) for the unique-movies-24h gauge.
# asyncio is single-threaded for the background task; sync endpoints append from
# a thread-pool thread — deque.append is atomic under the GIL, so no lock needed.
_recent_movies: collections.deque = collections.deque()

# ── business metrics ──────────────────────────────────────────────────────────

IMPRESSIONS = Counter("recsys_impressions_total", "Impression events received")
CLICKS = Counter("recsys_clicks_total", "Click events received")
RECOMMENDATIONS_SERVED = Counter(
    "recsys_recommendations_served_total",
    "Recommendation requests served",
    ["model_version"],
)
RECOMMENDATION_SCORE = Histogram(
    "recsys_recommendation_score",
    "Distribution of recommendation scores served",
    ["model_version"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
UNIQUE_MOVIES_24H = Gauge(
    "recsys_unique_movies_recommended_24h",
    "Unique movies recommended in the last 24 hours",
)


async def _update_unique_movies() -> None:
    """Background task: maintain a 24h sliding window of unique recommended movies."""
    while True:
        await asyncio.sleep(60)
        cutoff = time.time() - 86400
        while _recent_movies and _recent_movies[0][0] < cutoff:
            _recent_movies.popleft()
        UNIQUE_MOVIES_24H.set(len({m for _, m in _recent_movies}))


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
    task = asyncio.create_task(_update_unique_movies())
    yield
    task.cancel()
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
    recommendation_id: str
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

    recommendation_id = str(uuid.uuid4())

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
    candidate_ids = [mid for mid in candidate_ids if mid in scorer.movie_cache]
    if not candidate_ids:
        return RecommendationResponse(
            user_id=user_id, recommendation_id=recommendation_id, recommendations=[]
        )

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

    # Business metrics
    mv = scorer.model_version
    RECOMMENDATIONS_SERVED.labels(model_version=mv).inc()
    ts_now = time.time()
    for rec in recs:
        RECOMMENDATION_SCORE.labels(model_version=mv).observe(rec.score)
        _recent_movies.append((ts_now, rec.movie_id))

    # Log inference to model-predictions topic (includes user features for retraining)
    if _producer is not None:
        ts = int(ts_now)
        for pos, rec in enumerate(recs):
            try:
                _producer.send(
                    "model-predictions",
                    {
                        "recommendation_id": recommendation_id,
                        "user_id": user_id,
                        "movie_id": rec.movie_id,
                        "score": rec.score,
                        "position": pos,
                        "timestamp": ts,
                        "model_version": mv,
                        "user_features": json.dumps(user_data),
                    },
                )
            except Exception as exc:
                log.debug("model-predictions publish failed: %s", exc)

    return RecommendationResponse(
        user_id=user_id, recommendation_id=recommendation_id, recommendations=recs
    )


@app.post("/events", status_code=200)
def ingest_event(event: Event) -> dict:
    if event.event_type == "click":
        CLICKS.inc()
    elif event.event_type == "impression":
        IMPRESSIONS.inc()

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
