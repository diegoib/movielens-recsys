"""Tests for the FastAPI serving layer.

OnnxScorer and Redis are replaced with lightweight fakes so the tests run
without GCS access, a trained model, or a running Redis instance.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

# ── fake artifacts ────────────────────────────────────────────────────────────

_N_GENRES = 2
_USER_DIM = _N_GENRES * 2 + 3  # 7
_MOVIE_DIM = _N_GENRES + 20 + 3  # 25

_GENRE_INDEX = {"Action": 0, "Comedy": 1}
_NORM_STATS = {
    "n_clicks_last_7d": [0.0, 1.0],
    "avg_session_length": [2.0, 1.0],
    "days_since_last_activity": [1.0, 1.0],
    "year": [2010.0, 5.0],
    "popularity_last_30d": [5.0, 3.0],
    "avg_rating": [3.5, 0.5],
}
_VOCAB = {
    "user_vocab": {"1": 1, "2": 2, "3": 3},
    "movie_vocab": {"1": 1, "2": 2, "3": 3},
    "genre_index": _GENRE_INDEX,
    "norm_stats": _NORM_STATS,
    "user_behavior_dim": _USER_DIM,
    "movie_meta_dim": _MOVIE_DIM,
}

_MOVIES = [
    {
        "movie_id": 1,
        "movie_idx": 1,
        "title": "Movie A",
        "genres_vector": [1.0, 0.0],
        "genome_top20": [0.1] * 20,
        "year": 2010,
        "popularity_last_30d": 9,
        "avg_rating": 4.0,
    },
    {
        "movie_id": 2,
        "movie_idx": 2,
        "title": "Movie B",
        "genres_vector": [0.0, 1.0],
        "genome_top20": [0.2] * 20,
        "year": 2015,
        "popularity_last_30d": 5,
        "avg_rating": 3.5,
    },
    {
        "movie_id": 3,
        "movie_idx": 3,
        "title": "Movie C",
        "genres_vector": [1.0, 1.0],
        "genome_top20": None,
        "year": None,
        "popularity_last_30d": 2,
        "avg_rating": None,
    },
]


def _make_fake_scorer(tmp_path: Path) -> Any:
    """Write fake artifacts and return an OnnxScorer pointed at tmp_path."""
    from src.serving.scorer import OnnxScorer

    model_dir = str(tmp_path)

    # vocab.json
    (tmp_path / "vocab.json").write_bytes(json.dumps(_VOCAB).encode())

    # movie_features.parquet
    buf = io.BytesIO()
    pl.DataFrame(_MOVIES).write_parquet(buf)
    (tmp_path / "movie_features.parquet").write_bytes(buf.getvalue())

    # model.onnx — replaced with a mock session, so content doesn't matter
    (tmp_path / "model.onnx").write_bytes(b"placeholder")

    with patch("src.serving.scorer.ort.InferenceSession") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess.run.return_value = [np.array([0.9, 0.7, 0.3], dtype=np.float32)]
        mock_sess_cls.return_value = mock_sess
        scorer = OnnxScorer(model_dir)
        scorer._session = mock_sess  # keep the mock after __init__

    return scorer


@pytest.fixture()
def fake_scorer(tmp_path: Path) -> Any:
    return _make_fake_scorer(tmp_path)


@pytest.fixture()
def client(fake_scorer: Any) -> TestClient:
    import src.serving.app as app_module

    app_module._scorer = fake_scorer
    app_module._redis = None  # no Redis → cold start
    return TestClient(app_module.app)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_recommendations_cold_start(client: TestClient) -> None:
    resp = client.get("/recommendations/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == 1
    assert len(body["recommendations"]) == 3  # only 3 movies in fake cache
    assert all("movie_id" in r and "title" in r and "score" in r for r in body["recommendations"])


def test_recommendations_ordered_by_score(client: TestClient) -> None:
    resp = client.get("/recommendations/1")
    scores = [r["score"] for r in resp.json()["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_with_redis_features(client: TestClient) -> None:
    import src.serving.app as app_module

    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = {
        "genre_affinity_last_7d": "[0.8, 0.2]",
        "favorite_genres": "['Action']",
        "n_clicks_last_7d": "5",
        "avg_session_length": "3.0",
        "days_since_last_activity": "1.0",
    }
    original = app_module._redis
    app_module._redis = mock_redis
    try:
        resp = client.get("/recommendations/1")
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) > 0
    finally:
        app_module._redis = original


def test_events_valid(client: TestClient) -> None:
    payload = {
        "event_id": str(uuid.uuid4()),
        "timestamp": 1_700_000_000,
        "user_id": 1,
        "movie_id": 42,
        "event_type": "click",
        "rating": None,
        "session_id": str(uuid.uuid4()),
        "label": 1,
    }
    resp = client.post("/events", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


def test_events_invalid_event_type(client: TestClient) -> None:
    payload = {
        "event_id": str(uuid.uuid4()),
        "timestamp": 1_700_000_000,
        "user_id": 1,
        "movie_id": 42,
        "event_type": "purchase",  # invalid
        "rating": None,
        "session_id": str(uuid.uuid4()),
        "label": 1,
    }
    resp = client.post("/events", json=payload)
    assert resp.status_code == 422
