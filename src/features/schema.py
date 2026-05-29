"""Pydantic schemas for user and movie features.

These models define the feature contract shared across:
- Offline feature engineering (build_features.py)
- Online serving (src/serving/scorer.py reads UserFeatures from Redis)
- Stream processing (src/features/processor.py writes UserFeatures to Redis)
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

N_GENRES: int = 19  # unique genres in MovieLens (excluding "(no genres listed)")
N_GENOME: int = 20  # top-N genome tag scores stored per movie


class UserFeatures(BaseModel):
    """Point-in-time user features for one event row.

    genre_affinity_last_7d and n_clicks_last_7d are computed using only events
    with timestamp strictly before the current event — no future leakage.
    days_since_last_activity is null for a user's very first event.
    avg_session_length and favorite_genres are global per-user aggregates.
    """

    user_id: int
    event_id: str
    genre_affinity_last_7d: list[float]  # len=N_GENRES, normalized (sum ≤ 1.0)
    n_clicks_last_7d: int
    avg_session_length: float | None  # null for users with no clicks
    favorite_genres: list[str]  # len ≤ 3, top genres by click count
    days_since_last_activity: float | None  # null for user's first event

    @field_validator("genre_affinity_last_7d")
    @classmethod
    def check_affinity(cls, v: list[float]) -> list[float]:
        if len(v) != N_GENRES:
            raise ValueError(f"genre_affinity_last_7d must have {N_GENRES} elements, got {len(v)}")
        if sum(v) > 1.0 + 1e-5:
            raise ValueError(f"genre_affinity_last_7d must sum to ≤1.0, got {sum(v):.6f}")
        return v

    @field_validator("favorite_genres")
    @classmethod
    def check_fav_genres(cls, v: list[str]) -> list[str]:
        if len(v) > 3:
            raise ValueError(f"favorite_genres must have ≤3 elements, got {len(v)}")
        return v


class MovieFeatures(BaseModel):
    """Static movie features, computed once at the train/val/test split timestamp.

    genome_top20 is null for movies not present in genome-scores.csv (~46K of 61K
    event-movies). year is null for ~166 movies with unparseable titles.
    avg_rating is null for movies with no ratings in ratings.csv (~2K).
    The model graph handles all nulls — no imputation in the pipeline.
    """

    movie_id: int
    popularity_last_30d: int
    avg_rating: float | None  # null if movie has no ratings
    year: int | None  # null if title has no parseable year
    genres_vector: list[float]  # one-hot, len=N_GENRES, float32
    genome_top20: list[float] | None  # null if no genome data for this movie

    @field_validator("genres_vector")
    @classmethod
    def check_genres_vector(cls, v: list[float]) -> list[float]:
        if len(v) != N_GENRES:
            raise ValueError(f"genres_vector must have {N_GENRES} elements, got {len(v)}")
        return v

    @field_validator("genome_top20")
    @classmethod
    def check_genome(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != N_GENOME:
            raise ValueError(f"genome_top20 must have {N_GENOME} elements, got {len(v)}")
        return v
