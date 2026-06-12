"""Pydantic models for MovieLens source data and the synthetic events table."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, field_validator

_VALID_EVENT_TYPES = {"impression", "view", "click", "rating"}


class Rating(BaseModel):
    userId: int
    movieId: int
    rating: float
    timestamp: int

    @field_validator("rating")
    @classmethod
    def rating_in_range(cls, v: float) -> float:
        if not (0.5 <= v <= 5.0):
            raise ValueError(f"rating must be 0.5–5.0, got {v}")
        return v


class Movie(BaseModel):
    movieId: int
    title: str
    genres: str  # pipe-separated, e.g. "Action|Drama"


class GenomeScore(BaseModel):
    movieId: int
    tagId: int
    relevance: float


class Event(BaseModel):
    event_id: uuid.UUID
    timestamp: int
    user_id: int
    movie_id: int
    event_type: str
    rating: float | None = None
    session_id: uuid.UUID
    label: int  # 0 or 1
    recommendation_id: str | None = (
        None  # correlation ID linking events to the recommendation that triggered them
    )

    @field_validator("event_type")
    @classmethod
    def valid_event_type(cls, v: str) -> str:
        if v not in _VALID_EVENT_TYPES:
            raise ValueError(f"event_type must be one of {_VALID_EVENT_TYPES}")
        return v

    @field_validator("label")
    @classmethod
    def valid_label(cls, v: int) -> int:
        if v not in {0, 1}:
            raise ValueError(f"label must be 0 or 1, got {v}")
        return v
