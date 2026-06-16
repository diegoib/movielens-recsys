"""Candidate generation for the recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MovieRecord:
    movie_id: int
    movie_idx: int
    title: str
    genres_vector: list[float]
    genome_top20: list[float] | None
    year: float | None
    popularity_last_30d: int
    avg_rating: float | None
    embedding: np.ndarray


def generate_candidates(
    user_data: dict,
    movie_cache: dict[int, MovieRecord],
    n: int = 200,
) -> list[int]:
    """Return up to n movie_ids sorted by popularity, genre-filtered when possible.

    If the user has favorite_genres in their Redis data, only movies sharing at
    least one of those genres are considered. Falls back to global popularity
    ranking if genre filtering would yield fewer than n/2 candidates.
    """
    favorite_genres: list[str] = user_data.get("favorite_genres", [])

    if favorite_genres:
        genre_set = set(favorite_genres)
        filtered = [
            m
            for m in movie_cache.values()
            if _shares_genre(m.genres_vector, genre_set, movie_cache)
        ]
        if len(filtered) >= n // 2:
            filtered.sort(key=lambda m: m.popularity_last_30d, reverse=True)
            return [m.movie_id for m in filtered[:n]]

    all_movies = sorted(movie_cache.values(), key=lambda m: m.popularity_last_30d, reverse=True)
    return [m.movie_id for m in all_movies[:n]]


# genre filtering helpers ─────────────────────────────────────────────────────

_GENRE_NAMES: list[str] | None = None
_GENRE_INDEX: dict[str, int] | None = None


def set_genre_index(genre_index: dict[str, int]) -> None:
    """Called once at startup with the genre_index from vocab.json."""
    global _GENRE_NAMES, _GENRE_INDEX
    _GENRE_INDEX = genre_index
    _GENRE_NAMES = [g for g, _ in sorted(genre_index.items(), key=lambda x: x[1])]


def _shares_genre(genres_vector: list[float], genre_set: set[str], _cache: dict) -> bool:
    if _GENRE_NAMES is None:
        return True
    for i, val in enumerate(genres_vector):
        if val > 0.5 and i < len(_GENRE_NAMES) and _GENRE_NAMES[i] in genre_set:
            return True
    return False


def build_movie_meta(movie: MovieRecord, norm_stats: dict[str, tuple[float, float]]) -> np.ndarray:
    """Build the movie_meta feature vector [movie_meta_dim] for ONNX inference."""
    genres = list(movie.genres_vector)
    genome = list(movie.genome_top20) if movie.genome_top20 else [0.0] * 20

    def _norm(val: float | None, key: str) -> float:
        mean, std = norm_stats[key]
        v = val if val is not None else mean
        return (v - mean) / std

    year_n = _norm(movie.year, "year")
    pop_n = _norm(float(movie.popularity_last_30d), "popularity_last_30d")
    rat_n = _norm(movie.avg_rating, "avg_rating")

    return np.array(genres + genome + [year_n, pop_n, rat_n], dtype=np.float32)
