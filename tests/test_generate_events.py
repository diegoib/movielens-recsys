"""Validation tests for the synthetic event generation pipeline."""

from __future__ import annotations

import random

import polars as pl
import pytest

from src.data.generate_events import (
    RANDOM_SEED,
    _build_genre_structures,
    _build_popular_by_month,
    _build_sessions,
    _build_user_history,
    _generate_user_events,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

GENRES = ["Action", "Drama", "Comedy", "Thriller", "Romance"]
N_MOVIES = 60
N_USERS = 8
RATINGS_PER_USER = 12


def _make_movies_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "movieId": list(range(1, N_MOVIES + 1)),
            "title": [f"Movie {i} (200{i % 10})" for i in range(1, N_MOVIES + 1)],
            "genres": [GENRES[i % len(GENRES)] for i in range(N_MOVIES)],
        }
    )


def _make_ratings_df(rng: random.Random) -> pl.DataFrame:
    rows = []
    base_ts = 1_000_000
    for uid in range(1, N_USERS + 1):
        movies = rng.sample(range(1, N_MOVIES + 1), RATINGS_PER_USER)
        ts = base_ts + uid * 100_000
        for mid in movies:
            rows.append(
                {
                    "userId": uid,
                    "movieId": mid,
                    "rating": rng.choice([3.0, 3.5, 4.0, 4.5, 5.0]),
                    "timestamp": ts,
                }
            )
            ts += rng.randint(600, 7_200)
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def events_df() -> pl.DataFrame:
    rng_data = random.Random(0)
    movies_df = _make_movies_df()
    ratings_df = _make_ratings_df(rng_data)

    genre_map, genre_to_movies = _build_genre_structures(movies_df)
    popular_by_month = _build_popular_by_month(ratings_df)
    user_rated, movie_to_users = _build_user_history(ratings_df)

    # Build sorted per-user rating lists
    user_ratings_sorted: dict[int, list] = {}
    for uid in range(1, N_USERS + 1):
        rows = (
            ratings_df.filter(pl.col("userId") == uid)
            .sort("timestamp")
            .select(["movieId", "timestamp", "rating"])
            .rows()
        )
        user_ratings_sorted[uid] = [(m, t, r) for m, t, r in rows]

    rng = random.Random(RANDOM_SEED)
    all_events = []
    for uid in range(1, N_USERS + 1):
        sessions = _build_sessions(user_ratings_sorted[uid])
        all_events.extend(
            _generate_user_events(
                uid,
                sessions,
                user_rated[uid],
                popular_by_month,
                genre_map,
                genre_to_movies,
                movie_to_users,
                user_rated,
                rng,
            )
        )

    return pl.DataFrame(all_events)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_valid_event_types(events_df: pl.DataFrame) -> None:
    """All event_type values must be one of the four valid types."""
    valid = {"impression", "view", "click", "rating"}
    actual = set(events_df["event_type"].unique().to_list())
    assert actual.issubset(valid), f"Unexpected event types: {actual - valid}"


def test_no_negative_is_rated_movie(events_df: pl.DataFrame) -> None:
    """Negative impressions/views must never be a movie the user has rated."""
    rng_data = random.Random(0)
    ratings_df = _make_ratings_df(rng_data)
    user_rated, _ = _build_user_history(ratings_df)

    # Negative events = label 0 impression or view
    negatives = events_df.filter(
        (pl.col("label") == 0) & (pl.col("event_type").is_in(["impression", "view"]))
    )

    for row in negatives.iter_rows(named=True):
        uid, mid = row["user_id"], row["movie_id"]
        assert mid not in user_rated.get(uid, set()), (
            f"user {uid} has movie {mid} as a rated movie but it appears as a negative"
        )


def test_click_has_prior_view_and_impression(events_df: pl.DataFrame) -> None:
    """Each click event must have a view and an impression on the same movie
    in the same session, occurring before the click timestamp."""
    clicks = events_df.filter(pl.col("event_type") == "click")

    for row in clicks.iter_rows(named=True):
        uid, mid, sid, ts_click = (
            row["user_id"],
            row["movie_id"],
            row["session_id"],
            row["timestamp"],
        )
        same_session = events_df.filter(
            (pl.col("user_id") == uid)
            & (pl.col("movie_id") == mid)
            & (pl.col("session_id") == sid)
            & (pl.col("timestamp") < ts_click)
        )
        types = set(same_session["event_type"].to_list())
        assert "view" in types, f"click for user={uid} movie={mid} has no prior view"
        assert "impression" in types, f"click for user={uid} movie={mid} has no prior impression"


def test_negative_to_positive_ratio(events_df: pl.DataFrame) -> None:
    """Negative events should outnumber positive click events by ≥ 3x.

    We allow 3x (not 4x) to account for a possibly empty collaborative pool
    before _find_top_k_neighbors is implemented.
    """
    n_clicks = events_df.filter(pl.col("event_type") == "click").height
    n_neg_impressions = events_df.filter(
        (pl.col("label") == 0) & (pl.col("event_type") == "impression")
    ).height

    assert n_clicks > 0, "no click events generated"
    ratio = n_neg_impressions / n_clicks
    assert ratio >= 3.0, f"expected ≥3 negative impressions per click, got {ratio:.2f}"


def test_build_sessions_gap_creates_new_session() -> None:
    """Ratings more than 60 min apart must land in separate sessions."""
    ratings = [
        (1, 1_000, 4.0),
        (2, 2_000, 3.5),
        (3, 10_000, 5.0),  # 8_000 s gap (> 60 min)
        (4, 11_000, 4.0),
    ]
    sessions = _build_sessions(ratings)
    assert len(sessions) == 2
    assert len(sessions[0][1]) == 2
    assert len(sessions[1][1]) == 2
