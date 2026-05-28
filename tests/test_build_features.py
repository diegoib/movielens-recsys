"""Validation tests for Phase 3 feature engineering."""

from __future__ import annotations

import polars as pl
import pytest

from src.features.build_features import (
    _build_clicks_genre_table,
    _build_clicks_simple_table,
    _rolling_user_features,
    compute_genre_list,
    compute_movie_features,
    compute_user_features,
)
from src.features.schema import N_GENOME, N_GENRES, MovieFeatures, UserFeatures

# ── Synthetic dataset constants ───────────────────────────────────────────────

BASE_TS = 1_577_836_800  # 2020-01-01 00:00:00 UTC
WEEK_S = 604_800
DAY_S = 86_400

U1, U2, U3 = 1, 2, 3
M1, M2, M3, M4, M5 = 101, 102, 103, 104, 105

# Genres for the 5 test movies:
#   M1 → Action|Drama     M2 → Comedy    M3 → Action
#   M4 → Drama|Romance    M5 → (no genres listed)
GENRES_M1 = "Action|Drama"
GENRES_M2 = "Comedy"
GENRES_M3 = "Action"
GENRES_M4 = "Drama|Romance"
GENRES_M5 = "(no genres listed)"
ALL_GENRES_SORTED = sorted({"Action", "Comedy", "Drama", "Romance"})  # 4 genres


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def movies_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "movieId": [M1, M2, M3, M4, M5],
            "title": [
                "Film One (2010)",
                "Film Two (2015)",
                "Film Three (2000)",
                "Film Four (2020)",
                "Film Five",
            ],
            "genres": [GENRES_M1, GENRES_M2, GENRES_M3, GENRES_M4, GENRES_M5],
        }
    )


@pytest.fixture(scope="module")
def ratings_df() -> pl.DataFrame:
    """Small ratings covering M1–M4; M5 has no ratings (tests avg_rating null)."""
    return pl.DataFrame(
        {
            "userId": [U1, U1, U2, U2, U3],
            "movieId": [M1, M2, M3, M4, M1],
            "rating": [4.0, 3.5, 5.0, 2.0, 4.5],
            "timestamp": [BASE_TS, BASE_TS + 100, BASE_TS + 200, BASE_TS + 300, BASE_TS + 400],
        }
    )


@pytest.fixture(scope="module")
def events_df() -> pl.DataFrame:
    """Controlled event stream for 3 users.

    U1: two click events separated by > 7 days — tests the rolling window.
        - click_1 at BASE_TS (week 0)
        - click_2 at BASE_TS + WEEK_S + DAY_S (week 1 + 1d)
        - impression at BASE_TS + WEEK_S + 2*DAY_S → should see click_2 but NOT click_1

    U2: single-session with one click.

    U3: only impressions — no clicks at all (tests n_clicks=0, affinity=zeros).
    """
    rows = [
        # U1 — week-0 click on M1
        ("e01", BASE_TS, U1, M1, "impression", None, "s1", 1),
        ("e02", BASE_TS + 60, U1, M1, "view", None, "s1", 1),
        ("e03", BASE_TS + 120, U1, M1, "click", None, "s1", 1),
        # U1 — week-1+1d click on M2
        ("e04", BASE_TS + WEEK_S + DAY_S, U1, M2, "click", None, "s2", 1),
        # U1 — impression after both clicks (week-1 + 2d); 7d window includes e04 only
        ("e05", BASE_TS + WEEK_S + 2 * DAY_S, U1, M3, "impression", None, "s2", 0),
        # U2 — single session
        ("e06", BASE_TS + 2 * WEEK_S, U2, M3, "click", None, "s3", 1),
        ("e07", BASE_TS + 2 * WEEK_S + 300, U2, M4, "impression", None, "s3", 0),
        # U3 — impressions only (M5 included so avg_rating null is testable)
        ("e08", BASE_TS + 3 * WEEK_S, U3, M1, "impression", None, "s4", 0),
        ("e09", BASE_TS + 3 * WEEK_S + 600, U3, M2, "impression", None, "s4", 0),
        ("e10", BASE_TS + 4 * WEEK_S, U3, M5, "impression", None, "s5", 0),
    ]
    return pl.DataFrame(
        rows,
        orient="row",
        schema={
            "event_id": pl.String,
            "timestamp": pl.Int64,
            "user_id": pl.Int64,
            "movie_id": pl.Int64,
            "event_type": pl.String,
            "rating": pl.Float64,
            "session_id": pl.String,
            "label": pl.Int64,
        },
    )


@pytest.fixture(scope="module")
def all_genres(movies_df: pl.DataFrame) -> list[str]:
    return compute_genre_list(movies_df)


@pytest.fixture(scope="module")
def movie_features(
    movies_df: pl.DataFrame,
    ratings_df: pl.DataFrame,
    events_df: pl.DataFrame,
    all_genres: list[str],
) -> pl.DataFrame:
    empty_genome = pl.DataFrame(
        schema={"movieId": pl.Int64, "tagId": pl.Int64, "relevance": pl.Float64}
    )
    q80 = events_df["timestamp"].quantile(0.80)
    assert q80 is not None
    split_ts = int(q80)
    return compute_movie_features(
        movies_df, ratings_df, empty_genome, events_df, split_ts, all_genres
    )


@pytest.fixture(scope="module")
def user_features(
    events_df: pl.DataFrame, all_genres: list[str], movies_df: pl.DataFrame
) -> pl.DataFrame:
    return compute_user_features(events_df, all_genres, movies_df)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_genre_list_sorted_and_excludes_no_genres(movies_df: pl.DataFrame) -> None:
    genres = compute_genre_list(movies_df)
    assert genres == sorted(genres), "genre list must be sorted"
    assert "(no genres listed)" not in genres
    assert set(genres) == {"Action", "Comedy", "Drama", "Romance"}


def test_genres_vector_length(movie_features: pl.DataFrame, all_genres: list[str]) -> None:
    n = len(all_genres)
    lengths = movie_features["genres_vector"].list.len()
    assert (lengths == n).all(), f"Expected all genres_vector length={n}"


def test_genome_top20_null_for_missing(movie_features: pl.DataFrame) -> None:
    """All movies in the test fixture have no genome data → genome_top20 should be null."""
    assert movie_features["genome_top20"].null_count() == movie_features.height


def test_avg_rating_null_for_unrated_movie(
    movie_features: pl.DataFrame, ratings_df: pl.DataFrame
) -> None:
    """M5 has no entry in ratings_df → avg_rating should be null."""
    m5_row = movie_features.filter(pl.col("movie_id") == M5)
    assert m5_row.height == 1
    assert m5_row["avg_rating"][0] is None


def test_no_future_leakage_genre_affinity(
    events_df: pl.DataFrame, all_genres: list[str], movies_df: pl.DataFrame
) -> None:
    """U1's first event (e01, before any click) must have all-zero genre_affinity."""
    movies_exp = (
        movies_df.select(["movieId", "genres"])
        .with_columns(
            pl.col("movieId").cast(pl.Int64).alias("movie_id"),
            pl.col("genres").str.split("|").alias("genre_list"),
        )
        .select(["movie_id", "genre_list"])
    )

    u1_events = events_df.filter(pl.col("user_id") == U1).select(
        ["event_id", "user_id", "timestamp"]
    )
    u1_cg = _build_clicks_genre_table(events_df, movies_exp).filter(pl.col("click_user_id") == U1)
    u1_cs = _build_clicks_simple_table(events_df).filter(pl.col("click_user_id") == U1)

    result = _rolling_user_features(u1_events, u1_cg, u1_cs, all_genres)

    # e01 is the first event; no clicks precede it → affinity must be all zeros
    e01_row = result.filter(pl.col("event_id") == "e01")
    affinity = e01_row["genre_affinity_last_7d"][0]
    assert sum(affinity) == pytest.approx(0.0), (
        f"First event must have zero affinity, got {affinity}"
    )


def test_genre_affinity_sums_le_one(user_features: pl.DataFrame) -> None:
    row_sums = user_features["genre_affinity_last_7d"].list.sum()
    assert (row_sums <= 1.0 + 1e-5).all(), "genre_affinity_last_7d must sum to ≤1.0"


def test_temporal_split_three_partitions(
    events_df: pl.DataFrame,
    all_genres: list[str],
    movies_df: pl.DataFrame,
    ratings_df: pl.DataFrame,
) -> None:
    """max(train.ts) < min(val.ts) < min(test.ts)."""
    empty_genome = pl.DataFrame(
        schema={"movieId": pl.Int64, "tagId": pl.Int64, "relevance": pl.Float64}
    )
    q80 = events_df["timestamp"].quantile(0.80)
    q90 = events_df["timestamp"].quantile(0.90)
    assert q80 is not None and q90 is not None
    split_train_ts = int(q80)
    split_val_ts = int(q90)

    mf = compute_movie_features(
        movies_df, ratings_df, empty_genome, events_df, split_train_ts, all_genres
    )
    uf = compute_user_features(events_df, all_genres, movies_df)

    dataset = (
        events_df.join(uf.drop("user_id"), on="event_id")
        .join(mf, on="movie_id")
        .with_columns(
            pl.when(pl.col("timestamp") < split_train_ts)
            .then(pl.lit("train"))
            .when(pl.col("timestamp") < split_val_ts)
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
    )

    train_ts = dataset.filter(pl.col("split") == "train")["timestamp"]
    val_ts = dataset.filter(pl.col("split") == "val")["timestamp"]
    test_ts = dataset.filter(pl.col("split") == "test")["timestamp"]

    if train_ts.len() > 0 and val_ts.len() > 0:
        assert train_ts.max() < val_ts.min(), "train/val boundary violated"
    if val_ts.len() > 0 and test_ts.len() > 0:
        assert val_ts.max() < test_ts.min(), "val/test boundary violated"


def test_no_nulls_in_structural_columns(user_features: pl.DataFrame) -> None:
    """genre_affinity_last_7d and n_clicks_last_7d are always computable — no nulls."""
    assert user_features["genre_affinity_last_7d"].null_count() == 0
    assert user_features["n_clicks_last_7d"].null_count() == 0
    # U3 has no clicks → n_clicks_last_7d must be 0 for all U3 events
    u3_clicks = user_features.filter(pl.col("user_id") == U3)["n_clicks_last_7d"]
    assert (u3_clicks == 0).all(), "U3 has no clicks; n_clicks_last_7d must be 0"


def test_pydantic_schema_validates() -> None:
    """Pydantic validators enforce vector lengths and value constraints."""
    # UserFeatures: requires exactly N_GENRES elements summing to ≤1.0
    uf = UserFeatures(
        user_id=1,
        event_id="abc",
        genre_affinity_last_7d=[0.0] * N_GENRES,
        n_clicks_last_7d=3,
        avg_session_length=2.5,
        favorite_genres=["Action", "Drama"],
        days_since_last_activity=1.5,
    )
    assert len(uf.genre_affinity_last_7d) == N_GENRES

    # MovieFeatures: genres_vector length and genome length enforced
    mf = MovieFeatures(
        movie_id=101,
        popularity_last_30d=10,
        avg_rating=3.5,
        year=2010,
        genres_vector=[0.0] * N_GENRES,
        genome_top20=[0.1] * N_GENOME,
    )
    assert len(mf.genres_vector) == N_GENRES
    assert len(mf.genome_top20) == N_GENOME  # type: ignore[arg-type]

    # Validator must reject wrong-length affinity
    with pytest.raises(Exception):
        UserFeatures(
            user_id=1,
            event_id="bad",
            genre_affinity_last_7d=[0.5] * (N_GENRES - 1),  # wrong length
            n_clicks_last_7d=0,
            avg_session_length=None,
            favorite_genres=[],
            days_since_last_activity=None,
        )
