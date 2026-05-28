"""Offline feature engineering for the MovieLens recommender system.

Produces data/processed/train_dataset.parquet with one row per event.
Run via: uv run python src/features/build_features.py

Design notes:
- genre_affinity_last_7d and n_clicks_last_7d are point-in-time: only events
  with timestamp strictly before the current event are used (no future leakage).
- avg_session_length and favorite_genres are global per-user aggregates.
- Movie features are static, computed once anchored at the 80th-percentile timestamp.
- Temporal split: 80/10/10 (train/val/test) by timestamp quantiles.
- Nulls are NOT filled — the model graph handles them directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW_7D_S: int = 7 * 24 * 3600  # 604800 seconds
WINDOW_30D_S: int = 30 * 24 * 3600  # 2592000 seconds
N_GENOME_TOP: int = 20
N_FAV_GENRES: int = 3

_YEAR_RE = re.compile(r"\((\d{4})\)")


# ── Genre list ────────────────────────────────────────────────────────────────


def compute_genre_list(movies_df: pl.DataFrame) -> list[str]:
    """Return sorted, deterministic list of all genre strings.

    Excludes '(no genres listed)'. This ordering is the canonical index shared
    by genres_vector (one-hot) and genre_affinity_last_7d.
    """
    genres: set[str] = set()
    for genres_str in movies_df["genres"].to_list():
        for g in genres_str.split("|"):
            if g != "(no genres listed)":
                genres.add(g)
    return sorted(genres)


# ── Movie features ────────────────────────────────────────────────────────────


def compute_movie_features(
    movies_df: pl.DataFrame,
    ratings_df: pl.DataFrame,
    genome_df: pl.DataFrame,
    events_df: pl.DataFrame,
    split_ts: int,
    all_genres: list[str],
) -> pl.DataFrame:
    """Compute static movie features anchored at split_ts.

    Only movies that appear in events_df are returned (left join from events).

    Returns DataFrame with columns:
        movie_id (i64), popularity_last_30d (i32), avg_rating (f64, nullable),
        year (i32, nullable), genres_vector (list[f32]),
        genome_top20 (list[f32], nullable)
    """
    event_movies = events_df.select(pl.col("movie_id").cast(pl.Int64)).unique()

    # popularity_last_30d: events in a 30-day window centred on split_ts
    pop = (
        events_df.filter(
            pl.col("event_type").is_in(["rating", "view"])
            & (pl.col("timestamp") >= split_ts - WINDOW_30D_S)
            & (pl.col("timestamp") <= split_ts + WINDOW_30D_S)
        )
        .group_by("movie_id")
        .agg(pl.len().cast(pl.Int32).alias("popularity_last_30d"))
        .with_columns(pl.col("movie_id").cast(pl.Int64))
    )

    # avg_rating from raw ratings.csv
    avg_rat = (
        ratings_df.group_by("movieId")
        .agg(pl.col("rating").mean().alias("avg_rating"))
        .rename({"movieId": "movie_id"})
        .with_columns(pl.col("movie_id").cast(pl.Int64))
    )

    # year + genres_vector from movies.csv
    genre_index = {g: i for i, g in enumerate(all_genres)}
    n_genres = len(all_genres)

    def _genres_vec(genres_str: str) -> list[float]:
        vec = [0.0] * n_genres
        for g in genres_str.split("|"):
            idx = genre_index.get(g)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def _year(title: str) -> int | None:
        m = _YEAR_RE.search(title)
        return int(m.group(1)) if m else None

    movies_feat = (
        movies_df.select(["movieId", "title", "genres"])
        .with_columns(
            pl.col("movieId").cast(pl.Int64).alias("movie_id"),
            pl.col("genres")
            .map_elements(_genres_vec, return_dtype=pl.List(pl.Float32))
            .alias("genres_vector"),
            pl.col("title").map_elements(_year, return_dtype=pl.Int32).alias("year"),
        )
        .select(["movie_id", "genres_vector", "year"])
    )

    # genome_top20: top-N relevance scores per movie, sorted desc
    if genome_df.height > 0:
        genome_feat = (
            genome_df.sort(["movieId", "relevance"], descending=[False, True])
            .group_by("movieId")
            .agg(pl.col("relevance").head(N_GENOME_TOP).alias("genome_top20"))
            .rename({"movieId": "movie_id"})
            .with_columns(
                pl.col("movie_id").cast(pl.Int64),
                pl.col("genome_top20").cast(pl.List(pl.Float32)),
            )
        )
    else:
        genome_feat = pl.DataFrame(
            schema={"movie_id": pl.Int64, "genome_top20": pl.List(pl.Float32)}
        )

    return (
        event_movies.join(pop, on="movie_id", how="left")
        .join(avg_rat, on="movie_id", how="left")
        .join(movies_feat, on="movie_id", how="left")
        .join(genome_feat, on="movie_id", how="left")
        .with_columns(pl.col("popularity_last_30d").fill_null(0).cast(pl.Int32))
    )


# ── User features — pre-computation helpers ───────────────────────────────────


def _build_clicks_genre_table(events_df: pl.DataFrame, movies_exp: pl.DataFrame) -> pl.DataFrame:
    """One row per (click event, genre). Pre-computed once outside the user loop.

    movies_exp must have columns [movie_id, genre_list] where genre_list is list[str].
    """
    return (
        events_df.filter(pl.col("event_type") == "click")
        .select(["user_id", "timestamp", "movie_id"])
        .join(movies_exp, on="movie_id", how="left")
        .explode("genre_list")
        .filter(pl.col("genre_list").is_not_null() & (pl.col("genre_list") != "(no genres listed)"))
        .rename({"timestamp": "click_ts", "user_id": "click_user_id", "genre_list": "genre"})
        .select(["click_user_id", "click_ts", "genre"])
    )


def _build_clicks_simple_table(events_df: pl.DataFrame) -> pl.DataFrame:
    """One row per click event. Pre-computed once outside the user loop."""
    return (
        events_df.filter(pl.col("event_type") == "click")
        .select(["user_id", "timestamp"])
        .rename({"timestamp": "click_ts", "user_id": "click_user_id"})
    )


# ── User features — global (no point-in-time needed) ─────────────────────────


def _days_since_last_activity(events_df: pl.DataFrame) -> pl.DataFrame:
    """days_since_last_activity per event for all users in one vectorised pass.

    Uses shift(1).over('user_id') on timestamp-sorted events.
    First event per user gets null (correct: no prior activity).
    """
    return (
        events_df.sort(["user_id", "timestamp"])
        .with_columns(pl.col("timestamp").shift(1).over("user_id").alias("prev_ts"))
        .with_columns(
            ((pl.col("timestamp") - pl.col("prev_ts")) / 86400.0).alias("days_since_last_activity")
        )
        .select(["event_id", "days_since_last_activity"])
    )


def _avg_session_length(events_df: pl.DataFrame) -> pl.DataFrame:
    """Mean click count per session, averaged per user (global, all data)."""
    return (
        events_df.filter(pl.col("event_type") == "click")
        .group_by(["user_id", "session_id"])
        .agg(pl.len().alias("sess_clicks"))
        .group_by("user_id")
        .agg(pl.col("sess_clicks").mean().alias("avg_session_length"))
    )


def _favorite_genres(events_df: pl.DataFrame, movies_exp: pl.DataFrame) -> pl.DataFrame:
    """Top-N_FAV_GENRES genres by click count per user (global, all data)."""
    return (
        events_df.filter(pl.col("event_type") == "click")
        .select(["user_id", "movie_id"])
        .join(movies_exp, on="movie_id", how="left")
        .explode("genre_list")
        .filter(pl.col("genre_list").is_not_null() & (pl.col("genre_list") != "(no genres listed)"))
        .group_by(["user_id", "genre_list"])
        .agg(pl.len().alias("cnt"))
        .sort(["user_id", "cnt"], descending=[False, True])
        .group_by("user_id")
        .agg(pl.col("genre_list").head(N_FAV_GENRES).alias("favorite_genres"))
    )


# ── User features — per-user rolling window ───────────────────────────────────


def _rolling_user_features(
    user_events: pl.DataFrame,
    user_clicks_genre: pl.DataFrame,
    user_clicks_simple: pl.DataFrame,
    all_genres: list[str],
) -> pl.DataFrame:
    """Compute genre_affinity_last_7d and n_clicks_last_7d for one user.

    Uses join_where with temporal predicates [event.ts - 7d, event.ts) so only
    past clicks contribute — no future leakage.

    Returns DataFrame with columns:
        event_id (str), genre_affinity_last_7d (list[f32]), n_clicks_last_7d (i32)
    """
    n_genres = len(all_genres)
    event_ids = user_events.select("event_id")

    # Initialise with zeros / 0 for all events; overwrite with real values below.
    genre_affinity = event_ids.with_columns(
        pl.Series(
            "genre_affinity_last_7d",
            [[0.0] * n_genres] * event_ids.height,
            dtype=pl.List(pl.Float32),
        )
    )
    n_clicks = event_ids.with_columns(pl.lit(0).cast(pl.Int32).alias("n_clicks_last_7d"))

    # ── genre_affinity_last_7d ────────────────────────────────────────────────
    if user_clicks_genre.height > 0:
        joined = user_events.join_where(
            user_clicks_genre,
            pl.col("user_id") == pl.col("click_user_id"),
            pl.col("click_ts") >= (pl.col("timestamp") - WINDOW_7D_S),
            pl.col("click_ts") < pl.col("timestamp"),
        )
        if joined.height > 0:
            genre_counts = joined.group_by(["event_id", "genre"]).agg(pl.len().alias("count"))
            genre_pivot = genre_counts.pivot(
                index="event_id", on="genre", values="count", aggregate_function="sum"
            ).fill_null(0)
            # Pivot omits genres absent from this user's window; add them as zeros.
            for g in all_genres:
                if g not in genre_pivot.columns:
                    genre_pivot = genre_pivot.with_columns(pl.lit(0).cast(pl.UInt32).alias(g))
            # Normalise row by its sum → proportions in [0, 1]; divide-by-zero → 0.
            total = pl.sum_horizontal(*[pl.col(g) for g in all_genres])
            genre_pivot = (
                genre_pivot.with_columns(
                    [
                        (pl.col(g).cast(pl.Float32) / total.cast(pl.Float32)).alias(g)
                        for g in all_genres
                    ]
                )
                .fill_nan(0.0)
                .with_columns(
                    pl.concat_list([pl.col(g) for g in all_genres]).alias("genre_affinity_last_7d")
                )
                .select(["event_id", "genre_affinity_last_7d"])
            )
            # update() overwrites matching rows; events outside the window keep zeros.
            genre_affinity = genre_affinity.update(genre_pivot, on="event_id")

    # ── n_clicks_last_7d ──────────────────────────────────────────────────────
    if user_clicks_simple.height > 0:
        joined_simple = user_events.join_where(
            user_clicks_simple,
            pl.col("user_id") == pl.col("click_user_id"),
            pl.col("click_ts") >= (pl.col("timestamp") - WINDOW_7D_S),
            pl.col("click_ts") < pl.col("timestamp"),
        )
        if joined_simple.height > 0:
            n_clicks_computed = joined_simple.group_by("event_id").agg(
                pl.len().cast(pl.Int32).alias("n_clicks_last_7d")
            )
            n_clicks = n_clicks.update(n_clicks_computed, on="event_id")

    return genre_affinity.join(n_clicks, on="event_id")


# ── User features — public ────────────────────────────────────────────────────


def compute_user_features(
    events_df: pl.DataFrame,
    all_genres: list[str],
    movies_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute per-event point-in-time user features.

    Args:
        events_df:  the full events table
        all_genres: canonical genre list from compute_genre_list()
        movies_df:  optional movies DataFrame; if None, reads from RAW_DIR/movies.csv
    """
    if movies_df is None:
        movies_df = pl.read_csv(RAW_DIR / "movies.csv")

    movies_exp = (
        movies_df.select(["movieId", "genres"])
        .with_columns(
            pl.col("movieId").cast(pl.Int64).alias("movie_id"),
            pl.col("genres").str.split("|").alias("genre_list"),
        )
        .select(["movie_id", "genre_list"])
    )

    clicks_genre = _build_clicks_genre_table(events_df, movies_exp)
    clicks_simple = _build_clicks_simple_table(events_df)
    days_since = _days_since_last_activity(events_df)
    avg_sess = _avg_session_length(events_df)
    fav_genres = _favorite_genres(events_df, movies_exp)

    all_users = events_df["user_id"].unique().to_list()
    per_user: list[pl.DataFrame] = []

    for uid in tqdm(all_users, desc="Rolling user features", unit="user"):
        user_ev = events_df.filter(pl.col("user_id") == uid).select(
            ["event_id", "user_id", "timestamp"]
        )
        per_user.append(
            _rolling_user_features(
                user_ev,
                clicks_genre.filter(pl.col("click_user_id") == uid),
                clicks_simple.filter(pl.col("click_user_id") == uid),
                all_genres,
            )
        )

    rolling = pl.concat(per_user)

    return (
        events_df.select(["event_id", "user_id"])
        .join(rolling, on="event_id")
        .join(days_since, on="event_id")
        .join(avg_sess, on="user_id", how="left")
        .join(fav_genres, on="user_id", how="left")
        .with_columns(pl.col("favorite_genres").fill_null([]))
    )


# ── Main entry point ───────────────────────────────────────────────────────────


def build_features(
    events_path: Path = PROCESSED_DIR / "events.parquet",
    movies_path: Path = RAW_DIR / "movies.csv",
    ratings_path: Path = RAW_DIR / "ratings.csv",
    genome_path: Path = RAW_DIR / "genome-scores.csv",
    output_path: Path = PROCESSED_DIR / "train_dataset.parquet",
    years: list[int] | None = None,
) -> None:
    """Load sources, compute features, write train_dataset.parquet.

    One row per event. Adds all user and movie feature columns plus 'split'
    ('train'/'val'/'test'). Temporal split: 80th/90th timestamp quantiles.

    Args:
        years: if given, keep only events whose timestamp falls in these calendar
               years (e.g. [2023]). Useful to work on ~1% of the full dataset.
               The temporal split is recalculated on the filtered subset.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source data...")
    events_df = pl.read_parquet(events_path)
    if years is not None:
        event_year = pl.from_epoch(pl.col("timestamp"), time_unit="s").dt.year()
        events_df = events_df.filter(event_year.is_in(years))
        print(f"Filtered to years {years}: {events_df.height:,} events remaining.")
    movies_df = pl.read_csv(movies_path)
    ratings_df = pl.read_csv(ratings_path)
    genome_df = (
        pl.read_csv(genome_path)
        if genome_path.exists()
        else pl.DataFrame(schema={"movieId": pl.Int64, "tagId": pl.Int64, "relevance": pl.Float64})
    )

    all_genres = compute_genre_list(movies_df)
    print(f"Genres ({len(all_genres)}): {all_genres}")

    q80 = events_df["timestamp"].quantile(0.80)
    q90 = events_df["timestamp"].quantile(0.90)
    assert q80 is not None and q90 is not None
    split_train_ts = int(q80)
    split_val_ts = int(q90)

    print("Computing movie features...")
    movie_feats = compute_movie_features(
        movies_df, ratings_df, genome_df, events_df, split_train_ts, all_genres
    )

    print("Computing user features (rolling windows per user)...")
    user_feats = compute_user_features(events_df, all_genres, movies_df)

    print("Joining and writing output...")
    dataset = (
        events_df.join(user_feats.drop("user_id"), on="event_id")
        .join(movie_feats, on="movie_id")
        .with_columns(
            pl.when(pl.col("timestamp") < split_train_ts)
            .then(pl.lit("train"))
            .when(pl.col("timestamp") < split_val_ts)
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
    )

    dataset.write_parquet(output_path)
    counts = dataset["split"].value_counts().sort("split")
    print(f"Wrote {dataset.height:,} rows to {output_path}")
    print(f"Split distribution:\n{counts}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 — offline feature engineering")
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        dest="years",
        metavar="YEAR",
        help="Keep only events from this calendar year (repeatable: --year 2022 --year 2023)",
    )
    args = parser.parse_args()

    kwargs: dict[str, Any] = {}
    if args.events:
        kwargs["events_path"] = args.events
    if args.output:
        kwargs["output_path"] = args.output
    if args.years:
        kwargs["years"] = args.years
    build_features(**kwargs)
