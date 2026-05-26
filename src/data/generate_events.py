"""Simulator 1 — generate synthetic event table from MovieLens ratings.

Pipeline:
  1. Pre-compute lookup structures (monthly popularity, genre pools, user history).
  2. Group each user's ratings into sessions (gap > 60 min = new session).
  3. Per session: generate positive funnel + type-A/B negatives.
  4. Write events to Parquet in batches to stay within memory limits.
"""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SESSION_GAP_S = 3_600  # seconds; gap > 60 min → new session
NEG_A_RANGE = (4, 6)  # type-A negatives per rated movie
TYPE_B_DIVISOR = (2, 3)  # 1 type-B per 2-3 clicks
FRAC_POPULAR = 0.40
FRAC_GENRE = 0.40
FRAC_COLLAB = 0.20
POPULAR_TOP_K = 500  # movies kept per monthly bucket
BATCH_SIZE = 1_000  # users per Parquet chunk
RANDOM_SEED = 42


# ── Pre-computation ────────────────────────────────────────────────────────────


def _build_genre_structures(
    movies_df: pl.DataFrame,
) -> tuple[dict[int, list[str]], dict[str, list[int]]]:
    """Return (genre_map, genre_to_movies).

    genre_map       : movie_id → list of genre strings
    genre_to_movies : genre   → list of movie_ids
    """
    genre_map: dict[int, list[str]] = {}
    genre_to_movies: dict[str, list[int]] = defaultdict(list)
    for row in movies_df.iter_rows(named=True):
        genres = [g for g in row["genres"].split("|") if g != "(no genres listed)"]
        genre_map[row["movieId"]] = genres
        for g in genres:
            genre_to_movies[g].append(row["movieId"])
    return genre_map, dict(genre_to_movies)


def _build_popular_by_month(ratings_df: pl.DataFrame) -> dict[tuple[int, int], list[int]]:
    """Top-POPULAR_TOP_K movies per calendar month, ranked by rating count."""
    df = (
        ratings_df.with_columns(pl.from_epoch("timestamp", time_unit="s").alias("dt"))
        .with_columns(
            [
                pl.col("dt").dt.year().cast(pl.Int32).alias("year"),
                pl.col("dt").dt.month().cast(pl.Int8).alias("month"),
            ]
        )
        .group_by(["year", "month", "movieId"])
        .agg(pl.len().alias("cnt"))
        .sort(["year", "month", "cnt"], descending=[False, False, True])
    )
    popular: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in df.iter_rows(named=True):
        key = (row["year"], row["month"])
        if len(popular[key]) < POPULAR_TOP_K:
            popular[key].append(row["movieId"])
    return dict(popular)


def _build_user_history(
    ratings_df: pl.DataFrame,
) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    """Return (user_rated, movie_to_users).

    user_rated    : user_id  → set of rated movie_ids (used to filter negatives)
    movie_to_users: movie_id → list of user_ids (inverted index for collab)

    Uses Polars groupby (Rust/Arrow) instead of a Python row loop to avoid
    materialising 20M Python dicts during construction.
    """
    user_grouped = ratings_df.group_by("userId").agg(pl.col("movieId"))
    user_rated = {int(r[0]): set(r[1]) for r in user_grouped.iter_rows()}

    movie_grouped = ratings_df.group_by("movieId").agg(pl.col("userId"))
    movie_to_users = {int(r[0]): list(r[1]) for r in movie_grouped.iter_rows()}

    return user_rated, movie_to_users


# ── Collaborative neighborhood ─────────────────────────────────────────────────


def _find_top_k_neighbors(
    user_rated: set[int],
    movie_to_users: dict[int, list[int]],
    all_user_rated: dict[int, set[int]],
    current_user_id: int,
    k: int = 20,
) -> list[int]:
    """Return IDs of the k users most similar to current_user_id.

    Strategy: use movie_to_users (inverted index) to find candidate neighbors
    efficiently — only users who share at least one rated movie are worth
    comparing. Then rank candidates by co-rated count (approximation of Jaccard).
    Exclude current_user_id from the result.
    """
    users: set[int] = set()
    for film in user_rated:
        movie_users = movie_to_users.get(film, [])
        if len(movie_users) < 1000:
            users.update(movie_users)
    users.discard(current_user_id)  # never include the query user itself

    users_jaccard: list[tuple[int, int]] = []
    for user in users:
        other = all_user_rated.get(user)
        if other is None:  # user not in active subset (e.g. --max-users filtering)
            continue
        users_jaccard.append((user, len(other.intersection(user_rated))))
    users_jaccard.sort(key=lambda x: x[1], reverse=True)  # highest overlap first
    return [user[0] for user in users_jaccard[:k]]


def _collab_pool(
    user_rated: set[int],
    movie_to_users: dict[int, list[int]],
    all_user_rated: dict[int, set[int]],
    current_user_id: int,
) -> list[int]:
    neighbors = _find_top_k_neighbors(user_rated, movie_to_users, all_user_rated, current_user_id)
    if not neighbors:
        return []
    pool: set[int] = set()
    for nid in neighbors:
        pool |= all_user_rated.get(nid, set())
    return list(pool - user_rated)


# ── Session construction ───────────────────────────────────────────────────────


def _build_sessions(
    user_ratings: list[tuple[int, int, float]],
) -> list[tuple[str, list[tuple[int, int, float]]]]:
    """Group (movie_id, timestamp, rating_val) tuples into sessions.

    A new session starts when the gap to the previous rating exceeds SESSION_GAP_S.
    Returns list of (session_id_str, ratings_in_session).
    """
    sessions: list[tuple[str, list[tuple[int, int, float]]]] = []
    sid = str(uuid.uuid4())
    current: list[tuple[int, int, float]] = [user_ratings[0]]
    for i in range(1, len(user_ratings)):
        if user_ratings[i][1] - user_ratings[i - 1][1] > SESSION_GAP_S:
            sessions.append((sid, current))
            sid, current = str(uuid.uuid4()), []
        current.append(user_ratings[i])
    sessions.append((sid, current))
    return sessions


# ── Event helpers ──────────────────────────────────────────────────────────────


def _sample(pool: list[int], k: int, exclude: set[int], rng: random.Random) -> list[int]:
    candidates = [m for m in pool if m not in exclude]
    return rng.sample(candidates, min(k, len(candidates)))


def _evt(
    event_type: str,
    timestamp: int,
    user_id: int,
    movie_id: int,
    rating: float | None,
    session_id: str,
    label: int,
) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "user_id": user_id,
        "movie_id": movie_id,
        "event_type": event_type,
        "rating": rating,
        "session_id": session_id,
        "label": label,
    }


# ── Per-user event generation ──────────────────────────────────────────────────


def _generate_user_events(  # noqa: PLR0913
    user_id: int,
    sessions: list[tuple[str, list[tuple[int, int, float]]]],
    user_rated: set[int],
    popular_by_month: dict[tuple[int, int], list[int]],
    genre_map: dict[int, list[str]],
    genre_to_movies: dict[str, list[int]],
    movie_to_users: dict[int, list[int]],
    all_user_rated: dict[int, set[int]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cp = _collab_pool(user_rated, movie_to_users, all_user_rated, user_id)

    for sid, session_ratings in sessions:
        n_clicks = len(session_ratings)
        n_type_b = max(0, n_clicks // rng.randint(*TYPE_B_DIVISOR))

        # Genres seen in this session (used for type-B pool)
        session_genres: set[str] = set()
        for m, _, _ in session_ratings:
            session_genres.update(genre_map.get(m, []))

        ts_session_start = session_ratings[0][1] - rng.randint(20, 40) * 60

        for movie_id, ts_rating, rating_val in session_ratings:
            # ── Positive funnel (backward from rating timestamp) ─────────
            ts_imp = ts_rating - rng.randint(20, 40) * 60
            ts_view = ts_imp + rng.randint(1, 5) * 60
            ts_click = ts_view + rng.randint(5, 15) * 60

            # label=1 for all four: this movie was ultimately clicked/rated
            events += [
                _evt("impression", ts_imp, user_id, movie_id, None, sid, 1),
                _evt("view", ts_view, user_id, movie_id, None, sid, 1),
                _evt("click", ts_click, user_id, movie_id, None, sid, 1),
                _evt("rating", ts_rating, user_id, movie_id, rating_val, sid, 1),
            ]

            # ── Type-A negatives (impressions without view) ──────────────
            dt = datetime.fromtimestamp(ts_rating, tz=UTC)
            pop_pool = popular_by_month.get((dt.year, dt.month), [])

            genre_pool: list[int] = []
            for g in genre_map.get(movie_id, []):
                genre_pool.extend(genre_to_movies.get(g, []))

            n_neg = rng.randint(*NEG_A_RANGE)
            n_pop = max(1, round(n_neg * FRAC_POPULAR))
            n_gen = max(1, round(n_neg * FRAC_GENRE))
            n_col = max(0, n_neg - n_pop - n_gen)

            neg_movies = (
                _sample(pop_pool, n_pop, user_rated, rng)
                + _sample(genre_pool, n_gen, user_rated, rng)
                + _sample(cp, n_col, user_rated, rng)
            )
            seen: set[int] = set()
            deduped = [m for m in neg_movies if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]

            for neg_movie in deduped[:n_neg]:
                ts_neg = ts_session_start + rng.randint(-5, 20) * 60
                events.append(_evt("impression", ts_neg, user_id, neg_movie, None, sid, 0))

        # ── Type-B negatives (views without click) ───────────────────────
        if session_ratings and n_type_b > 0:
            type_b_pool: list[int] = []
            for g in session_genres:
                type_b_pool.extend(genre_to_movies.get(g, []))
            type_b_pool = list(set(type_b_pool) - user_rated)

            ts_mid = session_ratings[len(session_ratings) // 2][1]
            for neg_movie in rng.sample(type_b_pool, min(n_type_b, len(type_b_pool))):
                ts_view_b = ts_mid + rng.randint(-10, 10) * 60
                events.append(_evt("view", ts_view_b, user_id, neg_movie, None, sid, 0))

    return events


# ── Main entry point ───────────────────────────────────────────────────────────


ROW_CHUNK_SIZE = 5_000_000  # write a parquet chunk every ~5M events


def generate_events(
    output_path: Path | None = None,
    max_users: int | None = None,
) -> None:
    """Generate the synthetic events table.

    Args:
        output_path: destination parquet file (default: data/processed/events.parquet)
        max_users:   limit to first N users — useful for smoke-testing locally
    """
    out = output_path or PROCESSED_DIR / "events.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source data...")
    ratings_df = pl.read_csv(RAW_DIR / "ratings.csv")
    movies_df = pl.read_csv(RAW_DIR / "movies.csv")

    print("Pre-computing lookup structures...")
    genre_map, genre_to_movies = _build_genre_structures(movies_df)
    popular_by_month = _build_popular_by_month(ratings_df)
    user_rated, movie_to_users = _build_user_history(ratings_df)

    if max_users is not None:
        keep = set(list(user_rated.keys())[:max_users])
        ratings_df = ratings_df.filter(pl.col("userId").is_in(keep))
        user_rated = {k: v for k, v in user_rated.items() if k in keep}
        print(f"Limited to {max_users} users ({ratings_df.height:,} ratings).")

    # Sort once; extract numpy arrays (zero-copy from Arrow) to avoid
    # building a Python dict of 20M tuples (would cost ~3 GB).
    ratings_sorted = ratings_df.sort(["userId", "timestamp"])
    uid_arr = ratings_sorted["userId"].to_numpy()
    mid_arr = ratings_sorted["movieId"].to_numpy()
    ts_arr = ratings_sorted["timestamp"].to_numpy()
    rat_arr = ratings_sorted["rating"].to_numpy()
    del ratings_sorted, ratings_df

    rng = random.Random(RANDOM_SEED)
    chunk_paths: list[Path] = []
    batch_rows: list[dict[str, Any]] = []

    def _flush_chunk() -> None:
        chunk = PROCESSED_DIR / f"_chunk_{len(chunk_paths):04d}.parquet"
        pl.DataFrame(batch_rows).write_parquet(chunk)
        chunk_paths.append(chunk)
        batch_rows.clear()

    def _process_user(uid: int, buf: list[tuple[int, int, float]]) -> None:
        sessions = _build_sessions(buf)
        batch_rows.extend(
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
        if len(batch_rows) >= ROW_CHUNK_SIZE:
            _flush_chunk()

    prev_uid = -1
    user_buf: list[tuple[int, int, float]] = []
    n_users = len(user_rated)

    with tqdm(total=n_users, desc="Generating events", unit="user") as pbar:
        for i in range(len(uid_arr)):
            uid = int(uid_arr[i])
            if uid != prev_uid:
                if prev_uid != -1:
                    _process_user(prev_uid, user_buf)
                    pbar.update(1)
                prev_uid = uid
                user_buf = []
            user_buf.append((int(mid_arr[i]), int(ts_arr[i]), float(rat_arr[i])))
        if user_buf:
            _process_user(prev_uid, user_buf)
            pbar.update(1)

    if batch_rows:
        _flush_chunk()

    print("Merging chunks...")
    pl.scan_parquet([str(p) for p in chunk_paths]).sink_parquet(out)
    for p in chunk_paths:
        p.unlink()
    print(f"Events saved to {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulator 1 — generate events table")
    parser.add_argument(
        "--max-users", type=int, default=None, help="Limit to N users (for smoke tests)"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output parquet path")
    args = parser.parse_args()
    generate_events(output_path=args.output, max_users=args.max_users)
