"""Simulator 1 — generate synthetic event table from MovieLens ratings.

Pipeline:
  1. Pre-compute lookup structures (monthly popularity, genre pools, user history).
  2. Group each user's ratings into sessions (gap > 60 min = new session).
  3. Per session: generate positive funnel + type-A/B negatives.
  4. Workers write parquet chunks directly to disk; main process merges at the end.
"""

from __future__ import annotations

import os
import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from multiprocessing import Pool
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
USERS_PER_BATCH = 500  # users processed per worker task
RANDOM_SEED = 42

# Shared read-only state populated before Pool creation; inherited by fork workers
# via copy-on-write — no pickling of the large lookup dicts required.
_G: dict[str, Any] = {}


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


# ── Parallel worker ───────────────────────────────────────────────────────────


def _process_batch_task(args: tuple[int, list[tuple[int, list[tuple[int, int, float]]]]]) -> str:
    """Generate events for a batch of users and write them to a parquet chunk.

    Returns the path of the written file so the main process only receives a
    short string through IPC — not the ~700 dicts per user that the previous
    design sent.  Workers free the event list from memory as soon as the
    parquet file is flushed.
    """
    batch_idx, user_batch = args
    all_events: list[dict[str, Any]] = []
    for uid, buf in user_batch:
        rng = random.Random(RANDOM_SEED ^ uid)
        sessions = _build_sessions(buf)
        all_events.extend(
            _generate_user_events(
                uid,
                sessions,
                _G["user_rated"][uid],
                _G["popular_by_month"],
                _G["genre_map"],
                _G["genre_to_movies"],
                _G["movie_to_users"],
                _G["user_rated"],
                rng,
            )
        )
    chunk = Path(_G["chunk_dir"]) / f"_batch_{batch_idx:04d}.parquet"
    pl.DataFrame(all_events).write_parquet(chunk)
    return str(chunk)


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

    # chunk_dir must exist before forking so workers can write to it.
    chunk_dir = PROCESSED_DIR / "_batches"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Populate shared globals before forking so workers inherit them via COW.
    _G.update(
        genre_map=genre_map,
        genre_to_movies=genre_to_movies,
        popular_by_month=popular_by_month,
        movie_to_users=movie_to_users,
        user_rated=user_rated,
        chunk_dir=str(chunk_dir),
    )

    # Build per-user rating lists from sorted arrays.
    print("Building per-user rating buffers...")
    ratings_sorted = ratings_df.sort(["userId", "timestamp"])
    uid_arr = ratings_sorted["userId"].to_numpy()
    mid_arr = ratings_sorted["movieId"].to_numpy()
    ts_arr = ratings_sorted["timestamp"].to_numpy()
    rat_arr = ratings_sorted["rating"].to_numpy()
    del ratings_sorted, ratings_df

    user_ratings: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for i in range(len(uid_arr)):
        user_ratings[int(uid_arr[i])].append((int(mid_arr[i]), int(ts_arr[i]), float(rat_arr[i])))
    del uid_arr, mid_arr, ts_arr, rat_arr

    tasks = list(user_ratings.items())
    del user_ratings

    # Group users into batches; each worker writes one parquet file per batch
    # and returns only a path string — avoiding 67 GB of IPC serialization.
    batched_tasks = [
        (batch_idx, tasks[i : i + USERS_PER_BATCH])
        for batch_idx, i in enumerate(range(0, len(tasks), USERS_PER_BATCH))
    ]

    # Cap at 4 workers: beyond that, fork-induced COW copies of the shared dicts
    # consume more RAM than the parallelism is worth on this dataset.
    n_workers = min(os.cpu_count() or 4, 4)
    print(
        f"Processing {len(tasks):,} users in {len(batched_tasks)} batches"
        f" across {n_workers} workers..."
    )

    chunk_paths: list[Path] = []
    with (
        Pool(processes=n_workers) as pool,
        tqdm(total=len(batched_tasks), desc="Generating events", unit="batch") as pbar,
    ):
        for path_str in pool.imap_unordered(_process_batch_task, batched_tasks):
            chunk_paths.append(Path(path_str))
            pbar.update(1)

    print("Merging chunks...")
    pl.scan_parquet([str(p) for p in chunk_paths]).sink_parquet(out)
    for p in chunk_paths:
        p.unlink()
    chunk_dir.rmdir()
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
