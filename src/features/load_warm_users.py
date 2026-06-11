"""One-shot script: load warm user features from the training dataset into Redis.

Reads the training parquet (local or GCS), takes the most recent row per user
(which has the most up-to-date point-in-time user features), and writes them
to Redis so the serving layer starts with personalised recommendations instead
of cold-start zero vectors.

Written keys per user (hash user:{id}):
  genre_affinity_last_7d  → JSON list[float]
  n_clicks_last_7d        → str(int)
  days_since_last_activity → str(float)
  avg_session_length      → str(float)
  favorite_genres         → JSON list[str]

Format matches exactly what the streaming processor writes so features from
both sources are interchangeable.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys

import polars as pl
import redis as redis_lib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_FEATURE_COLS = [
    "user_id",
    "timestamp",
    "genre_affinity_last_7d",
    "n_clicks_last_7d",
    "days_since_last_activity",
    "avg_session_length",
    "favorite_genres",
]

_TTL_SECONDS = 30 * 24 * 3600  # 30 days; streaming processor will refresh on activity


@dataclasses.dataclass
class WarmupConfig:
    parquet_path: str = "data/processed/train_dataset.parquet"
    redis_host: str = os.environ.get("REDIS_HOST", "localhost")
    redis_port: int = int(os.environ.get("REDIS_PORT", "6379"))
    batch_size: int = 500


def _to_float_list(val: object) -> list[float]:
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    if isinstance(val, str):
        return [float(x) for x in val.strip().split()]
    return []


def _to_str_list(val: object) -> list[str]:
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    if isinstance(val, str):
        return val.strip().split()
    return []


def load(config: WarmupConfig) -> None:
    log.info("Reading user features from %s", config.parquet_path)

    df = (
        pl.scan_parquet(config.parquet_path)
        .select(_FEATURE_COLS)
        .sort("timestamp", descending=True)
        .unique(subset=["user_id"], keep="first")  # most recent row per user
        .collect()
    )

    n_users = len(df)
    log.info(
        "Loaded %d unique users — writing to Redis %s:%d",
        n_users,
        config.redis_host,
        config.redis_port,
    )

    r = redis_lib.Redis(host=config.redis_host, port=config.redis_port, decode_responses=True)
    try:
        r.ping()
    except redis_lib.ConnectionError as exc:
        log.error("Cannot connect to Redis: %s", exc)
        sys.exit(1)

    pipe = r.pipeline(transaction=False)
    written = 0

    for row in df.iter_rows(named=True):
        uid = row["user_id"]
        key = f"user:{uid}"

        pipe.hset(
            key,
            mapping={
                "genre_affinity_last_7d": json.dumps(_to_float_list(row["genre_affinity_last_7d"])),
                "n_clicks_last_7d": str(int(row["n_clicks_last_7d"] or 0)),
                "days_since_last_activity": str(float(row["days_since_last_activity"] or 0.0)),
                "avg_session_length": str(float(row["avg_session_length"] or 0.0)),
                "favorite_genres": json.dumps(_to_str_list(row["favorite_genres"])),
            },
        )
        pipe.expire(key, _TTL_SECONDS)
        written += 1

        if written % config.batch_size == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
            log.info("  %d / %d users written...", written, n_users)

    pipe.execute()
    log.info("Done. %d warm users loaded into Redis.", written)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load warm user features into Redis")
    parser.add_argument("--parquet_path", default="data/processed/train_dataset.parquet")
    parser.add_argument("--redis_host", default=os.environ.get("REDIS_HOST", "localhost"))
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()
    load(
        WarmupConfig(
            parquet_path=args.parquet_path,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            batch_size=args.batch_size,
        )
    )
