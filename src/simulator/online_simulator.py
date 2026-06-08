"""Simulator 2: concurrent virtual users driving the live serving API.

Architecture:
  main()
    ├─ load warm user IDs from GCS training dataset (or local fallback)
    ├─ launch each warm user as a staggered asyncio task
    ├─ start background spawner for cold-start users
    └─ wait indefinitely (Ctrl+C cancels and closes cleanly)

  _run_user()       — infinite session loop per user; stops when lifetime expires (churn)
  _run_session()    — one page-browsing session: GET /recommendations → POST events
  _spawn_cold_users() — periodically introduces new cold-start users

Warm users (IDs from the training dataset) may already have features in Redis.
Cold-start users (IDs above warm_user_id_max) start with zero-vector recommendations;
their first click seeds Redis via the streaming processor.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import random
import time
import uuid
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclasses.dataclass
class SimulatorConfig:
    api_url: str = "http://localhost:8000"
    # ── Warm user pool ────────────────────────────────────────────────────────
    gcs_dataset_path: str | None = None  # gs://bucket/.../train_dataset.parquet
    n_warm_users: int = 50
    warm_user_id_max: int = 138493  # fallback when gcs_dataset_path is not set
    stagger_seconds: float = 2.0  # delay between launching each warm user
    # ── Churn model ───────────────────────────────────────────────────────────
    churn_fraction: float = 0.3  # fraction of warm users that eventually churn
    churn_lifetime_days: float = 7.0  # mean churn lifetime (exponential distribution)
    cold_churn_fraction: float = 0.8  # most cold-start users churn quickly
    # ── Cold-start user spawner ───────────────────────────────────────────────
    new_users_per_hour: float = 6.0  # 1 new user every 10 minutes by default
    cold_user_id_base: int = 200000  # IDs above the historical MovieLens range
    # ── Session behavior ──────────────────────────────────────────────────────
    temperature: float = 1.0
    max_pages_per_session: int = 5
    session_gap_min: float = 20.0
    session_gap_max: float = 120.0
    watch_time_seconds: float = 10.0
    # ── Kafka ─────────────────────────────────────────────────────────────────
    redpanda_brokers: str | None = None
    seed: int | None = None


# ── Pure helper functions (tested independently) ──────────────────────────────


def _click_prob(score: float, temperature: float) -> float:
    """Return click probability for a recommendation with the given model score.

    Args:
        score: model output in [0, 1] (sigmoid probability from ONNX model)
        temperature: controls how closely the simulator follows model scores.
            temperature=1.0  → probabilities roughly mirror the model score.
            temperature>1.0  → more deterministic (high scores → near-certain click).
            temperature→0    → near-random (≈0.5 regardless of score).

    Returns:
        Click probability in [0, 1].
    """
    score = max(1e-7, min(1 - 1e-7, score))  # guard log(0)
    logit = math.log(score / (1 - score))
    noise = random.gauss(0, 0.5)
    return 1.0 / (1.0 + math.exp(-(logit * temperature + noise)))


def _build_event(
    user_id: int,
    movie_id: int,
    event_type: str,
    session_id: uuid.UUID,
    label: int,
) -> dict:
    """Build the JSON payload for POST /events."""
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": int(time.time()),
        "user_id": user_id,
        "movie_id": movie_id,
        "event_type": event_type,
        "rating": None,
        "session_id": str(session_id),
        "label": label,
    }


def _sample_lifetime(churn_fraction: float, churn_lifetime_days: float) -> float | None:
    """Sample a user lifetime in seconds, or None for a permanent (power) user.

    Uses an exponential distribution: many users churn quickly, a few stay much longer.
    churn_fraction=0 → always None (all power users).
    churn_fraction=1 → always a finite lifetime drawn from the exponential.
    """
    if random.random() >= churn_fraction:
        return None  # power user: never churns
    days = random.expovariate(1.0 / churn_lifetime_days)
    return days * 86400  # convert to seconds


def _load_warm_user_ids(config: SimulatorConfig) -> list[int]:
    """Return a list of n_warm_users user IDs.

    Reads unique user_ids from the GCS training dataset when gcs_dataset_path is set.
    Falls back to a random sample from range(1, warm_user_id_max+1) otherwise.
    """
    if config.gcs_dataset_path:
        try:
            import polars as pl

            df = pl.scan_parquet(config.gcs_dataset_path)
            ids: list[int] = df.select("user_id").unique().collect()["user_id"].to_list()
            log.info("Loaded %d unique user IDs from %s", len(ids), config.gcs_dataset_path)
            return random.sample(ids, min(config.n_warm_users, len(ids)))
        except Exception as exc:
            log.warning("Failed to load user IDs from GCS (%s). Using local fallback.", exc)
    return random.sample(range(1, config.warm_user_id_max + 1), config.n_warm_users)


# ── Async simulation coroutines ───────────────────────────────────────────────


async def _run_session(
    user_id: int,
    config: SimulatorConfig,
    client: httpx.AsyncClient,
    producer: Any | None,
) -> None:
    """Run one page-browsing session for a user."""
    session_id = uuid.uuid4()
    total_seen = 0
    total_clicks = 0

    for _ in range(config.max_pages_per_session):
        # 1. Fetch recommendations
        try:
            resp = await client.get(
                f"{config.api_url}/recommendations/{user_id}",
                params={"n": 5},
            )
            resp.raise_for_status()
            recs: list[dict] = resp.json().get("recommendations", [])
        except Exception as exc:
            log.warning("user %d: recommendations request failed: %s", user_id, exc)
            break

        if not recs:
            break

        # 2. Publish inference log to model-predictions topic
        if producer is not None:
            ts = int(time.time())
            for pos, rec in enumerate(recs):
                try:
                    producer.send(
                        "model-predictions",
                        {
                            "user_id": user_id,
                            "movie_id": rec["movie_id"],
                            "score": rec["score"],
                            "position": pos,
                            "timestamp": ts,
                            "session_id": str(session_id),
                        },
                    )
                except Exception as exc:
                    log.debug("model-predictions publish failed: %s", exc)

        # 3. Decide clicks — emit impression for every rec, click for at most one
        clicked_page = False
        for rec in recs:
            total_seen += 1
            prob = _click_prob(rec["score"], config.temperature)

            imp = _build_event(user_id, rec["movie_id"], "impression", session_id, 0)
            try:
                await client.post(f"{config.api_url}/events", json=imp)
            except Exception:
                pass

            if random.random() < prob:
                total_clicks += 1
                click = _build_event(user_id, rec["movie_id"], "click", session_id, 1)
                try:
                    await client.post(f"{config.api_url}/events", json=click)
                except Exception:
                    pass

                clicked_page = True
                await asyncio.sleep(config.watch_time_seconds)
                break  # one click per page → return to home

        if not clicked_page:
            break  # user lost interest; end session early

    log.debug("user %d: session done (clicks=%d/%d)", user_id, total_clicks, total_seen)


async def _run_user(
    user_id: int,
    config: SimulatorConfig,
    client: httpx.AsyncClient,
    producer: Any | None,
    lifetime_seconds: float | None,
) -> None:
    """Run continuous sessions for a user until their lifetime expires (churn) or forever."""
    deadline = time.monotonic() + lifetime_seconds if lifetime_seconds is not None else float("inf")

    while time.monotonic() < deadline:
        await _run_session(user_id, config, client, producer)
        gap = random.uniform(config.session_gap_min, config.session_gap_max)
        remaining = deadline - time.monotonic()
        await asyncio.sleep(min(gap, remaining))

    if lifetime_seconds is not None:
        log.info("user %d churned after %.1f days", user_id, lifetime_seconds / 86400)


async def _spawn_cold_users(
    config: SimulatorConfig,
    client: httpx.AsyncClient,
    producer: Any | None,
) -> None:
    """Periodically introduce new cold-start users."""
    interval = 3600.0 / config.new_users_per_hour
    next_id = config.cold_user_id_base

    while True:
        await asyncio.sleep(interval)
        lifetime = _sample_lifetime(config.cold_churn_fraction, config.churn_lifetime_days)
        asyncio.create_task(_run_user(next_id, config, client, producer, lifetime))
        log.info(
            "new cold-start user %d spawned (lifetime=%s)",
            next_id,
            f"{lifetime / 86400:.1f}d" if lifetime else "permanent",
        )
        next_id += 1


async def main(config: SimulatorConfig) -> None:
    if config.seed is not None:
        random.seed(config.seed)

    producer: Any | None = None
    if config.redpanda_brokers:
        try:
            from kafka import KafkaProducer

            producer = KafkaProducer(
                bootstrap_servers=config.redpanda_brokers,
                value_serializer=lambda v: json.dumps(v).encode(),
            )
            log.info("Kafka producer connected to %s", config.redpanda_brokers)
        except Exception as exc:
            log.warning("Kafka producer init failed (%s). Inference log disabled.", exc)

    warm_ids = _load_warm_user_ids(config)
    log.info(
        "Starting simulator: %d warm users, stagger=%.1fs, new_users/h=%.1f",
        len(warm_ids),
        config.stagger_seconds,
        config.new_users_per_hour,
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Stagger warm user launches
            for uid in warm_ids:
                lifetime = _sample_lifetime(config.churn_fraction, config.churn_lifetime_days)
                asyncio.create_task(_run_user(uid, config, client, producer, lifetime))
                log.info(
                    "warm user %d launched (lifetime=%s)",
                    uid,
                    f"{lifetime / 86400:.1f}d" if lifetime else "permanent",
                )
                await asyncio.sleep(config.stagger_seconds)

            asyncio.create_task(_spawn_cold_users(config, client, producer))

            # Run until Ctrl+C
            await asyncio.Event().wait()
    except asyncio.CancelledError:
        log.info("Simulator shutting down.")
    finally:
        if producer is not None:
            producer.close()


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(tyro.cli(SimulatorConfig)))
