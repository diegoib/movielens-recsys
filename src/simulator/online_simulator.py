"""Simulator 2: concurrent virtual users driving the live serving API.

Architecture — worker pool:
  main()
    ├─ pool = UserPool(all_warm_ids)     ← all training dataset user IDs in memory
    ├─ for _ in range(max_concurrent):   ← exactly K active sessions at any time
    │    create_task(_worker(pool))
    ├─ create_task(_add_cold_users_to_pool())
    └─ await Event().wait()              ← runs until Ctrl+C

  _worker()  — picks a random user from the pool, runs their session, sleeps, repeats
  _run_session() — one browsing session: GET /recommendations → POST impression/click events

The pool starts with all MovieLens user IDs (warm users, may have Redis features) and
grows as cold-start users are added at rate new_users_per_hour.  With K=10 workers and
138K users in the pool, each user gets picked roughly once every 138K/10 ≈ 13K sessions.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import random
import time
import uuid

import httpx

log = logging.getLogger(__name__)


@dataclasses.dataclass
class SimulatorConfig:
    api_url: str = "http://localhost:8000"
    # ── Concurrency ───────────────────────────────────────────────────────────
    max_concurrent: int = 10  # number of active session workers
    # ── User pool ─────────────────────────────────────────────────────────────
    gcs_dataset_path: str | None = None  # gs://bucket/.../train_dataset.parquet
    warm_user_id_max: int = 138493  # fallback pool range when gcs_dataset_path unset
    # ── Cold-start users ──────────────────────────────────────────────────────
    new_users_per_hour: float = 6.0  # rate at which new cold-start IDs join the pool
    cold_user_id_base: int = 200_000  # cold-start IDs start here (above MovieLens range)
    # ── Session behavior ──────────────────────────────────────────────────────
    temperature: float = 1.0
    max_pages_per_session: int = 5
    session_gap_min: float = 10.0
    session_gap_max: float = 60.0
    watch_time_seconds: float = 3.0
    seed: int | None = None


# ── User pool ─────────────────────────────────────────────────────────────────


class UserPool:
    """Mutable pool of user IDs shared across all workers.

    asyncio is single-threaded so no locking is needed: random.choice and
    list.append are not interrupted between awaits.
    """

    def __init__(self, warm_ids: list[int], cold_user_id_base: int) -> None:
        self._ids = warm_ids
        self._next_cold = cold_user_id_base

    def pick(self) -> int:
        return random.choice(self._ids)

    def add_cold_user(self) -> int:
        uid = self._next_cold
        self._ids.append(uid)
        self._next_cold += 1
        return uid

    def __len__(self) -> int:
        return len(self._ids)


def _load_warm_user_ids(config: SimulatorConfig) -> list[int]:
    """Return all unique user IDs from the training dataset, or a range fallback."""
    if config.gcs_dataset_path:
        try:
            import polars as pl

            df = pl.scan_parquet(config.gcs_dataset_path)
            ids: list[int] = df.select("user_id").unique().collect()["user_id"].to_list()
            log.info("Loaded %d unique user IDs from %s", len(ids), config.gcs_dataset_path)
            return ids
        except Exception as exc:
            log.warning("Failed to load user IDs from GCS (%s). Using local fallback.", exc)
    ids = list(range(1, config.warm_user_id_max + 1))
    log.info("Using local fallback pool: %d user IDs (1..%d)", len(ids), config.warm_user_id_max)
    return ids


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
    recommendation_id: str | None = None,
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
        "recommendation_id": recommendation_id,
    }


# ── Async simulation coroutines ───────────────────────────────────────────────


async def _run_session(
    user_id: int,
    config: SimulatorConfig,
    client: httpx.AsyncClient,
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
            body = resp.json()
            recommendation_id: str | None = body.get("recommendation_id")
            recs: list[dict] = body.get("recommendations", [])
        except Exception as exc:
            log.warning("user %d: recommendations request failed: %s", user_id, exc)
            break

        if not recs:
            break

        # 2. Decide clicks — impression for every rec, at most one click per page
        clicked_page = False
        for rec in recs:
            total_seen += 1
            prob = _click_prob(rec["score"], config.temperature)

            imp = _build_event(
                user_id, rec["movie_id"], "impression", session_id, 0, recommendation_id
            )
            try:
                await client.post(f"{config.api_url}/events", json=imp)
            except Exception:
                pass

            if random.random() < prob:
                total_clicks += 1
                click = _build_event(
                    user_id, rec["movie_id"], "click", session_id, 1, recommendation_id
                )
                try:
                    await client.post(f"{config.api_url}/events", json=click)
                except Exception:
                    pass

                clicked_page = True
                await asyncio.sleep(config.watch_time_seconds)
                break  # one click per page → back to home

        if not clicked_page:
            break  # lost interest; end session early

    log.debug("user %d: session done (clicks=%d/%d)", user_id, total_clicks, total_seen)


async def _worker(
    pool: UserPool,
    config: SimulatorConfig,
    client: httpx.AsyncClient,
) -> None:
    """One active session slot: pick user → session → sleep → repeat."""
    while True:
        user_id = pool.pick()
        await _run_session(user_id, config, client)
        gap = random.uniform(config.session_gap_min, config.session_gap_max)
        await asyncio.sleep(gap)


async def _add_cold_users_to_pool(pool: UserPool, new_users_per_hour: float) -> None:
    """Periodically add new cold-start user IDs to the pool."""
    interval = 3600.0 / new_users_per_hour
    while True:
        await asyncio.sleep(interval)
        uid = pool.add_cold_user()
        log.info("cold-start user %d added to pool (pool size: %d)", uid, len(pool))


async def main(config: SimulatorConfig) -> None:
    if config.seed is not None:
        random.seed(config.seed)

    warm_ids = _load_warm_user_ids(config)
    pool = UserPool(warm_ids, config.cold_user_id_base)
    log.info(
        "Pool: %d warm users — launching %d concurrent workers",
        len(pool),
        config.max_concurrent,
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for _ in range(config.max_concurrent):
                asyncio.create_task(_worker(pool, config, client))
            asyncio.create_task(_add_cold_users_to_pool(pool, config.new_users_per_hour))
            await asyncio.Event().wait()
    except asyncio.CancelledError:
        log.info("Simulator shutting down.")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(tyro.cli(SimulatorConfig)))
