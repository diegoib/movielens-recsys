"""Simulator 2: concurrent virtual users driving the live serving API.

Each user runs as an asyncio coroutine:
  1. GET /recommendations/{user_id}         — fetch personalized top-N
  2. Publish to topic "model-predictions"   — inference log for retraining
  3. POST /events (impression + maybe click) — feeds back into PyFlink → Redis
  4. asyncio.sleep(watch_time)              — simulates watching the movie
  5. Repeat until session ends, then gap, then new session

Cold-start users (no Redis data) get zero-vector recommendations on the first
page; their first click seeds Redis via the streaming processor, so from page 2
onward they are semi-warm.  Set user_id_start to a range with existing offline
features (populated by make features) to simulate warm users from the start.
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
    n_users: int = 10
    user_id_start: int = 1
    temperature: float = 1.0
    max_pages_per_session: int = 5
    session_gap_seconds: float = 30.0
    watch_time_seconds: float = 10.0
    duration_seconds: int = 300
    redpanda_brokers: str | None = None
    seed: int | None = None


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


async def _simulate_user(
    user_id: int,
    config: SimulatorConfig,
    client: httpx.AsyncClient,
    producer: Any | None,
) -> None:
    """Run one virtual user for the full simulation duration."""
    deadline = time.monotonic() + config.duration_seconds
    total_seen = 0
    total_clicks = 0

    while time.monotonic() < deadline:
        session_id = uuid.uuid4()
        clicked_this_session = False

        for _ in range(config.max_pages_per_session):
            if time.monotonic() >= deadline:
                break

            # ── 1. Fetch recommendations ──────────────────────────────────────
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

            # ── 2. Publish inference log ──────────────────────────────────────
            if producer is not None:
                ts = int(time.time())
                for pos, rec in enumerate(recs):
                    try:
                        producer.send(  # type: ignore[union-attr]
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

            # ── 3. Decide clicks ──────────────────────────────────────────────
            clicked_page = False
            for rec in recs:
                total_seen += 1
                prob = _click_prob(rec["score"], config.temperature)

                # Always emit impression
                imp_payload = _build_event(user_id, rec["movie_id"], "impression", session_id, 0)
                try:
                    await client.post(f"{config.api_url}/events", json=imp_payload)
                except Exception:
                    pass

                if random.random() < prob:
                    total_clicks += 1
                    click_payload = _build_event(user_id, rec["movie_id"], "click", session_id, 1)
                    try:
                        await client.post(f"{config.api_url}/events", json=click_payload)
                    except Exception:
                        pass

                    clicked_page = True
                    clicked_this_session = True
                    await asyncio.sleep(config.watch_time_seconds)
                    break  # one click per page → return to home

            if not clicked_page:
                # No clicks on this page → user lost interest, end session
                break

        log.info(
            "user %d: session done — clicks=%d/%d (total)",
            user_id,
            total_clicks,
            total_seen,
        )

        if not clicked_this_session:
            # Short gap for disengaged session
            gap = config.session_gap_seconds * 0.5
        else:
            gap = random.uniform(config.session_gap_seconds * 0.5, config.session_gap_seconds * 1.5)
        await asyncio.sleep(min(gap, deadline - time.monotonic()))


async def main(config: SimulatorConfig) -> None:
    if config.seed is not None:
        random.seed(config.seed)

    producer = None
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

    user_ids = list(range(config.user_id_start, config.user_id_start + config.n_users))
    log.info(
        "Starting simulator: %d users (ids %d-%d), temperature=%.2f, duration=%ds",
        config.n_users,
        user_ids[0],
        user_ids[-1],
        config.temperature,
        config.duration_seconds,
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [_simulate_user(uid, config, client, producer) for uid in user_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for uid, result in zip(user_ids, results):
        if isinstance(result, Exception):
            log.error("user %d raised: %s", uid, result)

    if producer is not None:
        producer.close()

    log.info("Simulation complete.")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(tyro.cli(SimulatorConfig)))
