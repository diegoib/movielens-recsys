"""PyFlink streaming job: consumes click events from RedPanda, writes user features to Redis.

Architecture:
    RedPanda (topic: events)
        → KafkaSource
        → filter clicks
        → keyBy user_id
        → UserFeatureProcessor (KeyedProcessFunction)
        → Redis HSET user:{user_id}

Running locally (outside Docker):
    Requires Java 11+ and `uv sync --group streaming`.
    REDPANDA_BROKERS=localhost:9092 REDIS_HOST=localhost python src/features/processor.py

Running in Docker:
    Uses `docker/streaming/Dockerfile` which bundles the Flink runtime + Kafka connector JAR.
"""

from __future__ import annotations

import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_WINDOW_7D = 7 * 24 * 3600
_WINDOW_1H = 3600

REDPANDA_BROKERS = os.environ.get("REDPANDA_BROKERS", "redpanda:9092")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))


# ── Pure feature computation (no Flink, no Redis — fully testable) ────────────


def _compute_features(entries: list[dict], now_ts: int) -> dict:
    """Compute user features from a list of click history entries.

    Args:
        entries: list of {"ts": int, "genres_vector": list[float]} dicts,
                 already pruned to the last 7 days.
        now_ts: current unix timestamp in seconds.

    Returns:
        dict with keys: genre_affinity_last_7d, n_clicks_last_7d,
        genre_affinity_last_1h, n_clicks_last_1h, days_since_last_activity.
    """
    n_genres: int = next(
        (len(e["genres_vector"]) for e in entries if e.get("genres_vector")),
        20,
    )

    entries_1h = [e for e in entries if e["ts"] >= now_ts - _WINDOW_1H]

    def _avg_genre(subset: list[dict]) -> list[float]:
        vecs = [e["genres_vector"] for e in subset if e.get("genres_vector")]
        if not vecs:
            return [0.0] * n_genres
        return [sum(v[i] for v in vecs) / len(vecs) for i in range(n_genres)]

    return {
        "genre_affinity_last_7d": _avg_genre(entries),
        "n_clicks_last_7d": len(entries),
        "genre_affinity_last_1h": _avg_genre(entries_1h),
        "n_clicks_last_1h": len(entries_1h),
        "days_since_last_activity": (
            (now_ts - max(e["ts"] for e in entries)) / 86400 if entries else 7.0
        ),
    }


# ── PyFlink stateful processor ────────────────────────────────────────────────

try:
    import redis as redis_lib
    from pyflink.common import Types
    from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext
    from pyflink.datastream.state import ListStateDescriptor

    class UserFeatureProcessor(KeyedProcessFunction):  # type: ignore[misc]
        """Maintains a rolling list of click events per user in Flink ListState.

        On each click event:
        1. Deserializes the enriched event JSON (must include genres_vector).
        2. Appends a compact {"ts", "genres_vector"} entry to ListState.
        3. Prunes entries older than 7 days (event-time pruning on wall clock).
        4. Calls _compute_features and writes all features to Redis via HSET.
        5. Sets a 7-day TTL on the Redis key to auto-expire inactive users.

        Redis writes use a fire-and-forget pattern — errors are logged but do
        not stop the stream. In production, you would add a dead-letter sink.
        """

        def open(self, ctx: RuntimeContext) -> None:
            self._state = ctx.get_list_state(ListStateDescriptor("click_history", Types.STRING()))
            self._redis = redis_lib.Redis(
                host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=1.0
            )

        def process_element(  # type: ignore[override]
            self, event_json: str, ctx: KeyedProcessFunction.Context
        ) -> None:
            try:
                event = json.loads(event_json)
            except json.JSONDecodeError:
                log.warning("Malformed event JSON, skipping: %.200s", event_json)
                return

            now_ts = int(time.time())

            new_entry = json.dumps(
                {
                    "ts": event.get("timestamp", now_ts),
                    "genres_vector": event.get("genres_vector") or [],
                }
            )

            # Update ListState: append, prune entries older than 7d
            current: list[str] = list(self._state.get() or [])
            current.append(new_entry)
            cutoff = now_ts - _WINDOW_7D
            current = [e for e in current if json.loads(e)["ts"] >= cutoff]
            self._state.update(current)

            entries = [json.loads(e) for e in current]
            features = _compute_features(entries, now_ts)

            user_id = event["user_id"]
            try:
                self._redis.hset(
                    f"user:{user_id}",
                    mapping={
                        k: json.dumps(v) if isinstance(v, list) else str(v)
                        for k, v in features.items()
                    },
                )
                self._redis.expire(f"user:{user_id}", _WINDOW_7D)
            except redis_lib.RedisError as exc:
                log.error("Redis write failed for user %s: %s", user_id, exc)

except ImportError as exc:
    log.warning("PyFlink not installed (%s). Processor will only run inside Docker.", exc)

    class UserFeatureProcessor:  # type: ignore[no-redef]
        pass


# ── Pipeline definition ───────────────────────────────────────────────────────


def build_pipeline(env) -> None:  # type: ignore[no-untyped-def]
    """Wire up: Kafka source → filter clicks → keyBy user_id → UserFeatureProcessor."""
    from pyflink.common import SimpleStringSchema, WatermarkStrategy
    from pyflink.datastream.connectors.kafka import KafkaOffsetsInitializer, KafkaSource

    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(REDPANDA_BROKERS)
        .set_topics("events")
        .set_group_id("feature-processor")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    stream = env.from_source(kafka_source, WatermarkStrategy.no_watermarks(), "redpanda-events")

    (
        stream.filter(lambda s: json.loads(s).get("event_type") == "click")
        .key_by(lambda s: str(json.loads(s)["user_id"]))
        .process(UserFeatureProcessor())
    )

    env.execute("user-feature-processor")


if __name__ == "__main__":
    from pyflink.datastream import StreamExecutionEnvironment

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.enable_checkpointing(60_000)  # checkpoint every 60s for fault tolerance

    log.info("Starting user-feature-processor (broker=%s, redis=%s)", REDPANDA_BROKERS, REDIS_HOST)
    build_pipeline(env)
