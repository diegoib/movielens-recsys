"""events_sink — Kafka consumer that persists events and inference logs to GCS Parquet.

Consumes two topics from RedPanda and writes partitioned Parquet files:
  events          → gs://.../events/dt=YYYY-MM-DD/part-NNNN.parquet
  model-predictions → gs://.../inference-logs/dt=YYYY-MM-DD/part-NNNN.parquet

Each file is one batch (batch_size records OR flush_interval_seconds, whichever first).
Polars + gcsfs transparently handle both gs:// and local paths.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


@dataclasses.dataclass
class SinkConfig:
    redpanda_brokers: str = "localhost:9092"
    gcs_events_path: str = "/tmp/events"
    gcs_inference_path: str = "/tmp/inference-logs"
    batch_size: int = 500
    flush_interval_seconds: float = 60.0


def _parquet_path(base: str, part: int) -> str:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"{base}/dt={today}/part-{part:04d}.parquet"


def _write_batch(records: list[dict], base: str, part: int) -> None:
    path = _parquet_path(base, part)
    df = pl.DataFrame(records, infer_schema_length=None)
    if path.startswith("gs://"):
        df.write_parquet(path)
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)
    log.info("Wrote %d records → %s", len(records), path)


def run(config: SinkConfig) -> None:
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        "events",
        "model-predictions",
        bootstrap_servers=config.redpanda_brokers,
        group_id="events-sink",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode()),
        consumer_timeout_ms=int(config.flush_interval_seconds * 1000),
    )

    events_batch: list[dict] = []
    inference_batch: list[dict] = []
    events_part = 0
    inference_part = 0

    log.info("Sink started — brokers=%s", config.redpanda_brokers)

    while True:
        # Consume until batch_size reached or consumer_timeout_ms fires (StopIteration)
        for msg in consumer:
            if msg.topic == "events":
                events_batch.append(msg.value)
            else:
                inference_batch.append(msg.value)

            if len(events_batch) >= config.batch_size:
                _write_batch(events_batch, config.gcs_events_path, events_part)
                events_part += 1
                events_batch = []
            if len(inference_batch) >= config.batch_size:
                _write_batch(inference_batch, config.gcs_inference_path, inference_part)
                inference_part += 1
                inference_batch = []

        # Flush whatever remains after the polling timeout
        if events_batch:
            _write_batch(events_batch, config.gcs_events_path, events_part)
            events_part += 1
            events_batch = []
        if inference_batch:
            _write_batch(inference_batch, config.gcs_inference_path, inference_part)
            inference_part += 1
            inference_batch = []


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(tyro.cli(SinkConfig))
