"""Build retraining dataset from persisted inference logs and click events.

Pipeline:
  1. Read inference logs (GCS) — one row per recommendation shown, with user features
     captured at prediction time (exact snapshot, not recomputed).
  2. Read click events (GCS) — ground truth labels.
  3. _assign_labels: equi-join on (recommendation_id, movie_id) → label 0 or 1.
  4. Expand user_features JSON column → individual feature columns.
  5. Join with movie_features.parquet for static movie metadata.
  6. Write output Parquet with the same schema as train_dataset.parquet.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from datetime import datetime

import polars as pl

log = logging.getLogger(__name__)


@dataclasses.dataclass
class BuildConfig:
    gcs_inference_path: str = "/tmp/inference-logs"
    gcs_events_path: str = "/tmp/events"
    gcs_movies_path: str = "artifacts/models/movie_features.parquet"
    output_path: str = "/tmp/retrain.parquet"
    since_date: str = "2023-01-01"


# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_feature(s: str | None, field: str, default: object) -> object:
    """Extract one field from a JSON-serialized user_features string."""
    try:
        d = json.loads(s) if s else {}
        val = d.get(field, default)
        return val if val is not None else default
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def _read_parquet_since(base_path: str, since_date: str) -> pl.DataFrame:
    """Read all dt=YYYY-MM-DD partitioned Parquet files under base_path."""
    try:
        df = pl.scan_parquet(f"{base_path}/dt=*/*.parquet").collect()
    except Exception as exc:
        log.warning("No data at %s: %s", base_path, exc)
        return pl.DataFrame()
    since_ts = int(datetime.fromisoformat(since_date).timestamp())
    if "timestamp" in df.columns:
        df = df.filter(pl.col("timestamp") >= since_ts)
    return df


def _expand_user_features(df: pl.DataFrame) -> pl.DataFrame:
    """Parse user_features JSON string column into individual feature columns.

    All Redis values are JSON-serializable: scalars stored as plain JSON numbers,
    lists stored as JSON arrays. json.loads handles both cases.
    """
    N_GENRES = 19
    return df.with_columns(
        [
            pl.col("user_features")
            .map_elements(
                lambda s: _parse_feature(s, "genre_affinity_last_7d", [0.0] * N_GENRES),
                return_dtype=pl.List(pl.Float64),
            )
            .alias("genre_affinity_last_7d"),
            pl.col("user_features")
            .map_elements(
                lambda s: _parse_feature(s, "n_clicks_last_7d", 0),
                return_dtype=pl.Int32,
            )
            .alias("n_clicks_last_7d"),
            pl.col("user_features")
            .map_elements(
                lambda s: _parse_feature(s, "days_since_last_activity", 0.0),
                return_dtype=pl.Float64,
            )
            .alias("days_since_last_activity"),
            pl.col("user_features")
            .map_elements(
                lambda s: _parse_feature(s, "avg_session_length", 0.0),
                return_dtype=pl.Float64,
            )
            .alias("avg_session_length"),
            pl.col("user_features")
            .map_elements(
                lambda s: _parse_feature(s, "favorite_genres", []),
                return_dtype=pl.List(pl.String),
            )
            .alias("favorite_genres"),
        ]
    ).drop("user_features")


def _assign_labels(inference_df: pl.DataFrame, clicks_df: pl.DataFrame) -> pl.DataFrame:
    """Equi-join inference log with click events to assign binary training labels.

    Args:
        inference_df: One row per (recommendation_id, movie_id) pair shown to a user.
                      Columns include at minimum: recommendation_id, movie_id.
        clicks_df:    Click events filtered to event_type == "click".
                      Columns include at minimum: recommendation_id, movie_id.

    Returns:
        inference_df with a new integer "label" column (1 = clicked, 0 = not clicked).

    The join is exact — no time-window approximation. recommendation_id links a
    specific recommendation request to the events it triggered in the same session.
    """
    click_keys = clicks_df.select(["recommendation_id", "movie_id"]).with_columns(
        pl.lit(1).alias("label")
    )
    return inference_df.join(
        click_keys, on=["recommendation_id", "movie_id"], how="left"
    ).with_columns(pl.col("label").fill_null(0).cast(pl.Int32))


# ── main pipeline ─────────────────────────────────────────────────────────────


def build(config: BuildConfig) -> None:
    log.info(
        "Reading inference logs from %s (since %s)", config.gcs_inference_path, config.since_date
    )
    inference_df = _read_parquet_since(config.gcs_inference_path, config.since_date)
    if inference_df.is_empty():
        log.error("No inference logs found — run the serving stack to generate data.")
        return

    log.info("%d inference log rows loaded", len(inference_df))

    log.info("Reading click events from %s", config.gcs_events_path)
    events_df = _read_parquet_since(config.gcs_events_path, config.since_date)
    clicks_df = (
        events_df.filter(pl.col("event_type") == "click")
        if not events_df.is_empty()
        else pl.DataFrame(schema={"recommendation_id": pl.String, "movie_id": pl.Int64})
    )
    log.info("%d click events loaded", len(clicks_df))

    labeled_df = _assign_labels(inference_df, clicks_df)

    log.info("Expanding user features")
    labeled_df = _expand_user_features(labeled_df)

    log.info("Joining movie features from %s", config.gcs_movies_path)
    movies_df = pl.read_parquet(config.gcs_movies_path).select(
        ["movie_id", "genres_vector", "genome_top20", "year", "popularity_last_30d", "avg_rating"]
    )
    final_df = labeled_df.join(movies_df, on="movie_id", how="left")

    # Add columns required by train_dataset.parquet schema
    n = len(final_df)
    final_df = final_df.with_columns(
        [
            pl.Series("event_id", [str(uuid.uuid4()) for _ in range(n)]),
            pl.Series("session_id", [str(uuid.uuid4()) for _ in range(n)]),
            pl.lit("impression").alias("event_type"),
            pl.lit("train").alias("split"),
        ]
    )

    drop_cols = [
        c
        for c in ("recommendation_id", "score", "position", "model_version")
        if c in final_df.columns
    ]
    final_df = final_df.drop(drop_cols)

    log.info("Writing %d rows → %s", n, config.output_path)
    final_df.write_parquet(config.output_path)


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(tyro.cli(BuildConfig))
