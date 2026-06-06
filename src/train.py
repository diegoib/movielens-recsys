"""Training entrypoint for the two-tower recommender model.

Usage:
    uv run python src/train.py                          # full training
    uv run python src/train.py --max_rows 10000 --max_epochs 2 --fast_dev_run
    uv run python src/train.py --mlp_hidden_dims 128 64 --embedding_dim 32

GCS paths accepted for data_path, movies_path, and onnx_output (gs://bucket/path).
Set MLFLOW_TRACKING_URI to enable MLflow logging and model registration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import tyro
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger

from src.data.dataset import RecSysDataModule
from src.models.export_onnx import export_onnx
from src.models.lightning_module import TwoTowerLightningModule
from src.models.two_tower import MovieTower, TwoTowerModel, UserTower

# Env var overrides for Cloud Run Job (take precedence over CLI defaults)
_GCS_DATA = os.environ.get("GCS_DATA_PATH")
_GCS_MOVIES = os.environ.get("GCS_MOVIES_PATH")
_GCS_ONNX = os.environ.get("GCS_ONNX_OUTPUT")


@dataclass
class TrainConfig:
    # Data — accepts local paths or gs:// URIs
    data_path: str = _GCS_DATA or "data/processed/train_dataset.parquet"
    movies_path: str = _GCS_MOVIES or "data/raw/movies.csv"
    max_rows: int | None = None

    # Training
    batch_size: int = 1024
    lr: float = 1e-3
    max_epochs: int = 10
    fast_dev_run: bool = False
    num_workers: int = 4

    # Architecture
    embedding_dim: int = 64
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    output_dim: int = 64

    # Output — accepts local paths or gs:// URIs
    checkpoint_dir: str = "artifacts/checkpoints"
    onnx_output: str = _GCS_ONNX or "artifacts/models/model.onnx"
    export_onnx_after_train: bool = True


def main(cfg: TrainConfig) -> None:
    num_workers = 0 if cfg.fast_dev_run else cfg.num_workers

    dm = RecSysDataModule(
        data_path=cfg.data_path,
        movies_path=cfg.movies_path,
        batch_size=cfg.batch_size,
        max_rows=cfg.max_rows,
        num_workers=num_workers,
    )
    dm.setup()
    print(f"Vocab: {dm.n_users:,} users, {dm.n_movies:,} movies")

    hidden_dims = tuple(cfg.mlp_hidden_dims)
    model = TwoTowerModel(
        user_tower=UserTower(
            dm.n_users, cfg.embedding_dim, dm.user_behavior_dim, hidden_dims, cfg.output_dim
        ),
        movie_tower=MovieTower(
            dm.n_movies, cfg.embedding_dim, dm.movie_meta_dim, hidden_dims, cfg.output_dim
        ),
    )
    lit = TwoTowerLightningModule(model, lr=cfg.lr)

    if os.environ.get("MLFLOW_TRACKING_URI"):
        logger: CSVLogger | MLFlowLogger = MLFlowLogger(experiment_name="two-tower-recsys")
    else:
        logger = CSVLogger(save_dir="artifacts/logs")

    callbacks = []
    if not cfg.fast_dev_run:
        checkpoint_dir = cfg.checkpoint_dir
        if not str(checkpoint_dir).startswith("gs://"):
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best-{epoch:02d}-{val_auc:.4f}",
                monitor="val_auc",
                mode="max",
                save_top_k=1,
            )
        )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        fast_dev_run=cfg.fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    trainer.fit(lit, dm)

    if cfg.export_onnx_after_train and not cfg.fast_dev_run:
        output_dir = str(cfg.onnx_output).rsplit("/", 1)[0]
        export_onnx(lit, cfg.onnx_output, dm)
        _export_serving_artifacts(dm, cfg.data_path, cfg.movies_path, output_dir)
        _register_mlflow_model(trainer, cfg.onnx_output, output_dir)


def _register_mlflow_model(trainer: L.Trainer, onnx_output: str, output_dir: str) -> None:
    """Register ONNX + serving artifacts in MLflow (only when MLFLOW_TRACKING_URI is set)."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return

    import io
    import tempfile

    import fsspec
    import mlflow
    import mlflow.onnx
    import onnx as onnx_lib

    output_str = str(onnx_output)
    if output_str.startswith("gs://"):
        with fsspec.open(output_str, "rb") as f:
            model_proto = onnx_lib.load(io.BytesIO(f.read()))  # type: ignore[arg-type]
    else:
        model_proto = onnx_lib.load(output_str)

    run_id = trainer.logger.run_id  # type: ignore[union-attr]
    with mlflow.start_run(run_id=run_id):
        mlflow.onnx.log_model(
            model_proto,
            artifact_path="model",
            registered_model_name="two-tower-recsys",
        )
        # Log vocab and movie features alongside the model so they're versioned together
        for filename in ["vocab.json", "movie_features.parquet"]:
            src = f"{output_dir}/{filename}"
            if src.startswith("gs://"):
                with fsspec.open(src, "rb") as f:
                    data = f.read()  # type: ignore[union-attr]
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, filename)
                    with open(tmp_path, "wb") as tmp:
                        tmp.write(data)
                    mlflow.log_artifact(tmp_path, artifact_path="model")
            else:
                mlflow.log_artifact(src, artifact_path="model")
    print(f"Model registered in MLflow: two-tower-recsys (run_id={run_id})")


def _export_serving_artifacts(
    dm: RecSysDataModule,
    data_path: str,
    movies_path: str,
    output_dir: str,
) -> None:
    """Save vocab.json and movie_features.parquet alongside the ONNX model.

    Both files are needed by the serving layer to reconstruct the exact feature
    space and vocabulary indices that the model was trained with.
    """
    import json

    import fsspec
    import polars as pl

    # ── vocab.json ────────────────────────────────────────────────────────────
    vocab: dict = {
        "user_vocab": {str(k): v for k, v in dm.user_vocab.items()},
        "movie_vocab": {str(k): v for k, v in dm.movie_vocab.items()},
        "genre_index": dm._genre_index,
        "norm_stats": {k: list(v) for k, v in dm._norm_stats.items()},
        "user_behavior_dim": dm.user_behavior_dim,
        "movie_meta_dim": dm.movie_meta_dim,
    }
    vocab_path = f"{output_dir}/vocab.json"
    vocab_bytes = json.dumps(vocab).encode()
    if vocab_path.startswith("gs://"):
        with fsspec.open(vocab_path, "wb") as f:
            f.write(vocab_bytes)  # type: ignore[union-attr]
    else:
        import pathlib

        pathlib.Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(vocab_path).write_bytes(vocab_bytes)
    print(f"Saved vocab.json → {vocab_path}")

    # ── movie_features.parquet ────────────────────────────────────────────────
    # Read impressions parquet, deduplicate by movie_id to get one row per movie.
    # Join with movies.csv to add title. Add vocab index for each movie.
    imp_df = pl.read_parquet(data_path)
    movie_feat_cols = [
        "movie_id",
        "genres_vector",
        "genome_top20",
        "year",
        "popularity_last_30d",
        "avg_rating",
    ]
    movie_feats = imp_df.select(movie_feat_cols).unique(subset=["movie_id"])

    movies_df = pl.read_csv(movies_path).select(
        pl.col("movieId").cast(pl.Int64).alias("movie_id"),
        pl.col("title"),
    )
    movie_feats = movie_feats.join(movies_df, on="movie_id", how="left")

    movie_vocab_series = pl.Series(
        "movie_idx",
        [dm.movie_vocab.get(mid, 0) for mid in movie_feats["movie_id"].to_list()],
        dtype=pl.Int64,
    )
    movie_feats = movie_feats.with_columns(movie_vocab_series)

    feat_path = f"{output_dir}/movie_features.parquet"
    import io as _io

    feat_buf = _io.BytesIO()
    movie_feats.write_parquet(feat_buf)
    feat_bytes = feat_buf.getvalue()
    if feat_path.startswith("gs://"):
        with fsspec.open(feat_path, "wb") as f:
            f.write(feat_bytes)  # type: ignore[union-attr]
    else:
        import pathlib

        pathlib.Path(feat_path).write_bytes(feat_bytes)
    print(f"Saved movie_features.parquet ({len(movie_feats):,} movies) → {feat_path}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
