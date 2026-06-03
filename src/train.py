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
        export_onnx(lit, cfg.onnx_output, dm)
        _register_mlflow_model(trainer, cfg.onnx_output)


def _register_mlflow_model(trainer: L.Trainer, onnx_output: str) -> None:
    """Register ONNX model in MLflow Model Registry (only when MLFLOW_TRACKING_URI is set)."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return

    import io

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
    print(f"Model registered in MLflow: two-tower-recsys (run_id={run_id})")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
