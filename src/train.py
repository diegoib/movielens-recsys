"""Training entrypoint for the two-tower recommender model.

Usage:
    uv run python src/train.py                          # full training
    uv run python src/train.py --max_rows 10000 --max_epochs 2 --fast_dev_run True
    uv run python src/train.py --mlp_hidden_dims 128 64 --embedding_dim 32
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

_PROCESSED_DIR = Path("data/processed")
_RAW_DIR = Path("data/raw")


@dataclass
class TrainConfig:
    # Data
    data_path: Path = _PROCESSED_DIR / "train_dataset.parquet"
    movies_path: Path = _RAW_DIR / "movies.csv"
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

    # Output
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    onnx_output: Path = Path("artifacts/models/model.onnx")
    export_onnx_after_train: bool = True


def main(cfg: TrainConfig) -> None:
    # Disable multiprocessing workers for fast_dev_run to avoid subprocess overhead
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
        logger = MLFlowLogger(experiment_name="two-tower-recsys")
    else:
        logger = CSVLogger(save_dir="artifacts/logs")

    callbacks = []
    if not cfg.fast_dev_run:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
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


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
