"""Lightning wrapper for TwoTowerModel: loss, metrics, optimizer, MLflow logging."""

from __future__ import annotations

import os

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.classification import BinaryAUROC

from src.models.two_tower import TwoTowerModel


class TwoTowerLightningModule(L.LightningModule):
    """Wraps TwoTowerModel with BCE training, BinaryAUROC validation, and Adam.

    MLflow logging is enabled automatically when MLFLOW_TRACKING_URI is set.
    Without it the module trains normally — metrics go to CSVLogger.
    """

    def __init__(self, model: TwoTowerModel, lr: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=["model"])

        self.val_auc = BinaryAUROC()

    def forward(
        self, user_ids: Tensor, behavior: Tensor, movie_ids: Tensor, meta: Tensor
    ) -> Tensor:
        return self.model(user_ids, behavior, movie_ids, meta)

    def _unpack(self, batch: tuple) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        user_ids, behavior, movie_ids, meta, labels = batch
        return user_ids, behavior.float(), movie_ids, meta.float(), labels.float()

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        user_ids, behavior, movie_ids, meta, labels = self._unpack(batch)
        scores = self.model(user_ids, behavior, movie_ids, meta)
        loss = F.binary_cross_entropy(scores, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        user_ids, behavior, movie_ids, meta, labels = self._unpack(batch)
        scores = self.model(user_ids, behavior, movie_ids, meta)
        loss = F.binary_cross_entropy(scores, labels)
        self.val_auc.update(scores, labels.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @property
    def mlflow_enabled(self) -> bool:
        return bool(os.environ.get("MLFLOW_TRACKING_URI"))
