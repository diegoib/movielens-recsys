"""RecSys DataModule: loads train_dataset.parquet and returns model-ready tensors."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.features.build_features import compute_genre_list

USER_BEHAVIOR_DIM: int = (
    41  # genre_affinity(19) + fav_multihot(19) + n_clicks(1) + avg_session(1) + days_since(1)
)
MOVIE_META_DIM: int = (
    42  # genres_vector(19) + genome_top20(20) + year(1) + popularity(1) + avg_rating(1)
)

_SCALAR_USER: tuple[str, ...] = (
    "n_clicks_last_7d",
    "avg_session_length",
    "days_since_last_activity",
)
_SCALAR_MOVIE: tuple[str, ...] = ("year", "popularity_last_30d", "avg_rating")


def _build_id_lookup(vocab: dict[int, int]) -> np.ndarray:
    """Build a dense lookup array: raw_id → vocab_index (0 = padding for unknowns)."""
    if not vocab:
        return np.zeros(1, dtype=np.int64)
    max_id = max(vocab.keys())
    table = np.zeros(max_id + 1, dtype=np.int64)
    for k, v in vocab.items():
        table[k] = v
    return table


class RecSysDataModule(L.LightningDataModule):
    """Loads train_dataset.parquet and exposes train/val/test DataLoaders.

    setup() must be called before accessing DataLoaders. It:
    - Builds user/movie ID vocabularies from the train split.
    - Computes z-score normalization stats from the train split only.
    - Converts all splits to TensorDatasets.
    """

    def __init__(
        self,
        data_path: str | Path,
        movies_path: str | Path,
        batch_size: int,
        max_rows: int | None = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.movies_path = movies_path
        self.batch_size = batch_size
        self.max_rows = max_rows
        self.num_workers = num_workers

        # Populated by setup()
        self.n_users: int = 0
        self.n_movies: int = 0
        self.user_vocab: dict[int, int] = {}
        self.movie_vocab: dict[int, int] = {}
        self._user_lookup: np.ndarray = np.zeros(1, dtype=np.int64)
        self._movie_lookup: np.ndarray = np.zeros(1, dtype=np.int64)
        self._norm_stats: dict[str, tuple[float, float]] = {}
        self._genre_index: dict[str, int] = {}
        self._n_genres: int = 0

        self._train_ds: TensorDataset | None = None
        self._val_ds: TensorDataset | None = None
        self._test_ds: TensorDataset | None = None

    @property
    def user_behavior_dim(self) -> int:
        """Actual behavior tensor width; valid after setup()."""
        return self._n_genres * 2 + 3

    @property
    def movie_meta_dim(self) -> int:
        """Actual movie-meta tensor width; valid after setup()."""
        return self._n_genres + 20 + 3

    def setup(self, stage: str | None = None) -> None:
        df = pl.read_parquet(self.data_path)
        if self.max_rows is not None:
            df = df.head(self.max_rows)

        train_df = df.filter(pl.col("split") == "train")
        val_df = df.filter(pl.col("split") == "val")
        test_df = df.filter(pl.col("split") == "test")

        # Genre vocabulary (for favorite_genres → multi-hot encoding)
        movies_df = pl.read_csv(self.movies_path)
        all_genres = compute_genre_list(movies_df)
        self._genre_index = {g: i for i, g in enumerate(all_genres)}
        self._n_genres = len(all_genres)

        # ID vocabularies from train split; unseen IDs → index 0 (padding_idx)
        train_user_ids = sorted(train_df["user_id"].unique().to_list())
        train_movie_ids = sorted(train_df["movie_id"].unique().to_list())
        self.user_vocab = {uid: idx + 1 for idx, uid in enumerate(train_user_ids)}
        self.movie_vocab = {mid: idx + 1 for idx, mid in enumerate(train_movie_ids)}
        self.n_users = len(train_user_ids)
        self.n_movies = len(train_movie_ids)
        self._user_lookup = _build_id_lookup(self.user_vocab)
        self._movie_lookup = _build_id_lookup(self.movie_vocab)

        # Z-score statistics from train split only (prevents data leakage into val/test)
        self._norm_stats = self._compute_norm_stats(train_df)

        self._train_ds = self._encode_split(train_df)
        self._val_ds = self._encode_split(val_df)
        self._test_ds = self._encode_split(test_df)

    def _compute_norm_stats(self, train_df: pl.DataFrame) -> dict[str, tuple[float, float]]:
        stats: dict[str, tuple[float, float]] = {}
        for col in _SCALAR_USER + _SCALAR_MOVIE:
            series = train_df[col].drop_nulls().cast(pl.Float64)
            mean = float(series.mean() or 0.0)
            std = max(float(series.std() or 1.0), 1e-8)
            stats[col] = (mean, std)
        return stats

    def _encode_split(self, df: pl.DataFrame) -> TensorDataset:
        n = len(df)

        # ── IDs ─────────────────────────────────────────────────────────────
        raw_u = df["user_id"].to_numpy()
        user_ids = np.zeros(n, dtype=np.int64)
        mask_u = raw_u < len(self._user_lookup)
        user_ids[mask_u] = self._user_lookup[raw_u[mask_u]]

        raw_m = df["movie_id"].to_numpy()
        movie_ids = np.zeros(n, dtype=np.int64)
        mask_m = raw_m < len(self._movie_lookup)
        movie_ids[mask_m] = self._movie_lookup[raw_m[mask_m]]

        # ── User behavior [N, 41] ────────────────────────────────────────────
        # genre_affinity_last_7d [N, 19]
        genre_aff = (
            df["genre_affinity_last_7d"].explode().to_numpy().reshape(n, -1).astype(np.float32)
        )

        # favorite_genres → multi-hot [N, 19]  (vectorized via Polars join)
        fav_multihot = np.zeros((n, self._n_genres), dtype=np.float32)
        genre_idx_df = pl.DataFrame(
            {
                "favorite_genres": list(self._genre_index.keys()),
                "col_idx": list(self._genre_index.values()),
            }
        )
        fav_df = (
            df.with_row_index("row_idx")
            .select(["row_idx", "favorite_genres"])
            .explode("favorite_genres")
            .filter(pl.col("favorite_genres").is_not_null())
            .join(genre_idx_df, on="favorite_genres", how="inner")
        )
        if fav_df.height > 0:
            fav_multihot[fav_df["row_idx"].to_numpy(), fav_df["col_idx"].to_numpy()] = 1.0

        # scalar user features [N, 3]: n_clicks, avg_session, days_since
        user_scalars = self._extract_norm_scalars(df, _SCALAR_USER)

        # ── Movie meta [N, 42] ───────────────────────────────────────────────
        # genres_vector [N, 19]
        genres_vec = df["genres_vector"].explode().to_numpy().reshape(n, -1).astype(np.float32)

        # genome_top20 [N, 20] — null rows → zero vectors
        genome_np = np.zeros((n, 20), dtype=np.float32)
        null_mask = df["genome_top20"].is_null().to_numpy()
        n_non_null = int((~null_mask).sum())
        if n_non_null > 0:
            flat = df["genome_top20"].drop_nulls().explode().to_numpy().astype(np.float32)
            genome_np[~null_mask] = flat.reshape(n_non_null, -1)

        # scalar movie features [N, 3]: year, popularity, avg_rating
        movie_scalars = self._extract_norm_scalars(df, _SCALAR_MOVIE)

        # ── Assemble tensors ─────────────────────────────────────────────────
        user_behavior = np.concatenate([genre_aff, fav_multihot, user_scalars], axis=1)
        movie_meta = np.concatenate([genres_vec, genome_np, movie_scalars], axis=1)
        labels = df["label"].cast(pl.Float32).to_numpy().astype(np.float32)

        return TensorDataset(
            torch.from_numpy(user_ids),
            torch.from_numpy(user_behavior),
            torch.from_numpy(movie_ids),
            torch.from_numpy(movie_meta),
            torch.from_numpy(labels),
        )

    def _extract_norm_scalars(self, df: pl.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
        """Z-score normalize scalar columns; null values → train mean → 0.0 after norm."""
        parts = []
        for col in cols:
            mean, std = self._norm_stats[col]
            arr = df[col].cast(pl.Float64).fill_null(mean).to_numpy().astype(np.float32)
            parts.append(((arr - mean) / std).astype(np.float32).reshape(-1, 1))
        return np.concatenate(parts, axis=1)

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
