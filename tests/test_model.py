"""Tests for Phase 4: TwoTowerModel architecture, ONNX export, and RecSysDataModule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from src.data.dataset import RecSysDataModule
from src.models.export_onnx import export_onnx, precompute_movie_embeddings
from src.models.lightning_module import TwoTowerLightningModule
from src.models.two_tower import MovieTower, TwoTowerModel, UserTower

# ── Small constants for architecture tests (independent of real data) ─────────
_EMBED = 8
_HIDDEN = (16,)
_OUT = 8
_B = 4
_BEHAVIOR = 41  # matches full dataset (19 genres)
_META = 42


def _make_model(n_users: int = 10, n_movies: int = 20) -> TwoTowerModel:
    return TwoTowerModel(
        user_tower=UserTower(n_users, _EMBED, _BEHAVIOR, _HIDDEN, _OUT),
        movie_tower=MovieTower(n_movies, _EMBED, _META, _HIDDEN, _OUT),
    )


def _random_batch(n: int = _B) -> tuple[torch.Tensor, ...]:
    return (
        torch.randint(1, 11, (n,)),
        torch.randn(n, _BEHAVIOR),
        torch.randint(1, 21, (n,)),
        torch.randn(n, _META),
    )


# ── Architecture ──────────────────────────────────────────────────────────────


def test_user_tower_shape() -> None:
    tower = UserTower(10, _EMBED, _BEHAVIOR, _HIDDEN, _OUT)
    out = tower(torch.randint(1, 11, (_B,)), torch.randn(_B, _BEHAVIOR))
    assert out.shape == (_B, _OUT)


def test_movie_tower_shape() -> None:
    tower = MovieTower(20, _EMBED, _META, _HIDDEN, _OUT)
    out = tower(torch.randint(1, 21, (_B,)), torch.randn(_B, _META))
    assert out.shape == (_B, _OUT)


def test_two_tower_forward() -> None:
    model = _make_model()
    scores = model(*_random_batch())
    assert scores.shape == (_B,)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


# ── ONNX roundtrip ────────────────────────────────────────────────────────────


def test_onnx_roundtrip(tmp_path: Path) -> None:
    model = _make_model().eval()
    lit = TwoTowerLightningModule(model)

    class _FakeDM:
        n_users = 10
        n_movies = 20
        user_behavior_dim = _BEHAVIOR
        movie_meta_dim = _META

    export_onnx(lit, tmp_path / "user_tower.onnx", _FakeDM())  # type: ignore[arg-type]
    assert (tmp_path / "user_tower.onnx").exists()


def test_precompute_movie_embeddings() -> None:
    model = _make_model()
    movie_idxs = np.arange(1, 6, dtype=np.int64)
    movie_metas = np.random.default_rng(0).standard_normal((5, _META)).astype(np.float32)

    embeddings = precompute_movie_embeddings(model, movie_idxs, movie_metas)

    assert embeddings.shape == (5, _OUT)
    with torch.no_grad():
        expected = model.encode_movie(
            torch.from_numpy(movie_idxs), torch.from_numpy(movie_metas)
        ).numpy()
    assert np.allclose(embeddings, expected, atol=1e-5)


# ── DataModule ────────────────────────────────────────────────────────────────

_N_GENRES_FAKE = 2  # "Action", "Comedy"
_N_ROWS = 30  # 20 train / 5 val / 5 test


def _write_movies_csv(path: Path) -> None:
    path.write_text(
        "movieId,title,genres\n"
        "1,Movie A (2010),Action\n"
        "2,Movie B (2015),Comedy\n"
        "3,Movie C (2020),Action|Comedy\n"
    )


def _write_fake_parquet(path: Path) -> None:
    rng = np.random.default_rng(42)
    n = _N_ROWS
    splits = ["train"] * 20 + ["val"] * 5 + ["test"] * 5

    affinity = [rng.random(_N_GENRES_FAKE).astype(np.float32).tolist() for _ in range(n)]
    genres_vec = [rng.integers(0, 2, _N_GENRES_FAKE).astype(np.float32).tolist() for _ in range(n)]
    genome = [rng.random(20).astype(np.float32).tolist() if i % 3 != 0 else None for i in range(n)]
    fav = [["Action"] if i % 2 == 0 else ["Comedy", "Action"] for i in range(n)]

    pl.DataFrame(
        {
            "event_id": [f"e{i:04d}" for i in range(n)],
            "timestamp": list(range(1_000_000, 1_000_000 + n)),
            "user_id": [i % 5 + 1 for i in range(n)],
            "movie_id": [i % 3 + 1 for i in range(n)],
            "event_type": ["impression"] * n,
            "rating": [None] * n,
            "session_id": [f"s{i % 5}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "genre_affinity_last_7d": affinity,
            "n_clicks_last_7d": [i % 5 for i in range(n)],
            "days_since_last_activity": [float(i) if i > 0 else None for i in range(n)],
            "avg_session_length": [2.0] * n,
            "favorite_genres": fav,
            "popularity_last_30d": [i % 10 for i in range(n)],
            "avg_rating": [3.5 if i % 4 != 0 else None for i in range(n)],
            "genres_vector": genres_vec,
            "year": [2010 if i % 5 != 0 else None for i in range(n)],
            "genome_top20": genome,
            "split": splits,
        }
    ).write_parquet(path)


@pytest.fixture(scope="module")
def fake_dm(tmp_path_factory: pytest.TempPathFactory) -> RecSysDataModule:
    tmp = tmp_path_factory.mktemp("recsys_data")
    _write_movies_csv(tmp / "movies.csv")
    _write_fake_parquet(tmp / "train_dataset.parquet")
    dm = RecSysDataModule(
        data_path=tmp / "train_dataset.parquet",
        movies_path=tmp / "movies.csv",
        batch_size=8,
        num_workers=0,
    )
    dm.setup()
    return dm


def test_datamodule_setup(fake_dm: RecSysDataModule) -> None:
    assert fake_dm.n_users == 5
    assert fake_dm.n_movies == 3
    assert fake_dm._n_genres == _N_GENRES_FAKE

    # Dimensions are computed from actual genre count, not hardcoded
    assert fake_dm.user_behavior_dim == _N_GENRES_FAKE * 2 + 3  # 7
    assert fake_dm.movie_meta_dim == _N_GENRES_FAKE + 20 + 3  # 25

    assert fake_dm._train_ds is not None
    user_ids, behavior, movie_ids, meta, labels = fake_dm._train_ds.tensors
    assert user_ids.shape == (20,)
    assert behavior.shape == (20, fake_dm.user_behavior_dim)
    assert movie_ids.shape == (20,)
    assert meta.shape == (20, fake_dm.movie_meta_dim)
    assert labels.shape == (20,)

    # No NaNs after null-filling + normalization
    assert not behavior.isnan().any(), "NaN in user_behavior tensor"
    assert not meta.isnan().any(), "NaN in movie_meta tensor"
