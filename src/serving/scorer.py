"""ONNX-based scorer: loads model + vocab + movie features at startup."""

from __future__ import annotations

import io
import json

import numpy as np
import onnxruntime as ort
import polars as pl

from src.serving.candidates import MovieRecord, build_movie_meta, set_genre_index


class OnnxScorer:
    """Loads model artifacts from model_dir (local path or gs://) and scores (user, movie) pairs."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir.rstrip("/")

        vocab = self._load_json("vocab.json")
        self.user_vocab: dict[str, int] = vocab["user_vocab"]
        self.movie_vocab: dict[str, int] = vocab["movie_vocab"]
        self.genre_index: dict[str, int] = vocab["genre_index"]
        self.norm_stats: dict[str, tuple[float, float]] = {
            k: (v[0], v[1]) for k, v in vocab["norm_stats"].items()
        }
        self.user_behavior_dim: int = vocab["user_behavior_dim"]
        self.movie_meta_dim: int = vocab["movie_meta_dim"]
        self.n_genres: int = len(self.genre_index)

        set_genre_index(self.genre_index)

        model_bytes = self._load_bytes("model.onnx")
        self._session = ort.InferenceSession(model_bytes)

        self.movie_cache: dict[int, MovieRecord] = self._load_movie_cache()

    # ── public API ────────────────────────────────────────────────────────────

    def score(
        self,
        user_idx: int,
        user_behavior: np.ndarray,
        movie_ids: list[int],
    ) -> np.ndarray:
        """Return P(click) scores for each movie_id. Shape: [len(movie_ids)]."""
        n = len(movie_ids)
        movie_idxs = np.array(
            [self.movie_vocab.get(str(mid), 0) for mid in movie_ids], dtype=np.int64
        )
        movie_metas = np.stack(
            [build_movie_meta(self.movie_cache[mid], self.norm_stats) for mid in movie_ids]
        )
        user_ids_arr = np.full(n, user_idx, dtype=np.int64)
        user_behavior_arr = np.tile(user_behavior, (n, 1)).astype(np.float32)

        outputs = self._session.run(
            ["score"],
            {
                "user_ids": user_ids_arr,
                "user_behavior": user_behavior_arr,
                "movie_ids": movie_idxs,
                "movie_meta": movie_metas,
            },
        )
        return outputs[0]

    def build_user_behavior(self, user_data: dict) -> np.ndarray:
        """Build user_behavior vector [user_behavior_dim] from Redis feature dict.

        If user_data is empty (cold start), returns a zero vector.
        """
        genre_names = [g for g, _ in sorted(self.genre_index.items(), key=lambda x: x[1])]

        # genre_affinity_last_7d [n_genres]
        raw_affinity = user_data.get("genre_affinity_last_7d", [])
        if isinstance(raw_affinity, str):
            import ast

            raw_affinity = ast.literal_eval(raw_affinity)
        affinity = np.array(raw_affinity or [0.0] * self.n_genres, dtype=np.float32)
        if len(affinity) != self.n_genres:
            affinity = np.zeros(self.n_genres, dtype=np.float32)

        # favorite_genres → multi-hot [n_genres]
        fav_raw = user_data.get("favorite_genres", [])
        if isinstance(fav_raw, str):
            import ast

            fav_raw = ast.literal_eval(fav_raw)
        fav_set = set(fav_raw or [])
        fav_multihot = np.array(
            [1.0 if g in fav_set else 0.0 for g in genre_names], dtype=np.float32
        )

        # scalar features [3]: n_clicks, avg_session, days_since
        def _norm(val: float | None, key: str) -> float:
            mean, std = self.norm_stats[key]
            v = float(val) if val is not None else mean
            return (v - mean) / std

        scalars = np.array(
            [
                _norm(user_data.get("n_clicks_last_7d"), "n_clicks_last_7d"),
                _norm(user_data.get("avg_session_length"), "avg_session_length"),
                _norm(user_data.get("days_since_last_activity"), "days_since_last_activity"),
            ],
            dtype=np.float32,
        )
        return np.concatenate([affinity, fav_multihot, scalars])

    def user_idx(self, user_id: int) -> int:
        return self.user_vocab.get(str(user_id), 0)

    # ── loading helpers ───────────────────────────────────────────────────────

    def _load_bytes(self, filename: str) -> bytes:
        path = f"{self.model_dir}/{filename}"
        if path.startswith("gs://"):
            import fsspec

            with fsspec.open(path, "rb") as f:
                return f.read()  # type: ignore[union-attr]
        return open(path, "rb").read()

    def _load_json(self, filename: str) -> dict:
        return json.loads(self._load_bytes(filename).decode())

    def _load_movie_cache(self) -> dict[int, MovieRecord]:
        parquet_bytes = self._load_bytes("movie_features.parquet")
        df = pl.read_parquet(io.BytesIO(parquet_bytes))
        cache: dict[int, MovieRecord] = {}
        for row in df.iter_rows(named=True):
            cache[row["movie_id"]] = MovieRecord(
                movie_id=row["movie_id"],
                movie_idx=row["movie_idx"],
                title=row["title"] or "",
                genres_vector=row["genres_vector"] or [],
                genome_top20=row["genome_top20"],
                year=row["year"],
                popularity_last_30d=row["popularity_last_30d"] or 0,
                avg_rating=row["avg_rating"],
            )
        return cache
